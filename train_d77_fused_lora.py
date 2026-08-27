# -*- coding: utf-8 -*-
"""D82：以 frozen D76/D64 logits 融合訓練最小 LoRA adapter 候選。"""

import argparse
import hashlib
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from audit_d67_d61_d64_tom_fusion import assert_same_selection, read_json
from model_conformer import ResidualDCNNDrumHybridConformer, load_hybrid_conformer_checkpoint
from run_six_class_smoke import build_window
from run_six_class_validation import (
    CHUNK_FRAMES,
    HOP_LENGTH,
    LABELS,
    LABEL_INDEX,
    SR,
    expected_events,
    load_fixed_windows,
    local_maxima,
    write_outputs,
)
from train_six_class_candidate import (
    batch_from_schedule,
    build_schedule,
    gaussian_smooth_targets,
    schedule_positive_weights,
)


TOM_INDEX = LABEL_INDEX['TOM']


class LoRAConv1x1(nn.Module):
    """中文註解：以低秩 1x1 Conv 修正 frozen onset head，保留原輸出介面。"""

    def __init__(self, base, rank=4, alpha=8.0):
        """中文註解：建立 A/B adapter；B 零初始化以保證初始輸出等同原 checkpoint。"""
        super().__init__()
        if not isinstance(base, nn.Conv1d) or base.kernel_size != (1,):
            raise TypeError('LoRAConv1x1 requires a Conv1d kernel_size=1 base head.')
        if rank <= 0:
            raise ValueError('rank must be positive.')
        self.base = base
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.down = nn.Conv1d(base.in_channels, rank, kernel_size=1, bias=False)
        self.up = nn.Conv1d(rank, base.out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=np.sqrt(5.0))
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        """中文註解：回傳原 onset logits 加上唯一可訓練的低秩修正量。"""
        return self.base(x) + self.scale * self.up(self.down(x))

    def adapter_state(self):
        """中文註解：只匯出可訓練 adapter，避免複製或覆寫基礎 checkpoint。"""
        return {
            'rank': self.rank,
            'scale': self.scale,
            'down.weight': self.down.weight.detach().cpu(),
            'up.weight': self.up.weight.detach().cpu(),
        }

    def load_adapter_state(self, state):
        """中文註解：嚴格載入既有 adapter，拒絕 rank、scale 或 tensor shape 不相容。"""
        if int(state['rank']) != self.rank or float(state['scale']) != self.scale:
            raise ValueError('LoRA adapter rank/scale mismatch.')
        tensors = (('down.weight', self.down.weight), ('up.weight', self.up.weight))
        with torch.no_grad():
            for key, target in tensors:
                source = state[key].to(device=target.device, dtype=target.dtype)
                if source.shape != target.shape:
                    raise ValueError(f'LoRA adapter tensor shape mismatch: {key}')
                target.copy_(source)


def fuse_tom_logits(five_class_logits, tom_logits):
    """中文註解：以 D64 TOM logits 精確替換 D76 TOM，其他五類完全保留 D76。"""
    if five_class_logits.shape != tom_logits.shape:
        raise ValueError('D76/D64 logits shapes differ.')
    if five_class_logits.shape[-1] != len(LABELS):
        raise ValueError('Fusion expects exactly six class logits.')
    return torch.cat((
        five_class_logits[..., :TOM_INDEX],
        tom_logits[..., TOM_INDEX:TOM_INDEX + 1],
        five_class_logits[..., TOM_INDEX + 1:],
    ), dim=-1)


def file_sha256(path):
    """中文註解：記錄不變基礎 checkpoint 的雜湊，讓 adapter 與來源權重可追溯。"""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_lora_model(checkpoint_path, device, rank, alpha):
    """中文註解：嚴格載入 D76/D64 後凍結全部原權重，只替換 onset head 為 LoRA wrapper。"""
    model = ResidualDCNNDrumHybridConformer(num_classes=len(LABELS)).to(device)
    load_hybrid_conformer_checkpoint(model, checkpoint_path, device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.onset_head = LoRAConv1x1(model.onset_head, rank=rank, alpha=alpha).to(device)
    model.eval()
    return model


def load_parent_adapter(path, args, d76_model, d64_model):
    """中文註解：驗證父 adapter 與兩個 base 雜湊後，續接其 LoRA 權重。"""
    payload = torch.load(path, map_location='cpu', weights_only=False)
    expected = {
        'base_d76_sha256': file_sha256(args.d76_checkpoint),
        'base_d64_sha256': file_sha256(args.d64_checkpoint),
    }
    for key, value in expected.items():
        if str(payload.get(key, '')).lower() != value.lower():
            raise ValueError(f'Parent adapter {key} mismatch.')
    if int(payload.get('rank', -1)) != args.rank or float(payload.get('alpha', -1.0)) != args.alpha:
        raise ValueError('Parent adapter rank/alpha mismatch.')
    d76_model.onset_head.load_adapter_state(payload['d76_onset_lora'])
    d64_model.onset_head.load_adapter_state(payload['d64_onset_lora'])
    return payload


def fused_logits(d76_model, d64_model, features):
    """中文註解：在 decoder 前融合兩份 onset logits，讓 BCE 可回傳到兩個 adapter。"""
    d76_onset, _ = d76_model(features)
    d64_onset, _ = d64_model(features)
    return fuse_tom_logits(d76_onset, d64_onset)


def assert_schedule_isolated(schedule, metadata, expected_windows=2800):
    """中文註解：拒絕任何非 train 視窗，避免封存 validation/test 混入 LoRA 更新。"""
    if len(schedule) != expected_windows:
        raise AssertionError(f'Expected {expected_windows} train windows, got {len(schedule)}.')
    leaked = [row['key'] for row in schedule if metadata[row['key']].get('split') != 'train']
    if leaked:
        raise AssertionError(f'Non-train rows entered schedule: {leaked[:3]}')


def validate_extra_schedule(schedule, metadata, per_class):
    """中文註解：驗證外部固定排程的形狀、欄位、train 隔離與逐類配額。"""
    expected_labels = [*LABELS, 'NEG']
    if not isinstance(schedule, list):
        raise TypeError('Extra schedule must be a JSON list.')
    for row in schedule:
        if not isinstance(row, dict) or not {'label', 'key', 'anchor'} <= set(row):
            raise ValueError('Extra schedule rows require label, key and anchor.')
        if row['label'] not in expected_labels or row['key'] not in metadata:
            raise ValueError(f'Invalid extra schedule row: {row}')
        if not np.isfinite(float(row['anchor'])):
            raise ValueError(f'Non-finite extra schedule anchor: {row}')
    counts = {label: sum(row['label'] == label for row in schedule) for label in expected_labels}
    if counts != {label: per_class for label in expected_labels}:
        raise AssertionError(f'Unexpected extra schedule label counts: {counts}')
    assert_schedule_isolated(schedule, metadata, len(expected_labels) * per_class)
    return schedule


def interleave_schedules(replay, extra):
    """中文註解：保留 replay 相對順序，將新資料均勻插入而非集中在 epoch 尾端。"""
    if not extra:
        return list(replay)
    interval = max(1, len(replay) // len(extra))
    output, extra_index = [], 0
    for replay_index, row in enumerate(replay, start=1):
        output.append(row)
        if replay_index % interval == 0 and extra_index < len(extra):
            output.append(extra[extra_index])
            extra_index += 1
    output.extend(extra[extra_index:])
    return output


def passes_parent_gate(parent_macro, parent_per_class, macro_f1, per_class):
    """中文註解：只有 Macro 嚴格提升且六類皆不退步，後代才可取代父研究基線。"""
    return macro_f1 > parent_macro and all(
        per_class[label] >= parent_per_class[label] for label in LABELS
    )


def evaluate_fixed_fusion(d76_model, d64_model, metadata, selection_path, output_dir):
    """中文註解：在固定 validation windows 量測 D77-style logits 融合。"""
    if os.path.exists(output_dir):
        raise FileExistsError(f'Validation output already exists: {output_dir}')
    windows = load_fixed_windows(metadata, selection_path, split='validation', per_class=8)
    device = next(d76_model.parameters()).device
    aggregate = {label: ([], []) for label in LABELS}
    selected_rows = []
    window_seconds = CHUNK_FRAMES * HOP_LENGTH / float(SR)
    d76_model.eval()
    d64_model.eval()
    with torch.no_grad():
        for window_index, selected in enumerate(windows):
            features, _, _, start_sec = build_window(
                selected['item'], selected['anchor'], use_true_superflux=True,
                use_multi_log_mel=False,
                input_mode=selected['item'].get('input_mode', 'drumsep-mix'),
            )
            feature_tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)
            probabilities = torch.sigmoid(fused_logits(d76_model, d64_model, feature_tensor)).squeeze(0).cpu().numpy()
            predicted = local_maxima(probabilities)
            expected = expected_events(selected['item'], start_sec)
            aggregate_offset = window_index * (window_seconds + 1.0)
            for label in LABELS:
                aggregate[label][0].extend(time + aggregate_offset for time in expected[label])
                aggregate[label][1].extend(time + aggregate_offset for time in predicted[label])
            selected_rows.append({
                'label': selected['label'], 'key': selected['key'], 'anchor': selected['anchor'],
                'window_start': start_sec, 'audio_path': selected['item']['audio_path'], 'split': 'validation',
                'aggregate_offset': aggregate_offset, 'architecture': 'dcnn-tcn-conformer',
                'feature_mode': 'true-superflux',
                'input_mode': selected['item'].get('input_mode', 'drumsep-mix'),
                'fusion_recipe': 'D76 LoRA KD/SD/HH/CRASH/RIDE + D64 LoRA TOM',
            })
    rows, gate = write_outputs(selected_rows, aggregate, output_dir)
    return float(gate['macro_f1']), {row['inst']: float(row['f1']) for row in rows}, gate


def save_adapter_candidate(path, args, d76_model, d64_model, best_epoch, macro_f1, per_class):
    """中文註解：只寫入 adapter state 與不可變來源資訊，禁止序列化完整 base model。"""
    payload = {
        'phase': args.phase,
        'base_d76_checkpoint': os.path.abspath(args.d76_checkpoint),
        'base_d76_sha256': file_sha256(args.d76_checkpoint),
        'base_d64_checkpoint': os.path.abspath(args.d64_checkpoint),
        'base_d64_sha256': file_sha256(args.d64_checkpoint),
        'rank': args.rank,
        'alpha': args.alpha,
        'best_epoch': best_epoch,
        'fixed_validation_macro_f1': macro_f1,
        'fixed_validation_per_class': per_class,
        'd76_onset_lora': d76_model.onset_head.adapter_state(),
        'd64_onset_lora': d64_model.onset_head.adapter_state(),
    }
    if args.init_adapter:
        payload['parent_adapter'] = os.path.abspath(args.init_adapter)
        payload['parent_adapter_sha256'] = file_sha256(args.init_adapter)
    torch.save(payload, path)


def run_self_check():
    """中文註解：驗證零初始化 LoRA 不改輸出，且 TOM 融合只替換指定欄位。"""
    torch.manual_seed(7)
    base = nn.Conv1d(3, len(LABELS), kernel_size=1)
    adapter = LoRAConv1x1(base, rank=2, alpha=4.0)
    features = torch.randn(2, 3, 5)
    assert torch.equal(base(features), adapter(features))
    with torch.no_grad():
        adapter.up.weight.fill_(0.25)
    assert not torch.equal(base(features), adapter(features))
    restored = LoRAConv1x1(nn.Conv1d(3, len(LABELS), kernel_size=1), rank=2, alpha=4.0)
    restored.base.load_state_dict(base.state_dict())
    restored.load_adapter_state(adapter.adapter_state())
    assert torch.equal(adapter(features), restored(features))
    d76 = torch.zeros(1, 4, len(LABELS))
    d64 = torch.ones(1, 4, len(LABELS))
    fused = fuse_tom_logits(d76, d64)
    assert torch.equal(fused[..., :TOM_INDEX], d76[..., :TOM_INDEX])
    assert torch.equal(fused[..., TOM_INDEX], d64[..., TOM_INDEX])
    assert torch.equal(fused[..., TOM_INDEX + 1:], d76[..., TOM_INDEX + 1:])
    replay = [{'key': f'old_{index}'} for index in range(6)]
    extra = [{'key': f'new_{index}'} for index in range(2)]
    mixed = interleave_schedules(replay, extra)
    assert len(mixed) == 8
    assert [row['key'] for row in mixed if row['key'].startswith('old_')] == [row['key'] for row in replay]
    baseline = {label: 0.5 for label in LABELS}
    assert passes_parent_gate(0.5, baseline, 0.51, baseline)
    assert not passes_parent_gate(0.5, baseline, 0.51, {**baseline, 'RIDE': 0.49})
    fixed_metadata = {'train': {'split': 'train'}}
    fixed_schedule = [
        {'label': label, 'key': 'train', 'anchor': float(index)}
        for index, label in enumerate([*LABELS, 'NEG'])
    ]
    assert validate_extra_schedule(fixed_schedule, fixed_metadata, 1) == fixed_schedule
    print('D82 self-check passed.')


def main():
    """中文註解：建立單一可重現 D82 候選，不覆寫現有模型或驗證輸出。"""
    parser = argparse.ArgumentParser(description='Train frozen D77 logits fusion with onset-head LoRA adapters.')
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--phase', default='D82')
    parser.add_argument('--candidate-name', default='d82_d77_fused_lora_adapter.pth')
    parser.add_argument('--metadata', default='mixed_d54_stem/metadata_d54.json')
    parser.add_argument('--extra-metadata', help='額外 train-only metadata；item 以自身 input_mode 覆寫 replay 模式')
    parser.add_argument('--extra-schedule', help='已稽核的固定 extra schedule JSON；必須搭配 extra-metadata')
    parser.add_argument('--extra-per-class', type=int, default=24)
    parser.add_argument('--init-adapter', help='嚴格續接既有 D82/D89 LoRA adapter')
    parser.add_argument('--d76-checkpoint', default='validation_runs/d76_crash_kd_retry_candidate/d76_crash_kd_retry_candidate.pth')
    parser.add_argument('--d64-checkpoint', default='validation_runs/d64_tom_competitor_candidate/d64_tom_competitor_candidate.pth')
    parser.add_argument('--d76-selection', default='validation_runs/d61_kd_negative_candidate/independent_validation/selected_windows.json')
    parser.add_argument('--d64-selection', default='validation_runs/d64_tom_competitor_candidate/independent_validation/selected_windows.json')
    parser.add_argument('--output-dir', default='validation_runs/d82_d77_fused_lora_candidate')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--alpha', type=float, default=8.0)
    parser.add_argument('--seed', type=int, default=1337)
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    if args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0 or args.lr <= 0:
        raise ValueError('epochs, patience, batch-size and lr must be positive.')
    if args.extra_per_class <= 0:
        raise ValueError('extra-per-class must be positive.')
    if os.path.exists(args.output_dir):
        raise FileExistsError(f'Output directory already exists: {args.output_dir}')
    for path in (args.metadata, args.d76_checkpoint, args.d64_checkpoint, args.d76_selection, args.d64_selection):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    for path in (args.extra_metadata, args.extra_schedule, args.init_adapter):
        if path and not os.path.isfile(path):
            raise FileNotFoundError(path)
    if args.extra_schedule and not args.extra_metadata:
        raise ValueError('--extra-schedule requires --extra-metadata.')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    d76_selection = read_json(args.d76_selection)
    d64_selection = read_json(args.d64_selection)
    assert_same_selection(d76_selection, d64_selection)
    metadata = read_json(args.metadata)
    replay_schedule = build_schedule(
        metadata, per_class=400, source_quota_profile='d37-real-first',
        negative_anchor_inst='KD', crash_kd_competitor=True,
    )
    assert_schedule_isolated(replay_schedule, metadata)
    extra_schedule = []
    if args.extra_metadata:
        extra_metadata = {
            key: {**item, 'input_mode': item.get('input_mode', 'mix')}
            for key, item in read_json(args.extra_metadata).items()
        }
        overlap = sorted(set(metadata) & set(extra_metadata))
        if overlap:
            raise ValueError(f'Extra metadata key collision: {overlap[:3]}')
        if args.extra_schedule:
            extra_schedule = validate_extra_schedule(
                read_json(args.extra_schedule), extra_metadata, args.extra_per_class,
            )
        else:
            extra_schedule = build_schedule(
                extra_metadata, per_class=args.extra_per_class, window_negative_from_train=True,
            )
            assert_schedule_isolated(extra_schedule, extra_metadata, 7 * args.extra_per_class)
        metadata.update(extra_metadata)
    schedule = interleave_schedules(replay_schedule, extra_schedule)
    positive_weights, event_counts = schedule_positive_weights(schedule, metadata)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d76_model = load_frozen_lora_model(args.d76_checkpoint, device, args.rank, args.alpha)
    d64_model = load_frozen_lora_model(args.d64_checkpoint, device, args.rank, args.alpha)
    parent_payload = None
    if args.init_adapter:
        parent_payload = load_parent_adapter(args.init_adapter, args, d76_model, d64_model)
    trainable = list(d76_model.onset_head.down.parameters()) + list(d76_model.onset_head.up.parameters())
    trainable += list(d64_model.onset_head.down.parameters()) + list(d64_model.onset_head.up.parameters())
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    pos_weight = torch.tensor([positive_weights[label] for label in LABELS], device=device).view(1, 1, -1)
    os.makedirs(args.output_dir)
    report = {
        'phase': args.phase, 'seed': args.seed, 'device': str(device), 'schedule_windows': len(schedule),
        'event_counts': event_counts, 'positive_weights': positive_weights, 'epochs': [],
        'base_d76_checkpoint': os.path.abspath(args.d76_checkpoint),
        'base_d64_checkpoint': os.path.abspath(args.d64_checkpoint),
        'selection_identical': True, 'release_status': 'research_only',
        'replay_windows': len(replay_schedule), 'extra_windows': len(extra_schedule),
        'extra_metadata': os.path.abspath(args.extra_metadata) if args.extra_metadata else None,
        'extra_schedule': os.path.abspath(args.extra_schedule) if args.extra_schedule else None,
        'extra_schedule_sha256': file_sha256(args.extra_schedule) if args.extra_schedule else None,
        'parent_adapter': os.path.abspath(args.init_adapter) if args.init_adapter else None,
    }
    parent_macro, parent_per_class = None, None
    if parent_payload is not None:
        parent_macro, parent_per_class, _ = evaluate_fixed_fusion(
            d76_model, d64_model, metadata, args.d76_selection,
            os.path.join(args.output_dir, 'parent_fixed_validation'),
        )
        expected_parent_macro = float(parent_payload['fixed_validation_macro_f1'])
        if abs(parent_macro - expected_parent_macro) > 1e-4:
            raise RuntimeError(
                f'Parent adapter baseline mismatch: expected {expected_parent_macro:.4f}, got {parent_macro:.4f}.'
            )
        report['parent_fixed_validation_macro_f1'] = parent_macro
        report['parent_fixed_validation_per_class'] = parent_per_class
    best_macro = parent_macro if parent_macro is not None else float('-inf')
    best_epoch, stale_epochs = (0 if parent_payload is not None else None), 0
    candidate_path = os.path.join(args.output_dir, args.candidate_name)
    for epoch in range(1, args.epochs + 1):
        d76_model.eval()
        d64_model.eval()
        d76_model.onset_head.train()
        d64_model.onset_head.train()
        losses = []
        for start in range(0, len(schedule), args.batch_size):
            features, onsets, _ = batch_from_schedule(
                schedule, metadata, start, args.batch_size, use_true_superflux=True, input_mode='drumsep-mix',
            )
            feature_tensor = torch.from_numpy(features).float().to(device)
            target_tensor = gaussian_smooth_targets(torch.from_numpy(onsets).float().to(device))
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(
                fused_logits(d76_model, d64_model, feature_tensor), target_tensor, pos_weight=pos_weight,
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_dir = os.path.join(args.output_dir, f'epoch_{epoch:02d}_fixed_validation')
        macro_f1, per_class, gate = evaluate_fixed_fusion(
            d76_model, d64_model, metadata, args.d76_selection, validation_dir,
        )
        epoch_report = {
            'epoch': epoch, 'mean_train_loss': float(np.mean(losses)), 'fixed_validation_macro_f1': macro_f1,
            'fixed_validation_per_class': per_class, 'fixed_validation_gate': gate,
        }
        report['epochs'].append(epoch_report)
        epoch_candidate = os.path.join(
            args.output_dir, f'{os.path.splitext(args.candidate_name)[0]}_epoch{epoch}.pth',
        )
        save_adapter_candidate(epoch_candidate, args, d76_model, d64_model, epoch, macro_f1, per_class)
        improves = (
            passes_parent_gate(parent_macro, parent_per_class, macro_f1, per_class)
            if parent_macro is not None else macro_f1 > best_macro
        )
        epoch_report['promotes_parent'] = improves
        if improves:
            best_macro, best_epoch, stale_epochs = macro_f1, epoch, 0
            save_adapter_candidate(candidate_path, args, d76_model, d64_model, best_epoch, macro_f1, per_class)
        else:
            stale_epochs += 1
        with open(os.path.join(args.output_dir, 'train_report.json'), 'w', encoding='utf-8') as handle:
            json.dump({**report, 'best_epoch': best_epoch, 'best_macro_f1': best_macro}, handle, indent=2)
        print(json.dumps(epoch_report, ensure_ascii=False))
        if stale_epochs >= args.patience:
            break
    print(json.dumps({
        'best_epoch': best_epoch, 'best_macro_f1': best_macro,
        'beats_d77_research_baseline': best_macro > 0.5386,
        'adapter_candidate': os.path.abspath(candidate_path) if os.path.isfile(candidate_path) else None,
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
