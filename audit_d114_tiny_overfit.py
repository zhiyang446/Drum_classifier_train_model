# -*- coding: utf-8 -*-
"""D114：以固定 28 個 train windows 驗證 D89 LoRA 是否具有基本可學習性。"""

import argparse
import csv
import json
import os
from collections import Counter
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from audit_d58_drumsep_errors import prepare_output_dir
from run_egmd_round4_validation import match_events
from run_six_class_smoke import CHUNK_FRAMES, HOP_LENGTH, SR, build_window
from run_six_class_validation import LABELS, expected_events, local_maxima
from train_d77_fused_lora import (
    file_sha256,
    fused_logits,
    load_frozen_lora_model,
    load_parent_adapter,
)
from train_six_class_candidate import (
    build_schedule,
    gaussian_smooth_targets,
    schedule_positive_weights,
)


EXPECTED_HASHES = {
    'real_metadata': 'd46a31a349bae6c254340d5b1eba87f4b81f4e4cf0e8fe42a9381d99e0c5e726',
    'enst_metadata': '00fd7ccdc955298884bc230720708802e52d3dca662af585acbfabe02ce1560a',
    'd89_adapter': '552900cb8a056364dd3ce0b7d880fc4d36b54f7f65b712c68b3fd75d97410177',
    'd76_checkpoint': '93a72bf661815608dd1546cf3fa30dd56cd805334a5bb247bccc223d47ca742a',
    'd64_checkpoint': '803cb4405693e2d3f450bd345d6cdd3120f87f3f636a43c55ca90d9cdb9a4fd3',
}
CHECKPOINT_STEPS = (0, 50, 100, 150, 200)


def read_json(path):
    """讀取 UTF-8 JSON 輸入。"""
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def interleave_domains(real_schedule, enst_schedule):
    """將兩個相同形狀的排程逐列交錯，避免單一 domain 集中。"""
    if len(real_schedule) != len(enst_schedule):
        raise ValueError('D114 domain schedules must have identical lengths.')
    return [
        {**row, 'domain': domain}
        for pair in zip(real_schedule, enst_schedule)
        for row, domain in zip(pair, ('real_song', 'enst'))
    ]


def build_tiny_schedule(real_metadata, enst_metadata):
    """建立兩域各六類＋NEG 各二的固定 28-window 排程。"""
    if set(real_metadata) & set(enst_metadata):
        raise ValueError('D114 metadata keys overlap.')
    if any(item.get('split') != 'train' for item in real_metadata.values()):
        raise ValueError('D114 real-song metadata must be train-only.')
    if any(item.get('split') != 'train' for item in enst_metadata.values()):
        raise ValueError('D114 ENST metadata must be train-only.')
    if any(
        token in f'{key} {item.get("audio_path", "")}'.lower()
        for key, item in enst_metadata.items()
        for token in ('drummer_2', 'drummer_3')
    ):
        raise ValueError('D114 forbids ENST validation/test drummers.')
    real_schedule = build_schedule(real_metadata, 2, window_negative_from_train=True)
    enst_schedule = build_schedule(enst_metadata, 2, window_negative_from_train=True)
    schedule = interleave_domains(real_schedule, enst_schedule)
    expected = {label: 4 for label in [*LABELS, 'NEG']}
    counts = Counter(row['label'] for row in schedule)
    if len(schedule) != 28 or dict(counts) != expected:
        raise AssertionError(f'Unexpected D114 schedule: windows={len(schedule)}, counts={dict(counts)}')
    return schedule


def prepare_tensors(schedule, metadata):
    """一次建立固定特徵、平滑 target、窗口起點與真值，供 200 steps 重用。"""
    features, targets, starts, truths = [], [], [], []
    for row in schedule:
        item = metadata[row['key']]
        feature, onset, _, start_sec = build_window(
            item, row['anchor'], use_true_superflux=True, use_multi_log_mel=False,
            input_mode=item.get('input_mode', 'mix'),
        )
        features.append(feature)
        targets.append(onset)
        starts.append(start_sec)
        truths.append(expected_events(item, start_sec))
    target_tensor = gaussian_smooth_targets(torch.from_numpy(np.stack(targets)).float())
    return torch.from_numpy(np.stack(features)).float(), target_tensor, starts, truths


def load_d89(args, device):
    """載入 frozen D76／D64 與 D89 adapter，回傳唯一 560 個可訓練參數。"""
    d76_model = load_frozen_lora_model(args.d76_checkpoint, device, args.rank, args.alpha)
    d64_model = load_frozen_lora_model(args.d64_checkpoint, device, args.rank, args.alpha)
    adapter_args = SimpleNamespace(
        d76_checkpoint=args.d76_checkpoint,
        d64_checkpoint=args.d64_checkpoint,
        rank=args.rank,
        alpha=args.alpha,
    )
    load_parent_adapter(args.d89_adapter, adapter_args, d76_model, d64_model)
    trainable = list(d76_model.onset_head.down.parameters()) + list(d76_model.onset_head.up.parameters())
    trainable += list(d64_model.onset_head.down.parameters()) + list(d64_model.onset_head.up.parameters())
    trainable_count = sum(parameter.numel() for parameter in trainable)
    if trainable_count != 560:
        raise AssertionError(f'Expected 560 D89 LoRA parameters, got {trainable_count}.')
    return d76_model, d64_model, trainable, trainable_count


def domain_loss(d76_model, d64_model, features, targets, indices, pos_weight, device, batch_size):
    """以訓練相同 BCE 與 positive weights 計算指定 domain 的平均 loss。"""
    total_loss = total_elements = 0
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            batch_features = features[batch_indices].to(device)
            batch_targets = targets[batch_indices].to(device)
            logits = fused_logits(d76_model, d64_model, batch_features)
            total_loss += float(F.binary_cross_entropy_with_logits(
                logits, batch_targets, pos_weight=pos_weight, reduction='sum',
            ).cpu())
            total_elements += batch_targets.numel()
    return total_loss / total_elements


def event_metrics(d76_model, d64_model, features, truths, indices, device, batch_size):
    """在同一 tiny train windows 上用固定 `.50/.05s` 解碼並計算六類事件 F1。"""
    aggregate = {label: ([], []) for label in LABELS}
    window_seconds = CHUNK_FRAMES * HOP_LENGTH / float(SR)
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            probabilities = torch.sigmoid(
                fused_logits(d76_model, d64_model, features[batch_indices].to(device))
            ).cpu().numpy()
            for local_index, schedule_index in enumerate(batch_indices):
                predicted = local_maxima(probabilities[local_index])
                offset = schedule_index * (window_seconds + 1.0)
                for label in LABELS:
                    aggregate[label][0].extend(time + offset for time in truths[schedule_index][label])
                    aggregate[label][1].extend(time + offset for time in predicted[label])
    per_class = {}
    for label in LABELS:
        *_, f1 = match_events(*aggregate[label], 0.05)
        per_class[label] = float(f1)
    return float(np.mean(list(per_class.values()))), per_class


def evaluate_step(
    step, d76_model, d64_model, features, targets, truths, schedule,
    pos_weight, device, batch_size,
):
    """記錄單一 step 的 combined／real／ENST loss 與事件 F1。"""
    indices = list(range(len(schedule)))
    real_indices = [index for index, row in enumerate(schedule) if row['domain'] == 'real_song']
    enst_indices = [index for index, row in enumerate(schedule) if row['domain'] == 'enst']
    combined_macro, combined_per_class = event_metrics(
        d76_model, d64_model, features, truths, indices, device, batch_size,
    )
    real_macro, _ = event_metrics(
        d76_model, d64_model, features, truths, real_indices, device, batch_size,
    )
    enst_macro, _ = event_metrics(
        d76_model, d64_model, features, truths, enst_indices, device, batch_size,
    )
    row = {
        'step': step,
        'combined_loss': domain_loss(
            d76_model, d64_model, features, targets, indices, pos_weight, device, batch_size,
        ),
        'real_song_loss': domain_loss(
            d76_model, d64_model, features, targets, real_indices, pos_weight, device, batch_size,
        ),
        'enst_loss': domain_loss(
            d76_model, d64_model, features, targets, enst_indices, pos_weight, device, batch_size,
        ),
        'combined_macro_f1': combined_macro,
        'real_song_macro_f1': real_macro,
        'enst_macro_f1': enst_macro,
    }
    row.update({f'{label}_f1': combined_per_class[label] for label in LABELS})
    return row


def learnability_pass(row):
    """套用預先宣告的 tiny-set Macro 與逐類 F1 門檻。"""
    values = [row['combined_macro_f1'], *(row[f'{label}_f1'] for label in LABELS)]
    return (
        all(np.isfinite(value) for value in values)
        and row['combined_macro_f1'] >= 0.90
        and all(row[f'{label}_f1'] >= 0.80 for label in LABELS)
    )


def write_csv(path, rows):
    """寫出固定 learning curve CSV。"""
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(args):
    """執行 D114 200-step tiny-set 診斷訓練並寫出不可覆寫證據。"""
    paths = {
        'real_metadata': args.real_metadata,
        'enst_metadata': args.enst_metadata,
        'd89_adapter': args.d89_adapter,
        'd76_checkpoint': args.d76_checkpoint,
        'd64_checkpoint': args.d64_checkpoint,
    }
    for path in paths.values():
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    actual_hashes = {name: file_sha256(path) for name, path in paths.items()}
    if actual_hashes != EXPECTED_HASHES:
        raise ValueError(f'D114 locked input hash mismatch: {actual_hashes}')
    if (
        args.steps != 200 or args.batch_size != 4 or args.lr != 0.001
        or args.rank != 4 or args.alpha != 8.0 or args.seed != 1337
    ):
        raise ValueError('D114 steps/batch/lr/rank/alpha/seed are locked to 200/4/.001/4/8/1337.')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    real_metadata, enst_metadata = read_json(args.real_metadata), read_json(args.enst_metadata)
    schedule = build_tiny_schedule(real_metadata, enst_metadata)
    metadata = {**real_metadata, **enst_metadata}
    features, targets, starts, truths = prepare_tensors(schedule, metadata)
    positive_weights, event_counts = schedule_positive_weights(schedule, metadata)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d76_model, d64_model, trainable, trainable_count = load_d89(args, device)
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    pos_weight = torch.tensor(
        [positive_weights[label] for label in LABELS], device=device,
    ).view(1, 1, -1)

    curve = [evaluate_step(
        0, d76_model, d64_model, features, targets, truths, schedule,
        pos_weight, device, args.batch_size,
    )]
    for step in range(1, args.steps + 1):
        offset = ((step - 1) * args.batch_size) % len(schedule)
        batch_indices = [(offset + index) % len(schedule) for index in range(args.batch_size)]
        batch_features = features[batch_indices].to(device)
        batch_targets = targets[batch_indices].to(device)
        d76_model.eval()
        d64_model.eval()
        d76_model.onset_head.train()
        d64_model.onset_head.train()
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(
            fused_logits(d76_model, d64_model, batch_features),
            batch_targets,
            pos_weight=pos_weight,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Non-finite D114 loss at step {step}.')
        loss.backward()
        optimizer.step()
        if step in CHECKPOINT_STEPS:
            curve.append(evaluate_step(
                step, d76_model, d64_model, features, targets, truths, schedule,
                pos_weight, device, args.batch_size,
            ))

    baseline, final = curve[0], curve[-1]
    losses_decreased = (
        final['real_song_loss'] < baseline['real_song_loss']
        and final['enst_loss'] < baseline['enst_loss']
    )
    passed = learnability_pass(final) and losses_decreased
    output_dir = prepare_output_dir(args.output_dir)
    selection = [
        {
            **row,
            'group_id': str(metadata[row['key']].get('group_id') or row['key']),
            'audio_path': metadata[row['key']]['audio_path'],
            'window_start': starts[index],
        }
        for index, row in enumerate(schedule)
    ]
    summary = {
        'phase': 'D114',
        'status': 'tiny_set_learnable_not_candidate' if passed else 'capacity_blocked_stop',
        'device': str(device),
        'training_started': True,
        'diagnostic_only': True,
        'optimizer_steps': args.steps,
        'checkpoint_written': False,
        'validation_or_test_read': False,
        'replay_windows': 0,
        'windows': len(schedule),
        'windows_by_domain': dict(Counter(row['domain'] for row in schedule)),
        'windows_by_label': dict(Counter(row['label'] for row in schedule)),
        'trainable_parameters': trainable_count,
        'input_hashes': actual_hashes,
        'event_counts': event_counts,
        'positive_weights': positive_weights,
        'baseline': baseline,
        'final': final,
        'real_and_enst_loss_decreased': losses_decreased,
        'current_lora_can_learn_tiny_set': passed,
        'ready_for_ratio_candidate_proposal': passed,
        'architecture_unfreeze_authorized': False,
        'ready_for_six_class_release': False,
    }
    with open(os.path.join(output_dir, 'selected_windows.json'), 'w', encoding='utf-8') as handle:
        json.dump(selection, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    write_csv(os.path.join(output_dir, 'learning_curve.csv'), curve)
    with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def run_self_check():
    """驗證兩域交錯、28-window 配額與 learnability gate 邊界。"""
    real = [{'label': label, 'key': f'real_{label}_{index}', 'anchor': float(index)}
            for index in range(2) for label in [*LABELS, 'NEG']]
    enst = [{'label': label, 'key': f'enst_{label}_{index}', 'anchor': float(index)}
            for index in range(2) for label in [*LABELS, 'NEG']]
    mixed = interleave_domains(real, enst)
    assert len(mixed) == 28
    assert [row['domain'] for row in mixed[:4]] == ['real_song', 'enst', 'real_song', 'enst']
    passing = {'combined_macro_f1': 0.90, **{f'{label}_f1': 0.80 for label in LABELS}}
    assert learnability_pass(passing)
    assert not learnability_pass({**passing, 'RIDE_f1': 0.79})
    print('D114 self-check passed.')


def main():
    """解析 CLI 並執行 D114 tiny-set 診斷或 self-check。"""
    parser = argparse.ArgumentParser(description='Run the D114 D89 tiny-set LoRA learnability audit.')
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--real-metadata', default='real-song/d104_five_fold/fold_01/train_metadata.json')
    parser.add_argument('--enst-metadata', default='enst_d107/metadata_d107_train.json')
    parser.add_argument('--d89-adapter', default='validation_runs/d89_d82_tim_gm_lora_retry/d89_d82_tim_gm_lora_retry_adapter.pth')
    parser.add_argument('--d76-checkpoint', default='validation_runs/d76_crash_kd_retry_candidate/d76_crash_kd_retry_candidate.pth')
    parser.add_argument('--d64-checkpoint', default='validation_runs/d64_tom_competitor_candidate/d64_tom_competitor_candidate.pth')
    parser.add_argument('--output-dir', default='validation_runs/d114_tiny_overfit_audit')
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--alpha', type=float, default=8.0)
    parser.add_argument('--seed', type=int, default=1337)
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    audit(args)


if __name__ == '__main__':
    main()
