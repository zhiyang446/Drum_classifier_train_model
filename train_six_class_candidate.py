# -*- coding: utf-8 -*-
"""以固定 STAR train 視窗訓練一個獨立六類候選模型。"""
import argparse
import json
import os

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from model_dcnn import DCNNDrumTCN, ResidualDCNNDrumTCN, transfer_residual_state, transfer_symmetric_state
from model_conformer import ResidualDCNNDrumConformer, ResidualDCNNDrumHybridConformer, transfer_d3r_hybrid_state, transfer_d3r_state
from run_six_class_smoke import CHUNK_FRAMES, LABELS, SR, TARGET_SAMPLES, build_window, load_accompaniment, load_compatible_weights
from run_six_class_confusion import evaluate_confusion
from run_six_class_validation import evaluate_model
from train_star_smoke import freeze_batchnorm_stats
from train_phase2 import SymmetricDrumTCN, propagate_velocity_targets


def build_schedule(metadata, per_class, balance_rare_sources=False, negative_source=None):
    """中文註解：建立固定排程，並可指定 window-local 真實負樣本來源。"""
    selected_by_label = {}
    info_cache = {}
    half_window_seconds = TARGET_SAMPLES / float(SR) / 2.0
    for label in LABELS:
        candidates = []
        for key, item in metadata.items():
            if item.get('split') != 'train':
                continue
            path = item.get('audio_path')
            if path:
                if path not in info_cache:
                    info_cache[path] = sf.info(path)
                duration = info_cache[path].frames / float(info_cache[path].samplerate)
            else:
                duration = None
            for event in item.get('events', []):
                anchor = float(event['time'])
                if event.get('inst') == label and (duration is None or half_window_seconds <= anchor <= duration - half_window_seconds):
                    candidates.append((key, anchor))
        candidates.sort()
        if balance_rare_sources and label in ('TOM', 'CRASH', 'RIDE'):
            if per_class % 2:
                raise ValueError('--balance-rare-sources requires an even --per-class value.')
            egmd = [row for row in candidates if metadata[row[0]].get('source') == 'egmd_pitch_weighted']
            star = [row for row in candidates if metadata[row[0]].get('source') != 'egmd_pitch_weighted']
            quota = per_class // 2
            if len(star) < quota or len(egmd) < quota:
                raise ValueError(f'Only STAR={len(star)} E-GMD={len(egmd)} centered events for {label}, need {quota} each.')
            star = [star[index * len(star) // quota] for index in range(quota)]
            egmd = [egmd[index * len(egmd) // quota] for index in range(quota)]
            selected = [row for pair in zip(star, egmd) for row in pair]
        else:
            if len(candidates) < per_class:
                raise ValueError(f'Only {len(candidates)} centered train events for {label}, need {per_class}.')
            selected = [candidates[index * len(candidates) // per_class] for index in range(per_class)]
        selected_by_label[label] = [
            {'label': label, 'key': key, 'anchor': anchor} for key, anchor in selected
        ]

    # 中文註解：預設保留整首無 rare 的舊邏輯；opt-in 來源改用窗口內無 rare 的真實混音。
    neg_candidates = []
    for key, item in metadata.items():
        if negative_source:
            if item.get('split') != 'negative_train' or item.get('source') != negative_source:
                continue
        elif item.get('split') != 'train':
            continue
        path = item.get('audio_path')
        if path:
            if path not in info_cache:
                info_cache[path] = sf.info(path)
            duration = info_cache[path].frames / float(info_cache[path].samplerate)
        else:
            duration = None
        events = item.get('events', [])
        has_neg_rare = any(event.get('inst') in ('TOM', 'CRASH', 'RIDE') for event in events)
        for event in events:
            anchor = float(event['time'])
            if event.get('inst') not in ('KD', 'SD', 'HH') or (duration is not None and not half_window_seconds <= anchor <= duration - half_window_seconds):
                continue
            if negative_source:
                start, end = anchor - half_window_seconds, anchor + half_window_seconds
                if any(rare.get('inst') in ('TOM', 'CRASH', 'RIDE') and start <= float(rare['time']) < end for rare in events):
                    continue
            elif has_neg_rare:
                continue
            neg_candidates.append((key, anchor))
    neg_candidates.sort()
    if len(neg_candidates) < per_class:
        raise ValueError(f'Only {len(neg_candidates)} negative centered train events for source={negative_source}, need {per_class}.')
    selected_by_label['NEG'] = []
    for index in range(per_class):
        key, anchor = neg_candidates[index * len(neg_candidates) // per_class]
        selected_by_label['NEG'].append({'label': 'NEG', 'key': key, 'anchor': anchor})

    ALL_TRAIN_CLASSES = list(LABELS) + ['NEG']
    return [selected_by_label[cls][index] for index in range(per_class) for cls in ALL_TRAIN_CLASSES]


def freeze_for_head_training(model):
    """中文註解：只訓練新六類輸出頭，保護轉移而來的聲學特徵。"""
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith('onset_head') or name.startswith('velocity_head')


def batch_from_schedule(
    schedule, metadata, start, batch_size, accompaniment_pool=None,
    gain_range=(0.10, 0.30), use_true_superflux=False,
):
    """中文註解：依固定 schedule 建立一個不含測試資料的訓練 batch。"""
    features, onsets, velocities = [], [], []
    for row in schedule[start:start + batch_size]:
        item = metadata[row['key']]
        accompaniment = None
        gain = 0.0
        offset = 0
        if accompaniment_pool:
            accompaniment = accompaniment_pool[np.random.randint(len(accompaniment_pool))]
            gain = float(np.random.uniform(*gain_range))
            offset = int(np.random.randint(max(1, len(accompaniment) - TARGET_SAMPLES + 1)))
        feature, onset, velocity, _ = build_window(
            item, row['anchor'], accompaniment=accompaniment,
            accompaniment_gain=gain, accompaniment_offset=offset,
            use_true_superflux=use_true_superflux,
        )
        features.append(feature)
        onsets.append(onset)
        velocities.append(velocity)
    return np.stack(features), np.stack(onsets), np.stack(velocities)


def gaussian_smooth_targets(targets):
    """中文註解：對任意類別數套用既有三分類相同的五框 onset target 平滑。"""
    kernel = torch.tensor([0.05, 0.25, 1.0, 0.25, 0.05], dtype=targets.dtype, device=targets.device).view(1, 1, 5)
    channels = targets.transpose(1, 2)
    padded = F.pad(channels, (2, 2))
    smoothed = [F.conv1d(padded[:, index:index + 1], kernel) for index in range(channels.shape[1])]
    return torch.clamp(torch.cat(smoothed, dim=1).transpose(1, 2), 0.0, 1.0)


def apply_frequency_mask(features, max_bins):
    """中文註解：對每個 train sample 的兩通道套用相同連續 Mel 遮罩。"""
    if max_bins <= 0:
        return features
    if features.ndim != 4 or max_bins > features.shape[2]:
        raise ValueError('frequency mask requires [batch, channel, frequency, time] and max_bins <= frequency bins')
    for sample in features:
        width = int(np.random.randint(0, max_bins + 1))
        if width:
            start = int(np.random.randint(0, sample.shape[1] - width + 1))
            sample[:, start:start + width, :] = 0.0
    return features


def balanced_positive_weight(event_count, window_count):
    """中文註解：以反比密度的平方根平衡稀有類別，避免線性權重造成假陽性爆增。"""
    return float(np.sqrt(CHUNK_FRAMES / max(event_count / window_count, 1e-6)))


def schedule_positive_weights(schedule, metadata):
    """中文註解：依固定窗口內每類實際正事件密度計算平方根平衡 onset 權重。"""
    counts = {label: 0 for label in LABELS}
    info_cache = {}
    for row in schedule:
        item = metadata[row['key']]
        path = item['audio_path']
        if path not in info_cache:
            info_cache[path] = sf.info(path)
        info = info_cache[path]
        source_window_samples = int(round(TARGET_SAMPLES * info.samplerate / SR))
        start = max(0, int(row['anchor'] * info.samplerate) - source_window_samples // 2)
        start = min(start, max(0, info.frames - source_window_samples))
        start_sec = start / float(info.samplerate)
        end_sec = start_sec + TARGET_SAMPLES / float(SR)
        for event in item['events']:
            label = event.get('inst')
            if label in counts and start_sec <= float(event['time']) < end_sec:
                counts[label] += 1
    weights = {label: balanced_positive_weight(counts[label], len(schedule)) for label in LABELS}
    return weights, counts


def run_self_check():
    """中文註解：確認固定 schedule 會為每個六類標籤產生相同數量窗口。"""
    metadata = {}
    for index, label in enumerate(LABELS):
        metadata[f'case_{index}'] = {
            'split': 'train',
            'events': [{'inst': label, 'time': float(time)} for time in range(3)],
        }
    schedule = build_schedule(metadata, 2)
    expected_labels = set(LABELS) | {'NEG'}
    assert len(schedule) == len(expected_labels) * 2
    assert {row['label'] for row in schedule} == expected_labels
    assert [row['label'] for row in schedule[:len(expected_labels)]] == list(LABELS) + ['NEG']
    for label in ('TOM', 'CRASH', 'RIDE'):
        metadata[f'egmd_{label}'] = {
            'split': 'train',
            'source': 'egmd_pitch_weighted',
            'events': [{'inst': label, 'time': float(time)} for time in range(3)],
        }
    balanced = build_schedule(metadata, 2, balance_rare_sources=True)
    for label in ('TOM', 'CRASH', 'RIDE'):
        rows = [row for row in balanced if row['label'] == label]
        assert sum(metadata[row['key']].get('source') == 'egmd_pitch_weighted' for row in rows) == 1
    crash_events = metadata['egmd_CRASH']['events']
    metadata['egmd_CRASH']['events'] = []
    try:
        build_schedule(metadata, 2, balance_rare_sources=True)
    except ValueError as error:
        assert 'E-GMD=0' in str(error)
    else:
        raise AssertionError('來源不足時必須拒絕排程')
    metadata['egmd_CRASH']['events'] = crash_events
    metadata['mdb_negative'] = {
        'split': 'negative_train',
        'source': 'mdbdrums_full_mix',
        'events': [
            {'inst': 'KD', 'time': 3.0}, {'inst': 'CRASH', 'time': 3.5},
            {'inst': 'SD', 'time': 10.0},
        ],
    }
    local_negative = build_schedule(metadata, 1, negative_source='mdbdrums_full_mix')
    assert [row for row in local_negative if row['label'] == 'NEG'][0]['anchor'] == 10.0
    try:
        build_schedule(metadata, 1, negative_source='missing_source')
    except ValueError as error:
        assert 'source=missing_source' in str(error)
    else:
        raise AssertionError('指定負樣本來源不足時必須拒絕排程')
    assert balanced_positive_weight(1, 1) == float(np.sqrt(CHUNK_FRAMES))
    targets = torch.zeros(1, 8, len(LABELS))
    targets[0, 4, 5] = 1.0
    assert gaussian_smooth_targets(targets)[0, 4, 5].item() == 1.0
    mask_input = np.ones((2, 2, 16, 4), dtype=np.float32)
    np.random.seed(1337)
    masked = apply_frequency_mask(mask_input, 12)
    assert np.array_equal(masked[:, 0] == 0.0, masked[:, 1] == 0.0)
    assert all(np.count_nonzero(sample[0, :, 0] == 0.0) <= 12 for sample in masked)
    assert np.all(apply_frequency_mask(np.ones((1, 2, 4, 4), dtype=np.float32), 0) == 1.0)
    print('Self-check passed.')


def create_model(architecture, checkpoint_path, device):
    """依架構建立六類模型，並從 Symmetric checkpoint 移植相容權重。"""
    if architecture == 'symmetric':
        model = SymmetricDrumTCN(num_classes=len(LABELS)).to(device)
        return model, load_compatible_weights(model, checkpoint_path, device)
    if architecture == 'dcnn-conformer':
        model = ResidualDCNNDrumConformer(num_classes=len(LABELS)).to(device)
        source_state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        return model, transfer_d3r_state(model, source_state)
    if architecture == 'dcnn-tcn-conformer':
        model = ResidualDCNNDrumHybridConformer(num_classes=len(LABELS)).to(device)
        source_state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        return model, transfer_d3r_hybrid_state(model, source_state)
    model_class = ResidualDCNNDrumTCN if architecture == 'dcnn-residual-tcn' else DCNNDrumTCN
    model = model_class(num_classes=len(LABELS)).to(device)
    source_state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    transfer = transfer_residual_state if architecture == 'dcnn-residual-tcn' else transfer_symmetric_state
    return model, transfer(model, source_state)


def resolve_feature_mode(architecture, feature_mode):
    """將特徵選擇與模型架構解耦，同時保留舊 D3 CLI 行為。"""
    if feature_mode:
        return feature_mode
    return 'true-superflux' if architecture == 'dcnn-tcn' else 'legacy-diff'


def build_full_model_optimizer(model, architecture, head_lr, backbone_lr, new_module_lr):
    """依 heads、新增 DCNN、既有網路三組學習率建立 optimizer。"""
    named = list(model.named_parameters())
    heads = [parameter for name, parameter in named if name.startswith(('onset_head', 'velocity_head'))]
    if architecture == 'dcnn-residual-tcn':
        new_prefixes = ('backbone.correction.', 'backbone.gate')
    elif architecture == 'dcnn-conformer':
        new_prefixes = ('onset_tcn.', 'velocity_tcn.')
    elif architecture == 'dcnn-tcn-conformer':
        new_prefixes = (
            'onset_tcn.conformer.', 'onset_tcn.gate',
            'velocity_tcn.conformer.', 'velocity_tcn.gate',
        )
    else:
        new_prefixes = ()
    new_modules = [parameter for name, parameter in named if new_prefixes and name.startswith(new_prefixes)]
    excluded = {id(parameter) for parameter in heads + new_modules}
    inherited = [parameter for _, parameter in named if id(parameter) not in excluded]
    groups = [{'params': heads, 'lr': head_lr}, {'params': inherited, 'lr': backbone_lr}]
    if new_modules:
        groups.append({'params': new_modules, 'lr': new_module_lr})
    return torch.optim.Adam(groups), {
        'heads': sum(parameter.numel() for parameter in heads),
        'inherited': sum(parameter.numel() for parameter in inherited),
        'new_modules': sum(parameter.numel() for parameter in new_modules),
    }


def main():
    """中文註解：執行一個固定預算的六類 head-only 候選訓練。"""
    parser = argparse.ArgumentParser(description='Train one bounded six-class STAR candidate.')
    parser.add_argument('--meta')
    parser.add_argument('--checkpoint', default='mixed_formal_kick375_snare18_hh12_candidate.pth')
    parser.add_argument('--output-dir', default='validation_runs/six_class_candidate_v1')
    parser.add_argument('--per-class', type=int, default=24)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--backbone-lr', type=float, default=2e-5)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--full-model', action='store_true')
    parser.add_argument('--gaussian-targets', action='store_true')
    parser.add_argument('--positive-weight', type=float, default=20.0)
    parser.add_argument('--schedule-balanced-weights', action='store_true')
    parser.add_argument('--balance-rare-sources', action='store_true', help='TOM/CRASH/RIDE 各取一半 STAR 與一半 E-GMD')
    parser.add_argument('--negative-source', help='Opt-in negative_train metadata source for window-local rare negatives')
    parser.add_argument('--freeze-bn', action='store_true')
    parser.add_argument('--log-every', type=int, default=1)
    parser.add_argument('--candidate-name', default='six_class_candidate.pth')
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--no-lock-three-class', action='store_true', help='停用 KD/SD/HH 物理梯度鎖定')
    parser.add_argument('--max-positive-weight', type=float, default=None, help='限制稀有鼓件正樣本損失權重的最大值')
    parser.add_argument('--accompaniment', help='Optional non-gate no-drums WAV for online domain mixing')
    parser.add_argument('--accompaniment-gain-min', type=float, default=0.10)
    parser.add_argument('--accompaniment-gain-max', type=float, default=0.30)
    parser.add_argument('--architecture', choices=('symmetric', 'dcnn-tcn', 'dcnn-residual-tcn', 'dcnn-conformer', 'dcnn-tcn-conformer'), default='symmetric')
    parser.add_argument('--feature-mode', choices=('legacy-diff', 'true-superflux'))
    parser.add_argument('--new-module-lr', type=float, help='Residual DCNN correction/gate learning rate; defaults to --lr')
    parser.add_argument('--validation-meta', help='每個 epoch 使用的 held-out STAR validation metadata')
    parser.add_argument('--validation-per-class', type=int, default=8)
    parser.add_argument('--early-stopping-patience', type=int, default=0, help='連續未刷新 validation Macro F1 的停止次數；0 表示停用')
    parser.add_argument('--frequency-mask-max-bins', type=int, default=0, help='訓練期同步遮罩兩通道的最大連續 Mel bins；0 表示停用')
    args = parser.parse_args()
    args.lock_three_class = not args.no_lock_three_class
    args.feature_mode = resolve_feature_mode(args.architecture, args.feature_mode)
    args.new_module_lr = args.lr if args.new_module_lr is None else args.new_module_lr

    if args.self_check:
        run_self_check()
        return
    if not args.meta:
        parser.error('--meta is required unless --self-check is used')
    if args.per_class <= 0 or args.batch_size <= 0 or args.epochs <= 0 or args.log_every <= 0:
        parser.error('--per-class, --batch-size, --epochs, and --log-every must be positive')
    if args.validation_per_class <= 0 or args.early_stopping_patience < 0 or args.frequency_mask_max_bins < 0:
        parser.error('--validation-per-class must be positive; patience and frequency mask cannot be negative')
    if args.early_stopping_patience and not args.validation_meta:
        parser.error('--early-stopping-patience requires --validation-meta')
    if not 0.0 <= args.accompaniment_gain_min <= args.accompaniment_gain_max:
        parser.error('accompaniment gain range is invalid')
    torch.manual_seed(1337)
    np.random.seed(1337)
    with open(args.meta, encoding='utf-8') as handle:
        metadata = json.load(handle)
    validation_metadata = None
    if args.validation_meta:
        with open(args.validation_meta, encoding='utf-8') as handle:
            validation_metadata = json.load(handle)
    schedule = build_schedule(
        metadata, args.per_class,
        balance_rare_sources=args.balance_rare_sources,
        negative_source=args.negative_source,
    )
    accompaniment_pool = [load_accompaniment(args.accompaniment)] if args.accompaniment else None
    if len(schedule) % args.batch_size:
        raise ValueError('Schedule length must divide evenly by batch size.')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'train_schedule.json'), 'w', encoding='utf-8') as handle:
        json.dump(schedule, handle, indent=2)
    if args.schedule_balanced_weights:
        class_weights, class_event_counts = schedule_positive_weights(schedule, metadata)
    else:
        class_weights = {label: args.positive_weight for label in LABELS}
        class_event_counts = None
    if args.max_positive_weight is not None:
        for label in LABELS:
            if class_weights[label] > args.max_positive_weight:
                class_weights[label] = args.max_positive_weight

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, transferred = create_model(args.architecture, args.checkpoint, device)
    if args.full_model:
        optimizer, optimizer_parameter_counts = build_full_model_optimizer(
            model, args.architecture, args.lr, args.backbone_lr, args.new_module_lr,
        )
    else:
        freeze_for_head_training(model)
        optimizer = torch.optim.Adam((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.lr)
        optimizer_parameter_counts = None
    losses = []
    validation_history = []
    best_macro_f1 = -1.0
    best_epoch = None
    epochs_without_improvement = 0
    early_stopped = False
    positive_weight = torch.tensor([class_weights[label] for label in LABELS], dtype=torch.float32, device=device).view(1, 1, -1)
    batches_per_epoch = len(schedule) // args.batch_size
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_bn:
            freeze_batchnorm_stats(model)
        for start in range(0, len(schedule), args.batch_size):
            feature, onset, velocity = batch_from_schedule(
                schedule, metadata, start, args.batch_size,
                accompaniment_pool=accompaniment_pool,
                gain_range=(args.accompaniment_gain_min, args.accompaniment_gain_max),
                use_true_superflux=args.feature_mode == 'true-superflux',
            )
            feature = apply_frequency_mask(feature, args.frequency_mask_max_bins)
            x = torch.from_numpy(feature).float().to(device)
            onset_target = torch.from_numpy(onset).float().to(device)
            velocity_target = torch.from_numpy(velocity).float().to(device)
            onset_logits, velocity_logits = model(x)
            onset_for_loss = gaussian_smooth_targets(onset_target) if args.gaussian_targets else onset_target
            bce = F.binary_cross_entropy(torch.sigmoid(onset_logits), onset_for_loss, reduction='none')
            onset_weight = torch.where(onset_for_loss > 0.0, positive_weight.expand_as(onset_for_loss), torch.ones_like(onset_for_loss))
            onset_loss = (bce * onset_weight).mean()
            active = onset_target.sum().clamp_min(1.0)
            velocity_for_loss = propagate_velocity_targets(velocity_target) if args.gaussian_targets else velocity_target
            velocity_loss = ((torch.sigmoid(velocity_logits) - velocity_for_loss).pow(2) * onset_for_loss).sum() / active
            loss = onset_loss + velocity_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            if args.lock_three_class:
                if model.onset_head.weight.grad is not None:
                    model.onset_head.weight.grad[:3].zero_()
                if model.onset_head.bias.grad is not None:
                    model.onset_head.bias.grad[:3].zero_()
                if model.velocity_head.weight.grad is not None:
                    model.velocity_head.weight.grad[:3].zero_()
                if model.velocity_head.bias.grad is not None:
                    model.velocity_head.bias.grad[:3].zero_()
            optimizer.step()
            losses.append(float(loss.item()))
            batch_number = start // args.batch_size + 1
            if batch_number % args.log_every == 0 or batch_number == batches_per_epoch:
                print(f'epoch={epoch}/{args.epochs} batch={batch_number}/{batches_per_epoch} loss={losses[-1]:.4f}', flush=True)
        # 中文註解：儲存每個 Epoch 的獨立候選權重以供自動篩選哨兵進行測試
        epoch_cand_name = f"{os.path.splitext(args.candidate_name)[0]}_epoch{epoch}.pth"
        torch.save(model.state_dict(), os.path.join(args.output_dir, epoch_cand_name))
        if validation_metadata is not None:
            rows, gate = evaluate_model(
                model, validation_metadata, os.path.join(args.output_dir, f'validation_epoch{epoch}'),
                per_class=args.validation_per_class, accompaniment=accompaniment_pool[0] if accompaniment_pool else None,
                accompaniment_path=args.accompaniment, architecture=args.architecture,
                feature_mode=args.feature_mode, device=device,
            )
            class_f1 = {row['inst']: float(row['f1']) for row in rows}
            macro_f1 = float(gate['macro_f1'])
            improved = macro_f1 > best_macro_f1
            validation_history.append({'epoch': epoch, 'macro_f1': macro_f1, 'per_class_f1': class_f1, 'improved': improved})
            print('validation ' + ' '.join(f'{label}={class_f1[label]:.4f}' for label in LABELS) + f' macro={macro_f1:.4f}', flush=True)
            if improved:
                best_macro_f1 = macro_f1
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(model.state_dict(), os.path.join(args.output_dir, args.candidate_name))
            else:
                epochs_without_improvement += 1
                if args.early_stopping_patience and epochs_without_improvement >= args.early_stopping_patience:
                    early_stopped = True
                    print(f'early_stopping epoch={epoch} best_epoch={best_epoch} best_macro_f1={best_macro_f1:.4f}', flush=True)
                    break
    candidate_path = os.path.join(args.output_dir, args.candidate_name)
    if validation_metadata is None:
        torch.save(model.state_dict(), candidate_path)
    confusion_report = None
    if validation_metadata is not None:
        model.load_state_dict(torch.load(candidate_path, map_location=device, weights_only=False))
        confusion_dir = os.path.join(args.output_dir, 'best_confusion')
        evaluate_confusion(
            model, validation_metadata, confusion_dir,
            accompaniment=accompaniment_pool[0] if accompaniment_pool else None,
            accompaniment_gain=0.17, per_class=args.validation_per_class,
            feature_mode=args.feature_mode, device=device,
        )
        confusion_report = os.path.abspath(os.path.join(confusion_dir, 'confusion_summary.json'))
    report = {
        'status': 'pass', 'labels': LABELS, 'schedule_windows': len(schedule),
        'per_class': args.per_class, 'batch_size': args.batch_size, 'epochs': args.epochs,
        'epochs_completed': len(validation_history) if validation_metadata is not None else args.epochs,
        'batches': len(losses),
        'head_learning_rate': args.lr, 'backbone_learning_rate': args.backbone_lr if args.full_model else None,
        'new_module_learning_rate': args.new_module_lr if args.full_model and args.architecture in ('dcnn-residual-tcn', 'dcnn-conformer', 'dcnn-tcn-conformer') else None,
        'optimizer_parameter_counts': optimizer_parameter_counts,
        'balance_rare_sources': args.balance_rare_sources,
        'negative_source': args.negative_source,
        'full_model': args.full_model, 'gaussian_targets': args.gaussian_targets,
        'class_positive_weights': class_weights, 'class_event_counts': class_event_counts, 'freeze_batchnorm': args.freeze_bn,
        'first_loss': losses[0], 'last_loss': losses[-1],
        'accompaniment': os.path.abspath(args.accompaniment) if args.accompaniment else None,
        'accompaniment_gain_range': [args.accompaniment_gain_min, args.accompaniment_gain_max],
        'architecture': args.architecture,
        'feature_mode': f'log_mel+{args.feature_mode.replace("-", "_")}',
        'frequency_mask_max_bins': args.frequency_mask_max_bins,
        'time_mask_max_frames': 0,
        'validation_meta': os.path.abspath(args.validation_meta) if args.validation_meta else None,
        'validation_per_class': args.validation_per_class if args.validation_meta else None,
        'early_stopping_patience': args.early_stopping_patience,
        'early_stopped': early_stopped, 'best_epoch': best_epoch, 'best_macro_f1': best_macro_f1 if best_epoch else None,
        'validation_history': validation_history,
        'best_confusion_report': confusion_report,
        'transferred_compatible_tensors': transferred, 'candidate': os.path.abspath(candidate_path),
    }
    with open(os.path.join(args.output_dir, 'train_report.json'), 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
