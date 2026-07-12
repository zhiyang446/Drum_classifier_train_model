# -*- coding: utf-8 -*-
"""以固定 STAR train 視窗訓練一個獨立六類候選模型。"""
import argparse
import json
import os

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from run_six_class_smoke import CHUNK_FRAMES, LABELS, SR, TARGET_SAMPLES, build_window, load_compatible_weights
from train_star_smoke import freeze_batchnorm_stats
from train_phase2 import SymmetricDrumTCN, propagate_velocity_targets


def build_schedule(metadata, per_class):
    """中文註解：只取可置中的 train 錨點，並以六類交錯順序建立固定排程。"""
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
        if len(candidates) < per_class:
            raise ValueError(f'Only {len(candidates)} centered train events for {label}, need {per_class}.')
        selected_by_label[label] = []
        for index in range(per_class):
            key, anchor = candidates[index * len(candidates) // per_class]
            selected_by_label[label].append({'label': label, 'key': key, 'anchor': anchor})
    return [selected_by_label[label][index] for index in range(per_class) for label in LABELS]


def freeze_for_head_training(model):
    """中文註解：只訓練新六類輸出頭，保護轉移而來的聲學特徵。"""
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith('onset_head') or name.startswith('velocity_head')


def batch_from_schedule(schedule, metadata, start, batch_size):
    """中文註解：依固定 schedule 建立一個不含測試資料的訓練 batch。"""
    features, onsets, velocities = [], [], []
    for row in schedule[start:start + batch_size]:
        item = metadata[row['key']]
        feature, onset, velocity, _ = build_window(item, row['anchor'])
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
    assert len(schedule) == len(LABELS) * 2
    assert {row['label'] for row in schedule} == set(LABELS)
    assert [row['label'] for row in schedule[:len(LABELS)]] == list(LABELS)
    assert balanced_positive_weight(1, 1) == float(np.sqrt(CHUNK_FRAMES))
    targets = torch.zeros(1, 8, len(LABELS))
    targets[0, 4, 5] = 1.0
    assert gaussian_smooth_targets(targets)[0, 4, 5].item() == 1.0
    print('Self-check passed.')


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
    parser.add_argument('--freeze-bn', action='store_true')
    parser.add_argument('--log-every', type=int, default=1)
    parser.add_argument('--candidate-name', default='six_class_candidate.pth')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    if not args.meta:
        parser.error('--meta is required unless --self-check is used')
    if args.per_class <= 0 or args.batch_size <= 0 or args.epochs <= 0 or args.log_every <= 0:
        parser.error('--per-class, --batch-size, --epochs, and --log-every must be positive')
    torch.manual_seed(1337)
    with open(args.meta, encoding='utf-8') as handle:
        metadata = json.load(handle)
    schedule = build_schedule(metadata, args.per_class)
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SymmetricDrumTCN(num_classes=len(LABELS)).to(device)
    transferred = load_compatible_weights(model, args.checkpoint, device)
    if args.full_model:
        head_params = list(model.onset_head.parameters()) + list(model.velocity_head.parameters())
        head_ids = {id(parameter) for parameter in head_params}
        backbone_params = [parameter for parameter in model.parameters() if id(parameter) not in head_ids]
        optimizer = torch.optim.Adam([
            {'params': head_params, 'lr': args.lr},
            {'params': backbone_params, 'lr': args.backbone_lr},
        ])
    else:
        freeze_for_head_training(model)
        optimizer = torch.optim.Adam((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.lr)
    losses = []
    positive_weight = torch.tensor([class_weights[label] for label in LABELS], dtype=torch.float32, device=device).view(1, 1, -1)
    batches_per_epoch = len(schedule) // args.batch_size
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_bn:
            freeze_batchnorm_stats(model)
        for start in range(0, len(schedule), args.batch_size):
            feature, onset, velocity = batch_from_schedule(schedule, metadata, start, args.batch_size)
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
            optimizer.step()
            losses.append(float(loss.item()))
            batch_number = start // args.batch_size + 1
            if batch_number % args.log_every == 0 or batch_number == batches_per_epoch:
                print(f'epoch={epoch}/{args.epochs} batch={batch_number}/{batches_per_epoch} loss={losses[-1]:.4f}', flush=True)
    candidate_path = os.path.join(args.output_dir, args.candidate_name)
    torch.save(model.state_dict(), candidate_path)
    report = {
        'status': 'pass', 'labels': LABELS, 'schedule_windows': len(schedule),
        'per_class': args.per_class, 'batch_size': args.batch_size, 'epochs': args.epochs, 'batches': len(losses),
        'head_learning_rate': args.lr, 'backbone_learning_rate': args.backbone_lr if args.full_model else None,
        'full_model': args.full_model, 'gaussian_targets': args.gaussian_targets,
        'class_positive_weights': class_weights, 'class_event_counts': class_event_counts, 'freeze_batchnorm': args.freeze_bn,
        'first_loss': losses[0], 'last_loss': losses[-1],
        'transferred_compatible_tensors': transferred, 'candidate': os.path.abspath(candidate_path),
    }
    with open(os.path.join(args.output_dir, 'train_report.json'), 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
