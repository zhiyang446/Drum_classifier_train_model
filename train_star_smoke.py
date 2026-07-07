# -*- coding: utf-8 -*-
"""
STAR Drums smoke training.

只確認 STAR metadata/audio/label 能通過現有 TCN 訓練路徑，不覆蓋正式模型。
"""
import argparse
import json
import os

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from dsp_utils import extract_features
from train_phase2 import SymmetricDrumTCN, gaussian_smooth_targets, propagate_velocity_targets


SR = 44100
HOP_LENGTH = 256
N_MELS = 256
CHUNK_FRAMES = 688
TARGET_SAMPLES = CHUNK_FRAMES * HOP_LENGTH
INST_INDICES = {'KD': 0, 'SD': 1, 'HH': 2}


def parse_channel_weights(text):
    """
    中文註解：解析 KD,SD,HH 三通道正樣本權重，避免為單一鼓組寫死訓練邏輯。
    """
    weights = [float(v.strip()) for v in text.split(',')]
    if len(weights) != 3:
        raise ValueError('--onset-pos-weights must contain three comma-separated numbers: KD,SD,HH')
    return weights


def summarize_events(events):
    """
    中文註解：統計單一 STAR 樣本的 KD/SD/HH 數量與 SD+HH 同時敲擊位置。
    """
    counts = {'KD': 0, 'SD': 0, 'HH': 0}
    bins = {}
    for ev in events:
        inst = ev['inst']
        if inst in counts:
            counts[inst] += 1
            bin_id = round(float(ev['time']) / 0.03)
            bins.setdefault(bin_id, {'time': float(ev['time']), 'insts': set()})['insts'].add(inst)
    sd_hh_times = [v['time'] for v in bins.values() if {'SD', 'HH'}.issubset(v['insts'])]
    return counts, sd_hh_times


def median_event_time(events, inst=None):
    """
    中文註解：取得指定鼓組的中間事件時間，用來把訓練切片中心放在有效標籤附近。
    """
    times = [float(ev['time']) for ev in events if inst is None or ev['inst'] == inst]
    if not times:
        return None
    times.sort()
    return times[len(times) // 2]


def with_anchor(item, bucket, anchor_time):
    """
    中文註解：複製 metadata 項目並附加抽樣 bucket 與切片中心時間。
    """
    copied = dict(item)
    copied['_bucket'] = bucket
    copied['_anchor_time'] = anchor_time
    return copied


def select_balanced_items(items, max_items):
    """
    中文註解：從 STAR train split 交錯挑選 SD、SD+HH、HH 與均衡樣本，避免單一分佈洗壞主模型。
    """
    scored = []
    for idx, item in enumerate(items):
        counts, sd_hh_times = summarize_events(item['events'])
        scored.append((idx, item, counts, sd_hh_times))

    buckets = [
        ('sd_dense', sorted(scored, key=lambda x: (x[2]['SD'], x[2]['HH']), reverse=True)),
        ('sd_hh', sorted(scored, key=lambda x: (len(x[3]), x[2]['SD']), reverse=True)),
        ('hh_dense', sorted(scored, key=lambda x: (x[2]['HH'], x[2]['SD']), reverse=True)),
        ('balanced', sorted(scored, key=lambda x: (min(x[2].values()), sum(x[2].values())), reverse=True)),
    ]
    selected = []
    used = set()
    cursors = {name: 0 for name, _ in buckets}

    while len(selected) < max_items and len(used) < len(items):
        progressed = False
        for name, bucket_items in buckets:
            while cursors[name] < len(bucket_items) and bucket_items[cursors[name]][0] in used:
                cursors[name] += 1
            if cursors[name] >= len(bucket_items):
                continue
            idx, item, counts, sd_hh_times = bucket_items[cursors[name]]
            if name == 'sd_hh' and not sd_hh_times:
                cursors[name] += 1
                continue
            used.add(idx)
            cursors[name] += 1
            if name == 'sd_hh':
                anchor_time = sd_hh_times[len(sd_hh_times) // 2]
            elif name == 'sd_dense':
                anchor_time = median_event_time(item['events'], 'SD')
            elif name == 'hh_dense':
                anchor_time = median_event_time(item['events'], 'HH')
            else:
                anchor_time = median_event_time(item['events'])
            selected.append(with_anchor(item, name, anchor_time))
            progressed = True
            if len(selected) >= max_items:
                break
        if not progressed:
            break

    for idx, item in enumerate(items):
        if len(selected) >= max_items:
            break
        if idx not in used:
            selected.append(with_anchor(item, 'fill', median_event_time(item['events'])))
    return selected


class StarSmokeDataset(Dataset):
    """
    STAR metadata 的最小訓練 Dataset，固定讀取短切片做 smoke test。
    """
    def __init__(self, meta_path, max_items=128, balanced=False):
        with open(meta_path, 'r', encoding='utf-8') as f:
            all_items = json.load(f)

        candidates = [v for v in all_items.values() if v.get('split') == 'train' and v.get('events')]
        self.items = select_balanced_items(candidates, max_items) if balanced else candidates[:max_items]
        if not self.items:
            raise ValueError(f'No training items found in {meta_path}')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx % len(self.items)]
        events = item['events']
        anchor = item.get('_anchor_time')
        if anchor is None:
            anchor = events[len(events) // 2]['time'] if events else 0.0

        with sf.SoundFile(item['audio_path']) as f:
            source_sr = f.samplerate
            total_samples = f.frames
            start_sample = max(0, int(anchor * source_sr) - TARGET_SAMPLES // 2)
            start_sample = min(start_sample, max(0, total_samples - TARGET_SAMPLES))
            f.seek(start_sample)
            y = f.read(TARGET_SAMPLES, dtype='float32')

        if y.ndim > 1:
            y = np.mean(y, axis=1)

        if source_sr != SR:
            y = librosa.resample(y, orig_sr=source_sr, target_sr=SR)

        if len(y) < TARGET_SAMPLES:
            y = np.pad(y, (0, TARGET_SAMPLES - len(y)), mode='constant')
        else:
            y = y[:TARGET_SAMPLES]

        onset_target = np.zeros((CHUNK_FRAMES, 3), dtype=np.float32)
        velocity_target = np.zeros((CHUNK_FRAMES, 3), dtype=np.float32)
        start_sec = start_sample / source_sr
        end_sec = start_sec + (TARGET_SAMPLES / SR)

        for ev in events:
            t_sec = float(ev['time'])
            if start_sec <= t_sec < end_sec:
                frame = int(round((t_sec - start_sec) * SR / HOP_LENGTH))
                if 0 <= frame < CHUNK_FRAMES:
                    inst_idx = INST_INDICES[ev['inst']]
                    onset_target[frame, inst_idx] = 1.0
                    velocity_target[frame, inst_idx] = float(ev['velocity']) / 127.0

        features = extract_features(y, sr=SR, hop_length=HOP_LENGTH, n_mels=N_MELS, use_hybrid=False)
        n_frames = features.shape[2]
        if n_frames != CHUNK_FRAMES:
            features = features[:, :, :CHUNK_FRAMES]
            if n_frames < CHUNK_FRAMES:
                features = np.pad(features, ((0, 0), (0, 0), (0, CHUNK_FRAMES - n_frames)), mode='constant')

        return (
            torch.from_numpy(features).float(),
            torch.from_numpy(onset_target).float(),
            torch.from_numpy(velocity_target).float(),
        )


def load_checkpoint(model, checkpoint_path, device):
    """
    載入現有 checkpoint，兼容舊 slot projection 權重。
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f'Checkpoint not found, training smoke model from scratch: {checkpoint_path}')
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'backbone.legacy_slot_proj.weight' in checkpoint:
        model.backbone.use_legacy_proj = True
    elif 'backbone.slot_proj.weight' in checkpoint and checkpoint['backbone.slot_proj.weight'].shape == torch.Size([64, 1024, 1, 1]):
        model.backbone.use_legacy_proj = True
        checkpoint['backbone.legacy_slot_proj.weight'] = checkpoint.pop('backbone.slot_proj.weight')
        checkpoint['backbone.legacy_slot_proj.bias'] = checkpoint.pop('backbone.slot_proj.bias')
    model.load_state_dict(checkpoint, strict=False)
    print(f'Loaded checkpoint: {checkpoint_path}')


def freeze_batchnorm_stats(model):
    """
    中文註解：凍結 BatchNorm running statistics，避免小批量 STAR 微調洗壞主模型分佈。
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.eval()


def train_heads_only(model):
    """
    中文註解：只訓練 onset/velocity 輸出頭，保留既有 backbone 與 TCN 節奏表徵。
    """
    for param in model.parameters():
        param.requires_grad = False
    for module in (model.onset_head, model.velocity_head):
        for param in module.parameters():
            param.requires_grad = True


def run_self_check():
    """
    最小自檢：確認 Dataset 對空 metadata 會拒絕。
    """
    tmp_path = os.path.join('processed_data', '_star_smoke_empty.json')
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    fake_items = [
        {'events': [{'time': 0.1, 'inst': 'SD', 'velocity': 90}, {'time': 0.2, 'inst': 'SD', 'velocity': 90}]},
        {'events': [{'time': 0.1, 'inst': 'SD', 'velocity': 90}, {'time': 0.1, 'inst': 'HH', 'velocity': 90}]},
        {'events': [{'time': 0.1, 'inst': 'HH', 'velocity': 90}, {'time': 0.2, 'inst': 'HH', 'velocity': 90}]},
        {'events': [{'time': 0.1, 'inst': 'KD', 'velocity': 90}, {'time': 0.2, 'inst': 'SD', 'velocity': 90}, {'time': 0.3, 'inst': 'HH', 'velocity': 90}]},
    ]
    selected = select_balanced_items(fake_items, 4)
    assert len(selected) == 4
    assert all('_anchor_time' in item for item in selected)
    assert {'sd_dense', 'sd_hh', 'hh_dense', 'balanced'}.issubset({item['_bucket'] for item in selected})
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump({}, f)
    try:
        try:
            StarSmokeDataset(tmp_path)
        except ValueError:
            print('Self-check passed.')
            return
        raise AssertionError('empty metadata should fail')
    finally:
        os.remove(tmp_path)


def main():
    parser = argparse.ArgumentParser(description='STAR smoke training.')
    parser.add_argument('--meta', default='processed_data/star_meta.json')
    parser.add_argument('--checkpoint', default='best_drum_model.pth')
    parser.add_argument('--output', default='star_smoke_model.pth')
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-batches', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--freeze-bn', action='store_true')
    parser.add_argument('--train-head-only', action='store_true')
    parser.add_argument('--balanced-sampler', action='store_true')
    parser.add_argument('--onset-pos-weights', default='1,1,1')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()

    if args.self_check:
        run_self_check()
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    dataset = StarSmokeDataset(args.meta, max_items=args.samples, balanced=args.balanced_sampler)
    if args.balanced_sampler:
        buckets = {}
        for item in dataset.items:
            buckets[item.get('_bucket', 'unknown')] = buckets.get(item.get('_bucket', 'unknown'), 0) + 1
        print(f'Balanced sampler buckets: {buckets}')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = SymmetricDrumTCN().to(device)
    load_checkpoint(model, args.checkpoint, device)
    if args.train_head_only:
        train_heads_only(model)

    # 中文註解：學習率由 CLI 控制，方便用極低 lr 做保守微調，避免主模型 collapse。
    optimizer = optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    onset_pos_weights = torch.tensor(parse_channel_weights(args.onset_pos_weights), device=device).view(1, 1, 3)
    model.train()
    if args.freeze_bn:
        freeze_batchnorm_stats(model)
    losses = []

    for batch_idx, (features, onset_targets, velocity_targets) in enumerate(loader):
        if batch_idx >= args.max_batches:
            break

        features = features.to(device)
        onset_targets = onset_targets.to(device)
        velocity_targets = velocity_targets.to(device)

        optimizer.zero_grad()
        onset_logits, velocity_logits = model(features)
        pred_onset = torch.sigmoid(onset_logits)
        pred_velocity = torch.sigmoid(velocity_logits)

        onset_smoothed = gaussian_smooth_targets(onset_targets, device)
        velocity_propagated = propagate_velocity_targets(velocity_targets)
        bce = nn.functional.binary_cross_entropy(pred_onset, onset_smoothed, reduction='none')
        active_weight = torch.where(onset_smoothed > 0.0, onset_pos_weights, torch.ones_like(onset_smoothed))
        loss_onset = (bce * active_weight).mean()
        loss_velocity = (onset_smoothed * (pred_velocity - velocity_propagated) ** 2).mean()
        loss = loss_onset + 10.0 * loss_velocity
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))
        print(f'Batch {batch_idx + 1}/{min(len(loader), args.max_batches)} loss={losses[-1]:.4f}')

    torch.save(model.state_dict(), args.output)
    print(f'Smoke training done. batches={len(losses)} first_loss={losses[0]:.4f} last_loss={losses[-1]:.4f}')
    print(f'Wrote temporary checkpoint: {args.output}')


if __name__ == '__main__':
    main()
