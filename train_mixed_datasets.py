# -*- coding: utf-8 -*-
"""
Mixed E-GMD / STAR / local XML smoke trainer.

This writes candidate checkpoints only. It never overwrites best_drum_model.pth.
"""
import argparse
import csv
import json
import os
import subprocess
import sys

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from dsp_utils import extract_features
from train_phase2 import SymmetricDrumTCN, gaussian_smooth_targets, propagate_velocity_targets
from train_star_smoke import freeze_batchnorm_stats, load_checkpoint, parse_channel_weights, select_balanced_items, train_heads_only


SR = 44100
HOP_LENGTH = 256
N_MELS = 256
CHUNK_FRAMES = 688
TARGET_SAMPLES = CHUNK_FRAMES * HOP_LENGTH
INST_INDICES = {'KD': 0, 'SD': 1, 'HH': 2}


def parse_mix_ratio(text):
    """
    中文註解：解析 egmd,star,local 三資料來源比例。
    """
    values = [int(v.strip()) for v in text.split(',')]
    if len(values) != 3 or sum(values) <= 0:
        raise ValueError('--mix-ratio must be three positive integers: egmd,star,local')
    return {'egmd': values[0], 'star': values[1], 'local': values[2]}


def parse_train_channels(text):
    """
    中文註解：解析要參與 loss 的鼓組通道，預設三通道全訓練。
    """
    if not text or text.strip().lower() in {'all', '*'}:
        return [0, 1, 2]
    mapping = {'KD': 0, 'KICK': 0, 'SD': 1, 'SNARE': 1, 'HH': 2, 'HIHAT': 2, 'HI-HAT': 2}
    channels = []
    for raw in text.split(','):
        key = raw.strip().upper()
        if key not in mapping:
            raise ValueError(f'Unknown train channel: {raw}')
        idx = mapping[key]
        if idx not in channels:
            channels.append(idx)
    if not channels:
        raise ValueError('--train-channels produced no channels')
    return channels


def channel_mask(indices, device):
    """
    中文註解：建立 [1,1,3] loss mask，只讓指定通道更新。
    """
    mask = torch.zeros((1, 1, 3), device=device)
    for idx in indices:
        mask[:, :, idx] = 1.0
    return mask


def load_meta_items(path, split='train', max_items=0):
    """
    中文註解：讀取 metadata JSON，保留指定 split 與有效 events。
    """
    with open(path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    items = [dict(v) for v in meta.values() if v.get('events') and (not split or v.get('split') == split)]
    if max_items > 0:
        items = items[:max_items]
    return items


def median_event_time(events):
    """
    中文註解：取得中間事件時間，讓切片中心落在有鼓點的位置。
    """
    if not events:
        return 0.0
    return float(events[len(events) // 2]['time'])


def snare_anchor_time(events):
    """
    中文註解：優先找 SD+HH 同時敲擊，其次找 SD，用來強化 Snare 共奏樣本。
    """
    bins = {}
    for ev in events:
        bin_id = round(float(ev['time']) / 0.03)
        bins.setdefault(bin_id, {'time': float(ev['time']), 'insts': set()})['insts'].add(ev.get('inst'))
    sd_hh_times = [v['time'] for v in bins.values() if {'SD', 'HH'}.issubset(v['insts'])]
    if sd_hh_times:
        sd_hh_times.sort()
        return sd_hh_times[len(sd_hh_times) // 2]
    sd_times = [float(ev['time']) for ev in events if ev.get('inst') == 'SD']
    if sd_times:
        sd_times.sort()
        return sd_times[len(sd_times) // 2]
    return median_event_time(events)


def make_schedule(ratio):
    """
    中文註解：依比例建立資料來源輪替表，例如 5/3/2。
    """
    schedule = []
    for source, count in ratio.items():
        schedule.extend([source] * count)
    return schedule


def source_probabilities(ratio):
    """
    中文註解：將資料來源比例轉成可重現隨機抽樣用的名稱與機率。
    """
    source_names = list(ratio.keys())
    counts = np.array([ratio[name] for name in source_names], dtype=np.float64)
    return source_names, counts / counts.sum()


class MixedMetaDataset(Dataset):
    """
    中文註解：從 E-GMD、STAR、local XML metadata 依固定比例讀取音訊切片。
    """
    def __init__(self, egmd_path, star_path, local_path, samples, ratio, max_items_per_source=0, snare_focus=False, random_sampling=False, balanced_sampler=False, seed=1337):
        self.sources = {
            'egmd': load_meta_items(egmd_path, split='train'),
            'star': load_meta_items(star_path, split='train'),
            'local': load_meta_items(local_path, split='train'),
        }
        if balanced_sampler:
            for name, items in self.sources.items():
                limit = max_items_per_source or len(items)
                self.sources[name] = select_balanced_items(items, min(limit, len(items)))
        elif max_items_per_source > 0:
            for name, items in self.sources.items():
                self.sources[name] = items[:max_items_per_source]
        missing = [name for name, items in self.sources.items() if not items]
        if missing:
            raise ValueError(f'Missing training items for: {", ".join(missing)}')
        if snare_focus:
            for items in self.sources.values():
                for item in items:
                    item['_anchor_time'] = snare_anchor_time(item['events'])
        self.samples = samples
        self.schedule = make_schedule(ratio)
        self.source_names, self.source_probs = source_probabilities(ratio)
        self.random_sampling = random_sampling
        self.seed = seed

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        if self.random_sampling:
            # 中文註解：用 idx 派生亂數，讓 DataLoader 順序固定時仍可覆蓋完整資料集且結果可重現。
            rng = np.random.default_rng(self.seed + idx)
            source = rng.choice(self.source_names, p=self.source_probs)
            items = self.sources[source]
            item = items[int(rng.integers(0, len(items)))]
        else:
            source = self.schedule[idx % len(self.schedule)]
            items = self.sources[source]
            item = items[(idx // len(self.schedule)) % len(items)]
        features, onset_target, velocity_target, event_weight_target = load_training_slice(item)
        return (
            torch.from_numpy(features).float(),
            torch.from_numpy(onset_target).float(),
            torch.from_numpy(velocity_target).float(),
            torch.from_numpy(event_weight_target).float(),
        )


def load_training_slice(item):
    """
    中文註解：讀取單一 metadata item 的 4 秒音訊切片並建立 onset/velocity target。
    """
    events = item['events']
    anchor = item.get('_anchor_time', median_event_time(events))
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
    event_weight_target = np.ones((CHUNK_FRAMES, 3), dtype=np.float32)
    start_sec = start_sample / source_sr
    end_sec = start_sec + (TARGET_SAMPLES / SR)
    for ev in events:
        t_sec = float(ev['time'])
        inst = ev.get('inst')
        if inst not in INST_INDICES or not (start_sec <= t_sec < end_sec):
            continue
        frame = int(round((t_sec - start_sec) * SR / HOP_LENGTH))
        if 0 <= frame < CHUNK_FRAMES:
            inst_idx = INST_INDICES[inst]
            onset_target[frame, inst_idx] = 1.0
            velocity_target[frame, inst_idx] = float(ev.get('velocity', 100.0)) / 127.0
            event_weight_target[frame, inst_idx] = max(
                event_weight_target[frame, inst_idx],
                float(ev.get('loss_weight', 1.0)),
            )

    features = extract_features(y, sr=SR, hop_length=HOP_LENGTH, n_mels=N_MELS, use_hybrid=False)
    n_frames = features.shape[2]
    if n_frames != CHUNK_FRAMES:
        features = features[:, :, :CHUNK_FRAMES]
        if n_frames < CHUNK_FRAMES:
            features = np.pad(features, ((0, 0), (0, 0), (0, CHUNK_FRAMES - n_frames)), mode='constant')
    return features, onset_target, velocity_target, event_weight_target


def run_self_check():
    """
    中文註解：最小自檢，確認比例排程與權重解析可用。
    """
    assert make_schedule(parse_mix_ratio('5,3,2')).count('egmd') == 5
    assert make_schedule(parse_mix_ratio('5,3,2')).count('star') == 3
    assert make_schedule(parse_mix_ratio('5,3,2')).count('local') == 2
    assert parse_train_channels('SD,HH') == [1, 2]
    names, probs = source_probabilities(parse_mix_ratio('5,3,2'))
    assert names == ['egmd', 'star', 'local']
    assert np.isclose(probs.sum(), 1.0)
    assert snare_anchor_time([{'time': 0.1, 'inst': 'SD'}, {'time': 0.1, 'inst': 'HH'}]) == 0.1
    bucketed = select_balanced_items([
        {'events': [{'time': 0.1, 'inst': 'SD', 'velocity': 90}]},
        {'events': [{'time': 0.1, 'inst': 'SD', 'velocity': 90}, {'time': 0.1, 'inst': 'HH', 'velocity': 90}]},
        {'events': [{'time': 0.1, 'inst': 'HH', 'velocity': 90}]},
        {'events': [{'time': 0.1, 'inst': 'KD', 'velocity': 90}, {'time': 0.2, 'inst': 'SD', 'velocity': 90}, {'time': 0.3, 'inst': 'HH', 'velocity': 90}]},
    ], 4)
    assert len(bucketed) == 4
    assert all('_anchor_time' in item and '_bucket' in item for item in bucketed)
    print('Self-check passed.')


def count_validation_failures(summary_csv):
    """
    中文註解：讀取 hard validation summary，回傳 gate fail 數量。
    """
    with open(summary_csv, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    return sum(1 for row in rows if row.get('gate_status') != 'pass'), len(rows)


def run_hard_validation(model_path, validation_root, epoch, star_limit):
    """
    中文註解：呼叫既有 hard validation runner，避免在訓練腳本重寫驗證邏輯。
    """
    output_dir = os.path.join(validation_root, f'epoch_{epoch:03d}')
    cmd = [
        sys.executable,
        'run_hard_validation.py',
        '--model', model_path,
        '--star-limit', str(star_limit),
        '--output-dir', output_dir,
    ]
    subprocess.run(cmd, check=True)
    summary_csv = os.path.join(output_dir, 'summary.csv')
    fail_count, total_count = count_validation_failures(summary_csv)
    return fail_count, total_count, summary_csv


def train_one_epoch(model, loader, optimizer, onset_pos_weights, onset_neg_weights, train_mask, device, max_batches, freeze_bn=False, hard_neg_boost=0.0):
    """
    中文註解：執行一個 mixed training epoch，回傳 loss 序列。
    """
    losses = []
    model.train()
    if freeze_bn:
        freeze_batchnorm_stats(model)
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        if len(batch) == 4:
            features, onset_targets, velocity_targets, event_weight_targets = batch
        else:
            features, onset_targets, velocity_targets = batch
            event_weight_targets = torch.ones_like(onset_targets)
        features = features.to(device)
        onset_targets = onset_targets.to(device)
        velocity_targets = velocity_targets.to(device)
        event_weight_targets = event_weight_targets.to(device)

        optimizer.zero_grad()
        onset_logits, velocity_logits = model(features)
        pred_onset = torch.sigmoid(onset_logits)
        pred_velocity = torch.sigmoid(velocity_logits)
        onset_smoothed = gaussian_smooth_targets(onset_targets, device)
        velocity_propagated = propagate_velocity_targets(velocity_targets)
        event_weight_propagated = propagate_velocity_targets(event_weight_targets)
        bce = nn.functional.binary_cross_entropy(pred_onset, onset_smoothed, reduction='none')
        active_weight = torch.where(onset_smoothed > 0.0, onset_pos_weights, onset_neg_weights)
        active_weight = torch.where(onset_smoothed > 0.0, active_weight * event_weight_propagated, active_weight)
        if hard_neg_boost > 0.0:
            # 中文註解：只強化模型已高分的負樣本，集中壓制 HH/SD 假陽性尖峰。
            hard_neg = 1.0 + hard_neg_boost * pred_onset.detach()
            active_weight = torch.where(onset_smoothed > 0.0, active_weight, active_weight * hard_neg)
        denom = train_mask.sum().clamp_min(1.0) * bce.shape[0] * bce.shape[1]
        loss_onset = (bce * active_weight * train_mask).sum() / denom
        loss_velocity = (onset_smoothed * (pred_velocity - velocity_propagated) ** 2 * train_mask).sum() / denom
        loss = loss_onset + 10.0 * loss_velocity
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(float(loss.item()))
        print(f'Batch {batch_idx + 1}/{min(len(loader), max_batches)} loss={losses[-1]:.4f}', flush=True)
    return losses


def main():
    """
    中文註解：CLI 入口，執行混合資料集 smoke/fine-tune 訓練。
    """
    parser = argparse.ArgumentParser(description='Mixed E-GMD/STAR/local XML trainer.')
    parser.add_argument('--egmd-meta', default='processed_data/egmd_meta.json')
    parser.add_argument('--star-meta', default='processed_data/star_meta.json')
    parser.add_argument('--local-meta', default='processed_data/local_xml_meta.json')
    parser.add_argument('--checkpoint', default='best_drum_model.pth')
    parser.add_argument('--output', default='mixed_candidate.pth')
    parser.add_argument('--samples', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--max-items-per-source', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-batches', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--mix-ratio', default='5,3,2')
    parser.add_argument('--onset-pos-weights', default='1,4,2')
    parser.add_argument('--onset-neg-weights', default='1,1,1')
    parser.add_argument('--hard-neg-boost', type=float, default=0.0)
    parser.add_argument('--train-channels', default='all')
    parser.add_argument('--snare-focus', action='store_true')
    parser.add_argument('--random-sampling', action='store_true')
    parser.add_argument('--balanced-sampler', action='store_true')
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--train-head-only', action='store_true')
    parser.add_argument('--freeze-bn', action='store_true')
    parser.add_argument('--validate-each-epoch', action='store_true')
    parser.add_argument('--validation-star-limit', type=int, default=2)
    parser.add_argument('--validation-root', default='validation_runs/mixed_formal')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()

    if args.self_check:
        run_self_check()
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    ratio = parse_mix_ratio(args.mix_ratio)
    dataset = MixedMetaDataset(
        args.egmd_meta,
        args.star_meta,
        args.local_meta,
        samples=args.samples,
        ratio=ratio,
        max_items_per_source=args.max_items_per_source,
        snare_focus=args.snare_focus,
        random_sampling=args.random_sampling,
        balanced_sampler=args.balanced_sampler,
        seed=args.seed,
    )
    print(f'Mix sources: egmd={len(dataset.sources["egmd"])}, star={len(dataset.sources["star"])}, local={len(dataset.sources["local"])}')
    print(f'Mix ratio: {ratio}')

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = SymmetricDrumTCN().to(device)
    load_checkpoint(model, args.checkpoint, device)
    if args.train_head_only:
        train_heads_only(model)
    optimizer = optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    onset_pos_weights = torch.tensor(parse_channel_weights(args.onset_pos_weights), device=device).view(1, 1, 3)
    onset_neg_weights = torch.tensor(parse_channel_weights(args.onset_neg_weights), device=device).view(1, 1, 3)
    train_mask = channel_mask(parse_train_channels(args.train_channels), device)
    print(f'Train channels mask: {train_mask.view(-1).detach().cpu().numpy().tolist()}')

    if args.freeze_bn:
        freeze_batchnorm_stats(model)
    best_fail_count = None
    all_losses = []
    for epoch in range(1, args.epochs + 1):
        print(f'Epoch {epoch}/{args.epochs}', flush=True)
        if args.freeze_bn:
            freeze_batchnorm_stats(model)
        losses = train_one_epoch(
            model, loader, optimizer, onset_pos_weights, onset_neg_weights,
            train_mask, device, args.max_batches, args.freeze_bn, args.hard_neg_boost
        )
        all_losses.extend(losses)
        epoch_path = args.output if args.epochs == 1 else f'{os.path.splitext(args.output)[0]}_epoch{epoch:03d}.pth'
        torch.save(model.state_dict(), epoch_path)
        print(f'Wrote epoch candidate: {epoch_path}', flush=True)
        if args.validate_each_epoch:
            fail_count, total_count, summary_csv = run_hard_validation(
                epoch_path,
                args.validation_root,
                epoch,
                args.validation_star_limit,
            )
            print(f'Hard validation: {total_count - fail_count}/{total_count} passed ({summary_csv})', flush=True)
            if best_fail_count is None or fail_count < best_fail_count:
                best_fail_count = fail_count
                torch.save(model.state_dict(), args.output)
                print(f'  -> Saved best mixed candidate so far: {args.output}', flush=True)
    print(f'Mixed training done. batches={len(all_losses)} first_loss={all_losses[0]:.4f} last_loss={all_losses[-1]:.4f}')
    if not args.validate_each_epoch:
        torch.save(model.state_dict(), args.output)
    print(f'Wrote candidate checkpoint: {args.output}')


if __name__ == '__main__':
    main()
