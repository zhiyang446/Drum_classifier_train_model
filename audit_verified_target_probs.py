# -*- coding: utf-8 -*-
"""
Audit raw model probabilities on score-confirmed user blind annotations.

This script does not change checkpoints. It answers whether the acoustic model
is giving high probability at verified KD/SD/HH target frames.
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
import torch
import librosa

from dsp_utils import extract_features
from train_mixed_datasets import INST_INDICES, load_training_slice
from train_phase2 import SymmetricDrumTCN
from train_star_smoke import load_checkpoint


INST_NAMES = ['KD', 'SD', 'HH']
SR = 44100
HOP_LENGTH = 256
N_MELS = 256


def parse_args():
    """中文註解：解析診斷用 CLI 參數。"""
    parser = argparse.ArgumentParser(description='Audit verified onset target probabilities.')
    parser.add_argument('--meta', default='processed_data/user_blind_precise_verified_windowed_meta.json')
    parser.add_argument('--model', action='append', required=True, help='Checkpoint path. Can be passed multiple times.')
    parser.add_argument('--output-dir', default='validation_runs/raw_ai_model_fix/target_prob_audit')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--nearby-frames', type=int, default=2)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--mode', choices=['slice', 'full'], default='slice')
    return parser.parse_args()


def load_meta(path):
    """中文註解：讀取已確認 metadata，保留 key 方便追蹤來源。"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [(key, value) for key, value in data.items() if value.get('events')]


def base_name(item_key):
    """中文註解：把 window key 還原成原始音訊名稱。"""
    return item_key.rsplit('_win', 1)[0] if '_win' in item_key else item_key


def positive_frames(onset_target):
    """中文註解：列出每個已標註鼓件的 target frame。"""
    hits = []
    for frame, inst_idx in np.argwhere(onset_target > 0.5):
        hits.append((int(frame), int(inst_idx)))
    return hits


def full_track_targets(item):
    """中文註解：把完整音訊的 verified event time 轉成 frame 與 channel。"""
    hits = []
    for ev in item.get('events', []):
        inst = ev.get('inst')
        if inst not in INST_INDICES:
            continue
        frame = int(round(float(ev['time']) * SR / HOP_LENGTH))
        hits.append((frame, INST_INDICES[inst]))
    return hits


def predict_full_track(model, item, device):
    """中文註解：使用正式轉寫相同方式對完整音訊抽特徵與推論。"""
    y, _ = librosa.load(item['audio_path'], sr=SR, mono=True)
    features = extract_features(y, sr=SR, hop_length=HOP_LENGTH, n_mels=N_MELS, use_hybrid=False)
    tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)
    onset_logits, _velocity_logits = model(tensor)
    return torch.sigmoid(onset_logits)[0].detach().cpu().numpy()


def max_near(prob, frame, inst_idx, radius):
    """中文註解：取 target frame 附近小範圍最大值，容忍一點點 frame 偏移。"""
    start = max(0, frame - radius)
    end = min(prob.shape[0], frame + radius + 1)
    return float(prob[start:end, inst_idx].max())


def summarize(values, threshold):
    """中文註解：將一組機率轉成可讀統計。"""
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return {
            'count': 0,
            'mean': '',
            'median': '',
            'p10': '',
            'p90': '',
            'min': '',
            'max': '',
            'hit_rate': '',
        }
    return {
        'count': int(arr.size),
        'mean': float(arr.mean()),
        'median': float(np.median(arr)),
        'p10': float(np.percentile(arr, 10)),
        'p90': float(np.percentile(arr, 90)),
        'min': float(arr.min()),
        'max': float(arr.max()),
        'hit_rate': float((arr >= threshold).mean()),
    }


def audit_model(model_path, items, device, radius, threshold, mode):
    """中文註解：對單一 checkpoint 跑完整 verified window 診斷。"""
    model = SymmetricDrumTCN().to(device)
    load_checkpoint(model, model_path, device)
    model.eval()

    exact_by_inst = defaultdict(list)
    near_by_inst = defaultdict(list)
    bg_by_inst = defaultdict(list)
    detail_rows = []

    with torch.no_grad():
        for item_key, item in items:
            if mode == 'full':
                prob = predict_full_track(model, item, device)
                hits = [(frame, inst_idx) for frame, inst_idx in full_track_targets(item) if frame < prob.shape[0]]
                target_mask = np.zeros_like(prob, dtype=bool)
                for frame, inst_idx in hits:
                    target_mask[frame, inst_idx] = True
            else:
                features, onset_target, _velocity_target = load_training_slice(item)
                tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)
                onset_logits, _velocity_logits = model(tensor)
                prob = torch.sigmoid(onset_logits)[0].detach().cpu().numpy()
                target_mask = onset_target > 0.5
                hits = positive_frames(onset_target)

            for inst_idx, inst in enumerate(INST_NAMES):
                bg = prob[~target_mask[:, inst_idx], inst_idx]
                if bg.size:
                    bg_by_inst[inst].extend(bg.tolist())

            for frame, inst_idx in hits:
                inst = INST_NAMES[inst_idx]
                exact = float(prob[frame, inst_idx])
                nearby = max_near(prob, frame, inst_idx, radius)
                exact_by_inst[inst].append(exact)
                near_by_inst[inst].append(nearby)
                detail_rows.append({
                    'model': os.path.basename(model_path),
                    'item_key': item_key,
                    'file': base_name(item_key),
                    'frame': frame,
                    'inst': inst,
                    'prob_exact': exact,
                    'prob_near': nearby,
                    'hit_exact': exact >= threshold,
                    'hit_near': nearby >= threshold,
                })

    summary_rows = []
    for inst in INST_NAMES:
        exact = summarize(exact_by_inst[inst], threshold)
        near = summarize(near_by_inst[inst], threshold)
        bg = summarize(bg_by_inst[inst], threshold)
        summary_rows.append({
            'model': os.path.basename(model_path),
            'inst': inst,
            'target_count': exact['count'],
            'exact_mean': exact['mean'],
            'exact_median': exact['median'],
            'exact_p10': exact['p10'],
            'exact_p90': exact['p90'],
            'exact_min': exact['min'],
            'exact_max': exact['max'],
            'exact_hit_rate': exact['hit_rate'],
            'near_mean': near['mean'],
            'near_median': near['median'],
            'near_p10': near['p10'],
            'near_p90': near['p90'],
            'near_min': near['min'],
            'near_max': near['max'],
            'near_hit_rate': near['hit_rate'],
            'background_mean': bg['mean'],
            'background_p90': bg['p90'],
            'background_max': bg['max'],
        })
    return summary_rows, detail_rows


def write_csv(path, rows):
    """中文註解：寫出 CSV，沒有資料時仍保留空檔案。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        open(path, 'w', encoding='utf-8').close()
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    """中文註解：主流程，逐一 checkpoint 診斷並輸出報告。"""
    args = parse_args()
    items = load_meta(args.meta)
    all_summary = []
    all_detail = []
    for model_path in args.model:
        summary_rows, detail_rows = audit_model(
            model_path=model_path,
            items=items,
            device=args.device,
            radius=args.nearby_frames,
            threshold=args.threshold,
            mode=args.mode,
        )
        all_summary.extend(summary_rows)
        all_detail.extend(detail_rows)

    write_csv(os.path.join(args.output_dir, 'summary.csv'), all_summary)
    write_csv(os.path.join(args.output_dir, 'details.csv'), all_detail)
    print(f'Wrote {len(all_summary)} summary rows and {len(all_detail)} detail rows to {args.output_dir}')


if __name__ == '__main__':
    main()
