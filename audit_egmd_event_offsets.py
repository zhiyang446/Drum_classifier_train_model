# -*- coding: utf-8 -*-
"""
Audit E-GMD metadata event alignment against model probability peaks.
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import librosa
import numpy as np
import torch

from dsp_utils import extract_features
from train_mixed_datasets import INST_INDICES
from train_phase2 import SymmetricDrumTCN
from train_star_smoke import load_checkpoint


SR = 44100
HOP_LENGTH = 256
N_MELS = 256
INST_NAMES = ['KD', 'SD', 'HH']


def parse_args():
    """中文註解：解析 E-GMD 事件偏移診斷參數。"""
    parser = argparse.ArgumentParser(description='Audit E-GMD event peak offsets.')
    parser.add_argument('--meta', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--velocity-min', type=float, default=30.0)
    parser.add_argument('--radius', type=int, default=8)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def load_model(path, device):
    """中文註解：載入與正式推論相同架構的 checkpoint。"""
    model = SymmetricDrumTCN().to(device)
    load_checkpoint(model, path, device)
    model.eval()
    return model


def predict_probs(model, audio_path, device):
    """中文註解：對完整音訊輸出逐 frame KD/SD/HH 機率。"""
    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    features = extract_features(y, sr=SR, hop_length=HOP_LENGTH, n_mels=N_MELS, use_hybrid=False)
    tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)
    with torch.no_grad():
        onset_logits, _ = model(tensor)
    return torch.sigmoid(onset_logits)[0].detach().cpu().numpy()


def event_rows(meta_items, model, device, radius, velocity_min):
    """中文註解：逐一事件找附近最高模型機率與 frame/time 偏移。"""
    rows = []
    for key, item in meta_items:
        probs = predict_probs(model, item['audio_path'], device)
        for ev in item.get('events', []):
            inst = ev.get('inst')
            if inst not in INST_INDICES or float(ev.get('velocity', 0.0)) < velocity_min:
                continue
            inst_idx = INST_INDICES[inst]
            event_frame = int(round(float(ev['time']) * SR / HOP_LENGTH))
            if not (0 <= event_frame < probs.shape[0]):
                continue
            start = max(0, event_frame - radius)
            end = min(probs.shape[0], event_frame + radius + 1)
            local = probs[start:end, inst_idx]
            best_rel = int(np.argmax(local))
            best_frame = start + best_rel
            rows.append({
                'item': key,
                'file': os.path.basename(item['audio_path']),
                'inst': inst,
                'event_time': float(ev['time']),
                'velocity': float(ev.get('velocity', 0.0)),
                'event_frame': event_frame,
                'best_frame': best_frame,
                'offset_frames': best_frame - event_frame,
                'offset_seconds': (best_frame - event_frame) * HOP_LENGTH / SR,
                'prob_exact': float(probs[event_frame, inst_idx]),
                'prob_best': float(probs[best_frame, inst_idx]),
            })
    return rows


def summary_rows(rows):
    """中文註解：依鼓件彙總偏移分佈與機率提升。"""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['inst']].append(row)
    out = []
    for inst in INST_NAMES:
        vals = grouped.get(inst, [])
        if not vals:
            continue
        offsets = np.asarray([r['offset_frames'] for r in vals], dtype=np.float32)
        best_probs = np.asarray([r['prob_best'] for r in vals], dtype=np.float32)
        exact_probs = np.asarray([r['prob_exact'] for r in vals], dtype=np.float32)
        out.append({
            'inst': inst,
            'count': len(vals),
            'median_offset_frames': float(np.median(offsets)),
            'mean_offset_frames': float(np.mean(offsets)),
            'p10_offset_frames': float(np.percentile(offsets, 10)),
            'p90_offset_frames': float(np.percentile(offsets, 90)),
            'exact_hit_050': float(np.mean(exact_probs >= 0.5)),
            'best_hit_050': float(np.mean(best_probs >= 0.5)),
            'exact_median_prob': float(np.median(exact_probs)),
            'best_median_prob': float(np.median(best_probs)),
        })
    return out


def write_csv(path, rows):
    """中文註解：寫出診斷 CSV。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        open(path, 'w', encoding='utf-8').close()
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    """中文註解：主流程，輸出 event-level offset details 與 summary。"""
    args = parse_args()
    with open(args.meta, 'r', encoding='utf-8') as f:
        data = json.load(f)
    items = [(key, value) for key, value in data.items() if value.get('events')]
    model = load_model(args.model, args.device)
    rows = event_rows(items, model, args.device, args.radius, args.velocity_min)
    write_csv(os.path.join(args.output_dir, 'details.csv'), rows)
    write_csv(os.path.join(args.output_dir, 'summary.csv'), summary_rows(rows))
    print(f'Wrote {len(rows)} detail rows to {args.output_dir}')


if __name__ == '__main__':
    main()
