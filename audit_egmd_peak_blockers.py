# -*- coding: utf-8 -*-
"""
Audit why E-GMD metadata events do or do not become TCN peak candidates.
"""
import argparse
import csv
import json
import os
from collections import Counter, defaultdict

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
    """中文註解：解析 peak/NMS blocker 診斷參數。"""
    parser = argparse.ArgumentParser(description='Audit E-GMD peak picking blockers.')
    parser.add_argument('--meta', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--velocity-min', type=float, default=30.0)
    parser.add_argument('--radius', type=int, default=8)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--self-check', action='store_true')
    return parser.parse_args()


def load_model(path, device):
    """中文註解：載入正式推論同架構 checkpoint。"""
    model = SymmetricDrumTCN().to(device)
    load_checkpoint(model, path, device)
    model.eval()
    return model


def predict_probs_and_audio(model, audio_path, device):
    """中文註解：輸出完整音訊 waveform 與逐 frame onset 機率。"""
    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    features = extract_features(y, sr=SR, hop_length=HOP_LENGTH, n_mels=N_MELS, use_hybrid=False)
    tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)
    with torch.no_grad():
        onset_logits, _ = model(tensor)
    return y, torch.sigmoid(onset_logits)[0].detach().cpu().numpy()


def get_mgpc_thresh(prob, c_type):
    """中文註解：複製 transcribe.py 的 MGPC threshold，用於只讀診斷。"""
    peaks = []
    for t in range(2, len(prob) - 2):
        if prob[t] > prob[t - 1] and prob[t] > prob[t + 1] and prob[t] > prob[t - 2] and prob[t] > prob[t + 2] and prob[t] >= 0.12:
            peaks.append(prob[t])
    peaks = np.array(sorted(peaks, reverse=True))
    if len(peaks) == 0:
        return 0.65
    if len(peaks) == 1:
        return float(np.clip(peaks[0] * 0.5, 0.30, 0.60))
    gaps = peaks[:-1] - peaks[1:]
    best_gap = -1
    best_thresh = 0.50
    for i in range(len(gaps)):
        mid = (peaks[i] + peaks[i + 1]) / 2.0
        if c_type == 0:
            valid = 0.22 <= mid <= 0.65
        elif c_type == 1:
            valid = 0.22 <= mid <= 0.60
        else:
            valid = 0.20 <= mid <= 0.60
        if valid and gaps[i] > best_gap:
            best_gap = gaps[i]
            best_thresh = mid
    if best_gap == -1:
        max_p = peaks[0]
        if max_p < 0.22:
            return 0.65
        if max_p < 0.45:
            return float(np.max([0.22, max_p * 0.48]))
        return float(np.clip(max_p * 0.48, 0.30, 0.55))
    if c_type == 0:
        return float(np.clip(best_thresh, 0.30, 0.50))
    return float(np.clip(best_thresh, 0.25, 0.50))


def adaptive_thresholds(y, probs):
    """中文註解：複製 transcribe.py 的動態 threshold 陣列。"""
    n_frames = len(probs)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP_LENGTH)[0]
    if len(rms) < n_frames:
        rms = np.pad(rms, (0, n_frames - len(rms)), mode='edge')
    elif len(rms) > n_frames:
        rms = rms[:n_frames]
    rms_db = 20 * np.log10(rms + 1e-5)
    max_db = np.max(rms_db)
    min_db = np.max([np.min(rms_db), max_db - 40.0])
    rms_db_norm = np.clip((rms_db - min_db) / (max_db - min_db + 1e-6), 0.0, 1.0)
    bases = [get_mgpc_thresh(probs[:, idx], idx) for idx in range(3)]
    return [
        np.clip(bases[0] + (0.08 - 0.16 * rms_db_norm), 0.25, 0.75),
        np.clip(bases[1] + (0.08 - 0.16 * rms_db_norm), 0.25, 0.75),
        np.clip(bases[2] + (0.10 - 0.25 * rms_db_norm), 0.25, 0.75),
    ]


def get_peaks(prob, threshold, peak_radius=2, min_dist=6, valley_coef=0.60):
    """中文註解：複製 transcribe.py 的 NMS peak picker。"""
    peaks = []
    last_trigger_frame = -999
    for t in range(peak_radius, len(prob) - peak_radius):
        thresh_t = threshold[t] if hasattr(threshold, '__len__') else threshold
        if prob[t] <= thresh_t:
            continue
        if any(prob[t] <= prob[t - r] or prob[t] <= prob[t + r] for r in range(1, peak_radius + 1)):
            continue
        if t - last_trigger_frame >= min_dist:
            if last_trigger_frame != -999:
                valley_val = np.min(prob[last_trigger_frame:t])
                if valley_coef is not None and valley_val > valley_coef * prob[t]:
                    continue
            peaks.append(t)
            last_trigger_frame = t
    return peaks


def classify_event(prob, threshold, peaks, event_frame, radius):
    """中文註解：分類 metadata event 在 peak picker 前後的狀態。"""
    start = max(0, event_frame - radius)
    end = min(len(prob), event_frame + radius + 1)
    local = prob[start:end]
    best_frame = start + int(np.argmax(local))
    if any(abs(peak - event_frame) <= radius for peak in peaks):
        status = 'emitted_peak'
    elif prob[best_frame] <= threshold[best_frame]:
        status = 'below_dynamic_threshold'
    elif any(prob[best_frame] <= prob[best_frame - r] or prob[best_frame] <= prob[best_frame + r] for r in range(1, 3) if 0 <= best_frame - r and best_frame + r < len(prob)):
        status = 'not_strict_local_max'
    else:
        status = 'nms_min_dist_or_valley'
    return status, best_frame, float(prob[best_frame]), float(threshold[best_frame])


def audit_rows(meta_items, model, device, radius, velocity_min):
    """中文註解：逐 metadata event 輸出 peak picker 分類。"""
    rows = []
    for key, item in meta_items:
        y, probs = predict_probs_and_audio(model, item['audio_path'], device)
        thresholds = adaptive_thresholds(y, probs)
        peaks_by_inst = [get_peaks(probs[:, idx], thresholds[idx]) for idx in range(3)]
        for ev in item.get('events', []):
            inst = ev.get('inst')
            if inst not in INST_INDICES or float(ev.get('velocity', 0.0)) < velocity_min:
                continue
            inst_idx = INST_INDICES[inst]
            event_frame = int(round(float(ev['time']) * SR / HOP_LENGTH))
            if not (0 <= event_frame < len(probs)):
                continue
            status, best_frame, best_prob, dyn_thresh = classify_event(
                probs[:, inst_idx],
                thresholds[inst_idx],
                peaks_by_inst[inst_idx],
                event_frame,
                radius,
            )
            rows.append({
                'item': key,
                'file': os.path.basename(item['audio_path']),
                'inst': inst,
                'event_time': f"{float(ev['time']):.6f}",
                'velocity': f"{float(ev.get('velocity', 0.0)):.1f}",
                'event_frame': event_frame,
                'best_frame': best_frame,
                'offset_frames': best_frame - event_frame,
                'best_prob': f"{best_prob:.6f}",
                'dynamic_threshold': f"{dyn_thresh:.6f}",
                'status': status,
            })
    return rows


def summary_rows(rows):
    """中文註解：依鼓件與 blocker 類型彙總。"""
    grouped = defaultdict(Counter)
    for row in rows:
        grouped[row['inst']][row['status']] += 1
    out = []
    for inst in INST_NAMES:
        total = sum(grouped[inst].values())
        if total <= 0:
            continue
        for status, count in sorted(grouped[inst].items()):
            out.append({'inst': inst, 'status': status, 'count': count, 'ratio': f'{count / total:.4f}'})
    return out


def write_csv(path, rows):
    """中文註解：寫出 CSV；無資料時建立空檔。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        open(path, 'w', encoding='utf-8').close()
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_self_check():
    """中文註解：確認 blocker 分類基本可用。"""
    prob = np.array([0.0, 0.2, 0.6, 0.2, 0.0], dtype=np.float32)
    threshold = np.full_like(prob, 0.5)
    peaks = get_peaks(prob, threshold, peak_radius=1, min_dist=1)
    assert peaks == [2]
    status, _, _, _ = classify_event(prob, threshold, peaks, 2, 1)
    assert status == 'emitted_peak'
    print('Self-check passed.')


def main():
    """中文註解：主流程，輸出 peak blocker details 與 summary。"""
    args = parse_args()
    if args.self_check:
        run_self_check()
        return
    with open(args.meta, 'r', encoding='utf-8') as f:
        data = json.load(f)
    items = [(key, value) for key, value in data.items() if value.get('events')]
    model = load_model(args.model, args.device)
    rows = audit_rows(items, model, args.device, args.radius, args.velocity_min)
    write_csv(os.path.join(args.output_dir, 'details.csv'), rows)
    write_csv(os.path.join(args.output_dir, 'summary.csv'), summary_rows(rows))
    print(f'Wrote {len(rows)} peak blocker rows to {args.output_dir}')


if __name__ == '__main__':
    main()
