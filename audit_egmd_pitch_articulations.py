# -*- coding: utf-8 -*-
"""
Audit E-GMD pitch/articulation coverage against model probabilities.
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import librosa
import numpy as np
import pretty_midi
import torch

from dsp_utils import extract_features
from train_mixed_datasets import INST_INDICES
from train_phase2 import SymmetricDrumTCN
from train_star_smoke import load_checkpoint


SR = 44100
HOP_LENGTH = 256
N_MELS = 256
PITCH_TO_INST = {
    35: 'KD',
    36: 'KD',
    37: 'SD',
    38: 'SD',
    40: 'SD',
    22: 'HH',
    26: 'HH',
    42: 'HH',
    44: 'HH',
    46: 'HH',
}


def parse_args():
    """中文註解：解析 pitch/articulation 診斷參數。"""
    parser = argparse.ArgumentParser(description='Audit E-GMD pitch/articulation probabilities.')
    parser.add_argument('--meta', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--velocity-min', type=float, default=30.0)
    parser.add_argument('--radius', type=int, default=8)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--self-check', action='store_true')
    return parser.parse_args()


def load_model(path, device):
    """中文註解：載入與正式推論一致的 SymmetricDrumTCN checkpoint。"""
    model = SymmetricDrumTCN().to(device)
    load_checkpoint(model, path, device)
    model.eval()
    return model


def predict_probs(model, audio_path, device):
    """中文註解：對完整音訊輸出逐 frame KD/SD/HH onset 機率。"""
    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    features = extract_features(y, sr=SR, hop_length=HOP_LENGTH, n_mels=N_MELS, use_hybrid=False)
    tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)
    with torch.no_grad():
        onset_logits, _ = model(tensor)
    return torch.sigmoid(onset_logits)[0].detach().cpu().numpy()


def midi_path_for_audio(audio_path):
    """中文註解：依 E-GMD 音訊路徑推導 sibling MIDI 路徑。"""
    return os.path.splitext(audio_path)[0] + '.midi'


def read_pitch_events(audio_path, velocity_min):
    """中文註解：從 sibling MIDI 讀取保留原始 pitch 的 KD/SD/HH 事件。"""
    midi_path = midi_path_for_audio(audio_path)
    if not os.path.exists(midi_path):
        return []
    midi = pretty_midi.PrettyMIDI(midi_path)
    events = []
    for instrument in midi.instruments:
        for note in instrument.notes:
            inst = PITCH_TO_INST.get(note.pitch)
            if inst is None or note.velocity < velocity_min:
                continue
            events.append({
                'time': float(note.start),
                'inst': inst,
                'pitch': int(note.pitch),
                'velocity': float(note.velocity),
            })
    events.sort(key=lambda row: (row['time'], row['pitch']))
    return events


def audit_item(key, item, model, device, radius, velocity_min):
    """中文註解：逐事件尋找附近最高機率，並保留原始 MIDI pitch。"""
    audio_path = item['audio_path']
    probs = predict_probs(model, audio_path, device)
    rows = []
    for ev in read_pitch_events(audio_path, velocity_min):
        inst_idx = INST_INDICES[ev['inst']]
        event_frame = int(round(ev['time'] * SR / HOP_LENGTH))
        if not (0 <= event_frame < probs.shape[0]):
            continue
        start = max(0, event_frame - radius)
        end = min(probs.shape[0], event_frame + radius + 1)
        local = probs[start:end, inst_idx]
        best_rel = int(np.argmax(local))
        best_frame = start + best_rel
        rows.append({
            'item': key,
            'file': os.path.basename(audio_path),
            'inst': ev['inst'],
            'pitch': ev['pitch'],
            'event_time': f"{ev['time']:.6f}",
            'velocity': f"{ev['velocity']:.1f}",
            'event_frame': event_frame,
            'best_frame': best_frame,
            'offset_frames': best_frame - event_frame,
            'prob_exact': f"{float(probs[event_frame, inst_idx]):.6f}",
            'prob_best': f"{float(probs[best_frame, inst_idx]):.6f}",
        })
    return rows


def summary_rows(rows):
    """中文註解：依 inst/pitch 彙總命中率、機率與力度分布。"""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['inst'], row['pitch'])].append(row)
    output = []
    for (inst, pitch), vals in sorted(grouped.items()):
        best = np.asarray([float(row['prob_best']) for row in vals], dtype=np.float32)
        exact = np.asarray([float(row['prob_exact']) for row in vals], dtype=np.float32)
        velocity = np.asarray([float(row['velocity']) for row in vals], dtype=np.float32)
        output.append({
            'inst': inst,
            'pitch': pitch,
            'events': len(vals),
            'best_hit_050': f"{float(np.mean(best >= 0.5)):.4f}",
            'best_hit_030': f"{float(np.mean(best >= 0.3)):.4f}",
            'exact_hit_050': f"{float(np.mean(exact >= 0.5)):.4f}",
            'best_median_prob': f"{float(np.median(best)):.6f}",
            'exact_median_prob': f"{float(np.median(exact)):.6f}",
            'median_velocity': f"{float(np.median(velocity)):.1f}",
        })
    return output


def write_csv(path, rows):
    """中文註解：寫出 CSV；無資料時仍建立空檔。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        open(path, 'w', encoding='utf-8').close()
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_self_check():
    """中文註解：確認 pitch 對應與 MIDI sibling path 推導。"""
    assert PITCH_TO_INST[36] == 'KD'
    assert PITCH_TO_INST[40] == 'SD'
    assert PITCH_TO_INST[46] == 'HH'
    assert midi_path_for_audio('a/b/c.wav').endswith('a/b/c.midi')
    print('Self-check passed.')


def main():
    """中文註解：主流程，輸出 pitch/articulation details 與 summary。"""
    args = parse_args()
    if args.self_check:
        run_self_check()
        return
    with open(args.meta, 'r', encoding='utf-8') as f:
        data = json.load(f)
    items = [(key, value) for key, value in data.items() if value.get('events')]
    if args.limit > 0:
        items = items[:args.limit]
    model = load_model(args.model, args.device)
    rows = []
    for key, item in items:
        rows.extend(audit_item(key, item, model, args.device, args.radius, args.velocity_min))
    write_csv(os.path.join(args.output_dir, 'details.csv'), rows)
    write_csv(os.path.join(args.output_dir, 'summary.csv'), summary_rows(rows))
    print(f'Wrote {len(rows)} pitch/articulation rows to {args.output_dir}')


if __name__ == '__main__':
    main()
