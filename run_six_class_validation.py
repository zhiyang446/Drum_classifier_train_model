# -*- coding: utf-8 -*-
"""使用 STAR test split 驗證獨立六類候選模型。"""
import argparse
import csv
import json
import os

import numpy as np
import torch

from run_egmd_round4_validation import match_events
from run_six_class_smoke import CHUNK_FRAMES, HOP_LENGTH, LABELS, LABEL_INDEX, SR, build_window, load_six_class_checkpoint
from train_phase2 import SymmetricDrumTCN


THRESHOLD = 0.50
TOLERANCE = 0.050
MACRO_F1_MIN = 0.70
PER_CLASS_F1_MIN = 0.55


def select_windows(metadata):
    """中文註解：每個鼓件挑一個最早的 STAR test 標註窗口，不依模型輸出選樣。"""
    selected = []
    for label in LABELS:
        candidates = []
        for key, item in metadata.items():
            if item.get('split') != 'test':
                continue
            for event in item.get('events', []):
                if event.get('inst') == label:
                    candidates.append((key, float(event['time']), item))
        if not candidates:
            raise ValueError(f'No STAR test event for {label}.')
        key, anchor, item = min(candidates, key=lambda row: (row[0], row[1]))
        selected.append({'label': label, 'key': key, 'anchor': anchor, 'item': item})
    return selected


def local_maxima(probabilities):
    """中文註解：以固定門檻擷取六類 onset 局部峰值，避免依個案調整。"""
    events = {label: [] for label in LABELS}
    for label, index in LABEL_INDEX.items():
        values = probabilities[:, index]
        for frame in range(1, len(values) - 1):
            if values[frame] >= THRESHOLD and values[frame] >= values[frame - 1] and values[frame] > values[frame + 1]:
                events[label].append(frame * HOP_LENGTH / float(SR))
    return events


def expected_events(item, start_sec):
    """中文註解：轉換指定物理窗口內的六類標註為窗口相對時間。"""
    end_sec = start_sec + CHUNK_FRAMES * HOP_LENGTH / float(SR)
    output = {label: [] for label in LABELS}
    for event in item.get('events', []):
        label = event.get('inst')
        time_sec = float(event['time'])
        if label in output and start_sec <= time_sec < end_sec:
            output[label].append(time_sec - start_sec)
    return output


def write_outputs(selected_rows, aggregate, output_dir):
    """中文註解：寫出可追溯的窗口選樣、逐類事件統計與固定門檻摘要。"""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'selected_windows.json'), 'w', encoding='utf-8') as handle:
        json.dump(selected_rows, handle, indent=2)
    fields = ['inst', 'expected', 'predicted', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'overall']
    rows = []
    for label in LABELS:
        expected, predicted = aggregate[label]
        tp, fp, fn, precision, recall, f1 = match_events(expected, predicted, TOLERANCE)
        rows.append({
            'inst': label, 'expected': len(expected), 'predicted': len(predicted), 'tp': tp, 'fp': fp, 'fn': fn,
            'precision': f'{precision:.4f}', 'recall': f'{recall:.4f}', 'f1': f'{f1:.4f}',
            'overall': 'pass' if f1 >= PER_CLASS_F1_MIN else 'fail',
        })
    with open(os.path.join(output_dir, 'event_compare.csv'), 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    macro_f1 = float(np.mean([float(row['f1']) for row in rows]))
    gate = {
        'gate': 'six_class_star_test_event',
        'overall': 'pass' if macro_f1 >= MACRO_F1_MIN and all(row['overall'] == 'pass' for row in rows) else 'fail',
        'macro_f1': round(macro_f1, 4),
        'macro_f1_min': MACRO_F1_MIN,
        'per_class_f1_min': PER_CLASS_F1_MIN,
        'threshold': THRESHOLD,
        'tolerance_seconds': TOLERANCE,
        'selected_windows': len(selected_rows),
    }
    with open(os.path.join(output_dir, 'gate_summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(gate, handle, indent=2)
    return rows, gate


def run_self_check():
    """中文註解：確認固定峰值與六類選樣邏輯可用。"""
    assert local_maxima(np.array([[0.0] * 6, [0.6] + [0.0] * 5, [0.2] + [0.0] * 5], dtype=np.float32))['KD'] == [HOP_LENGTH / float(SR)]
    test_meta = {f'case_{label}': {'split': 'test', 'events': [{'inst': label, 'time': 1.0}]} for label in LABELS}
    assert [row['label'] for row in select_windows(test_meta)] == list(LABELS)
    print('Self-check passed.')


def main():
    """中文註解：對六類候選執行 STAR test 物理事件驗收。"""
    parser = argparse.ArgumentParser(description='Validate an isolated six-class candidate on STAR test data.')
    parser.add_argument('--meta')
    parser.add_argument('--model')
    parser.add_argument('--output-dir')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    if not args.meta or not args.model or not args.output_dir:
        parser.error('--meta, --model, and --output-dir are required unless --self-check is used')
    with open(args.meta, encoding='utf-8') as handle:
        metadata = json.load(handle)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SymmetricDrumTCN(num_classes=len(LABELS)).to(device)
    load_six_class_checkpoint(model, args.model, device)
    model.eval()
    aggregate = {label: ([], []) for label in LABELS}
    selected_rows = []
    for selected in select_windows(metadata):
        features, _, _, start_sec = build_window(selected['item'], selected['anchor'])
        with torch.no_grad():
            logits, _ = model(torch.from_numpy(features).float().unsqueeze(0).to(device))
        predicted = local_maxima(torch.sigmoid(logits).squeeze(0).cpu().numpy())
        expected = expected_events(selected['item'], start_sec)
        for label in LABELS:
            aggregate[label][0].extend(expected[label])
            aggregate[label][1].extend(predicted[label])
        selected_rows.append({
            'label': selected['label'], 'key': selected['key'], 'anchor': selected['anchor'],
            'window_start': start_sec, 'audio_path': selected['item']['audio_path'],
            'expected_counts': {label: len(expected[label]) for label in LABELS},
        })
    rows, gate = write_outputs(selected_rows, aggregate, args.output_dir)
    print(json.dumps({'gate': gate, 'rows': rows}, indent=2))


if __name__ == '__main__':
    main()
