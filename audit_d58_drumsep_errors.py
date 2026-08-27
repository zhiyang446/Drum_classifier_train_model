# -*- coding: utf-8 -*-
"""D58：以固定 validation 窗口自動稽核 D56 的 CRASH 誤報與 TOM 漏檢。"""
import argparse
import csv
import json
import math
import os

import numpy as np
import torch

import run_six_class_validation as validator


def prepare_output_dir(path):
    """只建立新的空輸出目錄，避免覆寫既有驗收證據。"""
    if os.path.exists(path) and os.listdir(path):
        raise FileExistsError(f'Output directory is not empty: {path}')
    os.makedirs(path, exist_ok=True)
    return path


def match_indices(expected, predicted, tolerance):
    """以時間容忍度做一對一貪婪匹配，回傳已匹配的真值與預測索引。"""
    matched_expected, matched_predicted = set(), set()
    for pred_index, pred_time in enumerate(predicted):
        choices = [
            (abs(pred_time - target_time), target_index)
            for target_index, target_time in enumerate(expected)
            if target_index not in matched_expected and abs(pred_time - target_time) <= tolerance
        ]
        if choices:
            _, target_index = min(choices)
            matched_expected.add(target_index)
            matched_predicted.add(pred_index)
    return matched_expected, matched_predicted


def nearby_truth_labels(expected_by_label, event_time, excluded_label, tolerance):
    """列出指定時間附近的其他真值類別，供 CRASH 誤報歸因使用。"""
    return [
        label for label, times in expected_by_label.items()
        if label != excluded_label and any(abs(event_time - target_time) <= tolerance for target_time in times)
    ]


def local_scores(probabilities, event_time, tolerance):
    """取事件附近各類別最大機率，避免單一 frame 偏移造成錯誤歸因。"""
    center = int(round(event_time * validator.SR / validator.HOP_LENGTH))
    radius = int(math.ceil(tolerance * validator.SR / validator.HOP_LENGTH))
    start, end = max(0, center - radius), min(len(probabilities), center + radius + 1)
    return probabilities[start:end].max(axis=0)


def write_csv(path, rows, fieldnames):
    """寫出含固定表頭的 UTF-8 CSV，讓空結果也可被後續工具讀取。"""
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def count_by(rows, field):
    """統計指定欄位的非空值，輸出排序後的可序列化字典。"""
    counts = {}
    for row in rows:
        value = str(row.get(field, ''))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def audit_window(model, selected, window_index, tolerance):
    """對單一固定窗口輸出 CRASH 誤報與 TOM 漏檢的逐事件診斷列。"""
    # 重要變數：input_mode 固定為 D56 的 drumsep-mix，避免審計混入 raw-mix 行為。
    features, _, _, start_sec = validator.build_window(
        selected['item'], selected['anchor'], use_true_superflux=True, input_mode='drumsep-mix'
    )
    device = next(model.parameters()).device
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(features).float().unsqueeze(0).to(device))
    probabilities = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    expected = validator.expected_events(selected['item'], start_sec)
    predicted = validator.local_maxima(probabilities)

    base = {
        'window_index': window_index,
        'key': selected['key'],
        'group_id': str(selected['item'].get('group_id') or selected['key']),
        'audio_path': selected['item']['audio_path'],
        'anchor': round(float(selected['anchor']), 6),
        'window_start': round(float(start_sec), 6),
    }
    crash_rows, tom_rows = [], []
    crash_expected, crash_predicted = expected['CRASH'], predicted['CRASH']
    _, matched_crash_predictions = match_indices(crash_expected, crash_predicted, tolerance)
    for pred_index, event_time in enumerate(crash_predicted):
        if pred_index in matched_crash_predictions:
            continue
        nearby = nearby_truth_labels(expected, event_time, 'CRASH', tolerance)
        scores = local_scores(probabilities, event_time, tolerance)
        crash_rows.append({
            **base,
            'event_time': round(float(event_time), 6),
            'audio_time': round(float(start_sec + event_time), 6),
            'probability': round(float(scores[validator.LABELS.index('CRASH')]), 6),
            'nearby_truth_labels': '|'.join(nearby),
            'cause': 'cross_class' if nearby else 'unannotated',
        })

    tom_expected, tom_predicted = expected['TOM'], predicted['TOM']
    matched_tom_expected, _ = match_indices(tom_expected, tom_predicted, tolerance)
    tom_index = validator.LABELS.index('TOM')
    for target_index, event_time in enumerate(tom_expected):
        if target_index in matched_tom_expected:
            continue
        scores = local_scores(probabilities, event_time, tolerance)
        alternatives = [(float(score), label) for label, score in zip(validator.LABELS, scores) if label != 'TOM']
        alternative_probability, alternative_label = max(alternatives)
        tom_rows.append({
            **base,
            'event_time': round(float(event_time), 6),
            'audio_time': round(float(start_sec + event_time), 6),
            'tom_probability': round(float(scores[tom_index]), 6),
            'top_alternative': alternative_label,
            'top_alternative_probability': round(alternative_probability, 6),
        })
    return crash_rows, tom_rows


def run(args):
    """讀取封存窗口並執行 D56 checkpoint 的只讀錯誤審計。"""
    output_dir = prepare_output_dir(args.output_dir)
    with open(args.meta, encoding='utf-8') as handle:
        metadata = json.load(handle)
    windows = validator.load_fixed_windows(metadata, args.selected_windows, 'validation', args.per_class)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = validator.ResidualDCNNDrumHybridConformer(num_classes=len(validator.LABELS)).to(device)
    validator.load_hybrid_conformer_checkpoint(model, args.model, device)
    model.eval()

    crash_rows, tom_rows = [], []
    for window_index, selected in enumerate(windows):
        crash_part, tom_part = audit_window(model, selected, window_index, args.tolerance)
        crash_rows.extend(crash_part)
        tom_rows.extend(tom_part)

    crash_fields = ['window_index', 'key', 'group_id', 'audio_path', 'anchor', 'window_start', 'event_time', 'audio_time', 'probability', 'nearby_truth_labels', 'cause']
    tom_fields = ['window_index', 'key', 'group_id', 'audio_path', 'anchor', 'window_start', 'event_time', 'audio_time', 'tom_probability', 'top_alternative', 'top_alternative_probability']
    write_csv(os.path.join(output_dir, 'crash_false_positives.csv'), crash_rows, crash_fields)
    write_csv(os.path.join(output_dir, 'tom_misses.csv'), tom_rows, tom_fields)
    summary = {
        'phase': 'D58', 'status': 'complete_read_only_error_audit',
        'model': os.path.abspath(args.model), 'selected_windows': len(windows),
        'input_mode': 'drumsep-mix', 'threshold': validator.THRESHOLD, 'tolerance_seconds': args.tolerance,
        'crash_false_positives': len(crash_rows), 'crash_by_cause': count_by(crash_rows, 'cause'),
        'crash_by_nearby_truth': count_by(crash_rows, 'nearby_truth_labels'),
        'tom_misses': len(tom_rows), 'tom_by_top_alternative': count_by(tom_rows, 'top_alternative'),
    }
    with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def run_self_check():
    """驗證一對一匹配、鄰近類別與局部機率歸因的最小邊界案例。"""
    matched_expected, matched_predicted = match_indices([1.0], [1.02, 2.0], 0.05)
    assert matched_expected == {0} and matched_predicted == {0}
    assert nearby_truth_labels({'CRASH': [], 'HH': [1.0]}, 1.02, 'CRASH', 0.05) == ['HH']
    scores = local_scores(np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32), 0.0, 0.01)
    assert np.allclose(scores, [0.8, 0.9])
    print('Self-check passed.')


def main():
    """解析 D58 參數並防止沒有固定窗口的審計。"""
    parser = argparse.ArgumentParser(description='Audit D56 CRASH false positives and TOM misses on fixed validation windows.')
    parser.add_argument('--meta')
    parser.add_argument('--model')
    parser.add_argument('--selected-windows')
    parser.add_argument('--output-dir')
    parser.add_argument('--per-class', type=int, default=8)
    parser.add_argument('--tolerance', type=float, default=0.05)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    if not all((args.meta, args.model, args.selected_windows, args.output_dir)):
        parser.error('--meta, --model, --selected-windows, and --output-dir are required unless --self-check is used')
    run(args)


if __name__ == '__main__':
    main()
