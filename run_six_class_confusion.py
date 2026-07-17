# -*- coding: utf-8 -*-
"""產生六類事件分類的比例混淆矩陣。"""
import argparse
import csv
import json
import os
import tempfile

import numpy as np
import torch

from model_conformer import ResidualDCNNDrumHybridConformer, load_hybrid_conformer_checkpoint
from run_six_class_smoke import LABELS, TARGET_SAMPLES, build_window, load_accompaniment
from run_six_class_validation import TOLERANCE, expected_events, local_maxima, select_windows


def match_window(expected, predicted, tolerance=TOLERANCE):
    """中文註解：同類 TP 優先，再以最小時間差配對剩餘跨類事件。"""
    expected_rows = [(label, time) for label in LABELS for time in expected[label]]
    predicted_rows = [(label, time) for label in LABELS for time in predicted[label]]
    used_expected, used_predicted = set(), set()
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)

    for label_index, label in enumerate(LABELS):
        for expected_index, (expected_label, expected_time) in enumerate(expected_rows):
            if expected_label != label:
                continue
            candidates = [
                (abs(predicted_time - expected_time), predicted_index)
                for predicted_index, (predicted_label, predicted_time) in enumerate(predicted_rows)
                if predicted_index not in used_predicted and predicted_label == label
                and abs(predicted_time - expected_time) <= tolerance
            ]
            if candidates:
                _, predicted_index = min(candidates)
                used_expected.add(expected_index)
                used_predicted.add(predicted_index)
                matrix[label_index, label_index] += 1

    cross_candidates = sorted(
        (abs(predicted_time - expected_time), expected_index, predicted_index)
        for expected_index, (_, expected_time) in enumerate(expected_rows)
        if expected_index not in used_expected
        for predicted_index, (_, predicted_time) in enumerate(predicted_rows)
        if predicted_index not in used_predicted and abs(predicted_time - expected_time) <= tolerance
    )
    for _, expected_index, predicted_index in cross_candidates:
        if expected_index in used_expected or predicted_index in used_predicted:
            continue
        used_expected.add(expected_index)
        used_predicted.add(predicted_index)
        true_label = expected_rows[expected_index][0]
        predicted_label = predicted_rows[predicted_index][0]
        matrix[LABELS.index(true_label), LABELS.index(predicted_label)] += 1

    missed = {label: 0 for label in LABELS}
    extra = {label: 0 for label in LABELS}
    for index, (label, _) in enumerate(expected_rows):
        missed[label] += index not in used_expected
    for index, (label, _) in enumerate(predicted_rows):
        extra[label] += index not in used_predicted
    return matrix, missed, extra


def write_outputs(matrix, expected_totals, predicted_totals, missed, extra, output_dir):
    """中文註解：寫出計數、逐列百分比、錯誤配對與 unmatched 比例。"""
    os.makedirs(output_dir, exist_ok=True)
    row_totals = matrix.sum(axis=1)
    percentages = np.divide(
        matrix * 100.0, row_totals[:, None],
        out=np.zeros_like(matrix, dtype=np.float64), where=row_totals[:, None] != 0,
    )
    for filename, values, formatter in (
        ('confusion_counts.csv', matrix, lambda value: int(value)),
        ('confusion_row_percent.csv', percentages, lambda value: f'{value:.2f}'),
    ):
        with open(os.path.join(output_dir, filename), 'w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['true\\predicted', *LABELS])
            for label, row in zip(LABELS, values):
                writer.writerow([label, *(formatter(value) for value in row)])

    total_confusions = int(matrix.sum() - np.trace(matrix))
    pairs = []
    for true_index, true_label in enumerate(LABELS):
        for predicted_index, predicted_label in enumerate(LABELS):
            if true_index == predicted_index or matrix[true_index, predicted_index] == 0:
                continue
            count = int(matrix[true_index, predicted_index])
            pairs.append({
                'true': true_label,
                'predicted': predicted_label,
                'count': count,
                'within_true_matched_percent': round(100.0 * count / row_totals[true_index], 2),
                'all_confusions_percent': round(100.0 * count / total_confusions, 2),
            })
    pairs.sort(key=lambda row: (-row['count'], row['true'], row['predicted']))
    with open(os.path.join(output_dir, 'error_pairs.csv'), 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0].keys()) if pairs else ['true', 'predicted', 'count', 'within_true_matched_percent', 'all_confusions_percent'])
        writer.writeheader()
        writer.writerows(pairs)

    unmatched = []
    for label in LABELS:
        unmatched.append({
            'class': label,
            'expected': expected_totals[label],
            'missed': missed[label],
            'missed_percent': round(100.0 * missed[label] / expected_totals[label], 2) if expected_totals[label] else 0.0,
            'predicted': predicted_totals[label],
            'extra': extra[label],
            'extra_percent': round(100.0 * extra[label] / predicted_totals[label], 2) if predicted_totals[label] else 0.0,
        })
    with open(os.path.join(output_dir, 'unmatched_rates.csv'), 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(unmatched[0].keys()))
        writer.writeheader()
        writer.writerows(unmatched)

    class_health = []
    for index, label in enumerate(LABELS):
        tp = int(matrix[index, index])
        precision = tp / predicted_totals[label] if predicted_totals[label] else 0.0
        recall = tp / expected_totals[label] if expected_totals[label] else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        class_health.append({
            'class': label, 'f1': round(f1, 4), 'precision': round(precision, 4), 'recall': round(recall, 4),
            'matched_confusion_percent': round(100.0 * (row_totals[index] - tp) / row_totals[index], 2) if row_totals[index] else 0.0,
            'missed_percent': next(row['missed_percent'] for row in unmatched if row['class'] == label),
            'extra_percent': next(row['extra_percent'] for row in unmatched if row['class'] == label),
        })
    class_health.sort(key=lambda row: (row['f1'], row['class']))
    with open(os.path.join(output_dir, 'class_health.csv'), 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(class_health[0].keys()))
        writer.writeheader()
        writer.writerows(class_health)
    summary = {
        'labels': list(LABELS), 'normalization': 'row_percent_among_temporally_matched_events',
        'tolerance_seconds': TOLERANCE, 'total_temporally_matched': int(matrix.sum()),
        'total_class_confusions': total_confusions, 'matrix_row_percent': percentages.round(2).tolist(),
        'error_pairs': pairs, 'unmatched': unmatched, 'class_health': class_health,
    }
    with open(os.path.join(output_dir, 'confusion_summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
    return summary


def run_self_check():
    """中文註解：確認同類優先、跨類配對與 unmatched 計數。"""
    expected = {label: [] for label in LABELS}
    predicted = {label: [] for label in LABELS}
    expected['KD'], expected['SD'], expected['RIDE'] = [0.10], [0.20], [0.50]
    predicted['KD'], predicted['HH'], predicted['CRASH'] = [0.11], [0.21], [0.80]
    matrix, missed, extra = match_window(expected, predicted)
    assert matrix[LABELS.index('KD'), LABELS.index('KD')] == 1
    assert matrix[LABELS.index('SD'), LABELS.index('HH')] == 1
    assert missed['RIDE'] == 1 and extra['CRASH'] == 1
    expected_totals = {label: len(expected[label]) for label in LABELS}
    predicted_totals = {label: len(predicted[label]) for label in LABELS}
    with tempfile.TemporaryDirectory() as output_dir:
        summary = write_outputs(matrix, expected_totals, predicted_totals, missed, extra, output_dir)
        kd = next(row for row in summary['class_health'] if row['class'] == 'KD')
        assert kd['f1'] == 1.0 and os.path.exists(os.path.join(output_dir, 'class_health.csv'))
    print('Self-check passed.')


def evaluate_confusion(
    model, metadata, output_dir, accompaniment=None, accompaniment_gain=0.17,
    per_class=8, feature_mode='legacy-diff', device=None,
):
    """中文註解：以已載入的最佳模型產生完整六類問題報告。"""
    device = device or next(model.parameters()).device
    model.eval()
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    expected_totals = {label: 0 for label in LABELS}
    predicted_totals = {label: 0 for label in LABELS}
    missed = {label: 0 for label in LABELS}
    extra = {label: 0 for label in LABELS}
    for window_index, selected in enumerate(select_windows(metadata, 'validation', per_class)):
        features, _, _, start_sec = build_window(
            selected['item'], selected['anchor'], accompaniment=accompaniment,
            accompaniment_gain=accompaniment_gain,
            accompaniment_offset=window_index * TARGET_SAMPLES,
            use_true_superflux=feature_mode == 'true-superflux',
        )
        with torch.no_grad():
            logits, _ = model(torch.from_numpy(features).float().unsqueeze(0).to(device))
        predicted = local_maxima(torch.sigmoid(logits).squeeze(0).cpu().numpy())
        expected = expected_events(selected['item'], start_sec)
        window_matrix, window_missed, window_extra = match_window(expected, predicted)
        matrix += window_matrix
        for label in LABELS:
            expected_totals[label] += len(expected[label])
            predicted_totals[label] += len(predicted[label])
            missed[label] += window_missed[label]
            extra[label] += window_extra[label]
    return write_outputs(matrix, expected_totals, predicted_totals, missed, extra, output_dir)


def main():
    """中文註解：載入 D7 best 並產生 STAR mixed validation 六類混淆診斷。"""
    parser = argparse.ArgumentParser(description='Generate a row-normalized six-class confusion matrix.')
    parser.add_argument('--meta')
    parser.add_argument('--model')
    parser.add_argument('--output-dir')
    parser.add_argument('--accompaniment')
    parser.add_argument('--accompaniment-gain', type=float, default=0.17)
    parser.add_argument('--per-class', type=int, default=8)
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
    model = ResidualDCNNDrumHybridConformer(num_classes=len(LABELS)).to(device)
    load_hybrid_conformer_checkpoint(model, args.model, device)
    accompaniment = load_accompaniment(args.accompaniment) if args.accompaniment else None
    summary = evaluate_confusion(
        model, metadata, args.output_dir, accompaniment=accompaniment,
        accompaniment_gain=args.accompaniment_gain, per_class=args.per_class, device=device,
    )
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
