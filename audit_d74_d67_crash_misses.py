# -*- coding: utf-8 -*-
"""固定融合輸出之 CRASH 漏檢替代類別唯讀審計。"""

import argparse
import csv
import json
import os

import numpy as np
import torch

from audit_d58_drumsep_errors import local_scores, match_indices, nearby_truth_labels, prepare_output_dir, count_by, write_csv
from audit_d67_d61_d64_tom_fusion import assert_same_selection, fuse_tom_probabilities, load_model, read_json
from run_six_class_smoke import build_window
from run_six_class_validation import LABELS, expected_events, load_fixed_windows, local_maxima


CRASH_LABEL = 'CRASH'


def audit_window(d61_model, d64_model, selected, window_index, tolerance):
    """中文註解：列出單一封存窗口內所有未配對的 CRASH 真值事件。"""
    features, _, _, start_sec = build_window(
        selected['item'], selected['anchor'], use_true_superflux=True,
        use_multi_log_mel=False, input_mode='drumsep-mix',
    )
    device = next(d61_model.parameters()).device
    feature_tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)
    with torch.no_grad():
        d61_probabilities = torch.sigmoid(d61_model(feature_tensor)[0]).squeeze(0).cpu().numpy()
        d64_probabilities = torch.sigmoid(d64_model(feature_tensor)[0]).squeeze(0).cpu().numpy()
    probabilities = fuse_tom_probabilities(d61_probabilities, d64_probabilities)
    expected, predicted = expected_events(selected['item'], start_sec), local_maxima(probabilities)
    matched_expected, _ = match_indices(expected[CRASH_LABEL], predicted[CRASH_LABEL], tolerance)
    base = {
        'window_index': window_index, 'key': selected['key'],
        'group_id': str(selected['item'].get('group_id') or selected['key']),
        'audio_path': selected['item']['audio_path'], 'anchor': round(float(selected['anchor']), 6),
        'window_start': round(float(start_sec), 6),
    }
    rows = []
    for event_index, event_time in enumerate(expected[CRASH_LABEL]):
        if event_index in matched_expected:
            continue
        scores = local_scores(probabilities, event_time, tolerance)
        alternatives = [(float(score), label) for label, score in zip(LABELS, scores) if label != CRASH_LABEL]
        alternative_probability, alternative_label = max(alternatives)
        rows.append({
            **base, 'event_time': round(float(event_time), 6), 'audio_time': round(float(start_sec + event_time), 6),
            'crash_probability': round(float(scores[LABELS.index(CRASH_LABEL)]), 6),
            'nearby_truth_labels': '|'.join(nearby_truth_labels(expected, event_time, CRASH_LABEL, tolerance)),
            'top_alternative': alternative_label, 'top_alternative_probability': round(alternative_probability, 6),
        })
    return rows


def dominant_competitor(rows):
    """中文註解：僅在單一替代類別嚴格過半時，回傳可供 D75 驗證的競爭類別。"""
    counts = count_by(rows, 'top_alternative')
    if not rows or not counts:
        return None, 0.0, counts
    label, count = next(iter(counts.items()))
    share = count / len(rows)
    return (label if share > 0.5 else None), share, counts


def run(args):
    """中文註解：重建固定 D67 windows，寫出 CRASH FN 診斷，不改任何模型或資料。"""
    output_dir = prepare_output_dir(args.output_dir)
    d61_selection, d64_selection = read_json(args.d61_selection), read_json(args.d64_selection)
    assert_same_selection(d61_selection, d64_selection)
    windows = load_fixed_windows(read_json(args.metadata), args.d61_selection, 'validation', args.per_class)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d61_model, d64_model = load_model(args.d61_checkpoint, device), load_model(args.d64_checkpoint, device)
    rows = []
    for index, selected in enumerate(windows):
        rows.extend(audit_window(d61_model, d64_model, selected, index, args.tolerance))
    fields = ['window_index', 'key', 'group_id', 'audio_path', 'anchor', 'window_start', 'event_time', 'audio_time', 'crash_probability', 'nearby_truth_labels', 'top_alternative', 'top_alternative_probability']
    write_csv(os.path.join(output_dir, 'crash_misses.csv'), rows, fields)
    competitor, share, counts = dominant_competitor(rows)
    summary = {
        'phase': args.phase, 'status': 'complete_read_only_crash_fn_audit',
        'recipe': args.recipe, 'selected_windows': len(windows),
        'selection_identical': True, 'input_mode': 'drumsep-mix', 'threshold': 0.5,
        'tolerance_seconds': args.tolerance, 'crash_false_negatives': len(rows),
        'crash_fn_by_top_alternative': counts, 'dominant_competitor': competitor,
        'dominant_competitor_share': round(share, 4),
        'eligible_for_d75': competitor is not None, 'ready_for_six_class_release': False,
    }
    with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def run_self_check():
    """中文註解：確認只有嚴格過半的替代類別才能啟動 D75。"""
    assert dominant_competitor([{'top_alternative': 'HH'}] * 2 + [{'top_alternative': 'KD'}])[0] == 'HH'
    assert dominant_competitor([{'top_alternative': 'HH'}, {'top_alternative': 'KD'}])[0] is None
    print('Self-check passed.')


def main():
    """中文註解：解析固定 CRASH FN 審計參數或執行最小自檢。"""
    parser = argparse.ArgumentParser(description='Audit fused CRASH false negatives on fixed validation windows.')
    parser.add_argument('--d61-checkpoint', default='validation_runs/d61_kd_negative_candidate/d61_kd_negative_candidate.pth')
    parser.add_argument('--d64-checkpoint', default='validation_runs/d64_tom_competitor_candidate/d64_tom_competitor_candidate.pth')
    parser.add_argument('--metadata', default='mixed_d54_stem/metadata_d54.json')
    parser.add_argument('--d61-selection', default='validation_runs/d61_kd_negative_candidate/independent_validation/selected_windows.json')
    parser.add_argument('--d64-selection', default='validation_runs/d64_tom_competitor_candidate/independent_validation/selected_windows.json')
    parser.add_argument('--output-dir', default='validation_runs/d74_d67_crash_miss_audit')
    parser.add_argument('--phase', default='D74')
    parser.add_argument('--recipe', default='D61 KD/SD/HH/CRASH/RIDE + D64 TOM')
    parser.add_argument('--per-class', type=int, default=8)
    parser.add_argument('--tolerance', type=float, default=0.05)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    run(args)


if __name__ == '__main__':
    main()
