# -*- coding: utf-8 -*-
"""D79：以固定 D77 融合輸出審計 HH 的誤報與漏檢根因。"""

import argparse
import json
import os

import numpy as np
import torch

from audit_d58_drumsep_errors import count_by, local_scores, match_indices, nearby_truth_labels, prepare_output_dir, write_csv
from audit_d67_d61_d64_tom_fusion import assert_same_selection, fuse_tom_probabilities, load_model, read_json
from run_six_class_smoke import build_window
from run_six_class_validation import LABELS, expected_events, load_fixed_windows, local_maxima


HH_LABEL = 'HH'


def audit_window(five_class_model, tom_model, selected, window_index, tolerance):
    """中文註解：列出單一封存窗口中未配對的 HH 預測與真值事件。"""
    features, _, _, start_sec = build_window(
        selected['item'], selected['anchor'], use_true_superflux=True,
        use_multi_log_mel=False, input_mode='drumsep-mix',
    )
    device = next(five_class_model.parameters()).device
    feature_tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)
    with torch.no_grad():
        five_class_probabilities = torch.sigmoid(five_class_model(feature_tensor)[0]).squeeze(0).cpu().numpy()
        tom_probabilities = torch.sigmoid(tom_model(feature_tensor)[0]).squeeze(0).cpu().numpy()
    probabilities = fuse_tom_probabilities(five_class_probabilities, tom_probabilities)
    expected, predicted = expected_events(selected['item'], start_sec), local_maxima(probabilities)
    matched_expected, matched_predictions = match_indices(expected[HH_LABEL], predicted[HH_LABEL], tolerance)
    base = {
        'window_index': window_index, 'key': selected['key'],
        'group_id': str(selected['item'].get('group_id') or selected['key']),
        'audio_path': selected['item']['audio_path'], 'anchor': round(float(selected['anchor']), 6),
        'window_start': round(float(start_sec), 6),
    }
    hh_index = LABELS.index(HH_LABEL)
    false_positives, false_negatives = [], []
    for prediction_index, event_time in enumerate(predicted[HH_LABEL]):
        if prediction_index in matched_predictions:
            continue
        scores = local_scores(probabilities, event_time, tolerance)
        alternatives = [(float(score), label) for label, score in zip(LABELS, scores) if label != HH_LABEL]
        alternative_probability, alternative_label = max(alternatives)
        nearby = nearby_truth_labels(expected, event_time, HH_LABEL, tolerance)
        false_positives.append({
            **base, 'event_time': round(float(event_time), 6), 'audio_time': round(float(start_sec + event_time), 6),
            'hh_probability': round(float(scores[hh_index]), 6), 'nearby_truth_labels': '|'.join(nearby),
            'cause': 'cross_class' if nearby else 'unannotated',
            'top_alternative': alternative_label, 'top_alternative_probability': round(alternative_probability, 6),
        })
    for expected_index, event_time in enumerate(expected[HH_LABEL]):
        if expected_index in matched_expected:
            continue
        scores = local_scores(probabilities, event_time, tolerance)
        alternatives = [(float(score), label) for label, score in zip(LABELS, scores) if label != HH_LABEL]
        alternative_probability, alternative_label = max(alternatives)
        false_negatives.append({
            **base, 'event_time': round(float(event_time), 6), 'audio_time': round(float(start_sec + event_time), 6),
            'hh_probability': round(float(scores[hh_index]), 6),
            'nearby_truth_labels': '|'.join(nearby_truth_labels(expected, event_time, HH_LABEL, tolerance)),
            'top_alternative': alternative_label, 'top_alternative_probability': round(alternative_probability, 6),
        })
    return false_positives, false_negatives


def dominant_competitor(rows):
    """中文註解：只回報嚴格過半的 HH 漏檢替代類別，避免把分散錯誤當根因。"""
    counts = count_by(rows, 'top_alternative')
    if not rows or not counts:
        return None, 0.0
    label, count = next(iter(counts.items()))
    share = count / len(rows)
    return (label if share > 0.5 else None), round(share, 4)


def run(args):
    """中文註解：以 D77 固定融合重建封存窗口，輸出 HH FP/FN 根因證據。"""
    output_dir = prepare_output_dir(args.output_dir)
    five_class_selection, tom_selection = read_json(args.five_class_selection), read_json(args.tom_selection)
    assert_same_selection(five_class_selection, tom_selection)
    windows = load_fixed_windows(read_json(args.metadata), args.five_class_selection, 'validation', args.per_class)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    five_class_model, tom_model = load_model(args.five_class_checkpoint, device), load_model(args.tom_checkpoint, device)
    false_positives, false_negatives = [], []
    for window_index, selected in enumerate(windows):
        fp_rows, fn_rows = audit_window(five_class_model, tom_model, selected, window_index, args.tolerance)
        false_positives.extend(fp_rows)
        false_negatives.extend(fn_rows)
    fields = ['window_index', 'key', 'group_id', 'audio_path', 'anchor', 'window_start', 'event_time', 'audio_time', 'hh_probability', 'nearby_truth_labels', 'top_alternative', 'top_alternative_probability']
    write_csv(os.path.join(output_dir, 'hh_false_positives.csv'), false_positives, fields[:9] + ['cause'] + fields[9:])
    write_csv(os.path.join(output_dir, 'hh_false_negatives.csv'), false_negatives, fields)
    competitor, share = dominant_competitor(false_negatives)
    # ponytail: 只保留可決定下一步的兩種錯誤統計；無單一主因時不設計訓練配方。
    summary = {
        'phase': 'D79', 'status': 'complete_read_only_hh_error_audit',
        'recipe': 'D76 KD/SD/HH/CRASH/RIDE + D64 TOM', 'selected_windows': len(windows),
        'selection_identical': True, 'input_mode': 'drumsep-mix', 'threshold': 0.5,
        'tolerance_seconds': args.tolerance, 'hh_false_positives': len(false_positives),
        'hh_fp_by_cause': count_by(false_positives, 'cause'),
        'hh_fp_by_nearby_truth': count_by(false_positives, 'nearby_truth_labels'),
        'hh_fp_by_top_alternative': count_by(false_positives, 'top_alternative'),
        'hh_false_negatives': len(false_negatives),
        'hh_fn_by_top_alternative': count_by(false_negatives, 'top_alternative'),
        'hh_fn_dominant_competitor': competitor, 'hh_fn_dominant_competitor_share': share,
        'ready_for_training_candidate': False, 'ready_for_six_class_release': False,
    }
    with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def run_self_check():
    """中文註解：確認 HH 匹配與嚴格過半競爭類別判定不會反轉。"""
    matched_expected, matched_predictions = match_indices([1.0], [1.02, 2.0], 0.05)
    assert matched_expected == {0} and matched_predictions == {0}
    assert nearby_truth_labels({'HH': [], 'KD': [1.0]}, 1.02, HH_LABEL, 0.05) == ['KD']
    assert dominant_competitor([{'top_alternative': 'KD'}] * 2 + [{'top_alternative': 'SD'}])[0] == 'KD'
    assert dominant_competitor([{'top_alternative': 'KD'}, {'top_alternative': 'SD'}])[0] is None
    print('Self-check passed.')


def main():
    """中文註解：解析 D79 固定融合輸入並執行 HH 根因審計或自檢。"""
    parser = argparse.ArgumentParser(description='Audit D77 fused HH false positives and false negatives.')
    parser.add_argument('--five-class-checkpoint', default='validation_runs/d76_crash_kd_retry_candidate/d76_crash_kd_retry_candidate.pth')
    parser.add_argument('--tom-checkpoint', default='validation_runs/d64_tom_competitor_candidate/d64_tom_competitor_candidate.pth')
    parser.add_argument('--metadata', default='mixed_d54_stem/metadata_d54.json')
    parser.add_argument('--five-class-selection', default='validation_runs/d61_kd_negative_candidate/independent_validation/selected_windows.json')
    parser.add_argument('--tom-selection', default='validation_runs/d64_tom_competitor_candidate/independent_validation/selected_windows.json')
    parser.add_argument('--output-dir', default='validation_runs/d79_d77_hh_error_audit')
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
