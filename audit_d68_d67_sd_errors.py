# -*- coding: utf-8 -*-
"""D68：以固定 D67 融合輸出審計 SD false positive 的根因。"""

import argparse
import json
import os

import numpy as np
import torch

from audit_d58_drumsep_errors import (
    count_by,
    local_scores,
    match_indices,
    nearby_truth_labels,
    prepare_output_dir,
    write_csv,
)
from audit_d67_d61_d64_tom_fusion import (
    assert_same_selection,
    fuse_tom_probabilities,
    load_model,
    read_json,
)
from run_six_class_smoke import build_window
from run_six_class_validation import LABELS, expected_events, load_fixed_windows, local_maxima


SD_LABEL = 'SD'


def audit_window(d61_model, d64_model, selected, window_index, tolerance):
    """中文註解：列出單一封存窗口中所有未配對的 SD 預測。"""
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
    expected = expected_events(selected['item'], start_sec)
    predicted = local_maxima(probabilities)
    _, matched_predictions = match_indices(expected[SD_LABEL], predicted[SD_LABEL], tolerance)
    base = {
        'window_index': window_index,
        'key': selected['key'],
        'group_id': str(selected['item'].get('group_id') or selected['key']),
        'audio_path': selected['item']['audio_path'],
        'anchor': round(float(selected['anchor']), 6),
        'window_start': round(float(start_sec), 6),
    }
    rows = []
    for prediction_index, event_time in enumerate(predicted[SD_LABEL]):
        if prediction_index in matched_predictions:
            continue
        scores = local_scores(probabilities, event_time, tolerance)
        alternatives = [(float(score), label) for label, score in zip(LABELS, scores) if label != SD_LABEL]
        alternative_probability, alternative_label = max(alternatives)
        nearby = nearby_truth_labels(expected, event_time, SD_LABEL, tolerance)
        rows.append({
            **base,
            'event_time': round(float(event_time), 6),
            'audio_time': round(float(start_sec + event_time), 6),
            'sd_probability': round(float(scores[LABELS.index(SD_LABEL)]), 6),
            'nearby_truth_labels': '|'.join(nearby),
            'cause': 'cross_class' if nearby else 'unannotated',
            'top_alternative': alternative_label,
            'top_alternative_probability': round(alternative_probability, 6),
        })
    return rows


def run(args):
    """中文註解：驗證 D67 選窗後，對融合模型輸出寫入新的 SD 誤報證據。"""
    output_dir = prepare_output_dir(args.output_dir)
    d61_selection = read_json(args.d61_selection)
    d64_selection = read_json(args.d64_selection)
    assert_same_selection(d61_selection, d64_selection)
    metadata = read_json(args.metadata)
    windows = load_fixed_windows(metadata, args.d61_selection, 'validation', args.per_class)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d61_model = load_model(args.d61_checkpoint, device)
    d64_model = load_model(args.d64_checkpoint, device)
    rows = []
    for window_index, selected in enumerate(windows):
        rows.extend(audit_window(d61_model, d64_model, selected, window_index, args.tolerance))
    fields = [
        'window_index', 'key', 'group_id', 'audio_path', 'anchor', 'window_start',
        'event_time', 'audio_time', 'sd_probability', 'nearby_truth_labels', 'cause',
        'top_alternative', 'top_alternative_probability',
    ]
    write_csv(os.path.join(output_dir, 'sd_false_positives.csv'), rows, fields)
    # ponytail: 只輸出計數與逐筆 CSV；下一輪需要資料配方時才加其他圖表或聚合。
    summary = {
        'phase': 'D68',
        'status': 'complete_read_only_error_audit',
        'recipe': 'D61 KD/SD/HH/CRASH/RIDE + D64 TOM',
        'selected_windows': len(windows),
        'selection_identical': True,
        'input_mode': 'drumsep-mix',
        'threshold': 0.5,
        'tolerance_seconds': args.tolerance,
        'sd_false_positives': len(rows),
        'sd_by_cause': count_by(rows, 'cause'),
        'sd_by_nearby_truth': count_by(rows, 'nearby_truth_labels'),
        'sd_by_top_alternative': count_by(rows, 'top_alternative'),
        'ready_for_training_candidate': False,
        'ready_for_six_class_release': False,
    }
    with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def run_self_check():
    """中文註解：確認 SD 一對一匹配、鄰近真值與替代機率的基本邊界。"""
    matched_expected, matched_predictions = match_indices([1.0], [1.02, 2.0], 0.05)
    assert matched_expected == {0} and matched_predictions == {0}
    assert nearby_truth_labels({'SD': [], 'KD': [1.0]}, 1.02, SD_LABEL, 0.05) == ['KD']
    scores = local_scores(np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32), 0.0, 0.01)
    assert np.allclose(scores, [0.8, 0.9])
    print('Self-check passed.')


def main():
    """中文註解：解析 D68 參數，執行只讀審計或最小自檢。"""
    parser = argparse.ArgumentParser(description='Audit D67 fused SD false positives on fixed validation windows.')
    parser.add_argument('--d61-checkpoint', default='validation_runs/d61_kd_negative_candidate/d61_kd_negative_candidate.pth')
    parser.add_argument('--d64-checkpoint', default='validation_runs/d64_tom_competitor_candidate/d64_tom_competitor_candidate.pth')
    parser.add_argument('--metadata', default='mixed_d54_stem/metadata_d54.json')
    parser.add_argument('--d61-selection', default='validation_runs/d61_kd_negative_candidate/independent_validation/selected_windows.json')
    parser.add_argument('--d64-selection', default='validation_runs/d64_tom_competitor_candidate/independent_validation/selected_windows.json')
    parser.add_argument('--output-dir', default='validation_runs/d68_d67_sd_error_audit')
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
