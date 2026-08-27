# -*- coding: utf-8 -*-
"""D83：比較 D77/D82 固定驗收，並稽核 D82 RIDE regression 的根因。"""

import argparse
import csv
import json
import os

import torch

from audit_d58_drumsep_errors import count_by, local_scores, match_indices, nearby_truth_labels, prepare_output_dir, write_csv
from audit_d67_d61_d64_tom_fusion import assert_same_selection, read_json
from audit_d72_d70_delta import build_rows, read_metrics, validate_inputs
from run_six_class_smoke import build_window
from run_six_class_validation import LABELS, expected_events, load_fixed_windows, local_maxima
from train_d77_fused_lora import file_sha256, fuse_tom_logits, load_frozen_lora_model


RIDE_LABEL = 'RIDE'


def load_adapter(head, adapter_state, device):
    """中文註解：只把已驗證的低秩張量載入 D82 wrapper，不接觸 frozen base 權重。"""
    if int(adapter_state['rank']) != head.rank or float(adapter_state['scale']) != head.scale:
        raise ValueError('Adapter rank or scale differs from the D82 payload.')
    with torch.no_grad():
        head.down.weight.copy_(adapter_state['down.weight'].to(device))
        head.up.weight.copy_(adapter_state['up.weight'].to(device))


def strict_dominant(rows, field):
    """中文註解：只有單一替代原因嚴格過半才回報可審核主因。"""
    counts = count_by(rows, field)
    if not rows or not counts:
        return None, 0.0
    label, count = next(iter(counts.items()))
    share = count / len(rows)
    return (label if share > 0.5 else None), round(share, 4)


def audit_window(d76_model, d64_model, selected, window_index, tolerance):
    """中文註解：列出一個封存窗口的 RIDE 未配對預測與真值，保留局部競爭證據。"""
    features, _, _, start_sec = build_window(
        selected['item'], selected['anchor'], use_true_superflux=True,
        use_multi_log_mel=False, input_mode='drumsep-mix',
    )
    device = next(d76_model.parameters()).device
    feature_tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)
    with torch.no_grad():
        d76_logits, _ = d76_model(feature_tensor)
        d64_logits, _ = d64_model(feature_tensor)
        probabilities = torch.sigmoid(fuse_tom_logits(d76_logits, d64_logits)).squeeze(0).cpu().numpy()
    expected, predicted = expected_events(selected['item'], start_sec), local_maxima(probabilities)
    matched_expected, matched_predictions = match_indices(expected[RIDE_LABEL], predicted[RIDE_LABEL], tolerance)
    base = {
        'window_index': window_index, 'key': selected['key'],
        'group_id': str(selected['item'].get('group_id') or selected['key']),
        'audio_path': selected['item']['audio_path'], 'anchor': round(float(selected['anchor']), 6),
        'window_start': round(float(start_sec), 6),
    }
    ride_index = LABELS.index(RIDE_LABEL)
    false_positives, false_negatives = [], []
    for prediction_index, event_time in enumerate(predicted[RIDE_LABEL]):
        if prediction_index in matched_predictions:
            continue
        scores = local_scores(probabilities, event_time, tolerance)
        alternatives = [(float(score), label) for label, score in zip(LABELS, scores) if label != RIDE_LABEL]
        alternative_probability, alternative_label = max(alternatives)
        nearby = nearby_truth_labels(expected, event_time, RIDE_LABEL, tolerance)
        false_positives.append({
            **base, 'event_time': round(float(event_time), 6), 'audio_time': round(float(start_sec + event_time), 6),
            'ride_probability': round(float(scores[ride_index]), 6), 'nearby_truth_labels': '|'.join(nearby),
            'cause': 'cross_class' if nearby else 'unannotated', 'top_alternative': alternative_label,
            'top_alternative_probability': round(alternative_probability, 6),
        })
    for expected_index, event_time in enumerate(expected[RIDE_LABEL]):
        if expected_index in matched_expected:
            continue
        scores = local_scores(probabilities, event_time, tolerance)
        alternatives = [(float(score), label) for label, score in zip(LABELS, scores) if label != RIDE_LABEL]
        alternative_probability, alternative_label = max(alternatives)
        false_negatives.append({
            **base, 'event_time': round(float(event_time), 6), 'audio_time': round(float(start_sec + event_time), 6),
            'ride_probability': round(float(scores[ride_index]), 6),
            'nearby_truth_labels': '|'.join(nearby_truth_labels(expected, event_time, RIDE_LABEL, tolerance)),
            'top_alternative': alternative_label, 'top_alternative_probability': round(alternative_probability, 6),
        })
    return false_positives, false_negatives


def write_delta(path, rows):
    """中文註解：以既有固定 CSV 欄位寫出 D82 相對 D77 的逐類 delta。"""
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    """中文註解：驗證同一封存 gate 後，重建 D82 RIDE 未配對事件並輸出下一步判定。"""
    output_dir = prepare_output_dir(args.output_dir)
    d77_gate, d82_gate = read_json(args.d77_gate), read_json(args.d82_gate)
    d77_metrics, d82_metrics = read_metrics(args.d77_metrics), read_metrics(args.d82_metrics)
    validate_inputs(d77_gate, d82_gate, d77_metrics, d82_metrics)
    delta_rows = build_rows(d77_metrics, d82_metrics)
    write_delta(os.path.join(output_dir, 'per_class_delta.csv'), delta_rows)

    payload = torch.load(args.d82_adapter, map_location='cpu', weights_only=False)
    if payload.get('phase') != 'D82':
        raise ValueError('Adapter payload is not a D82 candidate.')
    if payload.get('base_d76_sha256') != file_sha256(args.d76_checkpoint):
        raise ValueError('D76 checkpoint SHA-256 differs from the D82 adapter payload.')
    if payload.get('base_d64_sha256') != file_sha256(args.d64_checkpoint):
        raise ValueError('D64 checkpoint SHA-256 differs from the D82 adapter payload.')
    d76_selection, d64_selection = read_json(args.d76_selection), read_json(args.d64_selection)
    assert_same_selection(d76_selection, d64_selection)
    metadata = read_json(args.metadata)
    windows = load_fixed_windows(metadata, args.d76_selection, 'validation', args.per_class)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d76_model = load_frozen_lora_model(args.d76_checkpoint, device, payload['rank'], payload['alpha'])
    d64_model = load_frozen_lora_model(args.d64_checkpoint, device, payload['rank'], payload['alpha'])
    load_adapter(d76_model.onset_head, payload['d76_onset_lora'], device)
    load_adapter(d64_model.onset_head, payload['d64_onset_lora'], device)
    d76_model.eval()
    d64_model.eval()
    false_positives, false_negatives = [], []
    for window_index, selected in enumerate(windows):
        fp_rows, fn_rows = audit_window(d76_model, d64_model, selected, window_index, args.tolerance)
        false_positives.extend(fp_rows)
        false_negatives.extend(fn_rows)
    fields = ['window_index', 'key', 'group_id', 'audio_path', 'anchor', 'window_start', 'event_time', 'audio_time', 'ride_probability', 'nearby_truth_labels', 'top_alternative', 'top_alternative_probability']
    write_csv(os.path.join(output_dir, 'ride_false_positives.csv'), false_positives, fields[:9] + ['cause'] + fields[9:])
    write_csv(os.path.join(output_dir, 'ride_false_negatives.csv'), false_negatives, fields)
    fn_competitor, fn_share = strict_dominant(false_negatives, 'top_alternative')
    fp_truth, fp_share = strict_dominant([row for row in false_positives if row['nearby_truth_labels']], 'nearby_truth_labels')
    ride_delta = next(row for row in delta_rows if row['inst'] == RIDE_LABEL)
    status = 'ride_competitor_feasibility_review' if fn_competitor else 'stop_same_data'
    summary = {
        'phase': 'D83', 'status': status, 'base_label': 'D77', 'candidate_label': 'D82',
        'selected_windows': len(windows), 'threshold': d77_gate['threshold'], 'tolerance_seconds': args.tolerance,
        'selection_identical': True, 'adapter_hashes_verified': True, 'ride_delta': ride_delta,
        'ride_false_positives': len(false_positives), 'ride_fp_by_cause': count_by(false_positives, 'cause'),
        'ride_fp_by_nearby_truth': count_by([row for row in false_positives if row['nearby_truth_labels']], 'nearby_truth_labels'),
        'ride_false_negatives': len(false_negatives), 'ride_fn_by_top_alternative': count_by(false_negatives, 'top_alternative'),
        'ride_fn_dominant_competitor': fn_competitor, 'ride_fn_dominant_competitor_share': fn_share,
        'ride_fp_dominant_nearby_truth': fp_truth, 'ride_fp_dominant_nearby_truth_share': fp_share,
        'next_step': 'audit_ride_competitor_data_feasibility' if fn_competitor else 'do_not_train_on_existing_data',
        'ready_for_training_candidate': False, 'ready_for_six_class_release': False,
    }
    with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def run_self_check():
    """中文註解：確認嚴格過半主因與 RIDE delta 取值不會誤判。"""
    assert strict_dominant([{'top_alternative': 'HH'}] * 2 + [{'top_alternative': 'SD'}], 'top_alternative')[0] == 'HH'
    assert strict_dominant([{'top_alternative': 'HH'}, {'top_alternative': 'SD'}], 'top_alternative')[0] is None
    print('D83 self-check passed.')


def main():
    """中文註解：解析固定 D77/D82 證據與 D82 adapter，執行唯讀 RIDE 審計。"""
    parser = argparse.ArgumentParser(description='Audit D82 RIDE regression against D77 without training.')
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--d77-metrics', default='validation_runs/d77_d76_d64_tom_fusion/event_compare.csv')
    parser.add_argument('--d82-metrics', default='validation_runs/d82_d77_fused_lora_candidate/epoch_05_fixed_validation/event_compare.csv')
    parser.add_argument('--d77-gate', default='validation_runs/d77_d76_d64_tom_fusion/gate_summary.json')
    parser.add_argument('--d82-gate', default='validation_runs/d82_d77_fused_lora_candidate/epoch_05_fixed_validation/gate_summary.json')
    parser.add_argument('--d82-adapter', default='validation_runs/d82_d77_fused_lora_candidate/d82_d77_fused_lora_adapter.pth')
    parser.add_argument('--d76-checkpoint', default='validation_runs/d76_crash_kd_retry_candidate/d76_crash_kd_retry_candidate.pth')
    parser.add_argument('--d64-checkpoint', default='validation_runs/d64_tom_competitor_candidate/d64_tom_competitor_candidate.pth')
    parser.add_argument('--metadata', default='mixed_d54_stem/metadata_d54.json')
    parser.add_argument('--d76-selection', default='validation_runs/d61_kd_negative_candidate/independent_validation/selected_windows.json')
    parser.add_argument('--d64-selection', default='validation_runs/d64_tom_competitor_candidate/independent_validation/selected_windows.json')
    parser.add_argument('--output-dir', default='validation_runs/d83_d77_d82_ride_audit')
    parser.add_argument('--per-class', type=int, default=8)
    parser.add_argument('--tolerance', type=float, default=0.05)
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    run(args)


if __name__ == '__main__':
    main()
