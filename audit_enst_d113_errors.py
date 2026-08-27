# -*- coding: utf-8 -*-
"""D113：比較 D89 與 D111 在固定 ENST validation 視窗的新增錯誤。"""

import argparse
import csv
import json
import os
from collections import Counter

import torch

from audit_d58_drumsep_errors import (
    local_scores,
    match_indices,
    nearby_truth_labels,
    prepare_output_dir,
    write_csv,
)
from evaluate_enst_d109_fixed import load_adapter_pair, read_json
from run_six_class_smoke import build_window
from run_six_class_validation import LABELS, THRESHOLD, expected_events, load_fixed_windows, local_maxima
from train_d77_fused_lora import file_sha256, fused_logits


EXPECTED_HASHES = {
    'selection': '08c97f46ccc677022e45ea4c1ec652b3379d647e7ef9d94dac4dafe49017d613',
    'd89': '552900cb8a056364dd3ce0b7d880fc4d36b54f7f65b712c68b3fd75d97410177',
    'd111': '44ce6da9a5b384410e3e1d29cf3ac2ce5eea475c329e199a6eaaac83b1a6fa0f',
}


def metric_counts(expected, predicted, tolerance):
    """以共用一對一匹配回傳事件數與 TP／FP／FN。"""
    matched_expected, matched_predicted = match_indices(expected, predicted, tolerance)
    return {
        'expected': len(expected),
        'predicted': len(predicted),
        'tp': len(matched_expected),
        'fp': len(predicted) - len(matched_predicted),
        'fn': len(expected) - len(matched_expected),
    }


def f1_from_counts(counts):
    """由 TP／FP／FN 計算固定事件 F1。"""
    precision_denominator = counts['tp'] + counts['fp']
    recall_denominator = counts['tp'] + counts['fn']
    precision = counts['tp'] / precision_denominator if precision_denominator else 0.0
    recall = counts['tp'] / recall_denominator if recall_denominator else 0.0
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def unmatched_times(expected, predicted, tolerance):
    """回傳未匹配真值與未匹配預測的時間及其原始索引。"""
    matched_expected, matched_predicted = match_indices(expected, predicted, tolerance)
    false_negatives = [(index, time) for index, time in enumerate(expected) if index not in matched_expected]
    false_positives = [(index, time) for index, time in enumerate(predicted) if index not in matched_predicted]
    return false_negatives, false_positives


def added_error_indices(base_times, candidate_times, tolerance):
    """找出候選錯誤中無法與父模型既有錯誤對齊的索引。"""
    _, matched_candidate = match_indices(base_times, candidate_times, tolerance)
    return [index for index in range(len(candidate_times)) if index not in matched_candidate]


def prepare_windows(metadata, selection_path):
    """重建固定 48 個 validation 視窗及共用特徵，拒絕 sealed drummer_3。"""
    if any(item.get('split') != 'validation' for item in metadata.values()):
        raise ValueError('D113 accepts validation metadata only.')
    if any('drummer_3' in f'{key} {item.get("audio_path", "")}'.lower() for key, item in metadata.items()):
        raise ValueError('Sealed ENST drummer_3 data is forbidden.')
    windows = load_fixed_windows(metadata, selection_path, split='validation', per_class=8)
    groups = {str(row['item'].get('group_id') or row['key']) for row in windows}
    if len(windows) != 48 or len(groups) != 48:
        raise AssertionError(f'Unexpected fixed selection: windows={len(windows)}, groups={len(groups)}')

    prepared = []
    for selected in windows:
        features, _, _, start_sec = build_window(
            selected['item'], selected['anchor'], use_true_superflux=True,
            use_multi_log_mel=False,
            input_mode=selected['item'].get('input_mode', 'drumsep-mix'),
        )
        prepared.append({
            'selected': selected,
            'features': features,
            'start_sec': start_sec,
            'expected': expected_events(selected['item'], start_sec),
        })
    return prepared


def infer_adapter(adapter_path, args, device, prepared):
    """依序載入單一 adapter pair 並推論，避免四個模型同時佔用 VRAM。"""
    d76_model, d64_model, _ = load_adapter_pair(adapter_path, args, device)
    probabilities = []
    with torch.no_grad():
        for window in prepared:
            feature_tensor = torch.from_numpy(window['features']).float().unsqueeze(0).to(device)
            probability = torch.sigmoid(fused_logits(d76_model, d64_model, feature_tensor))
            probabilities.append(probability.squeeze(0).cpu().numpy())
    del d76_model, d64_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return probabilities


def summarize_dominant(added_rows):
    """只接受嚴格過半的單一錯誤類型＋目標類別作為候選根因。"""
    counts = Counter(f"{row['error_type']}:{row['inst']}" for row in added_rows)
    ordered = dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
    if not added_rows:
        return ordered, None, 0.0
    key, count = next(iter(ordered.items()))
    share = count / len(added_rows)
    return ordered, key if share > 0.5 else None, round(share, 4)


def audit(args):
    """執行 D113 唯讀比較並輸出逐類、逐窗及新增錯誤證據。"""
    for path in (
        args.metadata, args.selection, args.d89_adapter, args.d111_adapter,
        args.d76_checkpoint, args.d64_checkpoint,
    ):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    actual_hashes = {
        'selection': file_sha256(args.selection),
        'd89': file_sha256(args.d89_adapter),
        'd111': file_sha256(args.d111_adapter),
    }
    if actual_hashes != EXPECTED_HASHES:
        raise ValueError(f'D113 locked input hash mismatch: {actual_hashes}')

    output_dir = prepare_output_dir(args.output_dir)
    metadata = read_json(args.metadata)
    prepared = prepare_windows(metadata, args.selection)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    probabilities = {
        'd89': infer_adapter(args.d89_adapter, args, device, prepared),
        'd111': infer_adapter(args.d111_adapter, args, device, prepared),
    }

    aggregate = {
        model: {label: {'expected': 0, 'predicted': 0, 'tp': 0, 'fp': 0, 'fn': 0} for label in LABELS}
        for model in ('d89', 'd111')
    }
    window_rows, added_rows = [], []
    recovered_fn_count = removed_fp_count = 0

    for window_index, window in enumerate(prepared):
        selected = window['selected']
        decoded = {
            model: local_maxima(probabilities[model][window_index])
            for model in ('d89', 'd111')
        }
        base = {
            'window_index': window_index,
            'key': selected['key'],
            'group_id': str(selected['item'].get('group_id') or selected['key']),
            'anchor': round(float(selected['anchor']), 6),
            'window_start': round(float(window['start_sec']), 6),
        }
        for label in LABELS:
            expected = window['expected'][label]
            model_counts = {
                model: metric_counts(expected, decoded[model][label], args.tolerance)
                for model in ('d89', 'd111')
            }
            for model in ('d89', 'd111'):
                for field in aggregate[model][label]:
                    aggregate[model][label][field] += model_counts[model][field]
            window_rows.append({
                **base, 'inst': label,
                **{f'd89_{field}': value for field, value in model_counts['d89'].items()},
                **{f'd111_{field}': value for field, value in model_counts['d111'].items()},
                'delta_tp': model_counts['d111']['tp'] - model_counts['d89']['tp'],
                'delta_fp': model_counts['d111']['fp'] - model_counts['d89']['fp'],
                'delta_fn': model_counts['d111']['fn'] - model_counts['d89']['fn'],
            })

            base_fn, base_fp = unmatched_times(expected, decoded['d89'][label], args.tolerance)
            candidate_fn, candidate_fp = unmatched_times(expected, decoded['d111'][label], args.tolerance)
            base_fn_indices = {index for index, _ in base_fn}
            candidate_fn_indices = {index for index, _ in candidate_fn}
            new_fn_indices = candidate_fn_indices - base_fn_indices
            recovered_fn_count += len(base_fn_indices - candidate_fn_indices)

            base_fp_times = [time for _, time in base_fp]
            candidate_fp_times = [time for _, time in candidate_fp]
            new_fp_positions = added_error_indices(base_fp_times, candidate_fp_times, args.tolerance)
            removed_fp_count += len(base_fp_times) - (len(candidate_fp_times) - len(new_fp_positions))

            for expected_index in sorted(new_fn_indices):
                event_time = expected[expected_index]
                base_scores = local_scores(probabilities['d89'][window_index], event_time, args.tolerance)
                candidate_scores = local_scores(probabilities['d111'][window_index], event_time, args.tolerance)
                top_index = int(candidate_scores.argmax())
                added_rows.append({
                    **base, 'error_type': 'added_fn', 'inst': label,
                    'event_time': round(float(event_time), 6),
                    'audio_time': round(float(window['start_sec'] + event_time), 6),
                    'd89_probability': round(float(base_scores[LABELS.index(label)]), 6),
                    'd111_probability': round(float(candidate_scores[LABELS.index(label)]), 6),
                    'd111_top_label': LABELS[top_index],
                    'd111_top_probability': round(float(candidate_scores[top_index]), 6),
                    'nearby_truth_labels': '|'.join(nearby_truth_labels(window['expected'], event_time, label, args.tolerance)),
                })
            for position in new_fp_positions:
                event_time = candidate_fp_times[position]
                base_scores = local_scores(probabilities['d89'][window_index], event_time, args.tolerance)
                candidate_scores = local_scores(probabilities['d111'][window_index], event_time, args.tolerance)
                top_index = int(candidate_scores.argmax())
                added_rows.append({
                    **base, 'error_type': 'added_fp', 'inst': label,
                    'event_time': round(float(event_time), 6),
                    'audio_time': round(float(window['start_sec'] + event_time), 6),
                    'd89_probability': round(float(base_scores[LABELS.index(label)]), 6),
                    'd111_probability': round(float(candidate_scores[LABELS.index(label)]), 6),
                    'd111_top_label': LABELS[top_index],
                    'd111_top_probability': round(float(candidate_scores[top_index]), 6),
                    'nearby_truth_labels': '|'.join(nearby_truth_labels(window['expected'], event_time, label, args.tolerance)),
                })

    per_class_rows = []
    for label in LABELS:
        base_counts, candidate_counts = aggregate['d89'][label], aggregate['d111'][label]
        base_f1, candidate_f1 = f1_from_counts(base_counts), f1_from_counts(candidate_counts)
        per_class_rows.append({
            'inst': label,
            **{f'd89_{field}': value for field, value in base_counts.items()},
            **{f'd111_{field}': value for field, value in candidate_counts.items()},
            'delta_tp': candidate_counts['tp'] - base_counts['tp'],
            'delta_fp': candidate_counts['fp'] - base_counts['fp'],
            'delta_fn': candidate_counts['fn'] - base_counts['fn'],
            'd89_f1': round(base_f1, 6),
            'd111_f1': round(candidate_f1, 6),
            'delta_f1': round(candidate_f1 - base_f1, 6),
        })

    concentration, dominant_root_cause, dominant_share = summarize_dominant(added_rows)
    ready_for_d114 = dominant_root_cause is not None
    status = (
        'concentrated_root_cause_candidate_not_training'
        if ready_for_d114 else
        'dispersed_regression_stop_same_recipe'
    )
    summary = {
        'phase': 'D113',
        'status': status,
        'device': str(device),
        'training_started': False,
        'sealed_test_read': False,
        'threshold': THRESHOLD,
        'tolerance_seconds': args.tolerance,
        'selection_windows': len(prepared),
        'selection_sha256': actual_hashes['selection'],
        'adapter_sha256': {'d89': actual_hashes['d89'], 'd111': actual_hashes['d111']},
        'added_errors': len(added_rows),
        'added_error_concentration': concentration,
        'dominant_root_cause': dominant_root_cause,
        'dominant_root_cause_share': dominant_share,
        'recovered_false_negatives': recovered_fn_count,
        'removed_false_positives': removed_fp_count,
        'ready_for_d114_proposal': ready_for_d114,
        'training_authorized': False,
        'ready_for_six_class_release': False,
        'per_class': per_class_rows,
    }

    write_csv(os.path.join(output_dir, 'per_class_delta.csv'), per_class_rows, list(per_class_rows[0]))
    write_csv(os.path.join(output_dir, 'per_window_delta.csv'), window_rows, list(window_rows[0]))
    error_fields = [
        'window_index', 'key', 'group_id', 'anchor', 'window_start', 'error_type', 'inst',
        'event_time', 'audio_time', 'd89_probability', 'd111_probability',
        'd111_top_label', 'd111_top_probability', 'nearby_truth_labels',
    ]
    write_csv(os.path.join(output_dir, 'added_errors.csv'), added_rows, error_fields)
    with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def run_self_check():
    """驗證事件計數、候選新增錯誤與嚴格過半判定。"""
    assert metric_counts([1.0, 2.0], [1.02, 3.0], 0.05) == {
        'expected': 2, 'predicted': 2, 'tp': 1, 'fp': 1, 'fn': 1,
    }
    assert added_error_indices([1.0], [1.02, 2.0], 0.05) == [1]
    rows = [{'error_type': 'added_fn', 'inst': 'SD'}] * 2 + [{'error_type': 'added_fp', 'inst': 'HH'}]
    assert summarize_dominant(rows)[1:] == ('added_fn:SD', 0.6667)
    assert summarize_dominant(rows[:1] + rows[-1:])[1] is None
    print('D113 self-check passed.')


def main():
    """解析 D113 CLI 並執行固定視窗唯讀根因稽核或 self-check。"""
    parser = argparse.ArgumentParser(description='Audit D89/D111 fixed ENST TP/FP/FN regression.')
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--metadata', default='enst_d107/metadata_d107_validation.json')
    parser.add_argument('--selection', default='validation_runs/d112_d111_enst_diagnostic/selected_windows.json')
    parser.add_argument('--d89-adapter', default='validation_runs/d89_d82_tim_gm_lora_retry/d89_d82_tim_gm_lora_retry_adapter.pth')
    parser.add_argument('--d111-adapter', default='validation_runs/d111_d89_enst_full_coverage_candidate/d111_d89_enst_full_coverage_adapter_epoch1.pth')
    parser.add_argument('--d76-checkpoint', default='validation_runs/d76_crash_kd_retry_candidate/d76_crash_kd_retry_candidate.pth')
    parser.add_argument('--d64-checkpoint', default='validation_runs/d64_tom_competitor_candidate/d64_tom_competitor_candidate.pth')
    parser.add_argument('--output-dir', default='validation_runs/d113_d111_enst_error_audit')
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--alpha', type=float, default=8.0)
    parser.add_argument('--tolerance', type=float, default=0.05)
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    audit(args)


if __name__ == '__main__':
    main()
