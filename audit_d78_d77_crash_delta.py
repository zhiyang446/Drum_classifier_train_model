# -*- coding: utf-8 -*-
"""D78：比較 D67 與 D77 的固定 CRASH 殘餘錯誤。"""

import argparse
import csv
import json
from pathlib import Path
import tempfile

from audit_d72_d70_delta import build_rows, read_json, read_metrics, validate_inputs


def read_summary(path):
    """中文註解：讀取既有 CRASH FN 摘要，避免重複解讀逐筆 CSV。"""
    return read_json(path)


def dominant_competitor(summary):
    """中文註解：取得殘餘 CRASH FN 中嚴格過半的單一替代類別。"""
    counts = summary['crash_fn_by_top_alternative']
    total = int(summary['crash_false_negatives'])
    if not counts or total <= 0:
        return None, 0.0
    label, count = max(counts.items(), key=lambda item: item[1])
    share = int(count) / total
    return (label if share > 0.5 else None), round(share, 4)


def route_status(rows, candidate_summary):
    """中文註解：只讓新的非 KD 過半競爭類別進入下一個可行性審計。"""
    crash = next(row for row in rows if row['inst'] == 'CRASH')
    competitor, share = dominant_competitor(candidate_summary)
    # ponytail: KD 已做過唯一一次候選；沒有新且過半的根因就停止，不再重跑訓練。
    eligible = competitor is not None and competitor != 'KD'
    status = 'eligible_for_new_competitor_feasibility' if eligible else 'crash_route_needs_non_kd_root_cause'
    return status, crash, competitor, share


def write_report(output_dir, payload, rows):
    """中文註解：只寫入新的 D78 JSON 與逐類 delta CSV，拒絕覆寫既有證據。"""
    directory = Path(output_dir)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {directory}')
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'per_class_delta.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (directory / 'summary.json').open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def run(args):
    """中文註解：驗證固定 gate 後輸出 D77 對 D67 的 CRASH delta 與下一步判定。"""
    base_gate, candidate_gate = read_json(args.base_gate), read_json(args.candidate_gate)
    base_metrics, candidate_metrics = read_metrics(args.base_metrics), read_metrics(args.candidate_metrics)
    validate_inputs(base_gate, candidate_gate, base_metrics, candidate_metrics)
    base_summary, candidate_summary = read_summary(args.base_summary), read_summary(args.candidate_summary)
    if base_summary['selected_windows'] != candidate_summary['selected_windows'] or base_summary['selected_windows'] != base_gate['selected_windows']:
        raise ValueError('CRASH audit selected-window mismatch')
    rows = build_rows(base_metrics, candidate_metrics)
    status, crash, competitor, share = route_status(rows, candidate_summary)
    payload = {
        'phase': 'D78', 'status': status,
        'base_label': args.base_label, 'candidate_label': args.candidate_label,
        'fixed_threshold': base_gate['threshold'], 'tolerance_seconds': base_gate['tolerance_seconds'],
        'selected_windows': base_gate['selected_windows'],
        'crash_delta': {field: crash[f'delta_{field}'] for field in ('tp', 'fp', 'fn', 'precision', 'recall', 'f1')},
        'base_crash_fn_by_top_alternative': base_summary['crash_fn_by_top_alternative'],
        'candidate_crash_fn_by_top_alternative': candidate_summary['crash_fn_by_top_alternative'],
        'remaining_dominant_competitor': competitor,
        'remaining_dominant_competitor_share': share,
        'new_competitor_feasibility_allowed': status == 'eligible_for_new_competitor_feasibility',
        'ready_for_six_class_release': False,
    }
    write_report(args.output_dir, payload, rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def run_self_check():
    """中文註解：確認 KD 會停止，而新的嚴格過半替代才可繼續。"""
    rows = [{'inst': 'CRASH', 'delta_tp': 0, 'delta_fp': 0, 'delta_fn': 0, 'delta_precision': 0.0, 'delta_recall': 0.0, 'delta_f1': 0.0}]
    kd = {'crash_fn_by_top_alternative': {'KD': 6, 'HH': 4}, 'crash_false_negatives': 10}
    hh = {'crash_fn_by_top_alternative': {'HH': 6, 'KD': 4}, 'crash_false_negatives': 10}
    assert route_status(rows, kd)[0] == 'crash_route_needs_non_kd_root_cause'
    assert route_status(rows, hh)[0] == 'eligible_for_new_competitor_feasibility'
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / 'report'
        write_report(target, {'phase': 'test'}, rows)
        assert (target / 'summary.json').is_file()
    print('Self-check passed.')


def main():
    """中文註解：解析 D78 唯讀輸入並執行 delta 審計或自檢。"""
    parser = argparse.ArgumentParser(description='Compare fixed D67 and D77 CRASH residual errors.')
    parser.add_argument('--base-metrics', default='validation_runs/d67_d61_d64_tom_fusion/event_compare.csv')
    parser.add_argument('--candidate-metrics', default='validation_runs/d77_d76_d64_tom_fusion/event_compare.csv')
    parser.add_argument('--base-gate', default='validation_runs/d67_d61_d64_tom_fusion/gate_summary.json')
    parser.add_argument('--candidate-gate', default='validation_runs/d77_d76_d64_tom_fusion/gate_summary.json')
    parser.add_argument('--base-summary', default='validation_runs/d74_d67_crash_miss_audit/summary.json')
    parser.add_argument('--candidate-summary', default='validation_runs/d78_d77_crash_residual_audit/crash_misses/summary.json')
    parser.add_argument('--base-label', default='D67')
    parser.add_argument('--candidate-label', default='D77')
    parser.add_argument('--output-dir', default='validation_runs/d78_d77_crash_residual_audit')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    run(args)


if __name__ == '__main__':
    main()
