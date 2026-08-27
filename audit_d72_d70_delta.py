# -*- coding: utf-8 -*-
"""D72：比較 D61 與 D70 固定 48-window 驗收的逐類 delta。"""

import argparse
import csv
import json
from pathlib import Path
import tempfile


LABELS = ('KD', 'SD', 'HH', 'TOM', 'CRASH', 'RIDE')
COUNT_FIELDS = ('tp', 'fp', 'fn')
SCORE_FIELDS = ('precision', 'recall', 'f1')


def read_json(path):
    """中文註解：讀取 gate JSON，讓門檻不一致在計算前立即失敗。"""
    with Path(path).open(encoding='utf-8') as handle:
        return json.load(handle)


def read_metrics(path):
    """中文註解：讀取固定驗收的逐類事件統計，並轉成依 label 索引的數值資料。"""
    with Path(path).open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    if tuple(row['inst'] for row in rows) != LABELS:
        raise ValueError(f'Unexpected label order in {path}')
    # 重要變數：metrics 保留每類的真值與事件匹配統計，避免把 CSV 字串直接相減。
    metrics = {}
    for row in rows:
        metrics[row['inst']] = {
            'expected': int(row['expected']), 'predicted': int(row['predicted']),
            **{field: int(row[field]) for field in COUNT_FIELDS},
            **{field: float(row[field]) for field in SCORE_FIELDS},
        }
    return metrics


def validate_inputs(base_gate, candidate_gate, base_metrics, candidate_metrics):
    """中文註解：確認比較的是同一批固定驗收，而非不同門檻或不同真值。"""
    for field in ('threshold', 'tolerance_seconds', 'selected_windows'):
        if base_gate[field] != candidate_gate[field]:
            raise ValueError(f'Gate mismatch for {field}: {base_gate[field]} != {candidate_gate[field]}')
    for label in LABELS:
        if base_metrics[label]['expected'] != candidate_metrics[label]['expected']:
            raise ValueError(f'Expected-event count mismatch for {label}')


def build_rows(base_metrics, candidate_metrics):
    """中文註解：計算候選減去基線的逐類事件與分數差異。"""
    rows = []
    for label in LABELS:
        base, candidate = base_metrics[label], candidate_metrics[label]
        row = {'inst': label, 'expected': base['expected']}
        for field in ('predicted',) + COUNT_FIELDS:
            row[f'base_{field}'] = base[field]
            row[f'candidate_{field}'] = candidate[field]
            row[f'delta_{field}'] = candidate[field] - base[field]
        for field in SCORE_FIELDS:
            row[f'base_{field}'] = base[field]
            row[f'candidate_{field}'] = candidate[field]
            row[f'delta_{field}'] = round(candidate[field] - base[field], 4)
        rows.append(row)
    return rows


def sd_route_status(rows):
    """中文註解：只以 SD 的 F1 與 FP/FN 同向改善決定 D73 是否有資格存在。"""
    sd = next(row for row in rows if row['inst'] == 'SD')
    # ponytail: 已有固定驗收 CSV 足以否決或通過 D73；不另跑模型、不搜尋新指標。
    improved = sd['delta_f1'] > 0 and sd['delta_fp'] <= 0 and sd['delta_fn'] <= 0
    return ('eligible_for_d73' if improved else 'd70_route_rejected'), sd


def write_report(output_dir, payload, rows):
    """中文註解：只建立新的 JSON 與 CSV 證據，避免覆寫既有驗收目錄。"""
    directory = Path(output_dir)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {directory}')
    directory.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (directory / 'per_class_delta.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (directory / 'summary.json').open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def run(args):
    """中文註解：執行 D72 的唯讀 CSV delta 審計並輸出 D73 決策。"""
    base_gate, candidate_gate = read_json(args.base_gate), read_json(args.candidate_gate)
    base_metrics, candidate_metrics = read_metrics(args.base_metrics), read_metrics(args.candidate_metrics)
    validate_inputs(base_gate, candidate_gate, base_metrics, candidate_metrics)
    rows = build_rows(base_metrics, candidate_metrics)
    status, sd = sd_route_status(rows)
    payload = {
        'phase': 'D72', 'status': status,
        'base_label': args.base_label, 'candidate_label': args.candidate_label,
        'fixed_threshold': base_gate['threshold'], 'tolerance_seconds': base_gate['tolerance_seconds'],
        'selected_windows': base_gate['selected_windows'],
        'sd_delta': {field: sd[f'delta_{field}'] for field in ('tp', 'fp', 'fn', 'precision', 'recall', 'f1')},
        'd73_training_allowed': status == 'eligible_for_d73',
        'reason': 'SD improves without FP/FN regression' if status == 'eligible_for_d73' else 'SD has mixed regression; do not repeat SD-vs-KD training.',
        'ready_for_six_class_release': False,
    }
    write_report(args.output_dir, payload, rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def run_self_check():
    """中文註解：確認 SD 改善與 mixed regression 的 D73 判定不會反轉。"""
    good = [{'inst': 'SD', 'delta_f1': 0.01, 'delta_fp': 0, 'delta_fn': -1}]
    bad = [{'inst': 'SD', 'delta_f1': -0.01, 'delta_fp': 1, 'delta_fn': 1}]
    assert sd_route_status(good)[0] == 'eligible_for_d73'
    assert sd_route_status(bad)[0] == 'd70_route_rejected'
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'report'
        write_report(path, {'phase': 'test'}, good)
        assert (path / 'summary.json').is_file() and (path / 'per_class_delta.csv').is_file()
    print('Self-check passed.')


def main():
    """中文註解：解析 D72 輸入路徑，執行審計或最小自檢。"""
    parser = argparse.ArgumentParser(description='Compare fixed D61 and D70 event reports without rerunning models.')
    parser.add_argument('--base-metrics', default='validation_runs/d61_kd_negative_candidate/independent_validation/event_compare.csv')
    parser.add_argument('--candidate-metrics', default='validation_runs/d71_d70_d64_tom_fusion/event_compare.csv')
    parser.add_argument('--base-gate', default='validation_runs/d61_kd_negative_candidate/independent_validation/gate_summary.json')
    parser.add_argument('--candidate-gate', default='validation_runs/d71_d70_d64_tom_fusion/gate_summary.json')
    parser.add_argument('--base-label', default='D61')
    parser.add_argument('--candidate-label', default='D70 five-class plus D64 TOM')
    parser.add_argument('--output-dir', default='validation_runs/d72_d70_vs_d61_delta_audit')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    run(args)


if __name__ == '__main__':
    main()
