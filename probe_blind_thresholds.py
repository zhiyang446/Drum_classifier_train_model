# -*- coding: utf-8 -*-
"""Probe a small list of global raw-AI thresholds against blind expectations."""
import argparse
import csv
import os
import subprocess
import sys


def parse_args():
    """中文註解：解析 threshold probe 參數。"""
    parser = argparse.ArgumentParser(description='Probe global thresholds for blind raw counts.')
    parser.add_argument('--model', required=True)
    parser.add_argument('--input', default='blind_user_tests')
    parser.add_argument('--expected-csv', default='blind_user_tests_expected.csv')
    parser.add_argument('--output-dir', default='validation_runs/raw_ai_model_fix/threshold_probe')
    parser.add_argument('--combo', action='append', required=True, help='kick,snare,hihat')
    return parser.parse_args()


def load_expected(path):
    """中文註解：讀取使用者給定的 raw count 目標。"""
    expected = {}
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            raw_name = row.get('file') or row.get('name')
            name = raw_name[:-4] if raw_name.endswith('.wav') else raw_name
            expected[name] = {
                'KD': int(row['expected_kick']),
                'SD': int(row['expected_snare']),
                'HH': int(row['expected_hihat']),
            }
    return expected


def parse_combo(text):
    """中文註解：把 k,s,h 字串轉成三個浮點門檻。"""
    values = [float(v.strip()) for v in text.split(',')]
    if len(values) != 3:
        raise ValueError(f'Bad combo: {text}')
    return values


def run_combo(args, combo, expected):
    """中文註解：執行一次 blind test 並計算和目標 count 的差距。"""
    k, s, h = combo
    safe_name = f'k{k}_s{s}_h{h}'.replace('.', 'p')
    out_dir = os.path.join(args.output_dir, safe_name)
    cmd = [
        sys.executable,
        'run_blind_test.py',
        '--input', args.input,
        '--model', args.model,
        '--output-dir', out_dir,
        '--thresh-kick', str(k),
        '--thresh-snare', str(s),
        '--thresh-hihat', str(h),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True)
    with open(os.path.join(out_dir, 'summary.csv'), newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    fail_fields = 0
    abs_diff = 0
    parts = []
    for row in rows:
        raw_name = row.get('file') or row.get('name')
        name = raw_name[:-4] if raw_name.endswith('.wav') else raw_name
        target = expected[name]
        got = {'KD': int(row['raw_kick']), 'SD': int(row['raw_snare']), 'HH': int(row['raw_hihat'])}
        fail_fields += sum(got[inst] != target[inst] for inst in got)
        abs_diff += sum(abs(got[inst] - target[inst]) for inst in got)
        parts.append(
            f'{name}:k={got["KD"]}/{target["KD"]},'
            f's={got["SD"]}/{target["SD"]},h={got["HH"]}/{target["HH"]}'
        )
    return {
        'combo': f'{k},{s},{h}',
        'fail_fields': fail_fields,
        'abs_diff': abs_diff,
        'detail': ' | '.join(parts),
    }


def main():
    """中文註解：主流程，排序列印最佳全域門檻探測結果。"""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    expected = load_expected(args.expected_csv)
    results = [run_combo(args, parse_combo(combo), expected) for combo in args.combo]
    results.sort(key=lambda row: (row['fail_fields'], row['abs_diff']))

    report_path = os.path.join(args.output_dir, 'probe_summary.csv')
    with open(report_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['combo', 'fail_fields', 'abs_diff', 'detail'])
        writer.writeheader()
        writer.writerows(results)

    for row in results:
        print(f'{row["combo"]} fail={row["fail_fields"]} diff={row["abs_diff"]} {row["detail"]}')
    print(f'Wrote {report_path}')


if __name__ == '__main__':
    main()
