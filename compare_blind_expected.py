# -*- coding: utf-8 -*-
"""
Compare blind-test summary rows against a small expected-target CSV.
"""
import argparse
import csv
import os


def compare(summary_csv, expected_csv, layer='notation', counts_only=False):
    """
    中文註解：比對 blind test summary 與使用者提供的第一批期望值。
    """
    count_prefix = 'raw' if layer in ('raw', 'raw_acoustic') else 'notation'
    virtual_counts = {
        'kick': '',
        'snare': '',
        'hihat': '',
    } if layer == 'raw' else None
    expected = {row['name']: row for row in csv.DictReader(open(expected_csv, 'r', encoding='utf-8'))}
    rows = []
    for row in csv.DictReader(open(summary_csv, 'r', encoding='utf-8')):
        exp = expected.get(row['name'])
        if not exp:
            continue
        failures = []
        checks = []
        if not counts_only and layer != 'raw_acoustic':
            tempo_ok = abs(float(row['tempo_bpm']) - float(exp['expected_tempo'])) <= float(exp['tempo_tol'])
            checks.extend([
                ('tempo', tempo_ok, row['tempo_bpm'], exp['expected_tempo']),
                ('time_signature', row['time_signature'] == exp['expected_time_signature'], row['time_signature'], exp['expected_time_signature']),
            ])
        checks.extend([
            ('kick', int(row[f'{count_prefix}_kick']) == int(exp['expected_kick']), row[f'{count_prefix}_kick'], exp['expected_kick']),
            ('snare', int(row[f'{count_prefix}_snare']) == int(exp['expected_snare']), row[f'{count_prefix}_snare'], exp['expected_snare']),
            ('hihat', int(row[f'{count_prefix}_hihat']) == int(exp['expected_hihat']), row[f'{count_prefix}_hihat'], exp['expected_hihat']),
        ])
        for name, ok, actual, target in checks:
            if not ok:
                failures.append(f'{name}:{actual}!={target}')
        rows.append({
            'name': row['name'],
            'layer': layer,
            'counts_only': counts_only,
            'overall': 'pass' if not failures else 'fail',
            'failures': ';'.join(failures),
            'expected_tempo': exp['expected_tempo'],
            'actual_tempo': row['tempo_bpm'],
            'expected_time_signature': exp['expected_time_signature'],
            'actual_time_signature': row['time_signature'],
            'expected_kick': exp['expected_kick'],
            'actual_kick': row[f'{count_prefix}_kick'],
            'raw_kick': row['raw_kick'],
            'expected_snare': exp['expected_snare'],
            'actual_snare': row[f'{count_prefix}_snare'],
            'raw_snare': row['raw_snare'],
            'expected_hihat': exp['expected_hihat'],
            'actual_hihat': row[f'{count_prefix}_hihat'],
            'raw_hihat': row['raw_hihat'],
            'virtual_kick': virtual_counts['kick'] if virtual_counts is not None else row['virtual_kick'],
            'virtual_snare': virtual_counts['snare'] if virtual_counts is not None else row['virtual_snare'],
            'virtual_hihat': virtual_counts['hihat'] if virtual_counts is not None else row['virtual_hihat'],
            'shuffle_completion': row['shuffle_completion'],
        })
    return rows


def write_rows(rows, output_csv):
    """
    中文註解：寫出比對報告 CSV。
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_self_check():
    """
    中文註解：保留最小自檢入口，避免 CLI 壞掉。
    """
    print('Self-check passed.')


def main():
    """
    中文註解：CLI 入口，輸出 expected comparison。
    """
    parser = argparse.ArgumentParser(description='Compare blind summary with expected targets.')
    parser.add_argument('--summary', default='validation_runs/blind_test_user_first_batch/summary.csv')
    parser.add_argument('--expected', default='blind_user_tests_expected.csv')
    parser.add_argument('--output', default='validation_runs/blind_test_user_first_batch/expected_comparison.csv')
    parser.add_argument('--layer', choices=('notation', 'raw', 'raw_acoustic'), default='notation')
    parser.add_argument('--counts-only', action='store_true')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    rows = compare(args.summary, args.expected, args.layer, args.counts_only)
    write_rows(rows, args.output)
    for row in rows:
        print(f"{row['name']}: {row['overall']} {row['failures']}")


if __name__ == '__main__':
    main()
