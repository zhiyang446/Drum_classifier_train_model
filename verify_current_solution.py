# -*- coding: utf-8 -*-
"""
One-command verifier for the currently accepted drum transcription solution.
"""
import argparse
import csv
import os
import subprocess
import sys


def run_command(cmd):
    """
    中文註解：執行子命令並在失敗時保留 stdout/stderr 方便定位。
    """
    print('> ' + ' '.join(cmd), flush=True)
    proc = subprocess.run(cmd, text=True, encoding='utf-8', errors='replace')
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def require_all_pass(csv_path, status_field):
    """
    中文註解：確認 CSV 每一列都通過指定 gate 欄位。
    """
    with open(csv_path, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f'No rows found in {csv_path}')
    failures = [row for row in rows if row.get(status_field) != 'pass']
    if failures:
        names = ', '.join(row.get('name', '<unknown>') for row in failures)
        raise SystemExit(f'{csv_path} has failing rows: {names}')
    print(f'PASS {csv_path} ({len(rows)} rows)', flush=True)


def run_self_check():
    """
    中文註解：最小自檢，確認 gate 檢查器能偵測 pass/fail。
    """
    os.makedirs(os.path.join('validation_runs', '_self_check'), exist_ok=True)
    path = os.path.join('validation_runs', '_self_check', 'verify_current_solution.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'overall'])
        writer.writeheader()
        writer.writerow({'name': 'demo', 'overall': 'pass'})
    require_all_pass(path, 'overall')
    gate_path = os.path.join('validation_runs', '_self_check', 'gate_summary.csv')
    with open(gate_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['gate', 'overall'])
        writer.writeheader()
        writer.writerow({'gate': 'round4_demo', 'overall': 'pass'})
    require_all_pass(gate_path, 'overall')
    print('Self-check passed.')


def main():
    """
    中文註解：CLI 入口，依序執行 raw acoustic、notation、hard validation 三個驗收 gate。
    """
    parser = argparse.ArgumentParser(description='Verify the accepted current drum transcription solution.')
    parser.add_argument('--model', default='drum_classifier.pth')
    parser.add_argument('--blind-input', default='blind_user_tests')
    parser.add_argument('--notation-expected', default='blind_user_tests_expected.csv')
    parser.add_argument('--raw-expected', default='validation_runs/raw_acoustic_expected_physical.csv')
    parser.add_argument('--output-dir', default='validation_runs/current_solution_verification')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()

    if args.self_check:
        run_self_check()
        return

    blind_dir = os.path.join(args.output_dir, 'blind')
    hard_dir = os.path.join(args.output_dir, 'hard')
    round4_first_dir = os.path.join(args.output_dir, 'round4_first5')
    round4_sixth_dir = os.path.join(args.output_dir, 'round4_offset5_single')
    raw_cmp = os.path.join(blind_dir, 'raw_acoustic_comparison.csv')
    notation_cmp = os.path.join(blind_dir, 'expected_comparison.csv')

    run_command([
        sys.executable, 'run_blind_test.py',
        '--input', args.blind_input,
        '--model', args.model,
        '--output-dir', blind_dir,
    ])
    run_command([
        sys.executable, 'compare_blind_expected.py',
        '--summary', os.path.join(blind_dir, 'summary.csv'),
        '--expected', args.raw_expected,
        '--output', raw_cmp,
        '--layer', 'raw_acoustic',
    ])
    run_command([
        sys.executable, 'compare_blind_expected.py',
        '--summary', os.path.join(blind_dir, 'summary.csv'),
        '--expected', args.notation_expected,
        '--output', notation_cmp,
        '--layer', 'notation',
    ])
    run_command([
        sys.executable, 'run_hard_validation.py',
        '--model', args.model,
        '--output-dir', hard_dir,
    ])
    run_command([
        sys.executable, 'run_egmd_round4_validation.py',
        '--model', args.model,
        '--output-dir', round4_first_dir,
        '--limit', '5',
    ])
    run_command([
        sys.executable, 'run_egmd_round4_validation.py',
        '--model', args.model,
        '--output-dir', round4_sixth_dir,
        '--limit', '1',
        '--offset', '5',
    ])

    require_all_pass(raw_cmp, 'overall')
    require_all_pass(notation_cmp, 'overall')
    require_all_pass(os.path.join(hard_dir, 'summary.csv'), 'gate_status')
    require_all_pass(os.path.join(round4_first_dir, 'gate_summary.csv'), 'overall')
    require_all_pass(os.path.join(round4_sixth_dir, 'gate_summary.csv'), 'overall')
    print('Current solution verification PASSED.')


if __name__ == '__main__':
    main()
