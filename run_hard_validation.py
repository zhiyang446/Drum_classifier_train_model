# -*- coding: utf-8 -*-
"""
Hard validation runner for drum transcription checkpoints.

Runs fixed regression audio and optional STAR hard-validation audio through
the existing transcribe.py CLI, then writes compact CSV/JSON reports.
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime


LOCAL_CASES = [
    {'name': 'test_shuffle', 'audio': 'test/test_shuffle.wav', 'time_signature': '4/4', 'tempo': 110.0, 'tempo_tol': 2.0, 'min_kick': 16, 'min_snare': 8, 'min_hihat': 32},
    {'name': 'test_3T', 'audio': 'test/test_3T.wav', 'time_signature': '12/8', 'tempo': 70.0, 'tempo_tol': 2.0},
    {'name': 'test_16', 'audio': 'test/test_16.wav', 'time_signature': '4/4', 'min_kick': 8, 'min_snare': 8, 'min_hihat': 60},
    {'name': 'test_58', 'audio': 'test/test_58.wav', 'time_signature': '5/8', 'min_kick': 40, 'min_snare': 25, 'min_hihat': 70},
]


def parse_transcribe_output(stdout):
    """
    中文註解：解析 transcribe.py stdout，取得 hard validation 報告需要的核心欄位。
    """
    result = {
        'tempo_bpm': '',
        'time_signature': '',
        'kick': '',
        'snare': '',
        'hihat': '',
        'mean_f1': '',
        'event_debug': '',
    }
    for line in stdout.splitlines():
        line = line.strip()
        tempo_match = re.search(r'\[Score Tempo\].*?=([0-9.]+) BPM', line)
        if tempo_match:
            result['tempo_bpm'] = tempo_match.group(1)
        if line.startswith('Effective Time Signature:'):
            result['time_signature'] = line.split(':', 1)[1].strip().split()[0]
        if line.startswith('File Mean F1-Score:'):
            result['mean_f1'] = line.split(':', 1)[1].strip()
        if line.startswith('Event debug CSV exported to:'):
            result['event_debug'] = line.split(':', 1)[1].strip()
        count_match = re.search(r'- (Kick|Snare|Hi-Hat) .*: ([0-9]+)$', line)
        if count_match:
            key = {'Kick': 'kick', 'Snare': 'snare', 'Hi-Hat': 'hihat'}[count_match.group(1)]
            result[key] = int(count_match.group(2))
    return result


def load_star_cases(path, limit, min_recall):
    """
    中文註解：讀取 STAR hard validation JSON，依 bucket/name 產生可執行案例。
    """
    if limit <= 0 or not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    cases = []
    for idx, (key, item) in enumerate(sorted(meta.items())):
        if len(cases) >= limit:
            break
        bucket = item.get('hard_bucket', 'star')
        stats = item.get('hard_stats', {})
        cases.append({
            'name': f'star_{idx:03d}_{bucket}',
            'audio': item['audio_path'],
            'gt_kick': int(stats.get('kd', 0)),
            'gt_snare': int(stats.get('sd', 0)),
            'gt_hihat': int(stats.get('hh', 0)),
            'min_recall_kick': min_recall,
            'min_recall_snare': min_recall,
            'min_recall_hihat': min_recall,
        })
    return cases


def run_case(case, model, output_dir):
    """
    中文註解：執行單一音訊驗證案例，回傳解析後的報告列。
    """
    case_dir = os.path.join(output_dir, case['name'])
    os.makedirs(case_dir, exist_ok=True)
    midi_path = os.path.join(case_dir, f'{case["name"]}.mid')
    debug_path = os.path.join(case_dir, f'{case["name"]}_event_debug.csv')
    cmd = [
        sys.executable,
        'transcribe.py',
        '--input', case['audio'],
        '--model', model,
        '--output', midi_path,
        '--event-debug', debug_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    parsed = parse_transcribe_output(proc.stdout)
    row = {
        'name': case['name'],
        'audio': case['audio'],
        'model': model,
        'status': 'pass' if proc.returncode == 0 else 'fail',
        'returncode': proc.returncode,
        'midi': midi_path,
        'gt_kick': case.get('gt_kick', ''),
        'gt_snare': case.get('gt_snare', ''),
        'gt_hihat': case.get('gt_hihat', ''),
        **parsed,
    }
    if proc.returncode != 0:
        row['error_tail'] = proc.stderr[-1000:]
    gate_status, gate_notes = evaluate_gate(case, row)
    row['gate_status'] = gate_status
    row['gate_notes'] = gate_notes
    return row


def evaluate_gate(case, row):
    """
    中文註解：依案例門檻判斷音樂結果是否通過，不把「程式有跑完」誤認為模型合格。
    """
    notes = []
    if row['status'] != 'pass':
        notes.append('transcribe failed')
    if case.get('time_signature') and row.get('time_signature') != case['time_signature']:
        notes.append(f'time_signature expected {case["time_signature"]} got {row.get("time_signature")}')
    if case.get('tempo') is not None and row.get('tempo_bpm'):
        tempo = float(row['tempo_bpm'])
        if abs(tempo - case['tempo']) > case.get('tempo_tol', 2.0):
            notes.append(f'tempo expected {case["tempo"]} got {tempo}')
    for inst in ('kick', 'snare', 'hihat'):
        minimum = case.get(f'min_{inst}')
        if minimum is not None:
            value = row.get(inst)
            if value == '' or int(value) < minimum:
                notes.append(f'{inst} expected >= {minimum} got {value}')
        min_recall = case.get(f'min_recall_{inst}')
        gt_count = case.get(f'gt_{inst}', 0)
        if min_recall is not None and gt_count:
            value = row.get(inst)
            recall = (int(value) / gt_count) if value != '' else 0.0
            if recall < min_recall:
                notes.append(f'{inst} recall expected >= {min_recall:.2f} got {recall:.2f} ({value}/{gt_count})')
    return ('pass' if not notes else 'fail'), '; '.join(notes)


def write_reports(rows, output_dir):
    """
    中文註解：輸出 CSV 與 JSON 報告，方便訓練中自動比較候選模型。
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'summary.csv')
    json_path = os.path.join(output_dir, 'summary.json')
    fields = [
        'name', 'status', 'gate_status', 'gate_notes', 'tempo_bpm', 'time_signature',
        'kick', 'snare', 'hihat', 'gt_kick', 'gt_snare', 'gt_hihat', 'mean_f1',
        'audio', 'model', 'midi', 'event_debug', 'returncode',
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    return csv_path, json_path


def run_self_check():
    """
    中文註解：最小自檢，確保 stdout parser 能抓到速度、拍號與鼓數。
    """
    sample = """
    Effective Time Signature: 4/4 (4.00 beats per measure)
    [Score Tempo] quarter=110.00 BPM (MIDI quarter=110.00 BPM)
    Event debug CSV exported to: out.csv
      - Kick (KD): 8
      - Snare (SD): 8
      - Hi-Hat (HH): 64
    File Mean F1-Score: 19.42%
    """
    parsed = parse_transcribe_output(sample)
    assert parsed['time_signature'] == '4/4'
    assert parsed['tempo_bpm'] == '110.00'
    assert parsed['kick'] == 8
    assert parsed['snare'] == 8
    assert parsed['hihat'] == 64
    assert parsed['mean_f1'] == '19.42%'
    gate_status, gate_notes = evaluate_gate(
        {'gt_snare': 10, 'min_recall_snare': 0.5},
        {'status': 'pass', 'snare': 2},
    )
    assert gate_status == 'fail'
    assert 'snare recall' in gate_notes
    print('Self-check passed.')


def main():
    """
    中文註解：CLI 入口，執行固定 hard validation 案例並產生報告。
    """
    parser = argparse.ArgumentParser(description='Run hard validation for a drum checkpoint.')
    parser.add_argument('--model', default='best_drum_model.pth')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--star-hard', default='processed_data/star_hard_validation.json')
    parser.add_argument('--star-limit', type=int, default=0)
    parser.add_argument('--star-min-recall', type=float, default=0.15)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()

    if args.self_check:
        run_self_check()
        return

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_stem = os.path.splitext(os.path.basename(args.model))[0]
    output_dir = args.output_dir or os.path.join('validation_runs', f'{stamp}_{model_stem}')
    cases = LOCAL_CASES + load_star_cases(args.star_hard, args.star_limit, args.star_min_recall)
    rows = []
    for case in cases:
        print(f'Running {case["name"]}: {case["audio"]}')
        rows.append(run_case(case, args.model, output_dir))
    csv_path, json_path = write_reports(rows, output_dir)
    print(f'Wrote hard validation CSV: {csv_path}')
    print(f'Wrote hard validation JSON: {json_path}')


if __name__ == '__main__':
    main()
