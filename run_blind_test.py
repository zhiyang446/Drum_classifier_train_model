# -*- coding: utf-8 -*-
"""
Batch blind-test runner for drum transcription candidates.
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys

from run_hard_validation import parse_transcribe_output


AUDIO_EXTS = {'.wav', '.flac', '.mp3', '.ogg', '.m4a', '.aiff', '.aif'}
INSTS = ('kick', 'snare', 'hihat')


def discover_audio(input_path, limit=0):
    """
    中文註解：收集單檔或資料夾中的音訊，排序後可用 limit 做 smoke test。
    """
    if os.path.isfile(input_path):
        return [input_path]
    paths = []
    for root, _, files in os.walk(input_path):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in AUDIO_EXTS:
                paths.append(os.path.join(root, name))
    paths.sort()
    return paths[:limit] if limit > 0 else paths


def safe_stem(path):
    """
    中文註解：把音檔名稱轉成穩定資料夾名稱。
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', stem)


def count_layer(csv_path):
    """
    中文註解：統計 raw/notation CSV 的 native、final、virtual 鼓件數。
    """
    if not os.path.exists(csv_path):
        return {'rows': 0, 'native': {}, 'final': {}, 'virtual': {}}
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    result = {'rows': len(rows)}
    for prefix in ('native', 'final', 'virtual'):
        result[prefix] = {
            inst: sum(row.get(f'{prefix}_{inst}') == 'True' for row in rows)
            for inst in INSTS
        }
    return result


def run_one(audio_path, model_path, output_dir, hint=None, thresholds=None, architecture='symmetric'):
    """
    中文註解：執行單一 blind test 音檔並回傳 summary row。
    """
    name = safe_stem(audio_path)
    case_dir = os.path.join(output_dir, name)
    os.makedirs(case_dir, exist_ok=True)
    midi_path = os.path.join(case_dir, f'{name}.mid')
    event_debug = os.path.join(case_dir, f'{name}_event_debug.csv')
    raw_csv = os.path.join(case_dir, f'{name}_raw_ai_events.csv')
    notation_csv = os.path.join(case_dir, f'{name}_notation_events.csv')
    stdout_path = os.path.join(case_dir, f'{name}_stdout.txt')
    stderr_path = os.path.join(case_dir, f'{name}_stderr.txt')

    cmd = [
        sys.executable, 'transcribe.py',
        '--input', audio_path,
        '--model', model_path,
        '--output', midi_path,
        '--event-debug', event_debug,
        '--raw-ai-events', raw_csv,
        '--notation-events', notation_csv,
        '--architecture', architecture,
    ]
    if hint:
        # 中文註解：診斷模式只使用 tempo/拍號提示，不用 KD/SD/HH 答案改寫輸出。
        cmd.extend(['--tempo', hint['expected_tempo'], '--time-signature', hint['expected_time_signature']])
    if thresholds:
        for arg_name, value in thresholds.items():
            if value is not None:
                cmd.extend([arg_name, str(value)])
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    with open(stdout_path, 'w', encoding='utf-8') as f:
        f.write(proc.stdout)
    with open(stderr_path, 'w', encoding='utf-8') as f:
        f.write(proc.stderr)

    parsed = parse_transcribe_output(proc.stdout)
    raw = count_layer(raw_csv)
    notation = count_layer(notation_csv)
    shuffle_completion = '[Shuffle Completion]' in proc.stdout

    row = {
        'name': name,
        'audio': audio_path,
        'status': 'pass' if proc.returncode == 0 else 'fail',
        'returncode': proc.returncode,
        'tempo_bpm': parsed.get('tempo_bpm', ''),
        'time_signature': parsed.get('time_signature', ''),
        'midi_kick': parsed.get('kick', ''),
        'midi_snare': parsed.get('snare', ''),
        'midi_hihat': parsed.get('hihat', ''),
        'raw_kick': raw['final'].get('kick', 0),
        'raw_snare': raw['final'].get('snare', 0),
        'raw_hihat': raw['final'].get('hihat', 0),
        'notation_kick': notation['final'].get('kick', 0),
        'notation_snare': notation['final'].get('snare', 0),
        'notation_hihat': notation['final'].get('hihat', 0),
        'virtual_kick': notation['virtual'].get('kick', 0),
        'virtual_snare': notation['virtual'].get('snare', 0),
        'virtual_hihat': notation['virtual'].get('hihat', 0),
        'shuffle_completion': shuffle_completion,
        'midi': midi_path,
        'event_debug': event_debug,
        'raw_ai_events': raw_csv,
        'notation_events': notation_csv,
        'stdout': stdout_path,
        'stderr': stderr_path,
    }
    if proc.returncode != 0:
        row['error_tail'] = proc.stderr[-1000:]
    return row


def load_expected_hints(path):
    """
    中文註解：讀取第一批 blind test 的 tempo/拍號提示；鼓件數只供外部比較，不在此改寫。
    """
    if not path:
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return {row['name']: row for row in csv.DictReader(f)}


def write_reports(rows, output_dir):
    """
    中文註解：輸出 blind test summary CSV/JSON。
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'summary.csv')
    json_path = os.path.join(output_dir, 'summary.json')
    fields = [
        'name', 'status', 'returncode', 'tempo_bpm', 'time_signature',
        'midi_kick', 'midi_snare', 'midi_hihat',
        'raw_kick', 'raw_snare', 'raw_hihat',
        'notation_kick', 'notation_snare', 'notation_hihat',
        'virtual_kick', 'virtual_snare', 'virtual_hihat',
        'shuffle_completion', 'audio', 'midi', 'event_debug',
        'raw_ai_events', 'notation_events', 'stdout', 'stderr', 'error_tail',
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
    中文註解：最小自檢，確認音檔探索與 CSV 統計可用。
    """
    assert safe_stem('a b#c.wav') == 'a_b_c'
    assert discover_audio(__file__) == [__file__] if os.path.splitext(__file__)[1].lower() in AUDIO_EXTS else True
    print('Self-check passed.')


def main():
    """
    中文註解：CLI 入口，批量執行 blind test。
    """
    parser = argparse.ArgumentParser(description='Run blind transcription tests.')
    parser.add_argument('--input', required=False, help='Audio file or directory.')
    parser.add_argument('--model', default='mixed_formal_kick375_snare18_hh12_candidate.pth')
    parser.add_argument('--output-dir', default='validation_runs/blind_test')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--expected-hints', default=None, help='Optional expected CSV used only for tempo/time-signature diagnostic hints.')
    parser.add_argument('--thresh-kick', type=float, default=None)
    parser.add_argument('--thresh-snare', type=float, default=None)
    parser.add_argument('--thresh-hihat', type=float, default=None)
    parser.add_argument('--thresh-tom', type=float, default=None)
    parser.add_argument('--thresh-crash', type=float, default=None)
    parser.add_argument('--thresh-ride', type=float, default=None)
    parser.add_argument('--architecture', default='symmetric')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()

    if args.self_check:
        run_self_check()
        return
    if not args.input:
        raise SystemExit('--input is required unless --self-check is used.')

    audio_files = discover_audio(args.input, args.limit)
    if not audio_files:
        raise SystemExit(f'No audio files found: {args.input}')
    hints = load_expected_hints(args.expected_hints)
    thresholds = {
        '--thresh-kick': args.thresh_kick,
        '--thresh-snare': args.thresh_snare,
        '--thresh-hihat': args.thresh_hihat,
        '--thresh-tom': args.thresh_tom,
        '--thresh-crash': args.thresh_crash,
        '--thresh-ride': args.thresh_ride,
    }
    rows = []
    for audio_path in audio_files:
        print(f'Running blind test: {audio_path}', flush=True)
        rows.append(run_one(audio_path, args.model, args.output_dir, hints.get(safe_stem(audio_path)), thresholds, architecture=args.architecture))
    csv_path, json_path = write_reports(rows, args.output_dir)
    print(f'Wrote blind summary CSV: {csv_path}')
    print(f'Wrote blind summary JSON: {json_path}')


if __name__ == '__main__':
    main()
