# -*- coding: utf-8 -*-
"""
Convert confirmed user annotation CSV rows into training metadata.
"""
import argparse
import csv
import glob
import json
import os

import soundfile as sf


VALID_INSTS = {'KD', 'SD', 'HH'}
PHYSICAL_TIME_SOURCES = {'raw_ai', 'audio_onset', 'grid_fill+audio_onset', 'notation_physical_map'}


def is_confirmed(value):
    """
    中文註解：接受常見布林文字，方便人工在 CSV 裡填 true/yes/1。
    """
    return str(value).strip().lower() in {'true', 'yes', '1', 'y'}


def load_confirmed_events(csv_path, allow_score_time=False):
    """
    中文註解：只讀取 confirmed=True 且已在音訊物理時間座標的 KD/SD/HH 事件。
    """
    events = []
    rejected_sources = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            inst = row.get('inst', '').strip()
            if inst not in VALID_INSTS or not is_confirmed(row.get('confirmed', '')):
                continue
            source = row.get('source', '').strip()
            if not allow_score_time and source not in PHYSICAL_TIME_SOURCES:
                rejected_sources[source or '<blank>'] = rejected_sources.get(source or '<blank>', 0) + 1
                continue
            events.append({
                'time': float(row['time']),
                'inst': inst,
                'velocity': float(row.get('velocity') or 90.0),
                'source': source,
            })
    if rejected_sources:
        details = ', '.join(f'{name}={count}' for name, count in sorted(rejected_sources.items()))
        raise ValueError(
            f'{csv_path} contains confirmed score-time rows ({details}). '
            'Convert them to physical audio time first, or pass --allow-score-time explicitly.'
        )
    events.sort(key=lambda item: (item['time'], item['inst']))
    return events


def build_meta(annotation_dir, audio_dir, allow_score_time=False):
    """
    中文註解：把每首已確認 CSV 轉成 train split metadata。
    """
    meta = {}
    chosen = {}
    for csv_path in sorted(glob.glob(os.path.join(annotation_dir, '*_annotations*.csv'))):
        filename = os.path.basename(csv_path)
        if filename.endswith('_annotations_physical.csv'):
            name = filename.replace('_annotations_physical.csv', '')
        elif filename.endswith('_annotations_score_confirmed.csv'):
            name = filename.replace('_annotations_score_confirmed.csv', '')
        elif filename.endswith('_annotations.csv'):
            name = filename.replace('_annotations.csv', '')
        else:
            continue
        if filename.endswith('_annotations_physical.csv') or filename.endswith('_annotations_score_confirmed.csv') or name not in chosen:
            chosen[name] = csv_path

    for name, csv_path in sorted(chosen.items()):
        events = load_confirmed_events(csv_path, allow_score_time=allow_score_time)
        if not events:
            continue
        audio_path = os.path.abspath(os.path.join(audio_dir, f'{name}.wav'))
        meta[name] = {
            'audio_path': audio_path,
            'split': 'train',
            'source': 'user_blind_verified',
            'events': events,
        }
    return meta


def audio_duration(path):
    """
    中文註解：取得音訊秒數，用來建立覆蓋整首的訓練視窗。
    """
    with sf.SoundFile(path) as f:
        return f.frames / float(f.samplerate)


def window_meta(meta, window_seconds):
    """
    中文註解：把每首 verified annotation 拆成多個訓練 item，避免長音檔只訓練中段。
    """
    if window_seconds <= 0:
        return meta
    output = {}
    step = window_seconds * 0.75
    for name, item in meta.items():
        duration = audio_duration(item['audio_path'])
        center = window_seconds / 2.0
        idx = 0
        while center < duration + window_seconds / 2.0:
            left = max(0.0, center - window_seconds / 2.0)
            right = min(duration, center + window_seconds / 2.0)
            events = [ev for ev in item['events'] if left <= float(ev['time']) < right]
            if events:
                copied = dict(item)
                copied['source'] = 'user_blind_verified_windowed'
                copied['_anchor_time'] = center
                copied['events'] = events
                output[f'{name}_win{idx:03d}'] = copied
                idx += 1
            center += step
    return output


def run_self_check():
    """
    中文註解：最小自檢，確認 confirmed 判斷不會誤吃 false。
    """
    assert is_confirmed('True')
    assert is_confirmed('yes')
    assert not is_confirmed('False')
    assert 'raw_ai' in PHYSICAL_TIME_SOURCES
    print('Self-check passed.')


def main():
    """
    中文註解：CLI 入口，輸出 verified metadata。
    """
    parser = argparse.ArgumentParser(description='Convert confirmed user annotations to metadata.')
    parser.add_argument('--annotation-dir', default='annotations/user_blind_precise')
    parser.add_argument('--audio-dir', default='blind_user_tests')
    parser.add_argument('--output', default='processed_data/user_blind_precise_verified_meta.json')
    parser.add_argument('--window-seconds', type=float, default=0.0)
    parser.add_argument('--allow-empty', action='store_true')
    parser.add_argument('--allow-score-time', action='store_true')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()

    if args.self_check:
        run_self_check()
        return

    meta = window_meta(
        build_meta(args.annotation_dir, args.audio_dir, allow_score_time=args.allow_score_time),
        args.window_seconds,
    )
    if not meta and not args.allow_empty:
        raise SystemExit('No confirmed annotations found. Mark rows with confirmed=True first.')
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f'Wrote {len(meta)} verified annotation items to {args.output}')


if __name__ == '__main__':
    main()
