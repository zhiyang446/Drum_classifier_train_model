# -*- coding: utf-8 -*-
"""
Build training metadata from passed notation event CSVs.
"""
import argparse
import csv
import json
import os


EVENTS = [
    ('KD', 'kick', 'vel_kick'),
    ('SD', 'snare', 'vel_snare'),
    ('HH', 'hihat', 'vel_hihat'),
]


def is_true(value):
    """中文註解：解析 CSV 內的布林字串。"""
    return str(value).strip().lower() in {'true', '1', 'yes', 'y'}


def read_csv(path):
    """中文註解：讀取 CSV 檔案。"""
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def load_events(path, include_virtual):
    """中文註解：從 notation_events CSV 取出 final triggered 事件作為 teacher labels。"""
    events = []
    for row in read_csv(path):
        for inst, key, vel_key in EVENTS:
            if not is_true(row.get(f'final_{key}')):
                continue
            if not include_virtual and is_true(row.get(f'virtual_{key}')):
                continue
            events.append({
                'time': float(row['raw_time']),
                'inst': inst,
                'velocity': float(row.get(vel_key) or 90.0),
            })
    events.sort(key=lambda item: (item['time'], item['inst']))
    return events


def window_items(name, item, window_seconds):
    """中文註解：將長音檔拆成多個含事件視窗，避免只訓練中段。"""
    if window_seconds <= 0:
        return {name: item}
    events = item['events']
    if not events:
        return {}
    last_time = max(float(ev['time']) for ev in events)
    output = {}
    step = window_seconds * 0.75
    center = window_seconds / 2.0
    idx = 0
    while center <= last_time + window_seconds / 2.0:
        left = max(0.0, center - window_seconds / 2.0)
        right = center + window_seconds / 2.0
        chunk_events = [ev for ev in events if left <= float(ev['time']) < right]
        if chunk_events:
            copied = dict(item)
            copied['_anchor_time'] = center
            copied['events'] = chunk_events
            output[f'{name}_win{idx:03d}'] = copied
            idx += 1
        center += step
    return output


def build_meta(args):
    """中文註解：依 blind audio 名稱建立 teacher metadata。"""
    meta = {}
    for name in args.names:
        csv_path = os.path.join(args.notation_root, name, f'{name}_notation_events.csv')
        audio_path = os.path.abspath(os.path.join(args.audio_dir, f'{name}.wav'))
        events = load_events(csv_path, args.include_virtual)
        item = {
            'audio_path': audio_path,
            'split': 'train',
            'source': 'notation_teacher',
            'events': events,
        }
        meta.update(window_items(name, item, args.window_seconds))
    return meta


def run_self_check():
    """中文註解：確認 window 切分會保留事件。"""
    item = {'events': [{'time': 0.1, 'inst': 'KD', 'velocity': 90.0}, {'time': 3.0, 'inst': 'HH', 'velocity': 80.0}]}
    windows = window_items('demo', item, 2.0)
    assert windows
    assert sum(len(v['events']) for v in windows.values()) >= 2
    print('Self-check passed.')


def main():
    """中文註解：CLI 入口。"""
    parser = argparse.ArgumentParser(description='Build notation teacher metadata.')
    parser.add_argument('--notation-root', default='validation_runs/single_checkpoint_brain_repair_blind6')
    parser.add_argument('--audio-dir', default='blind_user_tests')
    parser.add_argument('--output', default='processed_data/user_blind_notation_teacher_meta.json')
    parser.add_argument('--window-seconds', type=float, default=4.0)
    parser.add_argument('--include-virtual', action='store_true')
    parser.add_argument('--names', nargs='+', default=[
        'basic_shuffle',
        'basic_straight_16',
        'basic_straight_8',
        'ghost_snare',
        'syncopated_4_4',
    ])
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    meta = build_meta(args)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f'Wrote {len(meta)} teacher items to {args.output}')


if __name__ == '__main__':
    main()
