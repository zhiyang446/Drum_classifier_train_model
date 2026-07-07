# -*- coding: utf-8 -*-
"""
Build supervised metadata for user-confirmed blind drum examples.
"""
import argparse
import csv
import json
import os

import soundfile as sf


def audio_duration(path):
    """
    中文註解：讀取音訊長度，讓產生的事件不超出檔案尾端。
    """
    with sf.SoundFile(path) as f:
        return f.frames / float(f.samplerate)


def grid_times(start, interval, count, duration):
    """
    中文註解：產生固定網格事件時間，並把最後事件限制在音檔內。
    """
    return [t for t in (start + i * interval for i in range(count)) if 0.0 <= t < duration]


def load_raw_peaks(raw_csv):
    """
    中文註解：讀取現有 raw AI 事件，優先用模型已定位的實際 peak 時間當訓練標註。
    """
    peaks = {'KD': [], 'SD': [], 'HH': []}
    if not raw_csv or not os.path.exists(raw_csv):
        return peaks
    cols = {
        'KD': ('final_kick', 'prob_kick'),
        'SD': ('final_snare', 'prob_snare'),
        'HH': ('final_hihat', 'prob_hihat'),
    }
    with open(raw_csv, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            raw_time = float(row.get('raw_time') or row.get('quantized_time') or 0.0)
            for inst, (hit_col, prob_col) in cols.items():
                if row.get(hit_col) == 'True':
                    peaks[inst].append((raw_time, float(row.get(prob_col) or 0.0)))
    return peaks


def fit_events(inst, count, duration, start, end, raw_peaks):
    """
    中文註解：保留可信 raw peak；不足的事件用音檔內節奏網格補足，避免粗平均超出真實音訊。
    """
    selected = sorted(raw_peaks.get(inst, []), key=lambda item: item[1], reverse=True)[:count]
    times = sorted(t for t, _ in selected)
    if len(times) < count:
        interval = (end - start) / max(count - 1, 1)
        for candidate in grid_times(start, interval, count, duration):
            if len(times) >= count:
                break
            if all(abs(candidate - existing) > 0.035 for existing in times):
                times.append(candidate)
    return sorted(times[:count])


def build_events(row, audio_path, raw_peaks=None):
    """
    中文註解：依使用者確認的數量建立最小 KD/SD/HH 監督標註。
    """
    tempo = float(row['expected_tempo'])
    beat = 60.0 / tempo
    duration = audio_duration(audio_path)
    start = 0.30 if duration > 1.0 else 0.0

    hh_count = int(row['expected_hihat'])
    sd_count = int(row['expected_snare'])
    kd_count = int(row['expected_kick'])

    raw_peaks = raw_peaks or {'KD': [], 'SD': [], 'HH': []}
    all_raw_times = [t for items in raw_peaks.values() for t, _ in items]
    if all_raw_times:
        start = max(0.0, min(all_raw_times))

    hh_per_beat = 4 if hh_count >= 64 else 2
    beat_span = max(1, hh_count // hh_per_beat)
    score_span = beat_span * beat
    audio_span = max(beat, duration - start - 0.05)
    total_span = min(score_span, audio_span)
    end = min(duration - 0.05, start + total_span)

    events = []
    for t in fit_events('HH', hh_count, duration, start, end, raw_peaks):
        events.append({'time': t, 'inst': 'HH', 'velocity': 90.0})
    for t in fit_events('SD', sd_count, duration, start, end, raw_peaks):
        events.append({'time': t, 'inst': 'SD', 'velocity': 95.0})
    for t in fit_events('KD', kd_count, duration, start, end, raw_peaks):
        events.append({'time': t, 'inst': 'KD', 'velocity': 100.0})
    events.sort(key=lambda item: (item['time'], item['inst']))
    return events


def window_items(name, audio_path, events, window_seconds):
    """
    中文註解：把長音檔拆成多個訓練 item，確保訓練覆蓋整首而不是只取中間 4 秒。
    """
    if window_seconds <= 0:
        return {
            name: {
                'audio_path': audio_path,
                'split': 'train',
                'source': 'user_blind_hard',
                'events': events,
            }
        }
    duration = audio_duration(audio_path)
    items = {}
    step = window_seconds * 0.75
    idx = 0
    center = window_seconds / 2.0
    while center < duration + window_seconds / 2.0:
        left = max(0.0, center - window_seconds / 2.0)
        right = min(duration, center + window_seconds / 2.0)
        subset = [ev for ev in events if left <= float(ev['time']) < right]
        if subset:
            items[f'{name}_win{idx:03d}'] = {
                'audio_path': audio_path,
                'split': 'train',
                'source': 'user_blind_hard_windowed',
                '_anchor_time': center,
                'events': subset,
            }
            idx += 1
        center += step
    return items


def build_meta(input_dir, expected_csv, raw_root=None, window_seconds=0.0):
    """
    中文註解：把 blind_user_tests 與 expected CSV 合成 train split metadata。
    """
    rows = {}
    for row in csv.DictReader(open(expected_csv, 'r', encoding='utf-8')):
        rows[row['name']] = row

    meta = {}
    for name, row in rows.items():
        audio_path = os.path.abspath(os.path.join(input_dir, f'{name}.wav'))
        if not os.path.exists(audio_path):
            continue
        raw_csv = os.path.join(raw_root, name, f'{name}_raw_ai_events.csv') if raw_root else None
        events = build_events(row, audio_path, load_raw_peaks(raw_csv))
        meta.update(window_items(name, audio_path, events, window_seconds))
    return meta


def run_self_check():
    """
    中文註解：最小自檢，確認固定網格數量可控。
    """
    assert grid_times(0.0, 0.5, 3, 2.0) == [0.0, 0.5, 1.0]
    print('Self-check passed.')


def main():
    """
    中文註解：CLI 入口，輸出 user blind hard metadata JSON。
    """
    parser = argparse.ArgumentParser(description='Build user blind hard-example metadata.')
    parser.add_argument('--input-dir', default='blind_user_tests')
    parser.add_argument('--expected', default='blind_user_tests_expected.csv')
    parser.add_argument('--raw-root', default=None)
    parser.add_argument('--window-seconds', type=float, default=0.0)
    parser.add_argument('--output', default='processed_data/user_blind_hard_meta.json')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()

    if args.self_check:
        run_self_check()
        return

    meta = build_meta(args.input_dir, args.expected, args.raw_root, args.window_seconds)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f'Wrote {len(meta)} user hard examples to {args.output}')


if __name__ == '__main__':
    main()
