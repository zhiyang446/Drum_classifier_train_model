# -*- coding: utf-8 -*-
"""Build aligned, windowed training metadata from real separated drum WAV/MIDI pairs."""
import argparse
import csv
import json
import os
import re

import librosa
import numpy as np
import pretty_midi
import soundfile as sf


PITCH_TO_INST = {
    35: 'KD', 36: 'KD',
    37: 'SD', 38: 'SD', 40: 'SD',
    22: 'HH', 26: 'HH', 42: 'HH', 44: 'HH', 46: 'HH',
}


def normalized_stem(path):
    """將常見樂譜匯出尾碼移除，取得可配對的檔名。"""
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    stem = re.sub(r'[-_](drum[-_]?sheet[-_]?music|score)$', '', stem)
    return stem


def discover_pairs(input_dir, included):
    """依正規化檔名建立唯一 WAV/MIDI 配對。"""
    wavs = {}
    midis = {}
    for name in os.listdir(input_dir):
        path = os.path.join(input_dir, name)
        if not os.path.isfile(path):
            continue
        key = normalized_stem(name)
        suffix = os.path.splitext(name)[1].lower()
        if suffix == '.wav':
            wavs[key] = path
        elif suffix in {'.mid', '.midi'}:
            midis.setdefault(key, []).append(path)

    pairs = []
    for key in included:
        if key not in wavs:
            raise ValueError(f'Missing WAV for {key}')
        choices = midis.get(key, [])
        if len(choices) != 1:
            raise ValueError(f'Expected exactly one MIDI for {key}, found {len(choices)}')
        pairs.append((key, wavs[key], choices[0]))
    return pairs


def target_events(midi_path):
    """讀取目前三分類支援的 MIDI 事件與原始 pitch。"""
    pm = pretty_midi.PrettyMIDI(midi_path)
    events = []
    unsupported = 0
    for instrument in pm.instruments:
        for note in instrument.notes:
            inst = PITCH_TO_INST.get(note.pitch)
            if inst is None:
                unsupported += 1
                continue
            events.append({
                'time': float(note.start),
                'inst': inst,
                'velocity': int(note.velocity),
                'pitch': int(note.pitch),
            })
    events.sort(key=lambda event: event['time'])
    if not events:
        raise ValueError(f'No KD/SD/HH events in {midi_path}')
    return events, unsupported


def onset_envelope(audio_path):
    """計算對齊用的標準化 onset 強度時間序列。"""
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    hop_length = 256
    envelope = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    times = librosa.frames_to_time(np.arange(len(envelope)), sr=sr, hop_length=hop_length)
    envelope = (envelope - np.median(envelope)) / (np.std(envelope) + 1e-9)
    return times, envelope


def estimate_alignment(audio_path, events, scale_min, scale_max, scale_step, offset_min, offset_max, offset_step):
    """以 MIDI 鼓點對 audio onset 的平均反應搜尋全域時間映射。"""
    times, envelope = onset_envelope(audio_path)
    event_times = np.array([event['time'] for event in events], dtype=np.float64)
    scales = np.arange(scale_min, scale_max + scale_step * 0.5, scale_step)
    offsets = np.arange(offset_min, offset_max + offset_step * 0.5, offset_step)
    best_score, best_scale, best_offset = -np.inf, None, None
    for scale in scales:
        scaled_times = event_times * scale
        for offset in offsets:
            score = float(np.mean(np.interp(scaled_times + offset, times, envelope, left=-3.0, right=-3.0)))
            if score > best_score:
                best_score, best_scale, best_offset = score, float(scale), float(offset)
    return best_score, best_scale, best_offset


def build_windows(key, audio_path, midi_path, events, scale, offset, window_seconds, window_step):
    """將長歌展開成固定四秒、至少含一個目標事件的訓練窗口。"""
    duration = float(sf.info(audio_path).frames / sf.info(audio_path).samplerate)
    aligned = []
    for event in events:
        mapped_time = event['time'] * scale + offset
        if 0.0 <= mapped_time < duration:
            copied = dict(event)
            copied['time'] = float(mapped_time)
            aligned.append(copied)
    windows = {}
    last_start = max(0.0, duration - window_seconds)
    start = 0.0
    index = 0
    while start <= last_start + 1e-9:
        end = start + window_seconds
        if any(start <= event['time'] < end for event in aligned):
            windows[f'{key}_{index:03d}'] = {
                'audio_path': audio_path,
                'midi_path': midi_path,
                'duration': duration,
                'split': 'train',
                'source': 'real_audio_round1',
                '_anchor_time': float(start + window_seconds / 2.0),
                'events': aligned,
            }
        start += window_step
        index += 1
    if not windows:
        raise ValueError(f'No event-bearing windows for {key}')
    return windows, len(aligned)


def run_self_check():
    """執行不讀取資料集的最小配對與映射自檢。"""
    assert normalized_stem('blue-yung-kai-drum-sheet-music.mid') == 'blue-yung-kai'
    assert normalized_stem('counting-stars.wav') == 'counting-stars'
    assert PITCH_TO_INST[46] == 'HH'
    assert 49 not in PITCH_TO_INST
    print('Self-check passed.')


def main():
    """CLI 入口：輸出 real-audio 訓練 metadata 與對齊報表。"""
    parser = argparse.ArgumentParser(description='Build aligned real-audio drum training metadata.')
    parser.add_argument('--input-dir', required=False, default='test_real_audio')
    parser.add_argument('--include', default='')
    parser.add_argument('--output', default='validation_runs/real_audio_round1_meta.json')
    parser.add_argument('--report', default='validation_runs/real_audio_round1_alignment.csv')
    parser.add_argument('--window-seconds', type=float, default=4.0)
    parser.add_argument('--window-step', type=float, default=3.5)
    parser.add_argument('--scale-min', type=float, default=1.0)
    parser.add_argument('--scale-max', type=float, default=1.0)
    parser.add_argument('--scale-step', type=float, default=0.0025)
    parser.add_argument('--offset-min', type=float, default=-5.0)
    parser.add_argument('--offset-max', type=float, default=5.0)
    parser.add_argument('--offset-step', type=float, default=0.05)
    parser.add_argument('--min-alignment-score', type=float, default=0.5)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    included = [normalized_stem(value.strip()) for value in args.include.split(',') if value.strip()]
    if not included:
        raise SystemExit('--include requires the intended train pair names')
    if args.window_seconds <= 0 or args.window_step <= 0 or args.scale_step <= 0 or args.offset_step <= 0:
        raise SystemExit('Window and search steps must be positive')

    metadata = {}
    report_rows = []
    for key, audio_path, midi_path in discover_pairs(args.input_dir, included):
        events, unsupported = target_events(midi_path)
        score, scale, offset = estimate_alignment(
            audio_path, events, args.scale_min, args.scale_max, args.scale_step,
            args.offset_min, args.offset_max, args.offset_step,
        )
        if score < args.min_alignment_score:
            raise ValueError(f'Alignment score too low for {key}: {score:.3f}')
        windows, aligned_count = build_windows(
            key, audio_path, midi_path, events, scale, offset,
            args.window_seconds, args.window_step,
        )
        metadata.update(windows)
        report_rows.append({
            'name': key,
            'audio_path': audio_path,
            'midi_path': midi_path,
            'alignment_score': f'{score:.4f}',
            'scale': f'{scale:.4f}',
            'offset_seconds': f'{offset:.3f}',
            'target_events': aligned_count,
            'unsupported_events': unsupported,
            'windows': len(windows),
        })
        print(f'{key}: score={score:.3f} scale={scale:.4f} offset={offset:+.2f}s windows={len(windows)}')

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    os.makedirs(os.path.dirname(args.report) or '.', exist_ok=True)
    with open(args.report, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report_rows[0]))
        writer.writeheader()
        writer.writerows(report_rows)
    print(f'Wrote {len(metadata)} windows to {args.output}')


if __name__ == '__main__':
    main()
