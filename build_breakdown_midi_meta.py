# -*- coding: utf-8 -*-
"""建立 Breakdown MIDI Pack 的六類配對 metadata 與唯讀稽核報告。"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import librosa
import mido
import numpy as np


LABELS = ('KD', 'SD', 'HH', 'TOM', 'CRASH', 'RIDE')
PITCH_TO_INST = {
    36: 'KD',
    38: 'SD',
    26: 'HH', 42: 'HH', 44: 'HH', 46: 'HH',
    41: 'TOM', 43: 'TOM', 45: 'TOM', 47: 'TOM', 48: 'TOM', 50: 'TOM',
    49: 'CRASH', 52: 'CRASH', 55: 'CRASH', 57: 'CRASH',
    51: 'RIDE', 53: 'RIDE', 59: 'RIDE',
}
EXPECTED_PAIRS = 52
MAX_START_DELTA_SECONDS = 0.05


def track_id(path):
    """從套件檔名讀取固定配對編號，拒絕不符合規則的檔案。"""
    match = re.match(r'^(\d+)[\.,]', path.name)
    if match is None:
        raise ValueError(f'Cannot parse track id from {path.name}')
    return match.group(1)


def tempo_bpm(path):
    """從檔名讀取原作者標示的 BPM。"""
    match = re.search(r'(\d+)\s*BPM', path.name, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f'Cannot parse BPM from {path.name}')
    return float(match.group(1))


def indexed_files(directory, suffix):
    """以配對編號建立檔案索引，拒絕重複 ID。"""
    paths = {}
    for path in sorted(directory.glob(f'*{suffix}')):
        key = track_id(path)
        if key in paths:
            raise ValueError(f'Duplicate track id {key}: {paths[key]} and {path}')
        paths[key] = path
    return paths


def audio_stats(audio_path):
    """讀取 MP3 長度與首個可聽 onset，驗證 MIDI 起點對齊。"""
    samples, sample_rate = librosa.load(str(audio_path), sr=None, mono=True)
    if len(samples) == 0:
        raise ValueError(f'Empty audio: {audio_path}')
    rms = librosa.feature.rms(y=samples, frame_length=2048, hop_length=256)[0]
    active = np.flatnonzero(rms >= rms.max() * 0.02)
    if len(active) == 0:
        raise ValueError(f'No audible onset: {audio_path}')
    return float(len(samples) / sample_rate), float(active[0] * 256 / sample_rate)


def midi_events(midi_path, bpm, duration):
    """用檔名 BPM 轉換 MIDI tick，映射為六類並檢查音訊時間邊界。"""
    midi = mido.MidiFile(midi_path)
    events = []
    seen = set()
    for track in midi.tracks:
        ticks = 0
        for message in track:
            ticks += message.time
            if message.type != 'note_on' or message.velocity <= 0:
                continue
            inst = PITCH_TO_INST.get(message.note)
            if inst is None:
                raise ValueError(f'Unknown GM pitch {message.note} in {midi_path}')
            # ponytail: filename BPM is authoritative because this pack omits MIDI tempo events.
            time_sec = ticks * 60.0 / (midi.ticks_per_beat * bpm)
            if not 0.0 <= time_sec <= duration:
                raise ValueError(f'Event {time_sec:.6f}s outside audio {duration:.6f}s: {midi_path}')
            event_key = (round(time_sec, 9), int(message.note), int(message.velocity))
            if event_key in seen:
                continue
            seen.add(event_key)
            events.append({
                'time': float(time_sec),
                'inst': inst,
                'pitch': int(message.note),
                'velocity': float(message.velocity),
            })
    if not events:
        raise ValueError(f'No mapped MIDI events: {midi_path}')
    events.sort(key=lambda row: (row['time'], row['pitch']))
    return events


def split_for_index(index):
    """以完整配對為單位建立固定 42/5/5 split，避免同曲跨集合。"""
    if not 0 <= index < EXPECTED_PAIRS:
        raise ValueError(f'Unexpected split index: {index}')
    if index < 42:
        return 'train'
    if index < 47:
        return 'validation'
    return 'test'


def build_metadata(root):
    """建立 metadata 與 audit；缺配對、對齊失敗或未知音高立即停止。"""
    root = Path(root)
    audio_by_id = indexed_files(root / 'Reference Audio', '.mp3')
    midi_by_id = indexed_files(root / 'Breakdown MIDIs', '.mid')
    if set(audio_by_id) != set(midi_by_id):
        raise ValueError('MP3/MIDI pairing mismatch')
    if len(audio_by_id) != EXPECTED_PAIRS:
        raise ValueError(f'Expected {EXPECTED_PAIRS} pairs, found {len(audio_by_id)}')

    metadata = {}
    event_counts = {split: Counter() for split in ('train', 'validation', 'test')}
    alignment_rows = []
    for index, key in enumerate(sorted(audio_by_id, key=int)):
        audio_path = audio_by_id[key]
        midi_path = midi_by_id[key]
        bpm = tempo_bpm(midi_path)
        duration, audio_start = audio_stats(audio_path)
        events = midi_events(midi_path, bpm, duration)
        midi_start = events[0]['time']
        start_delta = audio_start - midi_start
        if abs(start_delta) > MAX_START_DELTA_SECONDS:
            raise ValueError(f'Start alignment exceeds {MAX_START_DELTA_SECONDS}s: {audio_path}')
        split = split_for_index(index)
        group_id = f'breakdown_{int(key):03d}'
        metadata[group_id] = {
            'audio_path': str(audio_path.resolve()),
            'midi_path': str(midi_path.resolve()),
            'duration': duration,
            'bpm': bpm,
            'split': split,
            'group_id': group_id,
            'source': 'breakdown_midi_pack_d25',
            'events': events,
        }
        event_counts[split].update(event['inst'] for event in events)
        alignment_rows.append({
            'group_id': group_id,
            'audio_start_sec': audio_start,
            'midi_start_sec': midi_start,
            'start_delta_sec': start_delta,
        })

    missing = {
        split: [label for label in LABELS if event_counts[split][label] == 0]
        for split in event_counts
    }
    audit = {
        'status': 'pass_with_coverage_gap' if any(missing.values()) else 'pass',
        'pairs': len(metadata),
        'splits': dict(Counter(item['split'] for item in metadata.values())),
        'events': {
            split: {label: event_counts[split][label] for label in LABELS}
            for split in event_counts
        },
        'missing_labels': missing,
        'max_abs_start_delta_sec': max(abs(row['start_delta_sec']) for row in alignment_rows),
        'alignment_rows': alignment_rows,
        'ready_for_training_candidate': True,
        'ready_for_six_class_release': False,
    }
    return metadata, audit


def write_json(path, payload):
    """寫入全新 JSON；拒絕覆寫既有資料準備輸出。"""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


def run_self_check():
    """驗證檔名解析、六類映射與固定歌曲級 split。"""
    assert track_id(Path('50, 198 BPM.mid')) == '50'
    assert tempo_bpm(Path('18. 173 BPM .mid')) == 173.0
    assert PITCH_TO_INST[41] == PITCH_TO_INST[47] == 'TOM'
    assert PITCH_TO_INST[49] == PITCH_TO_INST[57] == 'CRASH'
    assert PITCH_TO_INST[51] == PITCH_TO_INST[53] == 'RIDE'
    splits = [split_for_index(index) for index in range(EXPECTED_PAIRS)]
    assert Counter(splits) == Counter({'train': 42, 'validation': 5, 'test': 5})
    print('Self-check passed.')


def main():
    """CLI 入口，輸出 Breakdown MIDI Pack 的 metadata 與 audit JSON。"""
    parser = argparse.ArgumentParser(description='Build Breakdown MIDI Pack six-class metadata.')
    parser.add_argument('--root', default='Breakdown MIDI Pack')
    parser.add_argument('--output', default='processed_data/breakdown_midi_meta_d25.json')
    parser.add_argument('--audit', default='processed_data/breakdown_midi_audit_d25.json')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    metadata, audit = build_metadata(args.root)
    write_json(args.output, metadata)
    write_json(args.audit, audit)
    print(f'Wrote {len(metadata)} metadata items to {args.output}')
    print(f"Audit status: {audit['status']}; events: {audit['events']}")


if __name__ == '__main__':
    main()
