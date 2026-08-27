# -*- coding: utf-8 -*-
"""接入 Whack Studio Metal 的真實鼓 WAV/MIDI，建立六類資料稽核。"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import mido
import soundfile as sf


LABELS = ('KD', 'SD', 'HH', 'TOM', 'CRASH', 'RIDE')
PITCH_TO_INST = {
    36: 'KD',
    38: 'SD',
    26: 'HH', 42: 'HH', 44: 'HH', 46: 'HH',
    41: 'TOM', 43: 'TOM', 45: 'TOM', 47: 'TOM', 48: 'TOM', 50: 'TOM',
    49: 'CRASH', 52: 'CRASH', 55: 'CRASH', 57: 'CRASH',
    51: 'RIDE', 53: 'RIDE', 59: 'RIDE',
}
BPM_PATTERN = re.compile(r'(?<!\d)(\d{2,3})\s*BPM', flags=re.IGNORECASE)
BOUNDARY_TOLERANCE_SECONDS = 0.05


def split_for_group(group_id):
    """以歌曲群組的固定雜湊建立 80/10/10 split，避免跨歌洩漏。"""
    bucket = int(hashlib.sha256(group_id.encode('utf-8')).hexdigest()[:8], 16) % 10
    if bucket < 8:
        return 'train'
    if bucket == 8:
        return 'validation'
    return 'test'


def bpm_from_filename(midi_path):
    """解析 MIDI 檔名的作者 BPM；缺失時回傳 None。"""
    match = BPM_PATTERN.search(Path(midi_path).name)
    return float(match.group(1)) if match else None


def midi_tick_events(midi_path):
    """讀取 MIDI 絕對 tick 事件、未知 pitch 及完整時間軸終點。"""
    midi = mido.MidiFile(midi_path)
    ticks = 0
    events = []
    unknown_pitches = Counter()
    seen = set()
    tempo_messages = []
    for message in mido.merge_tracks(midi.tracks):
        ticks += message.time
        if message.type == 'set_tempo':
            tempo_messages.append(int(message.tempo))
            continue
        if message.type != 'note_on' or message.velocity <= 0:
            continue
        inst = PITCH_TO_INST.get(message.note)
        if inst is None:
            unknown_pitches[int(message.note)] += 1
            continue
        event_key = (ticks, int(message.note), int(message.velocity))
        if event_key in seen:
            continue
        seen.add(event_key)
        events.append({
            'tick': int(ticks),
            'inst': inst,
            'pitch': int(message.note),
            'velocity': float(message.velocity),
        })
    if not events:
        raise ValueError(f'No mapped six-class events: {midi_path}')
    events.sort(key=lambda row: (row['tick'], row['pitch']))
    return midi.ticks_per_beat, events, dict(sorted(unknown_pitches.items())), int(ticks), tempo_messages


def select_bpm(midi_path, ticks_per_beat, final_ticks, audio_duration, tempo_messages):
    """優先採用檔名 BPM；只有缺失時才使用單一 MIDI tempo 或音訊長度推算。"""
    filename_bpm = bpm_from_filename(midi_path)
    if filename_bpm is not None:
        return filename_bpm, 'filename_bpm', False
    if len(set(tempo_messages)) == 1:
        return 60000000.0 / tempo_messages[0], 'midi_tempo', False
    if final_ticks <= 0 or audio_duration <= 0:
        raise ValueError(f'Cannot infer BPM for {midi_path}')
    # ponytail: 沒有 tempo 的 MIDI 只用全曲長度推算單一 BPM；若有變速需求才加入逐段音訊對齊。
    inferred_bpm = final_ticks * 60.0 / (ticks_per_beat * audio_duration)
    return inferred_bpm, 'inferred_from_audio_duration', True


def timed_events(events, ticks_per_beat, bpm):
    """把絕對 tick 轉為秒，保留原 MIDI pitch 與 velocity。"""
    return [{
        'time': event['tick'] * 60.0 / (ticks_per_beat * bpm),
        'inst': event['inst'],
        'pitch': event['pitch'],
        'velocity': event['velocity'],
    } for event in events]


def paired_paths(track_dir):
    """驗證每首資料夾僅有一個 WAV 與一個 MIDI。"""
    wavs = sorted(Path(track_dir).glob('*.wav'))
    midis = sorted(Path(track_dir).glob('*.mid'))
    if len(wavs) != 1 or len(midis) != 1:
        raise ValueError(f'Expected one WAV and one MIDI in {track_dir}, found {len(wavs)}/{len(midis)}')
    return wavs[0], midis[0]


def write_json(path, payload):
    """只寫入新 JSON，拒絕覆寫既有接入結果。"""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {path}')
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


def build_metadata(root, output_dir):
    """建立真實 metal 鼓 metadata，並把時間軸不可用歌曲列入 audit。"""
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Dataset root not found: {root}')
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite existing output directory: {output_dir}')

    metadata = {}
    excluded = []
    alignment_rows = []
    event_counts = {split: Counter() for split in ('train', 'validation', 'test')}
    unknown_pitch_counts = Counter()
    bpm_sources = Counter()
    review_items = []
    track_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    for index, track_dir in enumerate(track_dirs, start=1):
        audio_path, midi_path = paired_paths(track_dir)
        audio_info = sf.info(str(audio_path))
        audio_duration = float(audio_info.duration)
        ticks_per_beat, events, unknown_pitches, final_ticks, tempo_messages = midi_tick_events(midi_path)
        bpm, bpm_source, review_required = select_bpm(
            midi_path, ticks_per_beat, final_ticks, audio_duration, tempo_messages
        )
        timed = timed_events(events, ticks_per_beat, bpm)
        last_event_time = timed[-1]['time']
        group_id = f'whack_metal_d28:{track_dir.name}'
        alignment_status = f'{bpm_source}_in_bounds'
        if last_event_time > audio_duration + BOUNDARY_TOLERANCE_SECONDS:
            alignment_status = f'{bpm_source}_outside_audio'
            excluded.append({
                'group_id': group_id,
                'audio_path': str(audio_path.resolve()),
                'midi_path': str(midi_path.resolve()),
                'bpm': bpm,
                'bpm_source': bpm_source,
                'last_event_time': last_event_time,
                'audio_duration': audio_duration,
                'outside_seconds': last_event_time - audio_duration,
            })
            continue
        item_id = f'whack_metal_d28_{index:03d}'
        split = split_for_group(group_id)
        metadata[item_id] = {
            'audio_path': str(audio_path.resolve()),
            'midi_path': str(midi_path.resolve()),
            'duration': audio_duration,
            'bpm': bpm,
            'bpm_source': bpm_source,
            'split': split,
            'group_id': group_id,
            'source': 'whack_studio_metal_d28',
            'sample_rate': int(audio_info.samplerate),
            'channels': int(audio_info.channels),
            'alignment_status': alignment_status,
            'review_required': review_required,
            'events': timed,
        }
        event_counts[split].update(event['inst'] for event in timed)
        unknown_pitch_counts.update(unknown_pitches)
        bpm_sources[bpm_source] += 1
        alignment_rows.append({
            'group_id': group_id,
            'bpm': bpm,
            'bpm_source': bpm_source,
            'review_required': review_required,
            'audio_duration': audio_duration,
            'last_event_time': last_event_time,
            'tail_seconds': audio_duration - last_event_time,
        })
        if review_required:
            review_items.append(group_id)

    if not metadata:
        raise ValueError('No in-bounds WAV/MIDI pairs available.')
    group_splits = {}
    for item in metadata.values():
        group_splits.setdefault(item['group_id'], set()).add(item['split'])
    leaked_groups = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    if leaked_groups:
        raise AssertionError(f'group_id crosses splits: {leaked_groups[:3]}')
    missing_labels = {
        split: [label for label in LABELS if event_counts[split][label] == 0]
        for split in event_counts
    }
    ready_for_training = not excluded and not review_items and not any(missing_labels.values())
    audit = {
        'phase': 'D28',
        'status': 'pass' if ready_for_training else 'pass_with_alignment_review',
        'source_root': str(root),
        'track_directories': len(track_dirs),
        'metadata_items': len(metadata),
        'excluded_outside_audio': excluded,
        'splits': dict(Counter(item['split'] for item in metadata.values())),
        'events': {split: {label: event_counts[split][label] for label in LABELS} for split in event_counts},
        'missing_labels': missing_labels,
        'group_split_leaks': 0,
        'bpm_sources': dict(sorted(bpm_sources.items())),
        'review_required_groups': review_items,
        'unknown_pitch_counts': dict(sorted(unknown_pitch_counts.items())),
        'alignment_rows': alignment_rows,
        'ready_for_training_candidate': ready_for_training,
        'ready_for_six_class_release': False,
    }
    output_dir.mkdir(parents=True)
    write_json(output_dir / 'metadata_d28.json', metadata)
    write_json(output_dir / 'audit_d28.json', audit)
    return metadata, audit


def run_self_check():
    """驗證 BPM 選擇、六類映射與群組 split 的固定性。"""
    assert bpm_from_filename(Path('Song MIDI - 158 BPM.mid')) == 158.0
    assert bpm_from_filename(Path('Song MIDI.mid')) is None
    assert PITCH_TO_INST[41] == PITCH_TO_INST[50] == 'TOM'
    assert PITCH_TO_INST[49] == PITCH_TO_INST[57] == 'CRASH'
    assert PITCH_TO_INST[51] == PITCH_TO_INST[59] == 'RIDE'
    assert split_for_group('whack_metal_d28:track') == split_for_group('whack_metal_d28:track')
    bpm, source, review = select_bpm(Path('No tempo.mid'), 480, 4800, 10.0, [])
    assert (bpm, source, review) == (60.0, 'inferred_from_audio_duration', True)
    print('Self-check passed.')


def main():
    """CLI 入口，建立 Whack Studio Metal 真實音訊資料的 metadata 與 audit。"""
    parser = argparse.ArgumentParser(description='Build Whack Studio Metal six-class WAV/MIDI metadata.')
    parser.add_argument('--root', default='Whack Studio Metal Drum Tracks/Whack Studio Metal Drum Tracks')
    parser.add_argument('--output-dir', default='whack_studio_metal_d28')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    metadata, audit = build_metadata(args.root, args.output_dir)
    print(f"Wrote {len(metadata)} metadata items; audit status: {audit['status']}")


if __name__ == '__main__':
    main()
