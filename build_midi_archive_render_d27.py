# -*- coding: utf-8 -*-
"""將使用者提供的鼓 MIDI Archive 去重、渲染成六類訓練候選音訊。"""
import argparse
import audioop
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import wave

import mido


LABELS = ('KD', 'SD', 'HH', 'TOM', 'CRASH', 'RIDE')
PITCH_TO_INST = {
    36: 'KD',
    38: 'SD',
    26: 'HH', 42: 'HH', 44: 'HH', 46: 'HH',
    41: 'TOM', 43: 'TOM', 45: 'TOM', 47: 'TOM', 48: 'TOM', 50: 'TOM',
    49: 'CRASH', 52: 'CRASH', 55: 'CRASH', 57: 'CRASH',
    51: 'RIDE', 53: 'RIDE', 59: 'RIDE',
}
SAMPLE_RATE = 44100


def sha256_file(path):
    """計算檔案內容雜湊，作為 MIDI 去重與音訊追溯依據。"""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def group_id_for(root, midi_path):
    """以來源 MIDI 的父資料夾作不可拆分歌曲／groove 群組。"""
    relative_parent = Path(midi_path).relative_to(root).parent.as_posix()
    return f'midi_archive_d27:{relative_parent}'


def split_for_group(group_id):
    """以群組內容的固定雜湊建立 80/10/10 split，避免跨群組洩漏。"""
    bucket = int(hashlib.sha256(group_id.encode('utf-8')).hexdigest()[:8], 16) % 10
    if bucket < 8:
        return 'train'
    if bucket == 8:
        return 'validation'
    return 'test'


def midi_events(midi_path):
    """解析 tempo-aware MIDI 時間，回傳六類事件、未知音高及結束時間。"""
    midi = mido.MidiFile(midi_path)
    tempo = 500000
    seconds = 0.0
    events = []
    unknown_pitches = Counter()
    seen = set()
    for message in mido.merge_tracks(midi.tracks):
        seconds += mido.tick2second(message.time, midi.ticks_per_beat, tempo)
        if message.type == 'set_tempo':
            tempo = message.tempo
            continue
        if message.type != 'note_on' or message.velocity <= 0:
            continue
        inst = PITCH_TO_INST.get(message.note)
        if inst is None:
            unknown_pitches[int(message.note)] += 1
            continue
        event_key = (round(seconds, 9), int(message.note), int(message.velocity))
        if event_key in seen:
            continue
        seen.add(event_key)
        events.append({
            'time': float(seconds),
            'inst': inst,
            'pitch': int(message.note),
            'velocity': float(message.velocity),
        })
    events.sort(key=lambda row: (row['time'], row['pitch']))
    return events, dict(sorted(unknown_pitches.items())), float(seconds)


def collect_canonical_midis(root):
    """遞迴讀取來源根目錄，保留每個 MIDI SHA-256 的第一個 canonical 檔案。"""
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'MIDI archive root not found: {root}')
    canonical = {}
    duplicates = []
    for midi_path in sorted(root.rglob('*.mid'), key=lambda path: path.relative_to(root).as_posix()):
        content_hash = sha256_file(midi_path)
        if content_hash in canonical:
            duplicates.append({
                'duplicate_path': str(midi_path.resolve()),
                'canonical_path': str(canonical[content_hash]['midi_path'].resolve()),
                'midi_sha256': content_hash,
            })
            continue
        canonical[content_hash] = {
            'midi_path': midi_path,
            'midi_sha256': content_hash,
            'group_id': group_id_for(root, midi_path),
        }
    if not canonical:
        raise ValueError(f'No MIDI files found under {root}')
    return list(canonical.values()), duplicates


def render_wav(renderer_path, soundfont_path, ffmpeg_path, midi_path, output_path):
    """以 FluidSynth 渲染並以 FFmpeg 轉為模型所需的單聲道 PCM WAV。"""
    output_path = Path(output_path)
    with tempfile.TemporaryDirectory(prefix='d27_midi_render_') as temporary_directory:
        # ponytail: 每個 MIDI 都用獨立暫存檔，避免保留雙聲道中間檔；若需速度才改成平行 worker。
        temporary_wav = Path(temporary_directory) / 'renderer.wav'
        renderer_command = [
            str(renderer_path), '-q', '-i', '-n', '-T', 'wav', '-F', str(temporary_wav),
            '-r', str(SAMPLE_RATE), str(soundfont_path), str(midi_path),
        ]
        subprocess.run(renderer_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        ffmpeg_command = [
            str(ffmpeg_path), '-nostdin', '-v', 'error', '-i', str(temporary_wav),
            '-ac', '1', '-ar', str(SAMPLE_RATE), '-c:a', 'pcm_s16le', '-n', str(output_path),
        ]
        subprocess.run(ffmpeg_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def validate_wav(audio_path, midi_duration):
    """驗證最終 WAV 格式、非靜音與不短於 MIDI 時間軸。"""
    with wave.open(str(audio_path), 'rb') as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        frames = handle.getnframes()
        raw_audio = handle.readframes(frames)
    if sample_rate != SAMPLE_RATE or channels != 1 or sample_width != 2:
        raise ValueError(f'Unexpected WAV format for {audio_path}: {sample_rate}Hz/{channels}ch/{sample_width * 8}bit')
    if frames == 0:
        raise ValueError(f'Empty WAV: {audio_path}')
    duration = frames / sample_rate
    rms = audioop.rms(raw_audio, sample_width)
    if rms <= 0:
        raise ValueError(f'Silent WAV: {audio_path}')
    if duration + 1e-6 < midi_duration:
        raise ValueError(f'WAV shorter than MIDI: {duration:.6f}s < {midi_duration:.6f}s for {audio_path}')
    return float(duration), int(rms)


def write_json(path, payload):
    """只寫入全新的 JSON 產物，拒絕覆寫既有 batch 結果。"""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {path}')
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


def build_archive(root, output_dir, renderer_path, soundfont_path, ffmpeg_path, resume=False):
    """批次渲染 canonical MIDI，並回傳 six-class metadata 與完整 audit。"""
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    renderer_path = Path(renderer_path).resolve()
    soundfont_path = Path(soundfont_path).resolve()
    ffmpeg_path = Path(ffmpeg_path).resolve()
    for required_path in (renderer_path, soundfont_path, ffmpeg_path):
        if not required_path.is_file():
            raise FileNotFoundError(f'Required renderer asset not found: {required_path}')
    if output_dir.exists() and not resume:
        raise FileExistsError(f'Refusing to overwrite existing output directory: {output_dir}')
    if resume and (output_dir / 'metadata_d27.json').exists():
        raise FileExistsError(f'Refusing to resume completed output: {output_dir}')

    canonical_rows, duplicates = collect_canonical_midis(root)
    prepared = []
    skipped_without_six_class_events = []
    for row in canonical_rows:
        events, unknown_pitches, midi_duration = midi_events(row['midi_path'])
        if not events:
            skipped_without_six_class_events.append({
                'midi_path': str(row['midi_path'].resolve()),
                'midi_sha256': row['midi_sha256'],
                'unknown_pitches': unknown_pitches,
            })
            continue
        prepared.append({
            **row,
            'events': events,
            'unknown_pitches': unknown_pitches,
            'midi_duration': midi_duration,
            'split': split_for_group(row['group_id']),
        })
    if not prepared:
        raise ValueError('No canonical MIDI contains mapped six-class events.')

    output_dir.mkdir(parents=True, exist_ok=resume)
    audio_dir = output_dir / 'audio'
    audio_dir.mkdir(exist_ok=resume)
    metadata = {}
    event_counts = {split: Counter() for split in ('train', 'validation', 'test')}
    unknown_pitch_counts = Counter()
    render_failures = []
    resumed_items = 0
    for index, row in enumerate(prepared, start=1):
        item_id = f"midi_archive_d27_{row['midi_sha256'][:16]}"
        audio_path = audio_dir / f'{item_id}.wav'
        if audio_path.exists():
            # ponytail: 只驗證既有完成 WAV；不重跑、不覆寫，失敗檔案留給人工檢查。
            duration, rms = validate_wav(audio_path, row['midi_duration'])
            resumed_items += 1
        else:
            try:
                render_wav(renderer_path, soundfont_path, ffmpeg_path, row['midi_path'], audio_path)
                duration, rms = validate_wav(audio_path, row['midi_duration'])
            except subprocess.CalledProcessError as error:
                render_failures.append({
                    'midi_path': str(row['midi_path'].resolve()),
                    'midi_sha256': row['midi_sha256'],
                    'returncode': error.returncode,
                    'stderr': (error.stderr or '').strip()[:500],
                })
                print(f'Render failed at {index}/{len(prepared)}; recorded and continuing.', flush=True)
                continue
        metadata[item_id] = {
            'audio_path': str(audio_path.resolve()),
            'midi_path': str(row['midi_path'].resolve()),
            'midi_sha256': row['midi_sha256'],
            'duration': duration,
            'midi_duration': row['midi_duration'],
            'split': row['split'],
            'group_id': row['group_id'],
            'source': 'drum_percussion_midi_archive_d27',
            'sample_rate': SAMPLE_RATE,
            'channels': 1,
            'rms': rms,
            'events': row['events'],
        }
        event_counts[row['split']].update(event['inst'] for event in row['events'])
        unknown_pitch_counts.update(row['unknown_pitches'])
        if index % 25 == 0 or index == len(prepared):
            print(f'Rendered {index}/{len(prepared)} canonical MIDI files.', flush=True)

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
    audit = {
        'phase': 'D27',
        'status': 'pass_with_render_failures' if render_failures else ('pass_with_coverage_gap' if any(missing_labels.values()) else 'pass'),
        'source_root': str(root),
        'source_midi_files': len(canonical_rows) + len(duplicates),
        'canonical_midi_files': len(canonical_rows),
        'rendered_items': len(metadata),
        'resumed_items': resumed_items,
        'render_failures': render_failures,
        'duplicates_skipped': len(duplicates),
        'without_six_class_events_skipped': len(skipped_without_six_class_events),
        'splits': dict(Counter(item['split'] for item in metadata.values())),
        'groups': len(group_splits),
        'group_split_leaks': 0,
        'events': {split: {label: event_counts[split][label] for label in LABELS} for split in event_counts},
        'missing_labels': missing_labels,
        'unknown_pitch_counts': dict(sorted(unknown_pitch_counts.items())),
        'renderer': {'path': str(renderer_path), 'sha256': sha256_file(renderer_path)},
        'soundfont': {'path': str(soundfont_path), 'sha256': sha256_file(soundfont_path)},
        'audio': {'sample_rate': SAMPLE_RATE, 'channels': 1, 'codec': 'pcm_s16le'},
        'ready_for_training_candidate': not render_failures and not any(missing_labels.values()),
        'ready_for_six_class_release': False,
    }
    write_json(output_dir / 'metadata_d27.json', metadata)
    write_json(output_dir / 'audit_d27.json', audit)
    return metadata, audit


def run_self_check():
    """驗證六類映射、固定 split 與 metadata 群組隔離規則。"""
    assert PITCH_TO_INST[36] == 'KD' and PITCH_TO_INST[38] == 'SD'
    assert PITCH_TO_INST[41] == PITCH_TO_INST[50] == 'TOM'
    assert PITCH_TO_INST[49] == PITCH_TO_INST[57] == 'CRASH'
    assert PITCH_TO_INST[51] == PITCH_TO_INST[59] == 'RIDE'
    group_id = 'midi_archive_d27:genre/song'
    assert split_for_group(group_id) == split_for_group(group_id)
    records = {'one': {'group_id': group_id, 'split': split_for_group(group_id)}}
    assert len({item['split'] for item in records.values() if item['group_id'] == group_id}) == 1
    print('Self-check passed.')


def main():
    """執行 Archive 的去重、渲染、metadata 與 audit 建置。"""
    parser = argparse.ArgumentParser(description='Render six-class MIDI Archive audio and metadata for D27.')
    parser.add_argument('--root', default='800000_Drum_Percussion_MIDI_Archive[6_19_15]')
    parser.add_argument('--output-dir', default='synthetic_midi_archive_d27')
    parser.add_argument('--renderer', default='third_party/fluidsynth-2.4.7/bin/fluidsynth.exe')
    parser.add_argument('--soundfont', default='assets/soundfonts/v1.471.sf2')
    parser.add_argument('--ffmpeg', default='C:/ffmpeg/bin/ffmpeg.exe')
    parser.add_argument('--resume', action='store_true', help='Verify and reuse existing WAVs in an interrupted D27 output directory.')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    metadata, audit = build_archive(args.root, args.output_dir, args.renderer, args.soundfont, args.ffmpeg, resume=args.resume)
    print(f"Wrote {len(metadata)} rendered items; audit status: {audit['status']}")


if __name__ == '__main__':
    main()
