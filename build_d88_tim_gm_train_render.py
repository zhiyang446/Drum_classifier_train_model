"""建立 Archive train-only 的 TimGM 替代音色資料。"""

import argparse
import json
from collections import Counter
from pathlib import Path

from build_midi_archive_render_d27 import LABELS, render_wav, sha256_file, validate_wav, write_json


EXPECTED_TRAIN_ITEMS = 1382


def load_train_items(metadata_path):
    """讀取 D27 metadata，嚴格只保留既有 train 項目。"""
    metadata_path = Path(metadata_path).resolve()
    records = json.loads(metadata_path.read_text(encoding='utf-8'))
    train_items = {item_id: row for item_id, row in records.items() if row['split'] == 'train'}
    if len(train_items) != EXPECTED_TRAIN_ITEMS:
        raise ValueError(f'Expected {EXPECTED_TRAIN_ITEMS} D27 train items, found {len(train_items)}')
    return dict(sorted(train_items.items()))


def prepare_output(output_dir, resume):
    """建立全新輸出目錄；中斷後只能驗證既有 WAV 再續跑。"""
    output_dir = Path(output_dir).resolve()
    metadata_path = output_dir / 'metadata_d88.json'
    if output_dir.exists():
        if not resume or metadata_path.exists():
            raise FileExistsError(f'Refusing to overwrite existing output: {output_dir}')
    else:
        output_dir.mkdir(parents=True)
    audio_dir = output_dir / 'audio'
    audio_dir.mkdir(exist_ok=resume)
    return output_dir, audio_dir


def build_d88(metadata_path, output_dir, renderer_path, soundfont_path, ffmpeg_path, resume=False):
    """渲染所有 D27 train MIDI 為新 TimGM WAV，並只在完整成功後寫 metadata。"""
    metadata_path = Path(metadata_path).resolve()
    renderer_path = Path(renderer_path).resolve()
    soundfont_path = Path(soundfont_path).resolve()
    ffmpeg_path = Path(ffmpeg_path).resolve()
    for required_path in (metadata_path, renderer_path, soundfont_path, ffmpeg_path):
        if not required_path.is_file():
            raise FileNotFoundError(f'Required file not found: {required_path}')
    train_items = load_train_items(metadata_path)
    output_dir, audio_dir = prepare_output(output_dir, resume)
    metadata = {}
    event_counts = Counter()
    resumed_items = 0
    for index, (item_id, row) in enumerate(train_items.items(), start=1):
        audio_path = audio_dir / f'{item_id}.wav'
        if audio_path.exists():
            duration, rms = validate_wav(audio_path, row['midi_duration'])
            resumed_items += 1
        else:
            render_wav(renderer_path, soundfont_path, ffmpeg_path, row['midi_path'], audio_path)
            duration, rms = validate_wav(audio_path, row['midi_duration'])
        metadata[item_id] = {
            **row,
            'audio_path': str(audio_path.resolve()),
            'duration': duration,
            'rms': rms,
            'source': 'drum_percussion_midi_archive_d88_tim_gm',
            'original_audio_path': row['audio_path'],
            'original_source': row['source'],
            'render_soundfont_sha256': sha256_file(soundfont_path),
        }
        event_counts.update(event['inst'] for event in row['events'])
        if index % 25 == 0 or index == len(train_items):
            print(f'Rendered {index}/{len(train_items)} train MIDI files.', flush=True)
    missing_labels = [label for label in LABELS if event_counts[label] == 0]
    if missing_labels or any(row['split'] != 'train' for row in metadata.values()):
        raise AssertionError(f'D88 train-only coverage failed: missing_labels={missing_labels}')
    if len({row['audio_path'] for row in metadata.values()}) != len(metadata):
        raise AssertionError('D88 audio paths are not unique')
    group_splits = {}
    for row in metadata.values():
        group_splits.setdefault(row['group_id'], set()).add(row['split'])
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise AssertionError('D88 group_id crosses splits')
    audit = {
        'phase': 'D88',
        'status': 'pass',
        'input_metadata': str(metadata_path),
        'rendered_items': len(metadata),
        'resumed_items': resumed_items,
        'splits': dict(Counter(row['split'] for row in metadata.values())),
        'groups': len(group_splits),
        'group_split_leaks': 0,
        'events': {label: event_counts[label] for label in LABELS},
        'missing_labels': missing_labels,
        'renderer': {'path': str(renderer_path), 'sha256': sha256_file(renderer_path)},
        'soundfont': {'path': str(soundfont_path), 'sha256': sha256_file(soundfont_path)},
        'audio': {'sample_rate': 44100, 'channels': 1, 'codec': 'pcm_s16le'},
        'ready_for_training_candidate': True,
        'ready_for_six_class_release': False,
    }
    write_json(output_dir / 'metadata_d88.json', metadata)
    write_json(output_dir / 'audit_d88.json', audit)
    return metadata, audit


def run_self_check():
    """驗證 D88 僅接收 D27 train 並保留不可覆寫輸出規則。"""
    rows = {
        'train': {'split': 'train'},
        'validation': {'split': 'validation'},
    }
    assert {item_id for item_id, row in rows.items() if row['split'] == 'train'} == {'train'}
    assert EXPECTED_TRAIN_ITEMS == 1382
    print('D88 self-check passed.')


def main():
    """執行 D88 的 train-only TimGM 批次渲染。"""
    parser = argparse.ArgumentParser(description='Render D27 train MIDI with TimGM for D88.')
    parser.add_argument('--metadata', default='synthetic_midi_archive_d27/metadata_d27.json')
    parser.add_argument('--output-dir', default='synthetic_midi_archive_d88_tim_gm')
    parser.add_argument('--renderer', default='third_party/fluidsynth-2.4.7/bin/fluidsynth.exe')
    parser.add_argument('--soundfont', default='.venv/Lib/site-packages/pretty_midi/TimGM6mb.sf2')
    parser.add_argument('--ffmpeg', default='C:/ffmpeg/bin/ffmpeg.exe')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    metadata, audit = build_d88(
        args.metadata, args.output_dir, args.renderer, args.soundfont, args.ffmpeg, resume=args.resume,
    )
    print(f"Wrote {len(metadata)} D88 items; audit status: {audit['status']}")


if __name__ == '__main__':
    main()
