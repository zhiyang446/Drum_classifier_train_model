# -*- coding: utf-8 -*-
"""將 MDB Drums full-mix 與 subclass 標註轉成現有六類 metadata。"""
import argparse
import json
from collections import Counter
from pathlib import Path

import soundfile as sf


LABELS = ('KD', 'SD', 'HH', 'TOM', 'CRASH', 'RIDE')
SUBCLASS_TO_INST = {
    'KD': 'KD',
    'SD': 'SD', 'SDB': 'SD', 'SDD': 'SD', 'SDF': 'SD',
    'SDG': 'SD', 'SDNS': 'SD', 'SST': 'SD',
    'CHH': 'HH', 'OHH': 'HH', 'PHH': 'HH',
    'HIT': 'TOM', 'MHT': 'TOM', 'HFT': 'TOM', 'LFT': 'TOM',
    'CRC': 'CRASH', 'CHC': 'CRASH', 'SPC': 'CRASH',
    'RDC': 'RIDE', 'RDB': 'RIDE',
}
IGNORED_SUBCLASSES = {'TMB'}
MIREX_TRAIN = {
    'MusicDelta_80sRock', 'MusicDelta_BebopJazz', 'MusicDelta_Britpop',
    'MusicDelta_CoolJazz', 'MusicDelta_Disco', 'MusicDelta_FunkJazz',
    'MusicDelta_FusionJazz', 'MusicDelta_Reggae', 'MusicDelta_Rock',
    'MusicDelta_Rockabilly', 'MusicDelta_Shadows', 'MusicDelta_Zeppelin',
}


def map_subclass(subclass):
    """中文註解：把官方 21 subclass 映射為產品六類，非目標 tambourine 回傳 None。"""
    if subclass in IGNORED_SUBCLASSES:
        return None
    if subclass not in SUBCLASS_TO_INST:
        raise ValueError(f'Unknown MDB subclass: {subclass}')
    return SUBCLASS_TO_INST[subclass]


def split_for_song(song):
    """中文註解：依官方 MIREX 2017 歌曲級清單指派 train/test。"""
    return 'train' if song in MIREX_TRAIN else 'test'


def parse_annotation(path, duration):
    """中文註解：解析單首 subclass 標註並拒絕未知標籤或超出音訊的事件。"""
    events = []
    with path.open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f'{path}:{line_number}: expected time and subclass')
            time_sec = float(parts[0])
            if not 0.0 <= time_sec <= duration:
                raise ValueError(f'{path}:{line_number}: event {time_sec} outside duration {duration}')
            inst = map_subclass(parts[1])
            if inst is not None:
                events.append({
                    'time': time_sec,
                    'inst': inst,
                    'subclass': parts[1],
                    'velocity': 100.0,
                })
    events.sort(key=lambda row: (row['time'], row['subclass']))
    return events


def build_metadata(root):
    """中文註解：建立 23 首 full-mix metadata 與歌曲級 split 稽核報告。"""
    audio_dir = root / 'audio' / 'full_mix'
    annotation_dir = root / 'annotations' / 'subclass'
    audio_paths = sorted(audio_dir.glob('*_MIX.wav'))
    if len(audio_paths) != 23:
        raise ValueError(f'Expected 23 MDB full mixes, found {len(audio_paths)}')

    metadata = {}
    split_counts = Counter()
    event_counts = {split: Counter() for split in ('train', 'test')}
    songs = []
    for audio_path in audio_paths:
        song = audio_path.stem.removesuffix('_MIX')
        annotation_path = annotation_dir / f'{song}_subclass.txt'
        if not annotation_path.is_file():
            raise FileNotFoundError(annotation_path)
        if song in metadata:
            raise ValueError(f'Duplicate MDB song: {song}')
        info = sf.info(audio_path)
        duration = info.frames / float(info.samplerate)
        events = parse_annotation(annotation_path, duration)
        split = split_for_song(song)
        metadata[song] = {
            'audio_path': str(audio_path.resolve()),
            'annotation_path': str(annotation_path.resolve()),
            'duration': duration,
            'split': split,
            'source': 'mdbdrums_full_mix',
            'events': events,
        }
        split_counts[split] += 1
        event_counts[split].update(event['inst'] for event in events)
        songs.append({'key': song, 'split': split, 'events': len(events)})

    if split_counts != Counter({'train': 12, 'test': 11}):
        raise ValueError(f'Expected MIREX split 12/11, found {dict(split_counts)}')
    missing = {
        split: sorted(set(LABELS) - set(event_counts[split]))
        for split in ('train', 'test')
    }
    if any(missing.values()):
        raise ValueError(f'Incomplete six-class coverage: {missing}')
    audit = {
        'status': 'pass',
        'source': 'MDB Drums CC BY-NC-SA 4.0',
        'songs': dict(split_counts),
        'events': {
            split: {label: event_counts[split][label] for label in LABELS}
            for split in ('train', 'test')
        },
        'song_rows': songs,
    }
    return metadata, audit


def merge_negative_metadata(base, mdb_metadata):
    """中文註解：只把 MDB 官方 train 歌曲以 negative_train 身分加入既有正樣本 metadata。"""
    output = dict(base)
    added = 0
    for song, item in sorted(mdb_metadata.items()):
        if item.get('split') != 'train':
            continue
        if item.get('source') != 'mdbdrums_full_mix':
            raise ValueError(f'Unexpected MDB source for {song}: {item.get("source")}')
        key = f'mdb_negative_{song}'
        if key in output:
            raise ValueError(f'Metadata key collision: {key}')
        copied = dict(item)
        copied['split'] = 'negative_train'
        output[key] = copied
        added += 1
    if added != 12:
        raise ValueError(f'Expected 12 MDB negative_train songs, found {added}')
    return output, added


def run_self_check():
    """中文註解：驗證官方映射、忽略標籤、未知標籤拒絕與歌曲級 split。"""
    assert {map_subclass(value) for value in ('HIT', 'MHT', 'HFT', 'LFT')} == {'TOM'}
    assert {map_subclass(value) for value in ('CRC', 'CHC', 'SPC')} == {'CRASH'}
    assert {map_subclass(value) for value in ('RDC', 'RDB')} == {'RIDE'}
    assert map_subclass('TMB') is None
    assert split_for_song('MusicDelta_Zeppelin') == 'train'
    assert split_for_song('MusicDelta_Beatles') == 'test'
    try:
        map_subclass('UNKNOWN')
    except ValueError:
        pass
    else:
        raise AssertionError('未知 MDB subclass 必須拒絕')
    mock_mdb = {
        f'song_{index}': {'split': 'train', 'source': 'mdbdrums_full_mix'}
        for index in range(12)
    }
    mock_mdb['held'] = {'split': 'test', 'source': 'mdbdrums_full_mix'}
    merged, added = merge_negative_metadata({'base': {'split': 'train'}}, mock_mdb)
    assert added == 12 and merged['mdb_negative_song_0']['split'] == 'negative_train'
    print('Self-check passed.')


def main():
    """中文註解：執行轉換並寫出全新 metadata 與稽核 JSON。"""
    parser = argparse.ArgumentParser(description='Build six-class MDB Drums metadata.')
    parser.add_argument('--root', default='MDBDrums/MDB Drums')
    parser.add_argument('--output', default='processed_data/mdbdrums_six_class_meta_d5b.json')
    parser.add_argument('--audit', default='validation_runs/mdbdrums_d5b_audit.json')
    parser.add_argument('--base-meta', help='Opt-in base metadata merged with MDB train as negative_train only')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return

    metadata, audit = build_metadata(Path(args.root))
    if args.base_meta:
        with Path(args.base_meta).open(encoding='utf-8') as handle:
            base = json.load(handle)
        metadata, added = merge_negative_metadata(base, metadata)
        audit['base_meta'] = str(Path(args.base_meta).resolve())
        audit['negative_train_songs'] = added
        audit['combined_items'] = len(metadata)
    for path, payload in ((Path(args.output), metadata), (Path(args.audit), audit)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
