# -*- coding: utf-8 -*-
"""
STAR Drums 前處理腳本。

將 STAR Drums 的 18 類 annotation 縮減成目前專案使用的 KD/SD/HH 三類，
並輸出與 preprocess_egmd.py 相容的 processed_data/star_meta.json。
"""
import argparse
import json
import os
from collections import Counter

import soundfile as sf


STAR_DIR = os.path.join('STAR_Drums_full', 'STAR_publication')
OUTPUT_DIR = 'processed_data'
OUTPUT_JSON = os.path.join(OUTPUT_DIR, 'star_meta.json')

# STAR 18 類到本專案三類的最小映射；tom/cymbal/ride 先不進三類模型。
CLASS_TO_INST = {
    'BD': 'KD',
    'SD': 'SD',
    'SS': 'SD',
    'CHH': 'HH',
    'PHH': 'HH',
    'OHH': 'HH',
}


def parse_annotation_file(annotation_path):
    """
    解析 STAR annotation txt，回傳 KD/SD/HH 事件與原始類別統計。

    :param annotation_path: str，STAR annotation txt 路徑。
    :return: tuple[list[dict], Counter]，三類事件與原始 18 類統計。
    """
    events = []
    raw_counts = Counter()

    with open(annotation_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue

            onset_sec = float(parts[0])
            raw_class = parts[1]
            velocity = float(parts[2])
            raw_counts[raw_class] += 1

            inst = CLASS_TO_INST.get(raw_class)
            if inst is None:
                continue

            events.append({
                'time': onset_sec,
                'inst': inst,
                'velocity': velocity
            })

    events.sort(key=lambda x: x['time'])
    return events, raw_counts


def get_audio_duration(audio_path):
    """
    使用 soundfile 讀取 FLAC 長度，避免載入整段音訊到記憶體。

    :param audio_path: str，音訊路徑。
    :return: float，音訊秒數。
    """
    info = sf.info(audio_path)
    return float(info.frames / info.samplerate)


def annotation_to_audio_path(annotation_path):
    """
    依 STAR 目錄命名規則，由 annotation txt 找到對應 mix FLAC。

    :param annotation_path: str，annotation txt 路徑。
    :return: str，對應 audio/mix/*.flac 路徑。
    """
    annotation_dir = os.path.dirname(annotation_path)
    split_root = os.path.dirname(annotation_dir)
    base_name = os.path.splitext(os.path.basename(annotation_path))[0]
    return os.path.join(split_root, 'audio', 'mix', f'{base_name}.flac')


def iter_annotation_files(star_dir):
    """
    掃描 STAR training/validation/test 底下所有 annotation txt。

    :param star_dir: str，STAR_publication 根目錄。
    :return: iterator[tuple[str, str]]，輸出 split 與 annotation 路徑。
    """
    data_dir = os.path.join(star_dir, 'data')
    split_specs = [
        ('train', os.path.join(data_dir, 'training')),
        ('validation', os.path.join(data_dir, 'validation')),
        ('test', os.path.join(data_dir, 'test')),
    ]

    for split, split_dir in split_specs:
        for root, _, files in os.walk(split_dir):
            if os.path.basename(root) != 'annotation':
                continue
            for name in files:
                if name.lower().endswith('.txt'):
                    yield split, os.path.join(root, name)


def build_star_metadata(star_dir):
    """
    建立 STAR 三類 metadata，格式對齊現有 E-GMD metadata。

    :param star_dir: str，STAR_publication 根目錄。
    :return: tuple[dict, dict]，metadata 與統計資訊。
    """
    meta = {}
    stats = {
        'processed': 0,
        'skipped_missing_audio': 0,
        'skipped_empty_events': 0,
        'events': Counter(),
        'raw_classes': Counter(),
        'splits': Counter(),
    }

    for split, annotation_path in iter_annotation_files(star_dir):
        audio_path = annotation_to_audio_path(annotation_path)
        if not os.path.isfile(audio_path):
            stats['skipped_missing_audio'] += 1
            continue

        events, raw_counts = parse_annotation_file(annotation_path)
        stats['raw_classes'].update(raw_counts)
        if not events:
            stats['skipped_empty_events'] += 1
            continue

        for ev in events:
            stats['events'][ev['inst']] += 1

        rel_key = os.path.relpath(annotation_path, os.path.join(star_dir, 'data'))
        song_key = 'star_' + os.path.splitext(rel_key)[0].replace('\\', '_').replace('/', '_')

        meta[song_key] = {
            'audio_path': os.path.abspath(audio_path),
            'annotation_path': os.path.abspath(annotation_path),
            'duration': get_audio_duration(audio_path),
            'bpm': 120.0,
            'split': split,
            'kit_name': os.path.basename(audio_path).replace('.flac', '').split('_mix_')[-1],
            'events': events
        }
        stats['processed'] += 1
        stats['splits'][split] += 1

    return meta, stats


def run_self_check():
    """
    執行最小 parser 自檢，確保類別映射與忽略類別行為不被破壞。
    """
    events = []
    raw_counts = Counter()
    lines = ['0.0\tBD\t100', '0.5\tSD\t90', '0.75\tSS\t85', '1.0\tOHH\t80', '1.5\tMT\t70']
    for line in lines:
        onset, raw_class, velocity = line.split()
        raw_counts[raw_class] += 1
        inst = CLASS_TO_INST.get(raw_class)
        if inst:
            events.append({'time': float(onset), 'inst': inst, 'velocity': float(velocity)})

    assert [e['inst'] for e in events] == ['KD', 'SD', 'SD', 'HH']
    assert raw_counts['MT'] == 1
    print('Self-check passed.')


def main():
    """
    CLI 入口：轉換 STAR annotation 並寫出 JSON metadata。
    """
    parser = argparse.ArgumentParser(description='Preprocess STAR Drums into KD/SD/HH metadata.')
    parser.add_argument('--star-dir', default=STAR_DIR, help='Path to STAR_publication directory.')
    parser.add_argument('--output', default=OUTPUT_JSON, help='Output JSON metadata path.')
    parser.add_argument('--self-check', action='store_true', help='Run parser self-check and exit.')
    args = parser.parse_args()

    if args.self_check:
        run_self_check()
        return

    if not os.path.isdir(args.star_dir):
        raise FileNotFoundError(f'STAR directory not found: {args.star_dir}')

    meta, stats = build_star_metadata(args.star_dir)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {len(meta)} STAR items to {args.output}")
    print(f"Splits: {dict(stats['splits'])}")
    print(f"KD/SD/HH events: {dict(stats['events'])}")
    print(f"Raw STAR classes: {dict(stats['raw_classes'])}")
    print(f"Skipped missing audio: {stats['skipped_missing_audio']}")
    print(f"Skipped empty KD/SD/HH: {stats['skipped_empty_events']}")


if __name__ == '__main__':
    main()
