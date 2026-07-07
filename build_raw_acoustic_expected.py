# -*- coding: utf-8 -*-
"""
Build raw-acoustic expected counts from physical-time confirmed annotations.
"""
import argparse
import csv
import glob
import os

from convert_user_annotations_to_meta import PHYSICAL_TIME_SOURCES, is_confirmed


INST_TO_FIELD = {
    'KD': 'expected_kick',
    'SD': 'expected_snare',
    'HH': 'expected_hihat',
}


def annotation_name(path):
    """中文註解：從 annotation CSV 檔名還原音檔名稱。"""
    name = os.path.basename(path)
    for suffix in ('_annotations_physical.csv', '_annotations_score_confirmed.csv', '_annotations.csv'):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return os.path.splitext(name)[0]


def count_physical_rows(path):
    """中文註解：只統計已確認且來源為音訊物理時間座標的 KD/SD/HH。"""
    counts = {field: 0 for field in INST_TO_FIELD.values()}
    skipped = 0
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            inst = row.get('inst', '').strip()
            source = row.get('source', '').strip()
            if inst not in INST_TO_FIELD or not is_confirmed(row.get('confirmed', '')):
                continue
            if source not in PHYSICAL_TIME_SOURCES:
                skipped += 1
                continue
            counts[INST_TO_FIELD[inst]] += 1
    return counts, skipped


def build_rows(annotation_dir):
    """中文註解：優先讀取 score_confirmed CSV，輸出 raw_acoustic 期望表。"""
    chosen = {}
    for path in sorted(glob.glob(os.path.join(annotation_dir, '*_annotations*.csv'))):
        name = annotation_name(path)
        if path.endswith('_annotations_score_confirmed.csv') or name not in chosen:
            chosen[name] = path

    rows = []
    for name, path in sorted(chosen.items()):
        counts, skipped = count_physical_rows(path)
        rows.append({
            'name': name,
            'expected_time_signature': '',
            'expected_tempo': '',
            'tempo_tol': '',
            **counts,
            'skipped_score_time_rows': skipped,
        })
    return rows


def write_rows(path, rows):
    """中文註解：輸出 raw acoustic expected CSV。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        'name', 'expected_time_signature', 'expected_tempo', 'tempo_tol',
        'expected_hihat', 'expected_snare', 'expected_kick', 'skipped_score_time_rows',
    ]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_self_check():
    """中文註解：確認欄位映射完整。"""
    assert INST_TO_FIELD['KD'] == 'expected_kick'
    assert INST_TO_FIELD['SD'] == 'expected_snare'
    assert INST_TO_FIELD['HH'] == 'expected_hihat'
    print('Self-check passed.')


def main():
    """中文註解：CLI 入口。"""
    parser = argparse.ArgumentParser(description='Build raw acoustic expected counts.')
    parser.add_argument('--annotation-dir', default='annotations/user_blind_precise')
    parser.add_argument('--output', default='validation_runs/raw_acoustic_expected.csv')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    rows = build_rows(args.annotation_dir)
    write_rows(args.output, rows)
    print(f'Wrote {len(rows)} rows to {args.output}')


if __name__ == '__main__':
    main()
