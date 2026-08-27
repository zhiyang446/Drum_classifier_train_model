# -*- coding: utf-8 -*-
"""唯讀稽核 Whack train/validation 的資料域與既有對齊欄位。"""
import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


LABELS = ('KD', 'SD', 'HH', 'TOM', 'CRASH', 'RIDE')
ROBUST_Z_LIMIT = 3.5


def robust_baseline(values):
    """中文註解：以中位數與 MAD 建立不受少數密度極端歌曲主導的基線。"""
    median = statistics.median(values)
    deviations = [abs(value - median) for value in values]
    return {'median': median, 'mad': statistics.median(deviations)}


def robust_z(value, baseline):
    """中文註解：回傳 MAD robust z；零 MAD 時只把不同值標為離群。"""
    mad = baseline['mad']
    if mad == 0.0:
        return 0.0 if value == baseline['median'] else math.inf
    return 0.67448975 * (value - baseline['median']) / mad


def whack_groups(metadata):
    """中文註解：從 D36 metadata 取出唯一 Whack group，並拒絕跨 split 洩漏。"""
    groups = {}
    for key, item in metadata.items():
        if item.get('source') != 'd36_whack_real':
            continue
        group_id = item.get('group_id')
        if not group_id:
            raise ValueError(f'Missing group_id for {key}.')
        if group_id in groups and groups[group_id]['split'] != item.get('split'):
            raise ValueError(f'Group split leak: {group_id}.')
        groups[group_id] = item
    if not groups:
        raise ValueError('No d36_whack_real items found.')
    return groups


def group_features(item, schedule_counts):
    """中文註解：產生單首歌曲的對齊、節奏與六類標註密度特徵。"""
    duration = float(item.get('duration', 0.0))
    if duration <= 0.0:
        raise ValueError(f"Invalid duration for {item['group_id']}.")
    required = ('bpm', 'alignment_score', 'alignment_offset_seconds')
    if any(item.get(field) is None for field in required):
        raise ValueError(f"Missing alignment fields for {item['group_id']}.")
    event_counts = Counter(event.get('inst') for event in item.get('events', []))
    if not sum(event_counts[label] for label in LABELS):
        raise ValueError(f"No six-class events for {item['group_id']}.")
    minutes = duration / 60.0
    densities = {label: event_counts[label] / minutes for label in LABELS}
    return {
        'group_id': item['group_id'], 'split': item['split'],
        'bpm': float(item['bpm']), 'duration_seconds': duration,
        'alignment_score': float(item['alignment_score']),
        'alignment_offset_seconds': float(item['alignment_offset_seconds']),
        'alignment_abs_offset_seconds': abs(float(item['alignment_offset_seconds'])),
        'event_counts': {label: event_counts[label] for label in LABELS},
        'events_per_minute': sum(event_counts[label] for label in LABELS) / minutes,
        'events_per_minute_by_class': densities,
        'schedule_windows': schedule_counts[item['group_id']],
    }


def audit_payload(metadata, schedule):
    """中文註解：比較 Whack validation 與 train robust 基線，僅輸出診斷旗標。"""
    groups = whack_groups(metadata)
    schedule_counts = Counter()
    for row in schedule:
        item = metadata[row['key']]
        if item.get('source') == 'd36_whack_real':
            schedule_counts[item['group_id']] += 1
    rows = [group_features(item, schedule_counts) for item in groups.values()]
    train_rows = [row for row in rows if row['split'] == 'train']
    validation_rows = [row for row in rows if row['split'] == 'validation']
    if not train_rows or not validation_rows:
        raise ValueError('Whack train and validation groups are both required.')
    feature_values = {
        'bpm': lambda row: row['bpm'],
        'duration_seconds': lambda row: row['duration_seconds'],
        'alignment_score': lambda row: row['alignment_score'],
        'alignment_abs_offset_seconds': lambda row: row['alignment_abs_offset_seconds'],
        'events_per_minute': lambda row: row['events_per_minute'],
    }
    feature_values.update({
        f'events_per_minute_{label}': lambda row, label=label: row['events_per_minute_by_class'][label]
        for label in LABELS
    })
    baselines = {
        name: robust_baseline([getter(row) for row in train_rows])
        for name, getter in feature_values.items()
    }
    validation_report = []
    for row in sorted(validation_rows, key=lambda value: value['group_id']):
        z_scores = {name: robust_z(getter(row), baselines[name]) for name, getter in feature_values.items()}
        flags = [name for name, score in z_scores.items() if abs(score) >= ROBUST_Z_LIMIT]
        validation_report.append({**row, 'robust_z': z_scores, 'outlier_features': flags})
    return {
        'phase': 'D41', 'status': 'audit_complete_not_training_ready',
        'method': 'metadata_robust_train_validation_comparison',
        'robust_z_limit': ROBUST_Z_LIMIT,
        'whack_group_counts': {'train': len(train_rows), 'validation': len(validation_rows)},
        'group_split_leaks': 0,
        'train_feature_baselines': baselines,
        'train_schedule_windows': {'total': sum(row['schedule_windows'] for row in train_rows)},
        'validation_groups': validation_report,
        'validation_outlier_count': sum(bool(row['outlier_features']) for row in validation_report),
        'ready_for_training_candidate': False,
        'ready_for_six_class_release': False,
    }


def run_self_check():
    """中文註解：以小型 metadata 驗證離群偵測與跨 split group 拒絕。"""
    metadata = {}
    for index, split in enumerate(('train', 'train', 'validation')):
        group_id = f'group_{index}'
        metadata[group_id] = {
            'source': 'd36_whack_real', 'group_id': group_id, 'split': split,
            'duration': 60.0, 'bpm': 120.0 + 10.0 * index,
            'alignment_score': 0.4, 'alignment_offset_seconds': 0.1,
            'events': [{'inst': label, 'time': 1.0 + index} for label in LABELS],
        }
    metadata['group_2']['bpm'] = 400.0
    payload = audit_payload(metadata, [{'key': 'group_0'}, {'key': 'group_1'}])
    assert payload['whack_group_counts'] == {'train': 2, 'validation': 1}
    assert 'bpm' in payload['validation_groups'][0]['outlier_features']
    leaked = {**metadata, 'leak': {**metadata['group_0'], 'split': 'validation'}}
    try:
        whack_groups(leaked)
    except ValueError as error:
        assert 'Group split leak' in str(error)
    else:
        raise AssertionError('跨 split group 必須被拒絕')
    print('Self-check passed.')


def main():
    """中文註解：執行唯讀 D41 audit，拒絕覆寫既有報告。"""
    parser = argparse.ArgumentParser(description='Audit D36 Whack train/validation metadata drift.')
    parser.add_argument('--meta', default='mixed_d36/metadata_d36.json')
    parser.add_argument('--schedule', default='validation_runs/d38_mixed_real_first_full_model/train_schedule.json')
    parser.add_argument('--output', default='whack_studio_metal_d41/audit_d41.json')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {output_path}')
    metadata = json.loads(Path(args.meta).read_text(encoding='utf-8'))
    schedule = json.loads(Path(args.schedule).read_text(encoding='utf-8'))
    payload = audit_payload(metadata, schedule)
    output_path.parent.mkdir(parents=True, exist_ok=False)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Wrote {output_path}; validation outliers {payload['validation_outlier_count']}/{len(payload['validation_groups'])}.")


if __name__ == '__main__':
    main()
