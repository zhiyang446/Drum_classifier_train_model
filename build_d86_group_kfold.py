# -*- coding: utf-8 -*-
"""建立 D54 train 的群組級五折交叉驗證指派，不修改原始 manifest。"""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


LABELS = ('KD', 'SD', 'HH', 'TOM', 'CRASH', 'RIDE')
DEFAULT_FOLDS = 5
DEFAULT_SEED = 86


def load_metadata(path):
    """讀取 D54 metadata，保留 item key 供輸出與隔離檢查使用。"""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def collect_groups(metadata):
    """只彙總 train item 為不可拆分的 group，並保留 validation 群組。"""
    groups = {}
    validation_groups = set()
    for item_key, item in metadata.items():
        split = item.get('split')
        group_id = item.get('group_id')
        if not group_id:
            raise ValueError(f'Missing group_id: {item_key}')
        if split == 'validation':
            validation_groups.add(group_id)
            continue
        if split != 'train':
            continue
        group = groups.setdefault(group_id, {
            'items': [], 'events': Counter(), 'sources': Counter(), 'audio_paths': set(),
        })
        group['items'].append((item_key, item))
        group['sources'][item.get('source', 'unknown')] += 1
        audio_path = item.get('audio_path')
        if not audio_path:
            raise ValueError(f'Missing audio_path: {item_key}')
        group['audio_paths'].add(audio_path)
        group['events'].update(
            event['inst'] for event in item.get('events', []) if event.get('inst') in LABELS
        )
    if not groups:
        raise ValueError('No train groups found')
    return groups, validation_groups


def assignment_cost(fold_stats, target_events, target_items, target_sources):
    """以六類事件為主、item 與來源為輔，量測目前五折偏離程度。"""
    cost = 0.0
    for stats in fold_stats:
        for label in LABELS:
            target = target_events[label]
            cost += (stats['events'][label] - target) ** 2 / max(target, 1.0)
        cost += 0.10 * (stats['items'] - target_items) ** 2 / max(target_items, 1.0)
        for source, target in target_sources.items():
            cost += 0.05 * (stats['sources'][source] - target) ** 2 / max(target, 1.0)
    return cost


def assign_folds(groups, folds, seed):
    """以固定種子與貪婪全域成本，將每個 group 恰好分派到一個 fold。"""
    if len(groups) < folds:
        raise ValueError(f'Need at least {folds} groups, got {len(groups)}')
    total_events = Counter()
    total_sources = Counter()
    total_items = 0
    for group in groups.values():
        total_events.update(group['events'])
        total_sources.update(group['sources'])
        total_items += len(group['items'])
    target_events = {label: total_events[label] / folds for label in LABELS}
    target_sources = {source: count / folds for source, count in total_sources.items()}
    target_items = total_items / folds
    fold_stats = [
        {'events': Counter(), 'sources': Counter(), 'items': 0, 'groups': []}
        for _ in range(folds)
    ]
    rng = random.Random(seed)
    tie_break = {group_id: rng.random() for group_id in groups}
    ordered_groups = sorted(
        groups,
        key=lambda group_id: (-sum(groups[group_id]['events'].values()), -len(groups[group_id]['items']), tie_break[group_id], group_id),
    )
    assignments = {}
    for group_id in ordered_groups:
        group = groups[group_id]
        candidates = []
        for fold_index, stats in enumerate(fold_stats):
            stats['events'].update(group['events'])
            stats['sources'].update(group['sources'])
            stats['items'] += len(group['items'])
            score = assignment_cost(fold_stats, target_events, target_items, target_sources)
            candidates.append((score, len(stats['groups']), fold_index))
            stats['events'].subtract(group['events'])
            stats['sources'].subtract(group['sources'])
            stats['items'] -= len(group['items'])
        _, _, selected_fold = min(candidates)
        selected = fold_stats[selected_fold]
        selected['events'].update(group['events'])
        selected['sources'].update(group['sources'])
        selected['items'] += len(group['items'])
        selected['groups'].append(group_id)
        assignments[group_id] = selected_fold
    return assignments, fold_stats


def build_audit(groups, validation_groups, assignments, fold_stats, folds, seed):
    """驗證群組、音訊路徑與舊 validation 的嚴格隔離，建立可追溯摘要。"""
    paths_to_folds = defaultdict(set)
    for group_id, group in groups.items():
        for audio_path in group['audio_paths']:
            paths_to_folds[audio_path].add(assignments[group_id])
    path_leaks = sorted(path for path, assigned in paths_to_folds.items() if len(assigned) > 1)
    validation_overlap = sorted(set(groups) & validation_groups)
    fold_missing = {
        str(index): [label for label in LABELS if fold_stats[index]['events'][label] == 0]
        for index in range(folds)
    }
    # assignments 是 group_id→單一 fold 的 dict；鍵集合相同即代表每個群組恰好一次。
    all_groups_assigned_once = len(assignments) == len(groups) and set(assignments) == set(groups)
    passed = all_groups_assigned_once and not path_leaks and not validation_overlap and not any(fold_missing.values())
    return {
        'phase': 'D86',
        'status': 'pass' if passed else 'rejected',
        'research_only': True,
        'ready_for_training_candidate': False,
        'ready_for_six_class_release': False,
        'folds': folds,
        'seed': seed,
        'train_groups': len(groups),
        'validation_groups': len(validation_groups),
        'all_train_groups_assigned_once': all_groups_assigned_once,
        'validation_group_overlap_count': len(validation_overlap),
        'validation_group_overlap': validation_overlap,
        'audio_path_cross_fold_leak_count': len(path_leaks),
        'audio_path_cross_fold_leaks': path_leaks,
        'missing_labels_by_fold': fold_missing,
        'folds_summary': {
            str(index): {
                'groups': len(stats['groups']),
                'items': stats['items'],
                'events': {label: stats['events'][label] for label in LABELS},
                'sources': dict(sorted(stats['sources'].items())),
            }
            for index, stats in enumerate(fold_stats)
        },
    }


def write_outputs(output_dir, groups, assignments, audit):
    """只寫入全新的 D86 目錄，拒絕覆寫既有證據。"""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite existing output directory: {output_dir}')
    output_dir.mkdir(parents=True)
    rows = []
    for group_id in sorted(groups):
        for item_key, item in sorted(groups[group_id]['items']):
            rows.append({
                'item_key': item_key,
                'fold': assignments[group_id],
                'group_id': group_id,
                'source': item.get('source', 'unknown'),
                'audio_path': item['audio_path'],
            })
    with (output_dir / 'fold_assignments.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['item_key', 'fold', 'group_id', 'source', 'audio_path'])
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / 'fold_summary.json').write_text(
        json.dumps(audit['folds_summary'], indent=2, ensure_ascii=False), encoding='utf-8'
    )
    (output_dir / 'audit_d86.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')


def run_self_check():
    """用合成資料驗證每群組唯一分派、五折覆蓋與舊 validation 隔離。"""
    metadata = {}
    for index in range(20):
        group_id = f'train_{index:02d}'
        metadata[f'item_{index:02d}'] = {
            'split': 'train', 'group_id': group_id, 'source': 'synthetic', 'audio_path': f'{group_id}.wav',
            'events': [{'inst': label} for label in LABELS for _ in range(index + 1)],
        }
    metadata['validation'] = {
        'split': 'validation', 'group_id': 'validation_00', 'source': 'synthetic', 'audio_path': 'validation.wav',
        'events': [{'inst': label} for label in LABELS],
    }
    groups, validation_groups = collect_groups(metadata)
    assignments, fold_stats = assign_folds(groups, folds=5, seed=DEFAULT_SEED)
    audit = build_audit(groups, validation_groups, assignments, fold_stats, folds=5, seed=DEFAULT_SEED)
    assert audit['status'] == 'pass'
    assert len(set(assignments.values())) == 5
    print('Self-check passed.')


def main():
    """CLI 入口：建立 D54 train 的群組級五折交叉驗證指派。"""
    parser = argparse.ArgumentParser(description='Build D86 group-level five-fold assignments from D54 train only.')
    parser.add_argument('--input', default='mixed_d54_stem/metadata_d54.json')
    parser.add_argument('--output-dir', default='validation_runs/d86_d54_group_kfold')
    parser.add_argument('--folds', type=int, default=DEFAULT_FOLDS)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    groups, validation_groups = collect_groups(load_metadata(args.input))
    assignments, fold_stats = assign_folds(groups, folds=args.folds, seed=args.seed)
    audit = build_audit(groups, validation_groups, assignments, fold_stats, folds=args.folds, seed=args.seed)
    if audit['status'] != 'pass':
        raise AssertionError(f'D86 isolation or coverage check failed: {audit}')
    write_outputs(args.output_dir, groups, assignments, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
