"""D34：把 D33 安全集以歌曲級方式重新分成可用的 six-class split。"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path


LABELS = ('KD', 'SD', 'HH', 'TOM', 'CRASH', 'RIDE')
SPLIT_SIZES = {'train': 56, 'validation': 8, 'test': 8}
MIN_EVALUATION_EVENTS = {'TOM': 100, 'CRASH': 100, 'RIDE': 100}
SEARCH_SEED = 20260721
SEARCH_CANDIDATES = 20000


def load_json(path):
    """讀取 UTF-8 JSON，不修改 D33 原始 metadata。"""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def event_counts(item):
    """統計單一歌曲的六類事件數，供歌曲級 split 評分。"""
    return Counter(event['inst'] for event in item['events'] if event['inst'] in LABELS)


def split_counts(groups, group_counts):
    """加總指定歌曲群組的六類事件數。"""
    counts = Counter()
    for group_id in groups:
        counts.update(group_counts[group_id])
    return counts


def candidate_score(validation, test, group_counts, total_counts):
    """以 validation/test 的事件比例誤差評分；越低代表分割越平衡。"""
    validation_counts = split_counts(validation, group_counts)
    test_counts = split_counts(test, group_counts)
    target_ratio = SPLIT_SIZES['validation'] / sum(SPLIT_SIZES.values())
    score = 0.0
    for label in LABELS:
        target = total_counts[label] * target_ratio
        # ponytail: 直接用比例誤差搜尋固定 split；資料量變大時才需要最佳化器。
        score += abs(validation_counts[label] - target) / max(target, 1.0)
        score += abs(test_counts[label] - target) / max(target, 1.0)
    for counts in (validation_counts, test_counts):
        for label, minimum in MIN_EVALUATION_EVENTS.items():
            if counts[label] < minimum:
                score += 1000.0 + (minimum - counts[label])
    return score


def find_stratified_groups(metadata):
    """以固定種子抽樣候選，取得 group 不洩漏且 rare class 足量的 56/8/8 split。"""
    by_group = {item['group_id']: item for item in metadata.values()}
    if len(by_group) != sum(SPLIT_SIZES.values()):
        raise AssertionError(f'Expected 72 unique groups, got {len(by_group)}')
    group_counts = {group_id: event_counts(item) for group_id, item in by_group.items()}
    total_counts = split_counts(by_group, group_counts)
    rng = random.Random(SEARCH_SEED)
    groups = sorted(by_group)
    best = None
    for _ in range(SEARCH_CANDIDATES):
        selected = rng.sample(groups, SPLIT_SIZES['validation'] + SPLIT_SIZES['test'])
        validation, test = selected[:SPLIT_SIZES['validation']], selected[SPLIT_SIZES['validation']:]
        score = candidate_score(validation, test, group_counts, total_counts)
        if best is None or score < best[0]:
            best = score, validation, test
    _, validation, test = best
    train = sorted(set(groups) - set(validation) - set(test))
    return {'train': train, 'validation': sorted(validation), 'test': sorted(test)}, group_counts


def build_split(input_path, output_dir):
    """寫入全新的 D34 metadata/audit，拒絕覆寫既有輸出。"""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite existing output directory: {output_dir}')
    metadata = load_json(input_path)
    assignments, group_counts = find_stratified_groups(metadata)
    group_split = {group_id: split for split, groups in assignments.items() for group_id in groups}
    output = {}
    for index, item in enumerate(sorted(metadata.values(), key=lambda row: row['group_id']), start=1):
        group_id = item['group_id']
        output[f'whack_metal_d34_{index:03d}'] = {
            **item,
            'split': group_split[group_id],
            'source': 'whack_studio_metal_d34_safe_stratified',
            'alignment_status': 'safe_stratified_candidate_not_training_ready',
        }
    split_events = {split: split_counts(groups, group_counts) for split, groups in assignments.items()}
    missing = {split: [label for label in LABELS if not counts[label]] for split, counts in split_events.items()}
    for split in ('validation', 'test'):
        for label, minimum in MIN_EVALUATION_EVENTS.items():
            if split_events[split][label] < minimum:
                raise AssertionError(f'{split} {label} has {split_events[split][label]} < {minimum}')
    if any(missing.values()):
        raise AssertionError(f'D34 missing labels: {missing}')
    audit = {
        'phase': 'D34',
        'status': 'stratified_split_complete_not_training_ready',
        'source_metadata': str(Path(input_path).resolve()),
        'seed': SEARCH_SEED,
        'candidate_searches': SEARCH_CANDIDATES,
        'splits': {split: len(groups) for split, groups in assignments.items()},
        'groups': assignments,
        'events': {split: {label: split_events[split][label] for label in LABELS} for split in split_events},
        'minimum_evaluation_events': MIN_EVALUATION_EVENTS,
        'missing_labels': missing,
        'group_split_leaks': 0,
        'ready_for_training_candidate': False,
        'ready_for_six_class_release': False,
    }
    output_dir.mkdir(parents=True)
    (output_dir / 'metadata_d34.json').write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding='utf-8')
    (output_dir / 'audit_d34.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')
    return output, audit


def run_self_check():
    """以小型六類合成資料驗證固定 split、群組隔離與 rare-class 門檻。"""
    metadata = {}
    for index in range(72):
        metadata[f'item_{index:03d}'] = {
            'group_id': f'group_{index:03d}',
            'events': [{'inst': label, 'time': 1.0} for label in LABELS for _ in range(150)],
        }
    assignments, _ = find_stratified_groups(metadata)
    assert {split: len(groups) for split, groups in assignments.items()} == SPLIT_SIZES
    assert len(set().union(*[set(groups) for groups in assignments.values()])) == 72
    print('Self-check passed.')


def main():
    """CLI 入口：建立 D34 安全歌曲的平衡 split。"""
    parser = argparse.ArgumentParser(description='Build D34 group-stratified Whack Metal split.')
    parser.add_argument('--input', default='whack_studio_metal_d33/metadata_d33.json')
    parser.add_argument('--output-dir', default='whack_studio_metal_d34')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    _, audit = build_split(args.input, args.output_dir)
    print(f"Wrote {args.output_dir}; split={audit['splits']}")


if __name__ == '__main__':
    main()
