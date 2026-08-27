# -*- coding: utf-8 -*-
"""建立 D36 訓練專用混合 manifest；不訓練也不覆寫來源資料。"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

LABELS = ('KD', 'SD', 'HH', 'TOM', 'CRASH', 'RIDE')


def load_json(path):
    """讀取 JSON，並在來源不存在時明確失敗。"""
    with Path(path).open(encoding='utf-8') as handle:
        return json.load(handle)


def add_items(output, items, source, allowed_splits):
    """加入指定 split 的來源資料，保留來源 group_id 並避免 key 碰撞。"""
    for key, item in sorted(items.items()):
        if item.get('split') not in allowed_splits:
            continue
        new_key = f'{source}:{key}'
        if new_key in output:
            raise ValueError(f'Duplicate D36 key: {new_key}')
        copied = dict(item)
        copied['source'] = source
        output[new_key] = copied


def counts(items):
    """計算每個來源與 split 的六類事件數，供訓練配方審核。"""
    result = defaultdict(lambda: defaultdict(Counter))
    for item in items.values():
        source, split = item['source'], item['split']
        result[source][split].update(event['inst'] for event in item['events'])
    return {
        source: {split: {label: value[label] for label in LABELS} for split, value in by_split.items()}
        for source, by_split in sorted(result.items())
    }


def build(archive_meta, archive_audit, breakdown_meta, whack_meta):
    """建立 D36 manifest，Archive/Breakdown 僅加入 train，Whack 保留 train/validation。"""
    failed_hashes = {row['midi_sha256'] for row in archive_audit['render_failures']}
    referenced_hashes = {item['midi_sha256'] for item in archive_meta.values()}
    leaked_failures = sorted(failed_hashes & referenced_hashes)
    if leaked_failures:
        raise ValueError(f'Render failures unexpectedly present in metadata: {leaked_failures[:3]}')
    output = {}
    add_items(output, archive_meta, 'd36_archive_synthetic', {'train'})
    add_items(output, breakdown_meta, 'd36_breakdown_real', {'train'})
    add_items(output, whack_meta, 'd36_whack_real', {'train', 'validation'})
    groups = defaultdict(set)
    for item in output.values():
        groups[item['group_id']].add(item['split'])
    leaks = sorted(group for group, splits in groups.items() if len(splits) > 1)
    if leaks:
        raise ValueError(f'group_id crosses D36 splits: {leaks[:3]}')
    audit = {
        'phase': 'D36', 'status': 'pass', 'items': len(output),
        'splits': dict(Counter(item['split'] for item in output.values())),
        'archive_render_failures_explicitly_excluded': len(failed_hashes),
        'archive_failure_hashes_referenced': leaked_failures,
        'group_split_leaks': len(leaks), 'events_by_source_split': counts(output),
        # ponytail: 僅輸出資料就緒證據；來源配額留待使用者核准的 D37 配方決定。
        'ready_for_training_recipe_review': True,
        'ready_for_training_candidate': False,
    }
    return output, audit


def write_new(path, payload):
    """只寫全新產物，避免覆寫歷史 manifest。"""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


def run_self_check():
    """驗證失敗 hash 排除與跨 split group 防線。"""
    archive = {'a': {'split': 'train', 'group_id': 'a', 'midi_sha256': 'ok', 'events': [{'inst': 'KD'}]}}
    breakdown = {'b': {'split': 'train', 'group_id': 'b', 'events': [{'inst': 'CRASH'}]}}
    whack = {'w': {'split': 'validation', 'group_id': 'w', 'events': [{'inst': 'RIDE'}]}}
    manifest, audit = build(archive, {'render_failures': [{'midi_sha256': 'failed'}]}, breakdown, whack)
    assert len(manifest) == 3 and audit['archive_render_failures_explicitly_excluded'] == 1
    print('Self-check passed.')


def main():
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description='Build D36 training-only mixed manifest.')
    parser.add_argument('--archive-meta', default='synthetic_midi_archive_d27/metadata_d27.json')
    parser.add_argument('--archive-audit', default='synthetic_midi_archive_d27/audit_d27.json')
    parser.add_argument('--breakdown-meta', default='processed_data/breakdown_midi_meta_d25.json')
    parser.add_argument('--whack-meta', default='whack_studio_metal_d34/metadata_d34.json')
    parser.add_argument('--output-dir', default='mixed_d36')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        run_self_check(); return
    manifest, audit = build(load_json(args.archive_meta), load_json(args.archive_audit), load_json(args.breakdown_meta), load_json(args.whack_meta))
    output_dir = Path(args.output_dir)
    write_new(output_dir / 'metadata_d36.json', manifest)
    write_new(output_dir / 'audit_d36.json', audit)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
