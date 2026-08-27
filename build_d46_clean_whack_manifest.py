"""D46：依 D45 穩定群組建立新的乾淨 Whack train manifest，不訓練。"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


LABELS = ('KD', 'SD', 'HH', 'TOM', 'CRASH', 'RIDE')


def load_json(path):
    """以 UTF-8 讀取既有 JSON，避免改寫 D36 或 D45。"""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def group_split_leaks(metadata):
    """回傳同一 group 跨 split 的清單，供 D46 fail-fast。"""
    splits = defaultdict(set)
    for item in metadata.values():
        splits[item['group_id']].add(item['split'])
    return sorted(group_id for group_id, values in splits.items() if len(values) > 1)


def event_counts(metadata):
    """統計每個來源／split 的六類 event，確認清理後仍可供配方審查。"""
    counts = defaultdict(Counter)
    for item in metadata.values():
        key = f"{item['source']}:{item['split']}"
        counts[key].update(event['inst'] for event in item['events'] if event['inst'] in LABELS)
    return {
        key: {label: count[label] for label in LABELS}
        for key, count in sorted(counts.items())
    }


def build_manifest(metadata, d45_audit):
    """只用 D45 白名單保留 Whack train，其他 D36 item 一律保持原樣。"""
    if d45_audit.get('phase') != 'D45':
        raise ValueError('D45 audit is unavailable.')
    stable_groups = set(d45_audit['stable_group_ids'])
    paused_groups = {row['group_id'] for row in d45_audit['paused_groups']}
    if len(stable_groups) != int(d45_audit['stable_group_count']) or len(paused_groups) != int(d45_audit['paused_group_count']):
        raise AssertionError('D45 group counts are inconsistent.')
    if stable_groups & paused_groups:
        raise AssertionError('D45 stable and paused groups overlap.')
    output, removed_keys = {}, []
    source_validation = {}
    for key, item in metadata.items():
        is_whack_train = item.get('source') == 'd36_whack_real' and item.get('split') == 'train'
        if is_whack_train:
            if item['group_id'] in stable_groups:
                output[key] = item
            elif item['group_id'] in paused_groups:
                removed_keys.append(key)
            else:
                raise AssertionError(f'D45 is missing Whack train group: {item["group_id"]}')
            continue
        output[key] = item
        if item.get('source') == 'd36_whack_real' and item.get('split') == 'validation':
            source_validation[key] = item
    output_validation = {
        key: item for key, item in output.items()
        if item.get('source') == 'd36_whack_real' and item.get('split') == 'validation'
    }
    if source_validation != output_validation:
        raise AssertionError('D46 must preserve D36 Whack validation exactly.')
    retained_whack_train = {
        item['group_id'] for item in output.values()
        if item.get('source') == 'd36_whack_real' and item.get('split') == 'train'
    }
    if retained_whack_train != stable_groups or len(removed_keys) != len(paused_groups):
        raise AssertionError('D46 Whack train filtering is incomplete.')
    leaks = group_split_leaks(output)
    if leaks:
        raise AssertionError(f'D46 group split leaks: {leaks[:3]}')
    counts = event_counts(output)
    train_totals = Counter()
    for key, values in counts.items():
        if key.endswith(':train'):
            train_totals.update(values)
    if any(train_totals[label] <= 0 for label in LABELS):
        raise AssertionError(f'D46 train is missing labels: {dict(train_totals)}')
    return output, {
        'removed_keys': sorted(removed_keys),
        'event_counts_by_source_split': counts,
        'train_event_counts': {label: train_totals[label] for label in LABELS},
        'group_split_leaks': 0,
    }


def build_output(metadata_path, d45_audit_path, output_dir):
    """寫入全新 D46 metadata/audit，拒絕覆寫任何既有結果。"""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite existing D46 directory: {output_dir}')
    metadata = load_json(metadata_path)
    d45_audit = load_json(d45_audit_path)
    output, checks = build_manifest(metadata, d45_audit)
    audit = {
        'phase': 'D46',
        'status': 'clean_train_manifest_complete_not_training_ready',
        'source_phase': 'D36',
        # ponytail: 只按 D45 已量測的 group 白名單過濾，其他來源與 validation 不做猜測式修改。
        'whack_train_stable_group_count': int(d45_audit['stable_group_count']),
        'whack_train_paused_group_count': int(d45_audit['paused_group_count']),
        'removed_item_count': len(checks['removed_keys']),
        'items': len(output),
        'whack_validation_unchanged': True,
        **checks,
        'ready_for_training_candidate': False,
        'ready_for_six_class_release': False,
    }
    output_dir.mkdir(parents=True)
    (output_dir / 'metadata_d46.json').write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding='utf-8',
    )
    (output_dir / 'audit_d46.json').write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8',
    )
    return output, audit


def run_self_check():
    """驗證 D45 白名單只過濾 Whack train，並完整保留 validation。"""
    metadata = {
        'stable': {'group_id': 'stable', 'source': 'd36_whack_real', 'split': 'train', 'events': [{'inst': label} for label in LABELS]},
        'paused': {'group_id': 'paused', 'source': 'd36_whack_real', 'split': 'train', 'events': [{'inst': 'KD'}]},
        'validation': {'group_id': 'validation', 'source': 'd36_whack_real', 'split': 'validation', 'events': [{'inst': 'RIDE'}]},
        'archive': {'group_id': 'archive', 'source': 'd36_archive_synthetic', 'split': 'train', 'events': [{'inst': 'KD'}]},
    }
    audit = {
        'phase': 'D45', 'stable_group_ids': ['stable'], 'stable_group_count': 1,
        'paused_groups': [{'group_id': 'paused'}], 'paused_group_count': 1,
    }
    output, checks = build_manifest(metadata, audit)
    assert set(output) == {'stable', 'validation', 'archive'}
    assert checks['removed_keys'] == ['paused']


def main():
    """提供 D46 self-check 與一次性乾淨 manifest 建置入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', default='mixed_d36/metadata_d36.json')
    parser.add_argument('--d45-audit', default='whack_studio_metal_d45/audit_d45.json')
    parser.add_argument('--output-dir', default='mixed_d46')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        print('D46 self-check passed.')
        return
    _, audit = build_output(args.metadata, args.d45_audit, args.output_dir)
    print(json.dumps({
        'phase': audit['phase'], 'items': audit['items'],
        'removed_item_count': audit['removed_item_count'],
        'ready_for_training_candidate': audit['ready_for_training_candidate'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
