"""D50：建立保留 D46 原始標註的 stem-aware 兩階段候選 manifest。"""

import argparse
import copy
import json
from collections import Counter
from pathlib import Path


LABEL_TO_STEM = {
    'KD': 'kick',
    'SD': 'snare',
    'TOM': 'toms',
    'HH': 'hh',
    'RIDE': 'ride',
    'CRASH': 'crash',
}
STEM_TO_LABEL = {stem: label for label, stem in LABEL_TO_STEM.items()}
EXPECTED_STEMS = tuple(LABEL_TO_STEM.values())
REVIEW_REASON = 'event_energy_not_above_background'


def load_json(path):
    """讀取 UTF-8 JSON；D50 只讀來源資料，不覆寫 D46/D49。"""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, payload):
    """寫入全新的 JSON 產物，避免覆寫既有候選結果。"""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


def select_whack_train(metadata):
    """選取 D46 唯一的 28 首穩定 Whack train，不讀 validation/test。"""
    selected = {
        key: item for key, item in metadata.items()
        if item['source'] == 'd36_whack_real' and item['split'] == 'train'
    }
    if len(selected) != 28 or len({item['group_id'] for item in selected.values()}) != 28:
        raise AssertionError(f'Expected 28 unique D46 Whack train items, got {len(selected)}')
    return selected


def review_labels_by_key(d49_audit):
    """從 D49 的品質 reason 自動推導需要遮罩的 stem 輔助類別。"""
    if d49_audit.get('phase') != 'D49':
        raise ValueError('D49 canonical audit is unavailable.')
    labels = {}
    for row in d49_audit['results']:
        if not row.get('review_required', False):
            continue
        for reason in row.get('review_reasons', []):
            stem, separator, detail = reason.partition(':')
            if separator and detail == REVIEW_REASON and stem in STEM_TO_LABEL:
                labels.setdefault(row['key'], set()).add(STEM_TO_LABEL[stem])
    return labels


def stem_paths_for_key(key, stem_root):
    """以 metadata key 找到 D48 六 stem，拒絕缺檔或錯誤數量。"""
    output_key = key.replace(':', '_')
    stem_dir = Path(stem_root) / output_key
    paths = {stem: stem_dir / f'{stem}.wav' for stem in EXPECTED_STEMS}
    missing = [stem for stem, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'Missing D48 stems for {key}: {missing}')
    if len(list(stem_dir.glob('*.wav'))) != len(EXPECTED_STEMS):
        raise AssertionError(f'Unexpected stem count for {key}: {stem_dir}')
    return {stem: str(path.resolve()) for stem, path in paths.items()}


def ignored_events(item, ignored_labels):
    """從既有 event 複製未來 stem 輔助 loss 要忽略的 event，原 event 不會被修改。"""
    return [copy.deepcopy(event) for event in item['events'] if event['inst'] in ignored_labels]


def verify_output(source, candidate, stem_targets, ignored_by_key):
    """驗證 D46 不變欄位、validation 隔離、stem 路徑與遮罩計數。"""
    if set(source) != set(candidate) or len(candidate) != 1460:
        raise AssertionError('D50 keys must be identical to all 1,460 D46 keys.')
    validation_keys = [key for key, item in source.items() if item['split'] == 'validation']
    if len(validation_keys) != 8:
        raise AssertionError(f'Expected 8 validation items, got {len(validation_keys)}')
    for key, original in source.items():
        item = candidate[key]
        if key not in stem_targets:
            if item != original:
                raise AssertionError(f'Non-stem item changed: {key}')
            continue
        auxiliary = item.pop('drumsep_stem_auxiliary')
        if item != original:
            raise AssertionError(f'D46 fields changed for stem target: {key}')
        candidate[key]['drumsep_stem_auxiliary'] = auxiliary
        if set(auxiliary['stem_paths']) != set(EXPECTED_STEMS):
            raise AssertionError(f'Invalid stem names: {key}')
        if any(not Path(path).is_file() for path in auxiliary['stem_paths'].values()):
            raise AssertionError(f'Missing path after build: {key}')
        expected = ignored_events(original, ignored_by_key.get(key, set()))
        if auxiliary['ignored_events'] != expected:
            raise AssertionError(f'Ignored events do not match D49 review: {key}')
    if any(candidate[key] != source[key] for key in validation_keys):
        raise AssertionError('D50 validation items must remain byte-equivalent.')
    groups = {}
    for item in candidate.values():
        groups.setdefault(item['group_id'], set()).add(item['split'])
    if any(len(splits) != 1 for splits in groups.values()):
        raise AssertionError('group_id split leak detected.')


def build_candidate(metadata_path, d49_audit_path, stem_root, output_dir):
    """建立 D50 candidate metadata/audit，不訓練且不改寫 D46、D48、D49。"""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite existing output directory: {output_dir}')
    source = load_json(metadata_path)
    d49_audit = load_json(d49_audit_path)
    stem_targets = select_whack_train(source)
    ignored_by_key = review_labels_by_key(d49_audit)
    candidate = copy.deepcopy(source)
    ignored_counts = Counter()
    for key, item in stem_targets.items():
        ignored = ignored_events(item, ignored_by_key.get(key, set()))
        ignored_counts.update(event['inst'] for event in ignored)
        candidate[key]['drumsep_stem_auxiliary'] = {
            'eligible': True,
            'stem_paths': stem_paths_for_key(key, stem_root),
            'ignored_events': ignored,
            'ignored_event_reason': REVIEW_REASON if ignored else None,
        }
    if sum(ignored_counts.values()) != 2 or ignored_counts != Counter({'RIDE': 2}):
        raise AssertionError(f'D49-derived D50 mask must contain exactly two RIDE events: {ignored_counts}')
    verify_output(source, candidate, stem_targets, ignored_by_key)
    output_dir.mkdir(parents=True)
    audit = {
        'phase': 'D50',
        'status': 'stem_aware_manifest_complete_not_training_ready',
        'source_metadata': str(Path(metadata_path).resolve()),
        'd49_canonical_audit': str(Path(d49_audit_path).resolve()),
        'items': len(candidate),
        'stem_auxiliary_tracks': len(stem_targets),
        'validation_items_unchanged': 8,
        'group_split_leaks': 0,
        'ignored_event_counts': dict(ignored_counts),
        'review_derived_labels_by_key': {key: sorted(labels) for key, labels in sorted(ignored_by_key.items())},
        # ponytail: D50 只保存 manifest；架構與 loss 配方需要獨立審查後才會有訓練程式。
        'ready_for_training_candidate': False,
        'ready_for_six_class_release': False,
    }
    write_json(output_dir / 'metadata_d50.json', candidate)
    write_json(output_dir / 'audit_d50.json', audit)
    return candidate, audit


def run_self_check():
    """驗證 D49 reason 對映與只遮罩目標類別 event 的最小案例。"""
    d49 = {
        'phase': 'D49',
        'results': [{'key': 'track', 'review_required': True, 'review_reasons': ['ride:event_energy_not_above_background']}],
    }
    assert review_labels_by_key(d49) == {'track': {'RIDE'}}
    item = {'events': [{'inst': 'RIDE', 'time': 1.0}, {'inst': 'KD', 'time': 1.0}]}
    assert ignored_events(item, {'RIDE'}) == [{'inst': 'RIDE', 'time': 1.0}]
    print('D50 self-check passed.')


def main():
    """提供 D50 自檢與 manifest 建置 CLI 入口。"""
    parser = argparse.ArgumentParser(description='Build D50 stem-aware candidate manifest from D46/D49/D48.')
    parser.add_argument('--metadata', default='mixed_d46/metadata_d46.json')
    parser.add_argument('--d49-audit', default='drumsep_d49/audit_d49_reclassified.json')
    parser.add_argument('--stem-root', default='drumsep_d48/output')
    parser.add_argument('--output-dir', default='mixed_d50_stem_candidate')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    _, audit = build_candidate(args.metadata, args.d49_audit, args.stem_root, args.output_dir)
    print(f"D50 manifest complete: {audit}")


if __name__ == '__main__':
    main()
