# -*- coding: utf-8 -*-
"""驗證真實鼓音訊分組，並把六類 raw AI CSV 轉成可審查 pseudo-label。"""
import argparse
import csv
import json
import os
import tempfile


LABELS = ('KD', 'SD', 'HH', 'TOM', 'CRASH', 'RIDE')
RARE_LABELS = {'TOM', 'CRASH', 'RIDE'}
LABEL_COLUMNS = {
    'KD': ('prob_kick', 'final_kick'),
    'SD': ('prob_snare', 'final_snare'),
    'HH': ('prob_hihat', 'final_hihat'),
    'TOM': ('prob_tom', 'final_tom'),
    'CRASH': ('prob_crash', 'final_crash'),
    'RIDE': ('prob_ride', 'final_ride'),
}
VALID_SPLITS = {'train', 'validation', 'test'}


def as_bool(value):
    """中文註解：解析 CSV 的布林欄位，避免字串 False 被當成真值。"""
    return str(value).strip().lower() in {'1', 'true', 'yes'}


def resolve_path(base_dir, value):
    """中文註解：將 manifest 相對路徑轉為絕對路徑。"""
    return os.path.abspath(value if os.path.isabs(value) else os.path.join(base_dir, value))


def validate_manifest(items, base_dir, check_files=True):
    """中文註解：驗證必要欄位、唯一 ID 與 group_id 的 split 隔離。"""
    if not isinstance(items, list) or not items:
        raise ValueError('manifest must contain a non-empty items list')
    ids, group_splits, normalized = set(), {}, []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError('every manifest item must be an object')
        missing = [key for key in ('id', 'audio_path', 'group_id', 'split', 'raw_events_csv') if not item.get(key)]
        if missing:
            raise ValueError(f"manifest item missing: {', '.join(missing)}")
        if item['id'] in ids:
            raise ValueError(f"duplicate item id: {item['id']}")
        if item['split'] not in VALID_SPLITS:
            raise ValueError(f"invalid split for {item['id']}: {item['split']}")
        ids.add(item['id'])
        group_splits.setdefault(item['group_id'], set()).add(item['split'])
        normalized_item = dict(item)
        for key in ('audio_path', 'raw_events_csv'):
            normalized_item[key] = resolve_path(base_dir, item[key])
            if check_files and not os.path.isfile(normalized_item[key]):
                raise FileNotFoundError(f"missing {key}: {normalized_item[key]}")
        normalized.append(normalized_item)
    leaked = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    if leaked:
        raise ValueError(f'group_id crosses splits: {", ".join(leaked)}')
    return normalized


def build_events(items, minimum_confidence):
    """中文註解：保留高置信原始事件，並強制標記三個弱類別供人工審查。"""
    events = []
    for item in items:
        with open(item['raw_events_csv'], newline='', encoding='utf-8-sig') as handle:
            for row in csv.DictReader(handle):
                for label, (probability_column, trigger_column) in LABEL_COLUMNS.items():
                    confidence = float(row.get(probability_column) or 0.0)
                    if not as_bool(row.get(trigger_column)) or confidence < minimum_confidence:
                        continue
                    events.append({
                        'item_id': item['id'],
                        'group_id': item['group_id'],
                        'split': item['split'],
                        'time': float(row['raw_time']),
                        'inst': label,
                        'confidence': confidence,
                        'review_required': label in RARE_LABELS,
                    })
    return events


def write_outputs(output_json, output_csv, items, events):
    """中文註解：寫出可供後續人工稽核與 metadata builder 使用的固定格式結果。"""
    payload = {'items': items, 'events': events, 'labels': list(LABELS)}
    with open(output_json, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    fields = ('item_id', 'group_id', 'split', 'time', 'inst', 'confidence', 'review_required')
    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(events)


def run_self_check():
    """中文註解：驗證群組隔離、低置信排除與稀有鼓件人工審查旗標。"""
    with tempfile.TemporaryDirectory() as directory:
        audio_path = os.path.join(directory, 'take.wav')
        csv_path = os.path.join(directory, 'events.csv')
        open(audio_path, 'wb').close()
        with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=('raw_time', 'prob_kick', 'final_kick', 'prob_tom', 'final_tom'))
            writer.writeheader()
            writer.writerow({'raw_time': '1.0', 'prob_kick': '0.95', 'final_kick': 'True', 'prob_tom': '0.93', 'final_tom': 'True'})
            writer.writerow({'raw_time': '2.0', 'prob_kick': '0.20', 'final_kick': 'True', 'prob_tom': '0.00', 'final_tom': 'False'})
        items = validate_manifest([{'id': 'take_1', 'audio_path': audio_path, 'raw_events_csv': csv_path, 'group_id': 'song_a', 'split': 'train'}], directory)
        events = build_events(items, 0.80)
        assert [event['inst'] for event in events] == ['KD', 'TOM']
        assert events[1]['review_required'] is True
        duplicate = [dict(items[0]), {'id': 'take_2', 'audio_path': audio_path, 'raw_events_csv': csv_path, 'group_id': 'song_a', 'split': 'test'}]
        try:
            validate_manifest(duplicate, directory)
        except ValueError as error:
            assert 'crosses splits' in str(error)
        else:
            raise AssertionError('group split leakage must be rejected')
    print('Self-check passed.')


def main():
    """中文註解：執行 manifest 驗證與 pseudo-label 審查檔輸出。"""
    parser = argparse.ArgumentParser(description='Validate grouped real-drum items and export six-class pseudo-label review files.')
    parser.add_argument('--manifest')
    parser.add_argument('--output-json')
    parser.add_argument('--output-review-csv')
    parser.add_argument('--minimum-confidence', type=float, default=0.80)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    if not all((args.manifest, args.output_json, args.output_review_csv)):
        parser.error('--manifest, --output-json and --output-review-csv are required unless --self-check is used')
    if not 0.0 <= args.minimum_confidence <= 1.0:
        parser.error('--minimum-confidence must be between 0 and 1')
    with open(args.manifest, encoding='utf-8') as handle:
        payload = json.load(handle)
    items = payload.get('items') if isinstance(payload, dict) else payload
    items = validate_manifest(items, os.path.dirname(os.path.abspath(args.manifest)))
    events = build_events(items, args.minimum_confidence)
    write_outputs(args.output_json, args.output_review_csv, items, events)
    print(f'Validated {len(items)} items and exported {len(events)} pseudo-label events.')


if __name__ == '__main__':
    main()
