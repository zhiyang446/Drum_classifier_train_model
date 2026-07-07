# -*- coding: utf-8 -*-
"""
從 STAR metadata 挑固定 hard validation 小樣本。

只使用 validation/test split，避免把 training 樣本拿來做回歸驗證。
"""
import argparse
import json
from collections import Counter, defaultdict


DEFAULT_INPUT = 'processed_data/star_meta.json'
DEFAULT_OUTPUT = 'processed_data/star_hard_validation.json'


def event_stats(events):
    """
    計算單一樣本的 KD/SD/HH 與同時敲擊統計。

    :param events: list[dict]，KD/SD/HH event list。
    :return: dict，統計結果。
    """
    counts = Counter(ev['inst'] for ev in events)
    by_time = defaultdict(set)
    for ev in events:
        by_time[round(float(ev['time']), 3)].add(ev['inst'])

    simultaneous = sum(1 for insts in by_time.values() if len(insts) >= 2)
    total = len(events)
    present = sum(1 for inst in ('KD', 'SD', 'HH') if counts[inst] > 0)

    return {
        'total': total,
        'kd': counts['KD'],
        'sd': counts['SD'],
        'hh': counts['HH'],
        'simultaneous': simultaneous,
        'present_classes': present,
        'balance': min(counts[inst] for inst in ('KD', 'SD', 'HH')) if present == 3 else 0,
    }


def ranked_pick(candidates, key_fn, limit, used):
    """
    依指定排序挑選未使用樣本。

    :param candidates: list[tuple[str, dict, dict]]，key/meta/stats。
    :param key_fn: callable，排序 key。
    :param limit: int，挑選數量。
    :param used: set，已挑選 key。
    :return: list[tuple[str, dict, dict]]，挑選結果。
    """
    picked = []
    for item in sorted(candidates, key=key_fn, reverse=True):
        key = item[0]
        if key in used:
            continue
        picked.append(item)
        used.add(key)
        if len(picked) >= limit:
            break
    return picked


def build_hard_validation(meta, per_bucket):
    """
    從 validation/test split 建立 hard validation 清單。

    :param meta: dict，STAR metadata。
    :param per_bucket: int，每個 bucket 挑選數量。
    :return: dict，hard validation metadata。
    """
    candidates = []
    for key, item in meta.items():
        if item.get('split') not in ('validation', 'test'):
            continue
        stats = event_stats(item.get('events', []))
        if stats['present_classes'] < 2:
            continue
        candidates.append((key, item, stats))

    used = set()
    buckets = {
        'snare_dense': ranked_pick(candidates, lambda x: (x[2]['sd'], x[2]['total']), per_bucket, used),
        'hihat_dense': ranked_pick(candidates, lambda x: (x[2]['hh'], x[2]['total']), per_bucket, used),
        'simultaneous': ranked_pick(candidates, lambda x: (x[2]['simultaneous'], x[2]['total']), per_bucket, used),
        'balanced': ranked_pick(candidates, lambda x: (x[2]['balance'], x[2]['total']), per_bucket, used),
    }

    output = {}
    for bucket, items in buckets.items():
        for key, item, stats in items:
            copied = dict(item)
            copied['hard_bucket'] = bucket
            copied['hard_stats'] = stats
            output[key] = copied
    return output


def run_self_check():
    """
    最小自檢，確保 ranking 不會選 training split。
    """
    meta = {
        'train_bad': {'split': 'train', 'events': [{'time': 0.0, 'inst': 'SD'}] * 99},
        'val_ok': {'split': 'validation', 'events': [
            {'time': 0.0, 'inst': 'KD'},
            {'time': 0.0, 'inst': 'SD'},
            {'time': 0.5, 'inst': 'HH'},
        ]},
    }
    out = build_hard_validation(meta, per_bucket=1)
    assert 'train_bad' not in out
    assert 'val_ok' in out
    print('Self-check passed.')


def main():
    """
    CLI 入口：挑選 STAR hard validation set。
    """
    parser = argparse.ArgumentParser(description='Select STAR hard validation samples.')
    parser.add_argument('--input', default=DEFAULT_INPUT, help='Input star_meta.json path.')
    parser.add_argument('--output', default=DEFAULT_OUTPUT, help='Output hard validation JSON path.')
    parser.add_argument('--per-bucket', type=int, default=8, help='Samples per bucket.')
    parser.add_argument('--self-check', action='store_true', help='Run self-check and exit.')
    args = parser.parse_args()

    if args.self_check:
        run_self_check()
        return

    with open(args.input, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    hard = build_hard_validation(meta, args.per_bucket)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(hard, f, indent=2)

    buckets = Counter(item['hard_bucket'] for item in hard.values())
    print(f"Wrote {len(hard)} hard validation items to {args.output}")
    print(f"Buckets: {dict(buckets)}")
    print(f"Splits: {dict(Counter(item['split'] for item in hard.values()))}")


if __name__ == '__main__':
    main()
