# -*- coding: utf-8 -*-
"""
Select train-split Snare/Hi-Hat hard examples for conservative fine-tuning.
"""
import argparse
import json
import os
from collections import Counter


def count_simultaneous_sdhh(events, window=0.03):
    """
    中文註解：以短時間窗統計 SD+HH 同時敲擊次數，避免只挑到單一鼓件密集片段。
    """
    bins = {}
    for ev in events:
        inst = ev.get('inst')
        if inst not in ('KD', 'SD', 'HH'):
            continue
        bin_id = round(float(ev.get('time', 0.0)) / window)
        bins.setdefault(bin_id, set()).add(inst)
    return sum(1 for insts in bins.values() if {'SD', 'HH'}.issubset(insts))


def score_item(item):
    """
    中文註解：給 SD/HH hard example 排序；SD/HH 是主目標，KD 作為守門保留。
    """
    counts = Counter(ev.get('inst') for ev in item.get('events', []))
    sdhh = count_simultaneous_sdhh(item.get('events', []))
    if counts['KD'] <= 0 or counts['SD'] <= 0 or counts['HH'] <= 0:
        return None
    if counts['SD'] < 4 and sdhh < 2:
        return None
    return counts['SD'] * 3.0 + counts['HH'] * 1.5 + sdhh * 6.0 + min(counts['KD'], 16) * 0.5


def select_items(meta, limit):
    """
    中文註解：只從 train split 選出分數最高的 hard examples。
    """
    scored = []
    for key, item in meta.items():
        if item.get('split') != 'train':
            continue
        score = score_item(item)
        if score is None:
            continue
        scored.append((score, key, item))
    scored.sort(key=lambda row: (-row[0], row[1]))
    return {key: item for _, key, item in scored[:limit]}


def write_selected(input_path, output_path, limit):
    """
    中文註解：讀取單一 metadata JSON 並寫出 hard-example 子集。
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    selected = select_items(meta, limit)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)
    return len(meta), len(selected)


def run_self_check():
    """
    中文註解：最小自檢，確保選擇器會保留 SD+HH+KD 訓練樣本並排除 validation。
    """
    meta = {
        'good': {'split': 'train', 'events': [
            {'time': 0.0, 'inst': 'KD'}, {'time': 0.5, 'inst': 'SD'}, {'time': 0.5, 'inst': 'HH'},
            {'time': 1.0, 'inst': 'SD'}, {'time': 1.0, 'inst': 'HH'},
            {'time': 1.5, 'inst': 'SD'}, {'time': 2.0, 'inst': 'SD'},
        ]},
        'no_kd': {'split': 'train', 'events': [{'time': 0.0, 'inst': 'SD'}, {'time': 0.0, 'inst': 'HH'}]},
        'holdout': {'split': 'test', 'events': [{'time': 0.0, 'inst': 'KD'}, {'time': 0.0, 'inst': 'SD'}, {'time': 0.0, 'inst': 'HH'}]},
    }
    selected = select_items(meta, 10)
    assert list(selected) == ['good']
    print('Self-check passed.')


def main():
    """
    中文註解：CLI 入口，為 E-GMD/STAR/local metadata 產生 SD/HH hard-example 子集。
    """
    parser = argparse.ArgumentParser(description='Select SD/HH hard-example train metadata.')
    parser.add_argument('--egmd-meta', default='processed_data/egmd_meta.json')
    parser.add_argument('--star-meta', default='processed_data/star_meta.json')
    parser.add_argument('--local-meta', default='processed_data/local_xml_meta.json')
    parser.add_argument('--output-dir', default='processed_data/sdhh_hard_examples')
    parser.add_argument('--limit', type=int, default=512)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()

    if args.self_check:
        run_self_check()
        return

    specs = [
        ('egmd', args.egmd_meta, os.path.join(args.output_dir, 'egmd_meta.json')),
        ('star', args.star_meta, os.path.join(args.output_dir, 'star_meta.json')),
        ('local', args.local_meta, os.path.join(args.output_dir, 'local_xml_meta.json')),
    ]
    for name, input_path, output_path in specs:
        total, selected = write_selected(input_path, output_path, args.limit)
        print(f'{name}: selected {selected}/{total} -> {output_path}')


if __name__ == '__main__':
    main()
