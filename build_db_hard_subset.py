# -*- coding: utf-8 -*-
"""Build a small database-derived hard subset for KD/SD/HH raw-AI repair."""
import argparse
import csv
import json
import os


INSTS = ('KD', 'SD', 'HH')


def parse_args():
    """中文註解：解析 hard subset 建立參數。"""
    parser = argparse.ArgumentParser(description='Build database hard subset metadata.')
    parser.add_argument('--source', action='append', required=True, help='name=path metadata source.')
    parser.add_argument('--output', default='processed_data/db_hard_subset_meta.json')
    parser.add_argument('--report', default='validation_runs/raw_ai_model_fix/db_hard_subset_report.csv')
    parser.add_argument('--per-bucket', type=int, default=40)
    parser.add_argument('--per-source-bucket', type=int, default=0)
    parser.add_argument('--max-source-items', type=int, default=5000)
    parser.add_argument('--split', default='train')
    return parser.parse_args()


def parse_source(text):
    """中文註解：解析 name=path 格式。"""
    if '=' not in text:
        raise ValueError(f'Bad source: {text}')
    name, path = text.split('=', 1)
    return name.strip(), path.strip()


def load_items(source_name, path, split, max_items):
    """中文註解：讀取 metadata 並保留可訓練 events。"""
    items = []
    if os.path.getsize(path) > 250 * 1024 * 1024:
        iterator = iter_json_object(path)
    else:
        with open(path, 'r', encoding='utf-8') as f:
            iterator = json.load(f).items()
    for key, item in iterator:
        if max_items and len(items) >= max_items:
            break
        if not item.get('events'):
            continue
        item_split = item.get('split')
        if split and item_split and item_split != split:
            continue
        copied = dict(item)
        copied['_source'] = source_name
        copied['_source_key'] = key
        items.append(copied)
    return items


def iter_json_object(path, chunk_size=1024 * 1024):
    """中文註解：串流讀取大型 top-level JSON object，避免一次載入整個 E-GMD metadata。"""
    decoder = json.JSONDecoder()
    buffer = ''
    started = False
    done = False

    with open(path, 'r', encoding='utf-8') as f:
        while not done:
            chunk = f.read(chunk_size)
            if chunk:
                buffer += chunk
            elif not buffer.strip():
                break

            while True:
                buffer = buffer.lstrip()
                if not started:
                    if not buffer:
                        break
                    if buffer[0] != '{':
                        raise ValueError(f'Expected JSON object in {path}')
                    buffer = buffer[1:]
                    started = True
                    continue
                if not buffer:
                    break
                if buffer[0] == '}':
                    done = True
                    buffer = buffer[1:]
                    break
                if buffer[0] == ',':
                    buffer = buffer[1:].lstrip()
                try:
                    key, idx = decoder.raw_decode(buffer)
                    rest = buffer[idx:].lstrip()
                    if not rest.startswith(':'):
                        raise ValueError(f'Expected colon after key in {path}')
                    rest = rest[1:].lstrip()
                    value, value_idx = decoder.raw_decode(rest)
                except json.JSONDecodeError:
                    if not chunk:
                        raise
                    break
                buffer = rest[value_idx:]
                yield key, value

            if not chunk and not done:
                break


def event_stats(events):
    """中文註解：統計鼓件數、SD+HH 同時點、SD-only 點。"""
    counts = {inst: 0 for inst in INSTS}
    bins = {}
    for ev in events:
        inst = ev.get('inst')
        if inst not in counts:
            continue
        counts[inst] += 1
        bin_id = round(float(ev['time']) / 0.03)
        bins.setdefault(bin_id, {'time': float(ev['time']), 'insts': set()})['insts'].add(inst)
    sd_hh = [v['time'] for v in bins.values() if {'SD', 'HH'}.issubset(v['insts'])]
    sd_only = [v['time'] for v in bins.values() if 'SD' in v['insts'] and 'HH' not in v['insts']]
    return counts, sd_hh, sd_only


def median_time(events, inst=None):
    """中文註解：取得指定鼓件的中間時間作為切片中心。"""
    times = [float(ev['time']) for ev in events if inst is None or ev.get('inst') == inst]
    if not times:
        return 0.0
    times.sort()
    return times[len(times) // 2]


def copy_with_bucket(item, bucket, anchor_time, stats):
    """中文註解：複製 item 並標記 bucket 與 anchor。"""
    copied = dict(item)
    copied['_bucket'] = bucket
    copied['_anchor_time'] = float(anchor_time)
    copied['_hard_counts'] = stats
    return copied


def select_bucket(scored, bucket, limit):
    """中文註解：依 bucket 規則選出最相關樣本。"""
    if bucket == 'hh_dense':
        ranked = sorted(scored, key=lambda x: (x['counts']['HH'], -x['counts']['SD']), reverse=True)
    elif bucket == 'sd_hh':
        ranked = sorted(scored, key=lambda x: (len(x['sd_hh']), x['counts']['SD']), reverse=True)
    elif bucket == 'sd_only':
        ranked = sorted(scored, key=lambda x: (len(x['sd_only']), -x['counts']['HH'], x['counts']['SD']), reverse=True)
    elif bucket == 'balanced':
        ranked = sorted(scored, key=lambda x: (min(x['counts'].values()), sum(x['counts'].values())), reverse=True)
    else:
        raise ValueError(bucket)

    selected = []
    for row in ranked:
        if len(selected) >= limit:
            break
        counts = row['counts']
        if bucket == 'sd_hh' and not row['sd_hh']:
            continue
        if bucket == 'sd_only' and not row['sd_only']:
            continue
        if bucket == 'balanced' and min(counts.values()) <= 0:
            continue
        if bucket == 'hh_dense' and counts['HH'] <= 0:
            continue
        if bucket == 'sd_hh':
            anchor = row['sd_hh'][len(row['sd_hh']) // 2]
        elif bucket == 'sd_only':
            anchor = row['sd_only'][len(row['sd_only']) // 2]
        elif bucket == 'hh_dense':
            anchor = median_time(row['item']['events'], 'HH')
        else:
            anchor = median_time(row['item']['events'])
        selected.append(copy_with_bucket(row['item'], bucket, anchor, counts))
    return selected


def write_report(path, selected):
    """中文註解：寫出 subset 統計 CSV。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = ['key', 'source', 'source_key', 'bucket', 'anchor_time', 'KD', 'SD', 'HH', 'audio_path']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key, item in selected.items():
            counts = item.get('_hard_counts', {})
            writer.writerow({
                'key': key,
                'source': item.get('_source', ''),
                'source_key': item.get('_source_key', ''),
                'bucket': item.get('_bucket', ''),
                'anchor_time': item.get('_anchor_time', ''),
                'KD': counts.get('KD', 0),
                'SD': counts.get('SD', 0),
                'HH': counts.get('HH', 0),
                'audio_path': item.get('audio_path', ''),
            })


def main():
    """中文註解：主流程，從多資料庫建立 hard subset。"""
    args = parse_args()
    all_items = []
    for source_text in args.source:
        name, path = parse_source(source_text)
        all_items.extend(load_items(name, path, args.split, args.max_source_items))

    scored = []
    for item in all_items:
        counts, sd_hh, sd_only = event_stats(item['events'])
        scored.append({'item': item, 'counts': counts, 'sd_hh': sd_hh, 'sd_only': sd_only})

    selected = {}
    used_audio_bucket = set()
    if args.per_source_bucket > 0:
        source_names = sorted({row['item'].get('_source', 'src') for row in scored})
        selection_jobs = []
        for source_name in source_names:
            source_scored = [row for row in scored if row['item'].get('_source', 'src') == source_name]
            for bucket in ('hh_dense', 'sd_hh', 'sd_only', 'balanced'):
                selection_jobs.append((source_name, bucket, source_scored, args.per_source_bucket))
    else:
        selection_jobs = [('all', bucket, scored, args.per_bucket) for bucket in ('hh_dense', 'sd_hh', 'sd_only', 'balanced')]

    for source_name, bucket, bucket_scored, limit in selection_jobs:
        for item in select_bucket(bucket_scored, bucket, limit):
            key_base = f'{item.get("_source", source_name)}_{bucket}_{len(selected):04d}'
            dedupe_key = (item.get('audio_path'), bucket)
            if dedupe_key in used_audio_bucket:
                continue
            used_audio_bucket.add(dedupe_key)
            selected[key_base] = item

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)
    write_report(args.report, selected)
    print(f'Wrote {len(selected)} items: {args.output}')
    print(f'Wrote report: {args.report}')


if __name__ == '__main__':
    main()
