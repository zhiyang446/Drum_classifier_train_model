# -*- coding: utf-8 -*-
"""
Convert confirmed score-time annotations into physical audio time.
"""
import argparse
import csv
import os

from convert_user_annotations_to_meta import PHYSICAL_TIME_SOURCES, is_confirmed


INST_TO_KEY = {
    'KD': 'kick',
    'SD': 'snare',
    'HH': 'hihat',
}


def read_csv(path):
    """中文註解：讀取 CSV 為 dict rows。"""
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    """中文註解：寫出 CSV，保留欄位順序。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def notation_events_by_inst(path):
    """中文註解：從已通過的 notation_events 依樂器整理 final events。"""
    events = {inst: [] for inst in INST_TO_KEY}
    for row in read_csv(path):
        for inst, key in INST_TO_KEY.items():
            if str(row.get(f'final_{key}', '')).strip().lower() == 'true':
                events[inst].append(row)
    for inst in events:
        events[inst].sort(key=lambda row: float(row['quantized_time']))
    return events


def convert_rows(annotation_rows, notation_by_inst):
    """中文註解：保留既有物理時間；只把 score-time rows 對齊到尚未使用的 notation event。"""
    used = {inst: set() for inst in INST_TO_KEY}
    converted = []
    skipped = []

    for row in annotation_rows:
        out = dict(row)
        inst = row.get('inst', '').strip()
        if inst not in INST_TO_KEY or not is_confirmed(row.get('confirmed', '')):
            converted.append(out)
            continue

        source = row.get('source', '').strip()
        notation_events = notation_by_inst.get(inst, [])

        if source in PHYSICAL_TIME_SOURCES:
            row_time = float(row['time'])
            candidates = [
                (abs(float(event['raw_time']) - row_time), idx, event)
                for idx, event in enumerate(notation_events)
                if idx not in used[inst]
            ]
            if candidates:
                delta, idx, notation = min(candidates, key=lambda item: item[0])
                if delta <= 0.12:
                    used[inst].add(idx)
                    out['score_time'] = row.get('time', '')
                    out['conversion_status'] = 'already_physical'
                    out['notation_quantized_time'] = notation.get('quantized_time', '')
                    out['notation_beat'] = notation.get('beat', '')
                else:
                    out['score_time'] = row.get('time', '')
                    out['conversion_status'] = 'already_physical_unmatched'
            converted.append(out)
            continue

        idx = next((i for i in range(len(notation_events)) if i not in used[inst]), None)
        if idx is None:
            out['conversion_status'] = 'missing_notation_event'
            skipped.append(out)
            converted.append(out)
            continue

        notation = notation_events[idx]
        used[inst].add(idx)
        out['score_time'] = row.get('time', '')
        out['time'] = notation['raw_time']
        out['source'] = 'notation_physical_map'
        out['conversion_status'] = 'physical_time'
        out['notation_quantized_time'] = notation.get('quantized_time', '')
        out['notation_beat'] = notation.get('beat', '')
        converted.append(out)

    return converted, skipped


def fieldnames_for(rows):
    """中文註解：合併原始與轉換新增欄位。"""
    names = []
    for row in rows:
        for key in row:
            if key not in names:
                names.append(key)
    for key in ('score_time', 'conversion_status', 'notation_quantized_time', 'notation_beat'):
        if key not in names:
            names.append(key)
    return names


def convert_file(annotation_path, notation_path, output_path):
    """中文註解：轉換單一標註檔。"""
    rows = read_csv(annotation_path)
    notation_by_inst = notation_events_by_inst(notation_path)
    converted, skipped = convert_rows(rows, notation_by_inst)
    write_csv(output_path, converted, fieldnames_for(converted))
    return len(converted), len(skipped)


def run(args):
    """中文註解：批次轉換五個使用者盲測標註。"""
    summary = []
    for name in args.names:
        annotation_path = os.path.join(args.annotation_dir, f'{name}_annotations_score_confirmed.csv')
        if not os.path.exists(annotation_path):
            annotation_path = os.path.join(args.annotation_dir, f'{name}_annotations.csv')
        notation_path = os.path.join(args.notation_root, name, f'{name}_notation_events.csv')
        output_path = os.path.join(args.output_dir, f'{name}_annotations_physical.csv')
        total, skipped = convert_file(annotation_path, notation_path, output_path)
        summary.append({'name': name, 'rows': total, 'missing_notation_events': skipped, 'output': output_path})

    summary_path = os.path.join(args.output_dir, 'conversion_summary.csv')
    write_csv(summary_path, summary, ['name', 'rows', 'missing_notation_events', 'output'])
    print(f'Wrote converted annotations to {args.output_dir}')


def run_self_check():
    """中文註解：確認序號對齊會把 time 改成 notation raw_time。"""
    rows = [{'time': '1.0', 'inst': 'HH', 'confirmed': 'True', 'source': 'score_image'}]
    notation = {'KD': [], 'SD': [], 'HH': [{'raw_time': '1.25', 'quantized_time': '1.0', 'beat': '2'}]}
    converted, skipped = convert_rows(rows, notation)
    assert not skipped
    assert converted[0]['time'] == '1.25'
    assert converted[0]['score_time'] == '1.0'
    assert converted[0]['source'] == 'notation_physical_map'
    print('Self-check passed.')


def main():
    """中文註解：CLI 入口。"""
    parser = argparse.ArgumentParser(description='Convert score-time annotations to physical time.')
    parser.add_argument('--annotation-dir', default='annotations/user_blind_precise')
    parser.add_argument('--notation-root', default='validation_runs/single_checkpoint_brain_repair_blind6')
    parser.add_argument('--output-dir', default='annotations/user_blind_physical')
    parser.add_argument('--names', nargs='+', default=[
        'basic_shuffle',
        'basic_straight_16',
        'basic_straight_8',
        'ghost_snare',
        'syncopated_4_4',
    ])
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    run(args)


if __name__ == '__main__':
    main()
