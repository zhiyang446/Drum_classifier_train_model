# -*- coding: utf-8 -*-
"""
Mine raw-AI false positives and missed confirmed annotations.
"""
import argparse
import csv
import os


INSTS = [
    ('KD', 'kick', 'prob_kick', 'thresh_kick'),
    ('SD', 'snare', 'prob_snare', 'thresh_snare'),
    ('HH', 'hihat', 'prob_hihat', 'thresh_hihat'),
]


def is_true(value):
    """中文註解：解析 CSV 內常見布林字串。"""
    return str(value).strip().lower() in {'true', '1', 'yes', 'y'}


def read_csv(path):
    """中文註解：讀取 CSV，若檔案不存在則回傳空列表。"""
    if not os.path.exists(path):
        return []
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def pick_annotation_path(annotation_dir, name):
    """中文註解：優先使用人工確認後的 score_confirmed 標註。"""
    preferred = os.path.join(annotation_dir, f'{name}_annotations_score_confirmed.csv')
    fallback = os.path.join(annotation_dir, f'{name}_annotations.csv')
    return preferred if os.path.exists(preferred) else fallback


def load_targets(annotation_path):
    """中文註解：載入 confirmed=True 的目標音符。"""
    targets = []
    for row in read_csv(annotation_path):
        inst = row.get('inst', '').strip()
        if inst in {'KD', 'SD', 'HH'} and is_true(row.get('confirmed')):
            targets.append({
                'time': float(row['time']),
                'inst': inst,
                'velocity': row.get('velocity', ''),
            })
    return sorted(targets, key=lambda row: (row['time'], row['inst']))


def load_debug_features(debug_path):
    """中文註解：依 frame 建立 event_debug 特徵索引，補足 Raw CSV 沒帶出的頻段資訊。"""
    by_frame = {}
    for row in read_csv(debug_path):
        frames = str(row.get('frames', '')).split(';')
        for frame in frames:
            frame = frame.strip()
            if frame:
                by_frame[frame] = row
    return by_frame


def load_predictions(raw_path, debug_path):
    """中文註解：將 raw_ai_events CSV 展開成單一樂器一列。"""
    preds = []
    debug_by_frame = load_debug_features(debug_path)
    for row in read_csv(raw_path):
        first_frame = str(row.get('frames', '')).split(';')[0].strip()
        debug_row = debug_by_frame.get(first_frame, {})
        for inst, key, prob_key, thresh_key in INSTS:
            if is_true(row.get(f'final_{key}')) or is_true(row.get(f'native_{key}')):
                preds.append({
                    'time': float(row['raw_time']),
                    'inst': inst,
                    'frames': row.get('frames', ''),
                    'prob': float(row.get(prob_key) or 0.0),
                    'thresh': float(row.get(thresh_key) or 0.0),
                    'beat': row.get('beat', ''),
                    'step_16th': row.get('step_16th', ''),
                    'low_rise': debug_row.get('low_rise', ''),
                    'mid_rise': debug_row.get('mid_rise', ''),
                    'hf_energy': debug_row.get('hf_energy', ''),
                    'global_hf_energy': debug_row.get('global_hf_energy', ''),
                })
    return sorted(preds, key=lambda row: (row['time'], row['inst']))


def classify(name, targets, preds, tolerance):
    """中文註解：以時間容忍窗貪婪匹配同樂器目標與預測。"""
    matched_target_ids = set()
    matched_pred_ids = set()
    matches = []

    for pred_idx, pred in enumerate(preds):
        candidates = []
        for target_idx, target in enumerate(targets):
            if target_idx in matched_target_ids or target['inst'] != pred['inst']:
                continue
            delta = abs(pred['time'] - target['time'])
            if delta <= tolerance:
                candidates.append((delta, target_idx, target))
        if not candidates:
            continue
        delta, target_idx, target = min(candidates, key=lambda item: item[0])
        matched_target_ids.add(target_idx)
        matched_pred_ids.add(pred_idx)
        matches.append({**pred, 'name': name, 'kind': 'match', 'target_time': target['time'], 'delta': delta})

    false_positives = [
        {**pred, 'name': name, 'kind': 'false_positive', 'target_time': '', 'delta': ''}
        for pred_idx, pred in enumerate(preds)
        if pred_idx not in matched_pred_ids
    ]
    misses = [
        {
            'name': name,
            'kind': 'miss',
            'inst': target['inst'],
            'time': target['time'],
            'target_time': target['time'],
            'delta': '',
            'frames': '',
            'prob': '',
            'thresh': '',
            'beat': '',
            'step_16th': '',
            'low_rise': '',
            'mid_rise': '',
            'hf_energy': '',
            'global_hf_energy': '',
        }
        for target_idx, target in enumerate(targets)
        if target_idx not in matched_target_ids
    ]
    return matches, false_positives, misses


def write_csv(path, rows, fieldnames):
    """中文註解：輸出 CSV，空資料也保留表頭方便後續查閱。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    """中文註解：主流程，逐檔比對人工確認標註與 Raw AI 輸出。"""
    summary_rows = []
    detail_rows = []
    for name in args.names:
        annotation_path = pick_annotation_path(args.annotation_dir, name)
        raw_path = os.path.join(args.raw_root, name, f'{name}_raw_ai_events.csv')
        debug_path = os.path.join(args.raw_root, name, f'{name}_event_debug.csv')
        targets = load_targets(annotation_path)
        preds = load_predictions(raw_path, debug_path)
        matches, false_positives, misses = classify(name, targets, preds, args.tolerance)
        detail_rows.extend(matches + false_positives + misses)

        for inst, _, _, _ in INSTS:
            inst_targets = sum(1 for row in targets if row['inst'] == inst)
            inst_preds = sum(1 for row in preds if row['inst'] == inst)
            inst_matches = sum(1 for row in matches if row['inst'] == inst)
            inst_fp = sum(1 for row in false_positives if row['inst'] == inst)
            inst_miss = sum(1 for row in misses if row['inst'] == inst)
            summary_rows.append({
                'name': name,
                'inst': inst,
                'targets': inst_targets,
                'predictions': inst_preds,
                'matches': inst_matches,
                'false_positives': inst_fp,
                'misses': inst_miss,
            })

    write_csv(
        os.path.join(args.output_dir, 'raw_false_positive_details.csv'),
        detail_rows,
        [
            'name', 'kind', 'inst', 'time', 'target_time', 'delta', 'frames', 'prob', 'thresh',
            'beat', 'step_16th', 'low_rise', 'mid_rise', 'hf_energy', 'global_hf_energy',
        ],
    )
    write_csv(
        os.path.join(args.output_dir, 'raw_false_positive_summary.csv'),
        summary_rows,
        ['name', 'inst', 'targets', 'predictions', 'matches', 'false_positives', 'misses'],
    )
    print(f'Wrote mining report to {args.output_dir}')


def run_self_check():
    """中文註解：確認匹配邏輯能分辨 match、false positive 與 miss。"""
    targets = [{'time': 1.0, 'inst': 'HH'}, {'time': 2.0, 'inst': 'SD'}]
    preds = [
        {
            'time': 1.01, 'inst': 'HH', 'frames': '', 'prob': 0.9, 'thresh': 0.5,
            'beat': '', 'step_16th': '', 'low_rise': '', 'mid_rise': '',
            'hf_energy': '', 'global_hf_energy': '',
        },
        {
            'time': 3.0, 'inst': 'HH', 'frames': '', 'prob': 0.8, 'thresh': 0.5,
            'beat': '', 'step_16th': '', 'low_rise': '', 'mid_rise': '',
            'hf_energy': '', 'global_hf_energy': '',
        },
    ]
    matches, false_positives, misses = classify('demo', targets, preds, 0.03)
    assert len(matches) == 1
    assert len(false_positives) == 1
    assert len(misses) == 1
    print('Self-check passed.')


def main():
    """中文註解：CLI 入口。"""
    parser = argparse.ArgumentParser(description='Mine raw AI false positives against confirmed annotations.')
    parser.add_argument('--annotation-dir', default='annotations/user_blind_precise')
    parser.add_argument('--raw-root', default='validation_runs/single_checkpoint_brain_repair_blind6')
    parser.add_argument('--output-dir', default='validation_runs/raw_ai_model_fix/false_positive_mining_20260701')
    parser.add_argument('--tolerance', type=float, default=0.04)
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
