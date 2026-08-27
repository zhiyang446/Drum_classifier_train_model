# -*- coding: utf-8 -*-
"""D63：唯讀盤點 D54 train 的 TOM-vs-KD/SD 共現訓練窗口。"""
import argparse
import csv
import json
import os

import soundfile as sf

from run_six_class_smoke import SR, TARGET_SAMPLES


def centered_in_audio(anchor, duration, half_window):
    """確認 TOM 錨點能置於既有四秒窗口中央，避免候選與 trainer 行為不一致。"""
    return half_window <= anchor <= duration - half_window


def competitor_flags(events, anchor, tolerance):
    """標記 TOM 事件附近是否有 KD 或 SD 真值，保留多標籤共現情境。"""
    labels = {
        event.get('inst') for event in events
        if abs(float(event['time']) - anchor) <= tolerance
    }
    return 'KD' in labels, 'SD' in labels


def audit_metadata(metadata, tolerance):
    """只讀 train split，回傳可重現的居中 TOM 候選與來源統計。"""
    half_window = TARGET_SAMPLES / float(SR) / 2.0
    durations = {}
    rows = []
    for key, item in sorted(metadata.items()):
        if item.get('split') != 'train' or not item.get('audio_path'):
            continue
        path = item['audio_path']
        if path not in durations:
            info = sf.info(path)
            durations[path] = info.frames / float(info.samplerate)
        for event in item.get('events', []):
            if event.get('inst') != 'TOM':
                continue
            anchor = float(event['time'])
            if not centered_in_audio(anchor, durations[path], half_window):
                continue
            has_kd, has_sd = competitor_flags(item['events'], anchor, tolerance)
            rows.append({
                'key': key,
                'source': str(item.get('source') or ''),
                'group_id': str(item.get('group_id') or key),
                'anchor': round(anchor, 6),
                'has_kd': has_kd,
                'has_sd': has_sd,
                'has_kd_or_sd': has_kd or has_sd,
            })
    return rows


def source_counts(rows):
    """依來源彙總 TOM、TOM+KD、TOM+SD 與任一競爭類別候選數。"""
    counts = {}
    for row in rows:
        summary = counts.setdefault(row['source'], {'total': 0, 'with_kd': 0, 'with_sd': 0, 'with_kd_or_sd': 0})
        summary['total'] += 1
        summary['with_kd'] += int(row['has_kd'])
        summary['with_sd'] += int(row['has_sd'])
        summary['with_kd_or_sd'] += int(row['has_kd_or_sd'])
    return dict(sorted(counts.items()))


def write_outputs(rows, output_dir, tolerance):
    """寫出新的審計證據，拒絕覆寫既有輸出目錄。"""
    if os.path.exists(output_dir) and os.listdir(output_dir):
        raise FileExistsError(f'Output directory is not empty: {output_dir}')
    os.makedirs(output_dir, exist_ok=True)
    fields = ['key', 'source', 'group_id', 'anchor', 'has_kd', 'has_sd', 'has_kd_or_sd']
    with open(os.path.join(output_dir, 'tom_candidates.csv'), 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        'phase': 'D63', 'status': 'complete_read_only_feasibility_audit',
        'tolerance_seconds': tolerance, 'centered_tom_candidates': len(rows),
        'with_kd_or_sd': sum(row['has_kd_or_sd'] for row in rows),
        'by_source': source_counts(rows),
    }
    with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def run_self_check():
    """驗證共現邊界與來源統計不會把 TOM 自身誤列為競爭類別。"""
    events = [{'inst': 'TOM', 'time': 1.0}, {'inst': 'KD', 'time': 1.051}, {'inst': 'SD', 'time': 2.0}]
    assert competitor_flags(events, 1.0, 0.05) == (False, False)
    assert competitor_flags(events, 1.0, 0.052) == (True, False)
    rows = [{'source': 'a', 'has_kd': True, 'has_sd': False, 'has_kd_or_sd': True}]
    assert source_counts(rows)['a'] == {'total': 1, 'with_kd': 1, 'with_sd': 0, 'with_kd_or_sd': 1}
    print('Self-check passed.')


def main():
    """解析 D63 唯讀審計參數。"""
    parser = argparse.ArgumentParser(description='Audit train TOM windows with nearby KD/SD events.')
    parser.add_argument('--meta')
    parser.add_argument('--output-dir')
    parser.add_argument('--tolerance', type=float, default=0.05)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    if not args.meta or not args.output_dir:
        parser.error('--meta and --output-dir are required unless --self-check is used')
    with open(args.meta, encoding='utf-8') as handle:
        metadata = json.load(handle)
    summary = write_outputs(audit_metadata(metadata, args.tolerance), args.output_dir, args.tolerance)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
