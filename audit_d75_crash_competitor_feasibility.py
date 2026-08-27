# -*- coding: utf-8 -*-
"""D75：盤點 D74 主導混淆類別的 CRASH 共現訓練窗口。"""

import argparse
import csv
import json
import os

import soundfile as sf

from audit_d63_tom_competitor_feasibility import centered_in_audio
from run_six_class_smoke import SR, TARGET_SAMPLES


SOURCE_QUOTAS = {'d36_whack_real': 260, 'd36_archive_synthetic': 80, 'd36_breakdown_real': 60}


def nearby_competitor(events, anchor, competitor, tolerance):
    """中文註解：確認 CRASH 錨點附近有 D74 指定的競爭真值類別。"""
    # ponytail: 僅補 IEEE 浮點端點誤差；語義仍是規格固定的 .05 秒含端點。
    return any(event.get('inst') == competitor and abs(float(event['time']) - anchor) <= tolerance + 1e-9 for event in events)


def audit_metadata(metadata, competitor, tolerance):
    """中文註解：只讀 D54 train，回傳可置中且符合既有來源配額的 CRASH 共現候選。"""
    half_window, durations, rows = TARGET_SAMPLES / float(SR) / 2.0, {}, []
    for key, item in sorted(metadata.items()):
        if item.get('split') != 'train' or not item.get('audio_path'):
            continue
        path = item['audio_path']
        if path not in durations:
            info = sf.info(path)
            durations[path] = info.frames / float(info.samplerate)
        for event in item.get('events', []):
            anchor = float(event['time'])
            if event.get('inst') != 'CRASH' or not centered_in_audio(anchor, durations[path], half_window):
                continue
            if nearby_competitor(item['events'], anchor, competitor, tolerance):
                rows.append({'key': key, 'source': str(item.get('source') or ''), 'group_id': str(item.get('group_id') or key), 'anchor': round(anchor, 6)})
    return rows


def source_counts(rows):
    """中文註解：只統計既有 D37 CRASH 來源，避免以新來源偽造可行性。"""
    return {source: sum(row['source'] == source for row in rows) for source in SOURCE_QUOTAS}


def write_outputs(rows, summary, output_dir):
    """中文註解：建立新的可行性 CSV/JSON，拒絕覆寫任何舊證據。"""
    if os.path.exists(output_dir) and os.listdir(output_dir):
        raise FileExistsError(f'Output directory is not empty: {output_dir}')
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'crash_competitor_candidates.csv'), 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=('key', 'source', 'group_id', 'anchor'))
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def run(args):
    """中文註解：依 D74 結論執行 CRASH 共現資料可行性盤點，絕不啟動訓練。"""
    d74 = json.load(open(args.d74_summary, encoding='utf-8'))
    competitor = d74.get('dominant_competitor')
    if not competitor:
        summary = {'phase': 'D75', 'status': 'route_rejected_no_dominant_competitor', 'd74_summary': args.d74_summary, 'eligible_for_later_recipe': False, 'ready_for_six_class_release': False}
        write_outputs([], summary, args.output_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    metadata = json.load(open(args.meta, encoding='utf-8'))
    rows = audit_metadata(metadata, competitor, args.tolerance)
    counts = source_counts(rows)
    eligible = all(counts[source] >= quota for source, quota in SOURCE_QUOTAS.items())
    summary = {
        'phase': 'D75', 'status': 'eligible_for_later_recipe' if eligible else 'route_rejected_insufficient_data',
        'competitor_label': competitor, 'tolerance_seconds': args.tolerance, 'centered_crash_competitor_candidates': len(rows),
        'source_quotas': SOURCE_QUOTAS, 'by_source': counts, 'eligible_for_later_recipe': eligible,
        'validation_or_test_read': False, 'ready_for_six_class_release': False,
    }
    write_outputs(rows, summary, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def run_self_check():
    """中文註解：確認共現邊界與既有 CRASH 來源配額判定。"""
    assert nearby_competitor([{'inst': 'HH', 'time': 1.05}], 1.0, 'HH', 0.05)
    assert not nearby_competitor([{'inst': 'HH', 'time': 1.051}], 1.0, 'HH', 0.05)
    rows = []
    for source, quota in SOURCE_QUOTAS.items():
        rows.extend([{'source': source}] * quota)
    assert source_counts(rows) == SOURCE_QUOTAS
    print('Self-check passed.')


def main():
    """中文註解：解析 D75 CLI 並執行 CRASH 共現候選盤點或自檢。"""
    parser = argparse.ArgumentParser(description='Audit train CRASH windows with D74 dominant competitor events.')
    parser.add_argument('--meta', default='mixed_d54_stem/metadata_d54.json')
    parser.add_argument('--d74-summary', default='validation_runs/d74_d67_crash_miss_audit/summary.json')
    parser.add_argument('--output-dir', default='validation_runs/d75_crash_competitor_feasibility')
    parser.add_argument('--tolerance', type=float, default=0.05)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    run(args)


if __name__ == '__main__':
    main()
