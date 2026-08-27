# -*- coding: utf-8 -*-
"""D84：唯讀盤點 D54 train 的 RIDE-vs-SD 共現訓練窗口。"""

import argparse
import csv
import json
import os

import soundfile as sf

from audit_d63_tom_competitor_feasibility import centered_in_audio
from run_six_class_smoke import SR, TARGET_SAMPLES


SOURCE_QUOTAS = {'d36_whack_real': 300, 'd36_archive_synthetic': 100}


def has_nearby_sd(events, anchor, tolerance):
    """中文註解：判斷 RIDE 錨點附近是否有 D83 指出的 SD 真值競爭事件。"""
    return any(
        event.get('inst') == 'SD' and abs(float(event['time']) - anchor) <= tolerance + 1e-9
        for event in events
    )


def audit_metadata(metadata, tolerance):
    """中文註解：只讀 train split，回傳可置中且 RIDE/SD 共現的候選窗口。"""
    half_window = TARGET_SAMPLES / float(SR) / 2.0
    durations, rows = {}, []
    for key, item in sorted(metadata.items()):
        if item.get('split') != 'train' or not item.get('audio_path'):
            continue
        path = item['audio_path']
        if path not in durations:
            info = sf.info(path)
            durations[path] = info.frames / float(info.samplerate)
        for event in item.get('events', []):
            anchor = float(event['time'])
            if event.get('inst') != 'RIDE' or not centered_in_audio(anchor, durations[path], half_window):
                continue
            if has_nearby_sd(item['events'], anchor, tolerance):
                rows.append({'key': key, 'source': str(item.get('source') or ''), 'group_id': str(item.get('group_id') or key), 'anchor': round(anchor, 6)})
    return rows


def source_counts(rows):
    """中文註解：只統計既有 RIDE 配額來源，避免以新資料偽造可行性。"""
    return {source: sum(row['source'] == source for row in rows) for source in SOURCE_QUOTAS}


def write_outputs(rows, output_dir, tolerance):
    """中文註解：只新建 D84 CSV/JSON 證據，拒絕覆寫既有驗證輸出。"""
    if os.path.exists(output_dir) and os.listdir(output_dir):
        raise FileExistsError(f'Output directory is not empty: {output_dir}')
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'ride_sd_candidates.csv'), 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=('key', 'source', 'group_id', 'anchor'))
        writer.writeheader()
        writer.writerows(rows)
    counts = source_counts(rows)
    eligible = all(counts[source] >= quota for source, quota in SOURCE_QUOTAS.items())
    summary = {
        'phase': 'D84', 'status': 'eligible_for_ride_only_adapter_spec' if eligible else 'stop_same_data_insufficient_ride_sd',
        'target_label': 'RIDE', 'competitor_label': 'SD', 'tolerance_seconds': tolerance,
        'centered_ride_sd_candidates': len(rows), 'source_quotas': SOURCE_QUOTAS, 'by_source': counts,
        'eligible_for_later_spec': eligible, 'validation_or_test_read': False,
        'ready_for_training_candidate': False, 'ready_for_six_class_release': False,
    }
    with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def run_self_check():
    """中文註解：確認 SD 共現端點與固定來源配額不會誤判。"""
    assert has_nearby_sd([{'inst': 'SD', 'time': 1.05}], 1.0, 0.05)
    assert not has_nearby_sd([{'inst': 'SD', 'time': 1.051}], 1.0, 0.05)
    rows = [{'source': 'd36_whack_real'}] * 300 + [{'source': 'd36_archive_synthetic'}] * 100
    assert source_counts(rows) == SOURCE_QUOTAS
    print('D84 self-check passed.')


def main():
    """中文註解：解析 D84 CLI 並執行唯讀 RIDE-vs-SD 可行性審計。"""
    parser = argparse.ArgumentParser(description='Audit train RIDE windows with nearby SD events.')
    parser.add_argument('--meta', default='mixed_d54_stem/metadata_d54.json')
    parser.add_argument('--output-dir', default='validation_runs/d84_ride_sd_competitor_feasibility')
    parser.add_argument('--tolerance', type=float, default=0.05)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    with open(args.meta, encoding='utf-8') as handle:
        metadata = json.load(handle)
    print(json.dumps(write_outputs(audit_metadata(metadata, args.tolerance), args.output_dir, args.tolerance), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
