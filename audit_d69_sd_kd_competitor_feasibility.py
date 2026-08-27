# -*- coding: utf-8 -*-
"""D69：唯讀盤點 D54 train 的 SD-vs-KD 共現訓練窗口。"""

import argparse
import csv
import json
import os

import soundfile as sf

from audit_d63_tom_competitor_feasibility import centered_in_audio
from run_six_class_smoke import SR, TARGET_SAMPLES


SOURCE_QUOTAS = {'d36_whack_real': 300, 'd36_archive_synthetic': 100}


def has_nearby_kd(events, anchor, tolerance):
    """中文註解：判斷 SD 錨點附近是否有 KD 真值，保留 D68 指出的主要混淆邊界。"""
    return any(
        event.get('inst') == 'KD' and abs(float(event['time']) - anchor) <= tolerance
        for event in events
    )


def audit_metadata(metadata, tolerance):
    """中文註解：只讀 train split，回傳可放入既有四秒窗口的 SD+KD 候選。"""
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
            if event.get('inst') != 'SD' or not centered_in_audio(anchor, durations[path], half_window):
                continue
            if has_nearby_kd(item['events'], anchor, tolerance):
                rows.append({
                    'key': key,
                    'source': str(item.get('source') or ''),
                    'group_id': str(item.get('group_id') or key),
                    'anchor': round(anchor, 6),
                })
    return rows


def source_counts(rows):
    """中文註解：只統計既有 D37 SD 來源，避免把新來源混入既定配方。"""
    return {source: sum(row['source'] == source for row in rows) for source in SOURCE_QUOTAS}


def write_outputs(rows, output_dir, tolerance):
    """中文註解：建立新的候選 CSV/JSON，拒絕覆寫既有驗證證據。"""
    if os.path.exists(output_dir) and os.listdir(output_dir):
        raise FileExistsError(f'Output directory is not empty: {output_dir}')
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'sd_kd_candidates.csv'), 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=('key', 'source', 'group_id', 'anchor'))
        writer.writeheader()
        writer.writerows(rows)
    counts = source_counts(rows)
    # ponytail: 只檢查既有 D37 配額；要改來源配方時才另立實驗。
    eligible = all(counts[source] >= quota for source, quota in SOURCE_QUOTAS.items())
    summary = {
        'phase': 'D69',
        'status': 'complete_read_only_feasibility_audit',
        'target_label': 'SD',
        'competitor_label': 'KD',
        'tolerance_seconds': tolerance,
        'centered_sd_kd_candidates': len(rows),
        'source_quotas': SOURCE_QUOTAS,
        'by_source': counts,
        'eligible_for_d70_recipe': eligible,
        'validation_or_test_read': False,
        'ready_for_training_candidate': False,
    }
    with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def run_self_check():
    """中文註解：確認 KD 共現邊界與來源配額判定。"""
    assert has_nearby_kd([{'inst': 'KD', 'time': 1.049}], 1.0, 0.05)
    assert not has_nearby_kd([{'inst': 'KD', 'time': 1.051}], 1.0, 0.05)
    rows = [{'source': 'd36_whack_real'}] * 300 + [{'source': 'd36_archive_synthetic'}] * 100
    assert source_counts(rows) == SOURCE_QUOTAS
    print('Self-check passed.')


def main():
    """中文註解：解析 D69 CLI 並執行只讀可行性審計。"""
    parser = argparse.ArgumentParser(description='Audit train SD windows with nearby KD events.')
    parser.add_argument('--meta', default='mixed_d54_stem/metadata_d54.json')
    parser.add_argument('--output-dir', default='validation_runs/d69_sd_kd_competitor_feasibility')
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
