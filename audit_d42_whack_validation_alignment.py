"""D42：只讀復核 D41 指出的 Whack validation 對齊離群歌曲。"""

import argparse
import json
import math
from pathlib import Path

from align_whack_metal_d29 import (
    candidate_alignment,
    local_offsets,
    midi_impulses,
    midi_tick_events,
    onset_envelope,
)


# 只有這兩項 D41 對齊欄位可決定 D42 目標；其他資料域旗標不納入本輪。
ALIGNMENT_OUTLIER_FEATURES = {'alignment_score', 'alignment_abs_offset_seconds'}
LOCAL_DRIFT_REVIEW_SECONDS = 0.25


def load_json(path):
    """以 UTF-8 讀取既有 audit 或 metadata，避免改寫輸入資料。"""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def select_targets(metadata, d41_audit):
    """從 D41 validation 離群清單挑出唯一且隔離的 D36 Whack 目標。"""
    if d41_audit.get('phase') != 'D41':
        raise ValueError('D41 audit is unavailable.')
    by_group = {item['group_id']: item for item in metadata.values()}
    targets = []
    for row in d41_audit['validation_groups']:
        flagged = set(row['outlier_features']) & ALIGNMENT_OUTLIER_FEATURES
        if not flagged:
            continue
        group_id = row['group_id']
        item = by_group.get(group_id)
        if item is None:
            raise KeyError(f'D41 group is absent from D36 metadata: {group_id}')
        if item['split'] != 'validation' or item['source'] != 'd36_whack_real':
            raise ValueError(f'D42 target is not isolated Whack validation: {group_id}')
        targets.append({
            'group_id': group_id,
            'audio_path': item['audio_path'],
            'midi_path': item['midi_path'],
            'bpm': float(item['bpm']),
            'metadata_alignment_score': float(item['alignment_score']),
            'metadata_alignment_offset_seconds': float(item['alignment_offset_seconds']),
            'd41_outlier_features': sorted(flagged),
        })
    group_ids = [item['group_id'] for item in targets]
    if len(group_ids) != len(set(group_ids)):
        raise AssertionError('D42 targets must be unique by group_id.')
    if len(targets) != int(d41_audit['validation_outlier_count']):
        raise AssertionError('D42 must include every D41 validation alignment outlier.')
    return sorted(targets, key=lambda item: item['group_id'])


def calculate_local_drift(offsets):
    """回傳三段 local offset 的最大差距；無有效量測時保留 None。"""
    if offsets is None:
        return None
    if len(offsets) != 3 or not all(math.isfinite(value) for value in offsets):
        raise ValueError('Expected three finite local offsets.')
    return float(max(offsets) - min(offsets))


def analyse_target(target, baseline):
    """在固定 metadata BPM 下量測全曲與三段 local 對齊，絕不搜尋或修正。"""
    audio_path = Path(target['audio_path'])
    midi_path = Path(target['midi_path'])
    if not audio_path.is_file() or not midi_path.is_file():
        raise FileNotFoundError(f'Missing D42 source pair: {audio_path} / {midi_path}')
    ticks_per_beat, events, _, _, _ = midi_tick_events(midi_path)
    envelope = onset_envelope(audio_path)
    fixed_bpm = candidate_alignment(envelope, events, ticks_per_beat, target['bpm'])
    pulses = midi_impulses(events, ticks_per_beat, target['bpm'], len(envelope))
    offsets = local_offsets(envelope, pulses)
    drift = calculate_local_drift(offsets)
    score_median = float(baseline['alignment_score']['median'])
    offset_median = float(baseline['alignment_abs_offset_seconds']['median'])
    return {
        **target,
        'fixed_bpm_alignment': fixed_bpm,
        'fixed_vs_metadata_score_delta': float(fixed_bpm['score'] - target['metadata_alignment_score']),
        'fixed_vs_metadata_offset_delta_seconds': float(
            fixed_bpm['offset_seconds'] - target['metadata_alignment_offset_seconds']
        ),
        'local_offsets_seconds': offsets,
        'local_offset_drift_seconds': drift,
        'review_flags': {
            'fixed_score_below_d41_train_median': bool(fixed_bpm['score'] < score_median),
            'fixed_abs_offset_above_d41_train_median': bool(abs(fixed_bpm['offset_seconds']) > offset_median),
            'local_drift_above_0_25_seconds': bool(
                drift is not None and drift > LOCAL_DRIFT_REVIEW_SECONDS
            ),
        },
    }


def build_report(metadata_path, d41_audit_path, output_path):
    """建立 D42 全新唯讀 JSON，拒絕覆寫既有結果或改動任何輸入。"""
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f'Refusing to overwrite existing D42 output: {output_path}')
    metadata = load_json(metadata_path)
    d41_audit = load_json(d41_audit_path)
    targets = select_targets(metadata, d41_audit)
    baseline = d41_audit['train_feature_baselines']
    results = [analyse_target(target, baseline) for target in targets]
    payload = {
        'phase': 'D42',
        'status': 'audit_complete_not_training_ready',
        'method': 'reuse_d29_fixed_bpm_fft_correlation_and_d32_three_segment_local_offsets',
        # ponytail: D42 只回答現存資料是否有局部漂移，不加入 BPM 搜尋或校正候選。
        'fixed_bpm_only': True,
        'event_or_metadata_changes': False,
        'local_drift_review_seconds': LOCAL_DRIFT_REVIEW_SECONDS,
        'd41_train_alignment_baseline': {
            'alignment_score': baseline['alignment_score'],
            'alignment_abs_offset_seconds': baseline['alignment_abs_offset_seconds'],
        },
        'target_count': len(results),
        'local_drift_review_count': sum(
            row['review_flags']['local_drift_above_0_25_seconds'] for row in results
        ),
        'results': results,
        'ready_for_training_candidate': False,
        'ready_for_six_class_release': False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    return payload


def run_self_check():
    """驗證 D42 只挑 D41 對齊離群、保持 validation 隔離並正確量測 drift。"""
    metadata = {
        'valid': {
            'group_id': 'validation_group', 'split': 'validation', 'source': 'd36_whack_real',
            'audio_path': 'audio.wav', 'midi_path': 'events.mid', 'bpm': 120.0,
            'alignment_score': 0.2, 'alignment_offset_seconds': 0.4,
        },
        'train': {
            'group_id': 'train_group', 'split': 'train', 'source': 'd36_whack_real',
            'audio_path': 'train.wav', 'midi_path': 'train.mid', 'bpm': 120.0,
            'alignment_score': 0.8, 'alignment_offset_seconds': 0.1,
        },
    }
    d41_audit = {
        'phase': 'D41', 'validation_outlier_count': 1,
        'validation_groups': [
            {'group_id': 'validation_group', 'outlier_features': ['alignment_score']},
            {'group_id': 'train_group', 'outlier_features': ['bpm']},
        ],
    }
    assert [item['group_id'] for item in select_targets(metadata, d41_audit)] == ['validation_group']
    assert calculate_local_drift([0.1, -0.1, 0.2]) == 0.30000000000000004
    assert calculate_local_drift(None) is None


def main():
    """提供 D42 self-check 與一次性的唯讀稽核命令列入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', default='mixed_d36/metadata_d36.json')
    parser.add_argument('--d41-audit', default='whack_studio_metal_d41/audit_d41.json')
    parser.add_argument('--output', default='whack_studio_metal_d42/audit_d42.json')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        print('D42 self-check passed.')
        return
    report = build_report(args.metadata, args.d41_audit, args.output)
    print(json.dumps({
        'phase': report['phase'],
        'target_count': report['target_count'],
        'local_drift_review_count': report['local_drift_review_count'],
        'ready_for_training_candidate': report['ready_for_training_candidate'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
