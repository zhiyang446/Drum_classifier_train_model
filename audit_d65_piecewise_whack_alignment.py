"""D65：唯讀建立暫停 Whack train 歌曲的密集局部對齊剖面。"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from align_whack_metal_d29 import (
    LOCAL_FRACTIONS,
    local_offset_profile,
    midi_impulses,
    midi_tick_events,
    onset_envelope,
)


PROFILE_FRACTIONS = tuple(sorted({*(index / 10.0 for index in range(1, 10)), *LOCAL_FRACTIONS}))


def load_json(path):
    """以 UTF-8 讀取既有 audit，不改寫任何輸入資料。"""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def select_paused_targets(d45_audit):
    """只選 D45 暫停的 Whack train 群組，拒絕混入 stable 或評估 split。"""
    if d45_audit.get('phase') != 'D45':
        raise ValueError('D45 audit is unavailable.')
    paused_ids = {row['group_id'] for row in d45_audit['paused_groups']}
    by_group = {row['group_id']: row for row in d45_audit['results']}
    if len(paused_ids) != int(d45_audit['paused_group_count']):
        raise AssertionError('D45 paused group count is inconsistent.')
    missing = paused_ids - set(by_group)
    if missing:
        raise KeyError(f'D45 paused groups missing results: {sorted(missing)}')
    targets = [by_group[group_id] for group_id in sorted(paused_ids)]
    # D45 的 target selector 已強制 source=d36_whack_real 且 split=train，
    # 其輸出列未重複保存 split；此處只接受該受驗證的 D45 audit。
    return targets


def linear_fit(profile):
    """量化 offset 軌跡能否以單一線性漂移近似；不據此自動修正事件。"""
    if profile is None or len(profile) != len(PROFILE_FRACTIONS):
        return None
    centers = np.array([row['center_seconds'] for row in profile], dtype=np.float64)
    offsets = np.array([row['offset_seconds'] for row in profile], dtype=np.float64)
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(offsets)):
        return None
    slope, intercept = np.polyfit(centers, offsets, deg=1)
    residuals = offsets - (slope * centers + intercept)
    return {
        'slope_seconds_per_second': float(slope),
        'intercept_seconds': float(intercept),
        'rmse_seconds': float(np.sqrt(np.mean(np.square(residuals)))),
        'maximum_abs_residual_seconds': float(np.max(np.abs(residuals))),
        'offset_range_seconds': float(np.max(offsets) - np.min(offsets)),
        'maximum_adjacent_jump_seconds': float(np.max(np.abs(np.diff(offsets)))),
    }


def analyse_target(target):
    """對單一暫停歌曲產生九點 local profile；不移動 MIDI event 時間。"""
    audio_path = Path(target['audio_path'])
    midi_path = Path(target['midi_path'])
    if not audio_path.is_file() or not midi_path.is_file():
        raise FileNotFoundError(f'Missing D65 source pair: {audio_path} / {midi_path}')
    ticks_per_beat, events, _, _, _ = midi_tick_events(midi_path)
    envelope = onset_envelope(audio_path)
    pulses = midi_impulses(events, ticks_per_beat, float(target['bpm']), len(envelope))
    profile = local_offset_profile(envelope, pulses, PROFILE_FRACTIONS)
    by_fraction = {row['fraction']: row['offset_seconds'] for row in profile or []}
    expected_d45 = target['local_offsets_seconds']
    reproduced_d45 = [by_fraction[fraction] for fraction in LOCAL_FRACTIONS]
    reproduction_delta = max(
        abs(actual - expected)
        for actual, expected in zip(reproduced_d45, expected_d45)
    )
    if reproduction_delta > 1e-12:
        raise AssertionError(f'D65 profile does not reproduce D45 offsets: {target["group_id"]}')
    return {
        'group_id': target['group_id'],
        'audio_path': str(audio_path),
        'midi_path': str(midi_path),
        'bpm': float(target['bpm']),
        'd45_three_segment_drift_seconds': float(target['local_offset_drift_seconds']),
        'd45_reproduction_max_delta_seconds': float(reproduction_delta),
        'profile': profile,
        'linear_fit': linear_fit(profile),
    }


def build_report(d45_audit_path, output_path):
    """寫入全新 D65 審計報告；禁止覆寫且永遠標記為不可訓練。"""
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f'Refusing to overwrite existing D65 output: {output_path}')
    d45_audit = load_json(d45_audit_path)
    targets = select_paused_targets(d45_audit)
    results = [analyse_target(target) for target in targets]
    usable = [row['linear_fit'] for row in results if row['linear_fit'] is not None]
    payload = {
        'phase': 'D65',
        'status': 'piecewise_profile_audit_complete_not_training_ready',
        'method': 'reuse_d29_fixed_bpm_fft_correlation_with_nine_local_centers',
        'profile_fractions': list(PROFILE_FRACTIONS),
        'target_count': len(results),
        'profile_complete_count': len(usable),
        'event_or_metadata_changes': False,
        'training_started': False,
        'results': results,
        'summary': {
            'median_linear_rmse_seconds': float(np.median([row['rmse_seconds'] for row in usable])) if usable else None,
            'maximum_linear_residual_seconds': float(max(row['maximum_abs_residual_seconds'] for row in usable)) if usable else None,
        },
        'ready_for_piecewise_metadata_candidate': False,
        'ready_for_training_candidate': False,
        'ready_for_six_class_release': False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    return payload


def run_self_check():
    """驗證 paused 選取隔離與線性殘差計算，不需要音訊或 MIDI 檔。"""
    audit = {
        'phase': 'D45', 'paused_group_count': 1,
        'paused_groups': [{'group_id': 'paused_train'}],
        'results': [{'group_id': 'paused_train', 'split': 'train'}],
    }
    assert [row['group_id'] for row in select_paused_targets(audit)] == ['paused_train']
    profile = [
        {'center_seconds': float(index), 'offset_seconds': 0.1 * index}
        for index in range(1, len(PROFILE_FRACTIONS) + 1)
    ]
    result = linear_fit(profile)
    assert result is not None and math.isclose(result['rmse_seconds'], 0.0, abs_tol=1e-12)


def main():
    """提供 D65 self-check 與一次性的唯讀審計命令列入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--d45-audit', default='whack_studio_metal_d45/audit_d45.json')
    parser.add_argument('--output', default='whack_studio_metal_d65/audit_d65_piecewise_profile_v2.json')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        print('D65 self-check passed.')
        return
    report = build_report(args.d45_audit, args.output)
    print(json.dumps({
        'phase': report['phase'],
        'target_count': report['target_count'],
        'profile_complete_count': report['profile_complete_count'],
        'ready_for_training_candidate': report['ready_for_training_candidate'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
