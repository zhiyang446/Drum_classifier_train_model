"""D66：唯讀驗證 D65 密集局部 offset 能否在記憶體中降低對齊殘差。"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from align_whack_metal_d29 import (
    HOP_LENGTH,
    SAMPLE_RATE,
    local_offset_profile,
    midi_tick_events,
    onset_envelope,
)
from build_whack_metal_meta_d28 import timed_events


RESIDUAL_LIMIT_SECONDS = 0.25


def load_json(path):
    """以 UTF-8 唯讀載入既有 audit，不修改任何輸入。"""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def profile_points(profile):
    """驗證並抽取 D65 的時間／offset 點，防止不完整 profile 被誤用。"""
    if not profile or len(profile) != 11:
        raise ValueError('D66 requires the complete 11-point D65 profile.')
    points = [(float(row['center_seconds']), float(row['offset_seconds'])) for row in profile]
    if any(not math.isfinite(value) for point in points for value in point):
        raise ValueError('D66 profile values must be finite.')
    if any(right[0] <= left[0] for left, right in zip(points, points[1:])):
        raise ValueError('D66 profile centers must be strictly increasing.')
    return points


def interpolated_offset(event_time, points):
    """在 D65 量測點間線性插值 offset，兩端維持最近量測值。"""
    if event_time <= points[0][0]:
        return points[0][1]
    if event_time >= points[-1][0]:
        return points[-1][1]
    for (left_time, left_offset), (right_time, right_offset) in zip(points, points[1:]):
        if left_time <= event_time <= right_time:
            ratio = (event_time - left_time) / (right_time - left_time)
            return left_offset + ratio * (right_offset - left_offset)
    raise AssertionError('D66 interpolation must cover every event time.')


def warp_events(events, duration, points):
    """只在記憶體調整 event 時間，檢查邊界與時間順序後回傳新事件。"""
    warped = []
    previous_time = -math.inf
    for event in events:
        adjusted_time = float(event['time']) + interpolated_offset(float(event['time']), points)
        if not 0.0 <= adjusted_time <= duration:
            return None, 'event_outside_audio'
        if adjusted_time + 1e-9 < previous_time:
            return None, 'event_time_reversal'
        warped.append({**event, 'time': adjusted_time})
        previous_time = adjusted_time
    return warped, None


def timed_event_impulses(events, frame_count):
    """把已校正秒數事件轉成 onset 格點脈衝，與既有 MIDI impulse 權重一致。"""
    frame_seconds = HOP_LENGTH / SAMPLE_RATE
    pulses = np.zeros(frame_count, dtype=np.float64)
    for event in events:
        frame = int(round(float(event['time']) / frame_seconds))
        if 0 <= frame < frame_count:
            pulses[frame] += math.sqrt(float(event['velocity']) / 127.0)
    return pulses


def analyse_target(target):
    """重建單首校正脈衝並以同一 D65 位置複測，不寫入 MIDI 或 metadata。"""
    audio_path = Path(target['audio_path'])
    midi_path = Path(target['midi_path'])
    if not audio_path.is_file() or not midi_path.is_file():
        raise FileNotFoundError(f'Missing D66 source pair: {audio_path} / {midi_path}')
    points = profile_points(target['profile'])
    fractions = [float(row['fraction']) for row in target['profile']]
    envelope = onset_envelope(audio_path)
    duration = len(envelope) * HOP_LENGTH / SAMPLE_RATE
    ticks_per_beat, raw_events, _, _, _ = midi_tick_events(midi_path)
    events = timed_events(raw_events, ticks_per_beat, float(target['bpm']))
    warped, failure = warp_events(events, duration, points)
    pre_max_abs = float(max(abs(row['offset_seconds']) for row in target['profile']))
    result = {
        'group_id': target['group_id'],
        'event_count': len(events),
        'pre_max_abs_local_offset_seconds': pre_max_abs,
        'event_or_metadata_changes': False,
    }
    if failure is not None:
        return {
            **result,
            'warp_failure': failure,
            'post_profile': None,
            'post_max_abs_local_residual_seconds': None,
            'probe_pass': False,
        }
    # ponytail: 用同一 FFT 量測點驗證插值修正；殘差不過 gate 前不建立任何資料候選。
    post_profile = local_offset_profile(
        envelope, timed_event_impulses(warped, len(envelope)), fractions,
    )
    post_residual = float(max(abs(row['offset_seconds']) for row in post_profile))
    return {
        **result,
        'warp_failure': None,
        'post_profile': post_profile,
        'post_max_abs_local_residual_seconds': post_residual,
        'probe_pass': post_residual <= RESIDUAL_LIMIT_SECONDS,
    }


def build_report(d65_audit_path, output_path):
    """建立不可覆寫的 D66 唯讀報告，永遠不標記為可訓練。"""
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f'Refusing to overwrite existing D66 output: {output_path}')
    d65 = load_json(d65_audit_path)
    if d65.get('phase') != 'D65' or d65.get('target_count') != 28:
        raise ValueError('D66 requires the completed D65 28-track audit.')
    results = [analyse_target(target) for target in d65['results']]
    post_values = [
        row['post_max_abs_local_residual_seconds'] for row in results
        if row['post_max_abs_local_residual_seconds'] is not None
    ]
    payload = {
        'phase': 'D66',
        'status': 'dense_piecewise_probe_complete_not_training_ready',
        'method': 'd65_11_point_piecewise_linear_in_memory_warp_then_same_local_fft_remeasure',
        'residual_limit_seconds': RESIDUAL_LIMIT_SECONDS,
        'target_count': len(results),
        'probe_pass_count': sum(row['probe_pass'] for row in results),
        'warp_failure_count': sum(row['warp_failure'] is not None for row in results),
        'event_or_metadata_changes': False,
        'training_started': False,
        'results': results,
        'summary': {
            'median_post_max_abs_local_residual_seconds': float(np.median(post_values)) if post_values else None,
            'maximum_post_max_abs_local_residual_seconds': float(max(post_values)) if post_values else None,
        },
        'ready_for_piecewise_metadata_candidate': False,
        'ready_for_training_candidate': False,
        'ready_for_six_class_release': False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    return payload


def run_self_check():
    """驗證插值端點／內插、順序與邊界拒絕，不需要音訊或 MIDI 檔。"""
    points = [(10.0, 1.0), (20.0, 3.0)]
    assert interpolated_offset(0.0, points) == 1.0
    assert interpolated_offset(15.0, points) == 2.0
    events = [{'time': 2.0, 'velocity': 100.0}, {'time': 4.0, 'velocity': 100.0}]
    warped, failure = warp_events(events, 20.0, points)
    assert failure is None and [row['time'] for row in warped] == [3.0, 5.0]
    rejected, failure = warp_events(events, 3.0, points)
    assert rejected is None and failure == 'event_outside_audio'


def main():
    """提供 D66 self-check 與一次性的唯讀密集插值復測入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--d65-audit', default='whack_studio_metal_d65/audit_d65_piecewise_profile_v2.json')
    parser.add_argument('--output', default='whack_studio_metal_d66/audit_d66_dense_piecewise_probe.json')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        print('D66 self-check passed.')
        return
    report = build_report(args.d65_audit, args.output)
    print(json.dumps({
        'phase': report['phase'],
        'target_count': report['target_count'],
        'probe_pass_count': report['probe_pass_count'],
        'warp_failure_count': report['warp_failure_count'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
