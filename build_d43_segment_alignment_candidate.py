"""D43：把 D42 已證實漂移的 Whack validation 標註寫成獨立候選。"""

import argparse
import json
from pathlib import Path

from build_whack_metal_meta_d28 import timed_events
from align_whack_metal_d29 import midi_tick_events


SEGMENT_FRACTIONS = (0.25, 0.50, 0.75)
LOCAL_DRIFT_REVIEW_SECONDS = 0.25


def load_json(path):
    """以 UTF-8 讀取既有 JSON，不修改任何輸入。"""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def segment_offset_seconds(event_time, duration, offsets):
    """在三個局部量測點之間線性插值，兩端維持最近的 offset。"""
    if duration <= 0.0 or len(offsets) != len(SEGMENT_FRACTIONS):
        raise ValueError('D43 requires a positive duration and three local offsets.')
    points = [duration * fraction for fraction in SEGMENT_FRACTIONS]
    if event_time <= points[0]:
        return float(offsets[0])
    if event_time >= points[-1]:
        return float(offsets[-1])
    for index in range(len(points) - 1):
        left_time, right_time = points[index], points[index + 1]
        if left_time <= event_time <= right_time:
            ratio = (event_time - left_time) / (right_time - left_time)
            return float(offsets[index] + ratio * (offsets[index + 1] - offsets[index]))
    raise AssertionError('D43 segment lookup must cover every event time.')


def select_targets(metadata, d42_audit):
    """只選 D42 已旗標為局部漂移的 Whack validation 歌曲。"""
    if d42_audit.get('phase') != 'D42':
        raise ValueError('D42 audit is unavailable.')
    by_group = {item['group_id']: item for item in metadata.values()}
    targets = []
    for result in d42_audit['results']:
        if not result['review_flags']['local_drift_above_0_25_seconds']:
            continue
        item = by_group.get(result['group_id'])
        if item is None or item['split'] != 'validation' or item['source'] != 'd36_whack_real':
            raise ValueError(f'D43 target is not isolated Whack validation: {result["group_id"]}')
        offsets = result['local_offsets_seconds']
        if offsets is None or float(result['local_offset_drift_seconds']) <= LOCAL_DRIFT_REVIEW_SECONDS:
            raise ValueError(f'D43 target lacks confirmed local drift: {result["group_id"]}')
        targets.append({
            'group_id': result['group_id'],
            'local_offsets_seconds': [float(value) for value in offsets],
            'local_drift_seconds': float(result['local_offset_drift_seconds']),
        })
    if len(targets) != int(d42_audit['local_drift_review_count']):
        raise AssertionError('D43 must include every D42 local-drift review target.')
    if len({target['group_id'] for target in targets}) != len(targets):
        raise AssertionError('D43 targets must be unique by group_id.')
    return sorted(targets, key=lambda target: target['group_id'])


def rebuild_segment_events(item, offsets):
    """從原始 MIDI 重建分段位移 events，嚴格保留數量、邊界與時間順序。"""
    ticks_per_beat, raw_events, _, _, _ = midi_tick_events(item['midi_path'])
    original_events = timed_events(raw_events, ticks_per_beat, float(item['bpm']))
    rebuilt = []
    previous_time = -1.0
    duration = float(item['duration'])
    for event in original_events:
        adjusted_time = float(event['time']) + segment_offset_seconds(
            float(event['time']), duration, offsets,
        )
        if not 0.0 <= adjusted_time <= duration:
            raise ValueError(f'D43 event leaves audio boundary: {item["group_id"]}')
        if adjusted_time + 1e-9 < previous_time:
            raise ValueError(f'D43 segment mapping reverses event order: {item["group_id"]}')
        rebuilt.append({**event, 'time': adjusted_time})
        previous_time = adjusted_time
    if len(rebuilt) != len(item['events']):
        raise ValueError(f'D43 event count differs from D36: {item["group_id"]}')
    return rebuilt


def assert_group_split_isolation(metadata):
    """確認候選 metadata 沒有讓同一歌曲跨 split。"""
    split_by_group = {}
    for item in metadata.values():
        split_by_group.setdefault(item['group_id'], set()).add(item['split'])
    leaks = sorted(group for group, splits in split_by_group.items() if len(splits) > 1)
    if leaks:
        raise AssertionError(f'D43 group split leaks: {leaks[:3]}')
    return 0


def build_candidate(metadata_path, d42_audit_path, output_dir):
    """建立全新 D43 metadata/audit，拒絕覆寫 D36 或既有候選。"""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite existing D43 directory: {output_dir}')
    metadata = load_json(metadata_path)
    d42_audit = load_json(d42_audit_path)
    targets = select_targets(metadata, d42_audit)
    target_by_group = {target['group_id']: target for target in targets}
    candidate = {}
    modified = []
    for key, item in metadata.items():
        replacement = dict(item)
        target = target_by_group.get(item['group_id'])
        if target is not None:
            replacement['events'] = rebuild_segment_events(item, target['local_offsets_seconds'])
            replacement['alignment_status'] = 'candidate_d43_segment_alignment_not_training_ready'
            replacement['alignment_source_phase'] = 'D43'
            replacement['alignment_segment_fractions'] = list(SEGMENT_FRACTIONS)
            replacement['alignment_segment_offsets_seconds'] = target['local_offsets_seconds']
            replacement['alignment_segment_drift_seconds'] = target['local_drift_seconds']
            modified.append({
                'key': key,
                'group_id': item['group_id'],
                'event_count': len(replacement['events']),
                'local_drift_seconds': target['local_drift_seconds'],
            })
        candidate[key] = replacement
    if len(modified) != len(targets) or len(candidate) != len(metadata):
        raise AssertionError('D43 must preserve every D36 item and modify only its targets.')
    group_split_leaks = assert_group_split_isolation(candidate)
    payload = {
        'phase': 'D43',
        'status': 'candidate_metadata_complete_not_training_ready',
        'method': 'D42_three_segment_local_offset_piecewise_linear_from_raw_midi',
        # ponytail: 只替換有實測 drift 的五首；低 score 但沒有 drift 的歌曲維持原資料等待獨立證據。
        'modified_validation_groups': modified,
        'modified_group_count': len(modified),
        'unchanged_item_count': len(candidate) - len(modified),
        'group_split_leaks': group_split_leaks,
        'event_count_preserved': True,
        'ready_for_training_candidate': False,
        'ready_for_six_class_release': False,
    }
    output_dir.mkdir(parents=True)
    (output_dir / 'metadata_d43.json').write_text(
        json.dumps(candidate, indent=2, ensure_ascii=False), encoding='utf-8',
    )
    (output_dir / 'audit_d43.json').write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8',
    )
    return candidate, payload


def run_self_check():
    """驗證插值端點、內插值與 D42 五首選取規則。"""
    assert segment_offset_seconds(0.0, 100.0, [1.0, 2.0, 3.0]) == 1.0
    assert segment_offset_seconds(100.0, 100.0, [1.0, 2.0, 3.0]) == 3.0
    assert abs(segment_offset_seconds(37.5, 100.0, [1.0, 2.0, 3.0]) - 1.5) < 1e-12
    metadata = {
        'valid': {
            'group_id': 'validation_group', 'split': 'validation', 'source': 'd36_whack_real',
        },
    }
    audit = {
        'phase': 'D42', 'local_drift_review_count': 1,
        'results': [{
            'group_id': 'validation_group', 'local_offsets_seconds': [0.0, 1.0, 2.0],
            'local_offset_drift_seconds': 2.0,
            'review_flags': {'local_drift_above_0_25_seconds': True},
        }],
    }
    assert [target['group_id'] for target in select_targets(metadata, audit)] == ['validation_group']


def main():
    """提供 D43 self-check 與一次性的候選 metadata 建置入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', default='mixed_d36/metadata_d36.json')
    parser.add_argument('--d42-audit', default='whack_studio_metal_d42/audit_d42.json')
    parser.add_argument('--output-dir', default='mixed_d43')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        print('D43 self-check passed.')
        return
    _, audit = build_candidate(args.metadata, args.d42_audit, args.output_dir)
    print(json.dumps({
        'phase': audit['phase'],
        'modified_group_count': audit['modified_group_count'],
        'ready_for_training_candidate': audit['ready_for_training_candidate'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
