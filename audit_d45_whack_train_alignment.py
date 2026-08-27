"""D45：唯讀量測 Whack train 的局部對齊漂移，建立下一輪資料品質證據。"""

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median

from audit_d42_whack_validation_alignment import analyse_target


DRIFT_PAUSE_SECONDS = 0.25
LABELS = ('KD', 'SD', 'HH', 'TOM', 'CRASH', 'RIDE')


def load_json(path):
    """以 UTF-8 讀取既有資料，不改寫任何輸入。"""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def select_train_targets(metadata, d41_audit):
    """只挑 D36 的 Whack train 群組，保留既有對齊欄位供 D42 共用量測。"""
    if d41_audit.get('phase') != 'D41':
        raise ValueError('D41 audit is unavailable.')
    targets = []
    for item in metadata.values():
        if item.get('source') != 'd36_whack_real' or item.get('split') != 'train':
            continue
        targets.append({
            'group_id': item['group_id'],
            'audio_path': item['audio_path'],
            'midi_path': item['midi_path'],
            'bpm': float(item['bpm']),
            'metadata_alignment_score': float(item['alignment_score']),
            'metadata_alignment_offset_seconds': float(item['alignment_offset_seconds']),
            'events': item['events'],
        })
    expected = int(d41_audit['whack_group_counts']['train'])
    if len(targets) != expected or len({target['group_id'] for target in targets}) != len(targets):
        raise AssertionError(f'Expected {expected} unique Whack train groups, got {len(targets)}')
    return sorted(targets, key=lambda target: target['group_id'])


def build_report(metadata_path, d41_audit_path, output_path):
    """建立 D45 唯讀 audit，僅以已驗證 drift 門檻列出暫停候選。"""
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f'Refusing to overwrite existing D45 output: {output_path}')
    metadata = load_json(metadata_path)
    d41_audit = load_json(d41_audit_path)
    baseline = d41_audit['train_feature_baselines']
    results = [analyse_target(target, baseline) for target in select_train_targets(metadata, d41_audit)]
    stable = [row for row in results if row['local_offset_drift_seconds'] <= DRIFT_PAUSE_SECONDS]
    paused = [row for row in results if row['local_offset_drift_seconds'] > DRIFT_PAUSE_SECONDS]
    stable_events = Counter(
        event['inst'] for row in stable for event in row['events'] if event['inst'] in LABELS
    )
    payload = {
        'phase': 'D45',
        'status': 'train_alignment_audit_complete_not_training_ready',
        'method': 'reuse_d42_fixed_bpm_fft_and_three_segment_local_offsets',
        # ponytail: 僅用已驗證的 drift 門檻暫停資料；低 score 先留作診斷，避免猜測式刪除。
        'drift_pause_seconds': DRIFT_PAUSE_SECONDS,
        'target_count': len(results),
        'stable_group_count': len(stable),
        'paused_group_count': len(paused),
        'stable_group_ids': [row['group_id'] for row in stable],
        'paused_groups': [
            {
                'group_id': row['group_id'],
                'local_offset_drift_seconds': row['local_offset_drift_seconds'],
                'fixed_bpm_score': row['fixed_bpm_alignment']['score'],
            }
            for row in paused
        ],
        'stable_event_counts': {label: stable_events[label] for label in LABELS},
        'local_drift_seconds': {
            'median': float(median(row['local_offset_drift_seconds'] for row in results)),
            'maximum': float(max(row['local_offset_drift_seconds'] for row in results)),
        },
        'results': results,
        'event_or_metadata_changes': False,
        'ready_for_training_candidate': False,
        'ready_for_six_class_release': False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    return payload


def run_self_check():
    """驗證只會挑 Whack train 的唯一群組，並保留正確數量。"""
    metadata = {
        'train': {
            'group_id': 'train_group', 'source': 'd36_whack_real', 'split': 'train',
            'audio_path': 'train.wav', 'midi_path': 'train.mid', 'bpm': 120.0,
            'alignment_score': 0.8, 'alignment_offset_seconds': 0.1, 'events': [],
        },
        'validation': {
            'group_id': 'validation_group', 'source': 'd36_whack_real', 'split': 'validation',
            'audio_path': 'validation.wav', 'midi_path': 'validation.mid', 'bpm': 120.0,
            'alignment_score': 0.8, 'alignment_offset_seconds': 0.1, 'events': [],
        },
    }
    audit = {'phase': 'D41', 'whack_group_counts': {'train': 1}}
    assert [target['group_id'] for target in select_train_targets(metadata, audit)] == ['train_group']


def main():
    """提供 D45 self-check 與 56 首 Whack train 唯讀稽核入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', default='mixed_d36/metadata_d36.json')
    parser.add_argument('--d41-audit', default='whack_studio_metal_d41/audit_d41.json')
    parser.add_argument('--output', default='whack_studio_metal_d45/audit_d45.json')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        print('D45 self-check passed.')
        return
    report = build_report(args.metadata, args.d41_audit, args.output)
    print(json.dumps({
        'phase': report['phase'],
        'stable_group_count': report['stable_group_count'],
        'paused_group_count': report['paused_group_count'],
        'ready_for_training_candidate': report['ready_for_training_candidate'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
