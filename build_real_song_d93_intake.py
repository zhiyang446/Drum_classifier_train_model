# -*- coding: utf-8 -*-
"""把 D93 原始鼓 MIDI 轉為帶音訊時間校正的六類 reference event，不進行訓練。"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import mido

from run_real_audio_validation import LABELS, PITCH_TO_LABEL_IDX


VALID_SPLITS = {'train', 'validation', 'test'}
EVENT_FIELDS = ('time', 'inst', 'velocity', 'midi_pitch', 'source', 'review_required')


def read_plan(path):
    """讀取 D93 intake 計畫並確認歌曲 ID、split 與來源檔皆安全可用。"""
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding='utf-8'))
    items = payload.get('items')
    if not isinstance(items, list) or not items:
        raise ValueError('plan must contain non-empty items')
    ids, group_splits, normalized = set(), {}, []
    for item in items:
        missing = [key for key in ('id', 'audio_file', 'midi_file', 'reference_offset_sec', 'group_id', 'split') if key not in item]
        if missing:
            raise ValueError(f"plan item missing: {', '.join(missing)}")
        if item['id'] in ids or item['split'] not in VALID_SPLITS:
            raise ValueError(f"invalid or duplicate item: {item['id']}")
        ids.add(item['id'])
        group_splits.setdefault(item['group_id'], set()).add(item['split'])
        audio = (path.parent / item['audio_file']).resolve()
        midi = (path.parent / item['midi_file']).resolve()
        if not audio.is_file() or not midi.is_file():
            raise FileNotFoundError(f"missing paired source for {item['id']}")
        normalized.append({**item, 'audio': audio, 'midi': midi, 'reference_offset_sec': float(item['reference_offset_sec'])})
    if set(item['split'] for item in normalized) != VALID_SPLITS:
        raise ValueError('D93 plan must retain train, validation, and test candidates')
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise ValueError('group_id crosses splits')
    return payload, normalized


def midi_events(item):
    """將來源 MIDI 的 note_on 依既有六類 GM 映射轉為校正後的真值事件。"""
    clock = 0.0
    events, unknown = [], Counter()
    for message in mido.MidiFile(item['midi']):
        clock += message.time
        if message.type != 'note_on' or message.velocity <= 0:
            continue
        class_index = PITCH_TO_LABEL_IDX.get(message.note)
        if class_index is None:
            unknown[message.note] += 1
            continue
        label = LABELS[class_index]
        events.append({
            'time': round(clock + item['reference_offset_sec'], 6),
            'inst': label,
            'velocity': message.velocity,
            'midi_pitch': message.note,
            'source': 'original_midi_offset_to_audio',
            'review_required': label in {'TOM', 'CRASH', 'RIDE'},
        })
    unexpected = sorted(set(unknown) - set(item.get('review_pitches', [])))
    missing_review = sorted(set(item.get('review_pitches', [])) - set(unknown))
    if unexpected or missing_review:
        raise ValueError(f"review pitch mismatch for {item['id']}: unexpected={unexpected}, missing={missing_review}")
    return events, dict(sorted(unknown.items()))


def write_csv(path, rows):
    """只寫入新的 reference event CSV，固定欄位以便後續 metadata builder 使用。"""
    with path.open('x', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build(plan_path, output_dir):
    """建立不可覆寫的 D93 manifest、逐歌事件 CSV 與一致性 audit。"""
    plan, items = read_plan(plan_path)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f'refusing to overwrite output: {output_dir}')
    events_dir = output_dir / 'reference_events'
    events_dir.mkdir(parents=True)
    manifest_items, song_audit = [], []
    total_counts = Counter()
    for item in items:
        events, unknown = midi_events(item)
        csv_path = events_dir / f"{item['id']}.csv"
        write_csv(csv_path, events)
        counts = Counter(event['inst'] for event in events)
        total_counts.update(counts)
        manifest_items.append({
            'id': item['id'],
            'audio_path': str(Path('..') / item['audio'].name),
            'reference_midi': str(Path('..') / item['midi'].name),
            'reference_events_csv': str(Path('reference_events') / csv_path.name),
            'reference_offset_sec': item['reference_offset_sec'],
            'group_id': item['group_id'],
            'split': item['split'],
            'review_pitches': item.get('review_pitches', []),
        })
        song_audit.append({
            'id': item['id'], 'split': item['split'], 'events': len(events),
            'class_counts': {label: counts[label] for label in LABELS}, 'unknown_pitches': unknown,
        })
    manifest = {'phase': plan.get('phase'), 'status': 'candidate_not_training', 'items': manifest_items}
    audit = {
        'phase': plan.get('phase'), 'status': 'pass_with_review' if any(row['unknown_pitches'] for row in song_audit) else 'pass',
        'items': len(manifest_items), 'group_split_leaks': 0, 'song_audit': song_audit,
        'total_class_counts': {label: total_counts[label] for label in LABELS},
        'ready_for_training_candidate': False, 'training_started': False,
    }
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output_dir / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return audit


def run_self_check():
    """確認六類映射、split 集合與未知音高 review 規則不會被靜默略過。"""
    assert tuple(LABELS) == ('KD', 'SD', 'HH', 'TOM', 'CRASH', 'RIDE')
    assert PITCH_TO_LABEL_IDX[35] == 0 and PITCH_TO_LABEL_IDX[51] == 5
    assert VALID_SPLITS == {'train', 'validation', 'test'}
    print('D93 intake builder self-check passed.')


def main():
    """提供 D93 的自檢或一次性不可覆寫 intake 建立入口。"""
    parser = argparse.ArgumentParser(description='Build D93 corrected real-song MIDI intake artifacts without training.')
    parser.add_argument('--plan', default='real-song/d93_intake_plan.json')
    parser.add_argument('--output-dir', default='real-song/d93_intake')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    print(json.dumps(build(args.plan, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
