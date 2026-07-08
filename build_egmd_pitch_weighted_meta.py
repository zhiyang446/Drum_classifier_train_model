# -*- coding: utf-8 -*-
"""
Build E-GMD train metadata with preserved MIDI pitch and optional loss weights.
"""
import argparse
import json
import os
import re
from collections import Counter

import pretty_midi


PITCH_TO_INST = {
    35: 'KD',
    36: 'KD',
    37: 'SD',
    38: 'SD',
    40: 'SD',
    22: 'HH',
    26: 'HH',
    42: 'HH',
    44: 'HH',
    46: 'HH',
}


def parse_pitch_weights(text):
    """中文註解：解析 pitch=weight 規則，例如 37=2.0,38=1.4。"""
    weights = {}
    if not text.strip():
        return weights
    for part in text.split(','):
        key, value = part.split('=', 1)
        weights[int(key.strip())] = float(value.strip())
    return weights


def midi_path_for_audio(audio_path):
    """中文註解：依 E-GMD 音訊路徑推導 sibling MIDI 路徑。"""
    return os.path.splitext(audio_path)[0] + '.midi'


def groove_key_for_audio(audio_path):
    """中文註解：移除 E-GMD render 編號，取得可跨 kit 去重的 groove key。"""
    name = os.path.basename(audio_path)
    return re.sub(r'_\d+\.[^.]+$', '', name)


def read_pitch_events(audio_path, pitch_weights, velocity_min):
    """中文註解：讀取 MIDI 並保留 pitch，同時套用通用 pitch 權重。"""
    midi_path = midi_path_for_audio(audio_path)
    if not os.path.exists(midi_path):
        return []
    midi = pretty_midi.PrettyMIDI(midi_path)
    events = []
    for instrument in midi.instruments:
        for note in instrument.notes:
            inst = PITCH_TO_INST.get(note.pitch)
            if inst is None or note.velocity < velocity_min:
                continue
            events.append({
                'time': float(note.start),
                'inst': inst,
                'pitch': int(note.pitch),
                'velocity': float(note.velocity),
                'loss_weight': float(pitch_weights.get(note.pitch, 1.0)),
            })
    events.sort(key=lambda row: (row['time'], row['pitch']))
    return events


def item_counts(events):
    """中文註解：統計單一 item 的鼓件與 pitch 分布。"""
    inst_counts = Counter(ev['inst'] for ev in events)
    pitch_counts = Counter(ev['pitch'] for ev in events)
    return inst_counts, pitch_counts


def keep_item(events, required_pitches):
    """中文註解：依通用 pitch 條件篩選 train item，不依測試檔名挑選。"""
    if not events:
        return False
    if not required_pitches:
        return True
    pitches = {ev['pitch'] for ev in events}
    return bool(pitches.intersection(required_pitches))


def window_items(key, item, window_seconds):
    """中文註解：將長音檔展開成固定 anchor 視窗，增加訓練覆蓋率。"""
    if window_seconds <= 0:
        return {key: item}
    duration = float(item.get('duration', 0.0) or 0.0)
    if duration <= window_seconds:
        copied = dict(item)
        copied['_anchor_time'] = duration / 2.0 if duration > 0 else 0.0
        return {key: copied}
    output = {}
    step = window_seconds
    center = window_seconds / 2.0
    idx = 0
    while center <= duration:
        copied = dict(item)
        copied['_anchor_time'] = center
        output[f'{key}_win{idx:03d}'] = copied
        center += step
        idx += 1
    return output


def density_score(inst_counts, duration, sort_by):
    """中文註解：依通用事件密度計算排序分數，不讀取驗證檔名或答案。"""
    duration = max(float(duration or 0.0), 1e-6)
    kd_rate = inst_counts.get('KD', 0) / duration
    sd_rate = inst_counts.get('SD', 0) / duration
    hh_rate = inst_counts.get('HH', 0) / duration
    if sort_by == 'kdsd_density':
        return kd_rate + sd_rate
    if sort_by == 'sd_density':
        return sd_rate
    if sort_by == 'kd_density':
        return kd_rate
    if sort_by == 'events_density':
        return kd_rate + sd_rate + hh_rate
    return 0.0


def build_meta(source_meta, limit, pitch_weights, required_pitches, velocity_min, max_kits_per_groove, window_seconds, min_kd_per_sec, min_sd_per_sec, sort_by):
    """中文註解：從 E-GMD train split 建立 pitch-aware candidate metadata。"""
    candidates = []
    groove_counts = Counter()
    for key, item in sorted(source_meta.items()):
        if item.get('split') != 'train':
            continue
        audio_path = item.get('audio_path', '')
        if not audio_path or not os.path.exists(audio_path):
            continue
        groove_key = groove_key_for_audio(audio_path)
        if max_kits_per_groove > 0 and groove_counts[groove_key] >= max_kits_per_groove:
            continue
        events = read_pitch_events(audio_path, pitch_weights, velocity_min)
        if not keep_item(events, required_pitches):
            continue
        inst_counts, pitch_counts = item_counts(events)
        duration = float(item.get('duration', 0.0) or 0.0)
        duration_for_rate = max(duration, 1e-6)
        kd_rate = inst_counts.get('KD', 0) / duration_for_rate
        sd_rate = inst_counts.get('SD', 0) / duration_for_rate
        if kd_rate < min_kd_per_sec or sd_rate < min_sd_per_sec:
            continue
        copied = dict(item)
        copied['events'] = events
        copied['source'] = 'egmd_pitch_weighted'
        groove_counts[groove_key] += 1
        score = density_score(inst_counts, duration, sort_by)
        candidates.append({
            'key': key,
            'item': copied,
            'score': score,
            'report': {
                'key': key,
                'events': len(events),
                'KD': inst_counts.get('KD', 0),
                'SD': inst_counts.get('SD', 0),
                'HH': inst_counts.get('HH', 0),
                'KD_per_sec': f'{kd_rate:.4f}',
                'SD_per_sec': f'{sd_rate:.4f}',
                'density_score': f'{score:.4f}',
                'pitches': ' '.join(f'{pitch}:{count}' for pitch, count in sorted(pitch_counts.items())),
                'audio_path': audio_path,
            },
        })

    if sort_by != 'key':
        candidates.sort(key=lambda row: (-row['score'], row['key']))

    output = {}
    report = []
    for candidate in candidates:
        for out_key, out_item in window_items(candidate['key'], candidate['item'], window_seconds).items():
            if limit and len(output) >= limit:
                break
            output[out_key] = out_item
        report.append(candidate['report'])
        if limit and len(output) >= limit:
            break
    return output, report


def write_report(path, rows):
    """中文註解：寫出 metadata 建立報告。"""
    import csv

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = ['key', 'events', 'KD', 'SD', 'HH', 'KD_per_sec', 'SD_per_sec', 'density_score', 'pitches', 'audio_path']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    """中文註解：解析 CLI 參數。"""
    parser = argparse.ArgumentParser(description='Build pitch-weighted E-GMD train metadata.')
    parser.add_argument('--meta', default='processed_data/egmd_meta.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--limit', type=int, default=300)
    parser.add_argument('--pitch-weights', default='')
    parser.add_argument('--require-pitches', default='')
    parser.add_argument('--velocity-min', type=float, default=30.0)
    parser.add_argument('--max-kits-per-groove', type=int, default=1)
    parser.add_argument('--window-seconds', type=float, default=0.0)
    parser.add_argument('--min-kd-per-sec', type=float, default=0.0)
    parser.add_argument('--min-sd-per-sec', type=float, default=0.0)
    parser.add_argument('--sort-by', choices=['key', 'kdsd_density', 'kd_density', 'sd_density', 'events_density'], default='key')
    parser.add_argument('--self-check', action='store_true')
    return parser.parse_args()


def run_self_check():
    """中文註解：確認 pitch 權重解析與篩選規則。"""
    weights = parse_pitch_weights('37=2.0,38=1.5')
    assert weights[37] == 2.0
    assert keep_item([{'pitch': 38}], {38})
    assert not keep_item([{'pitch': 40}], {38})
    assert groove_key_for_audio('x/10_jazz_110_beat_4-4_15.wav') == '10_jazz_110_beat_4-4'
    assert len(window_items('x', {'duration': 8.0}, 4.0)) == 2
    assert density_score(Counter({'KD': 8, 'SD': 4}), 4.0, 'kdsd_density') == 3.0
    print('Self-check passed.')


def main():
    """中文註解：主流程，寫出 pitch-aware metadata 與報告。"""
    args = parse_args()
    if args.self_check:
        run_self_check()
        return
    pitch_weights = parse_pitch_weights(args.pitch_weights)
    required_pitches = {int(value.strip()) for value in args.require_pitches.split(',') if value.strip()}
    with open(args.meta, 'r', encoding='utf-8') as f:
        source_meta = json.load(f)
    output, report = build_meta(
        source_meta,
        args.limit,
        pitch_weights,
        required_pitches,
        args.velocity_min,
        args.max_kits_per_groove,
        args.window_seconds,
        args.min_kd_per_sec,
        args.min_sd_per_sec,
        args.sort_by,
    )
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    write_report(args.report, report)
    print(f'Wrote {len(output)} pitch-weighted items to {args.output}')


if __name__ == '__main__':
    main()
