# -*- coding: utf-8 -*-
"""
Run Round4 validation on short E-GMD test-split clips.
"""
import argparse
import csv
import json
import os
import re
import tempfile

import soundfile as sf

from compare_blind_expected import compare, write_rows
from run_blind_test import run_one, write_reports


INST_MAP = {
    'KD': 'expected_kick',
    'SD': 'expected_snare',
    'HH': 'expected_hihat',
}
EVENT_FIELDS = {
    'KD': 'kick',
    'SD': 'snare',
    'HH': 'hihat',
}
STRONG_VELOCITY = {
    'KD': 30.0,
    'SD': 70.0,
    'HH': 30.0,
}
TARGET_PITCHES = {35, 36, 37, 38, 40, 22, 26, 42, 44, 46}


def count_unsupported_midi_events(audio_path):
    """
    中文註解：統計 E-GMD sibling MIDI 中目前三分類未支援的鼓件數。
    """
    midi_path = os.path.splitext(audio_path)[0] + '.midi'
    if not os.path.exists(midi_path):
        return 0
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(midi_path)
    return sum(
        1
        for instrument in pm.instruments
        for note in instrument.notes
        if note.pitch not in TARGET_PITCHES
    )


def safe_case_name(audio_path):
    """
    中文註解：產生和 run_blind_test.safe_stem 相同規則的案例名稱。
    """
    from run_blind_test import safe_stem
    return safe_stem(audio_path)


def select_cases(meta, limit, offset=0, max_unsupported=0):
    """
    中文註解：從 E-GMD metadata 選出固定、唯一、短秒數的 test split 4/4 音訊。
    """
    candidates = []
    seen_audio = set()
    seen_groove = set()
    for item in meta.values():
        audio_path = item.get('audio_path', '')
        name = os.path.basename(audio_path)
        groove_key = re.sub(r'_\d+\.[^.]+$', '', name)
        if item.get('split') != 'test':
            continue
        if audio_path in seen_audio:
            continue
        if groove_key in seen_groove:
            continue
        if not (20.0 <= float(item.get('duration', 0.0)) <= 40.0):
            continue
        if '_4-4_' not in name:
            continue
        if not os.path.exists(audio_path):
            continue
        unsupported = count_unsupported_midi_events(audio_path)
        if unsupported > max_unsupported:
            continue
        counts = {'expected_kick': 0, 'expected_snare': 0, 'expected_hihat': 0}
        for ev in item.get('events', []):
            field = INST_MAP.get(ev.get('inst'))
            if field:
                counts[field] += 1
        if min(counts.values()) <= 0:
            continue
        seen_audio.add(audio_path)
        seen_groove.add(groove_key)
        candidates.append({
            'name': safe_case_name(audio_path),
            'audio_path': audio_path,
            'expected_tempo': str(float(item['bpm'])),
            'tempo_tol': '2.0',
            'expected_time_signature': '4/4',
            'unsupported_midi_events': str(unsupported),
            **{k: str(v) for k, v in counts.items()},
        })
    candidates.sort(key=lambda row: (
        int(row['expected_kick']) + int(row['expected_snare']) + int(row['expected_hihat']),
        int(row['expected_hihat']),
        row['name'],
    ))
    return candidates[offset:offset + limit]


def write_expected(rows, path):
    """
    中文註解：寫出 compare_blind_expected.py 可讀的 expected CSV。
    """
    fields = [
        'name', 'expected_tempo', 'tempo_tol', 'expected_time_signature',
        'expected_kick', 'expected_snare', 'expected_hihat',
    ]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def materialize_excerpts(selected, meta, output_dir, seconds):
    """
    中文註解：將 E-GMD test 音訊裁成固定長度 excerpt，並同步更新 expected counts。
    """
    if seconds <= 0:
        return selected
    by_audio = {item['audio_path']: item for item in meta.values()}
    excerpt_dir = os.path.join(output_dir, 'excerpts')
    os.makedirs(excerpt_dir, exist_ok=True)
    output = []
    for case in selected:
        item = by_audio[case['audio_path']]
        events = item.get('events', [])
        latest_start = max(0.0, float(item.get('duration', seconds)) - seconds)
        best_start = 0.0
        best_counts = None
        best_total = -1
        step = 0.5
        pos = 0.0
        while pos <= latest_start + 1e-9:
            counts = {'expected_kick': 0, 'expected_snare': 0, 'expected_hihat': 0}
            for ev in events:
                t = float(ev['time'])
                if pos <= t < pos + seconds:
                    field = INST_MAP.get(ev.get('inst'))
                    if field:
                        counts[field] += 1
            total = min(counts.values()) * 1000 + sum(counts.values())
            if min(counts.values()) > 0 and total > best_total:
                best_start = pos
                best_counts = counts
                best_total = total
            pos += step
        if best_counts is None:
            best_counts = {'expected_kick': 0, 'expected_snare': 0, 'expected_hihat': 0}
            for ev in events:
                t = float(ev['time'])
                if 0.0 <= t < seconds:
                    field = INST_MAP.get(ev.get('inst'))
                    if field:
                        best_counts[field] += 1

        y, sr = sf.read(case['audio_path'], dtype='float32')
        start_sample = int(round(best_start * sr))
        end_sample = min(len(y), start_sample + int(round(seconds * sr)))
        excerpt_path = os.path.join(excerpt_dir, f"{case['name']}_excerpt.wav")
        sf.write(excerpt_path, y[start_sample:end_sample], sr)

        copied = dict(case)
        copied['name'] = safe_case_name(excerpt_path)
        copied['audio_path'] = excerpt_path
        copied['excerpt_source'] = case['audio_path']
        copied['excerpt_start'] = f'{best_start:.3f}'
        copied['excerpt_seconds'] = f'{seconds:.3f}'
        for key, value in best_counts.items():
            copied[key] = str(value)
        output.append(copied)
    return output


def read_predicted_events(csv_path, layer_prefix='final'):
    """
    中文註解：讀取 raw/notation event CSV，轉成各鼓件的物理時間列表。
    """
    events = {inst: [] for inst in EVENT_FIELDS}
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            t = float(row['raw_time'])
            for inst, field in EVENT_FIELDS.items():
                if row.get(f'{layer_prefix}_{field}') == 'True':
                    events[inst].append(t)
    return events


def match_events(expected_times, predicted_times, tolerance):
    """
    中文註解：用固定容差做一對一事件匹配，輸出 TP/FP/FN 與 F1。
    """
    used = [False] * len(predicted_times)
    tp = 0
    for expected in expected_times:
        best_idx = -1
        best_dist = tolerance + 1.0
        for idx, predicted in enumerate(predicted_times):
            if used[idx]:
                continue
            dist = abs(predicted - expected)
            if dist <= tolerance and dist < best_dist:
                best_idx = idx
                best_dist = dist
        if best_idx >= 0:
            used[best_idx] = True
            tp += 1
    fp = len(predicted_times) - tp
    fn = len(expected_times) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return tp, fp, fn, precision, recall, f1


def cluster_close_events(times, window):
    """
    中文註解：將同鼓件過近的 MIDI 裝飾音合併為一個物理事件診斷點。
    """
    clustered = []
    for t in sorted(times):
        if not clustered or t - clustered[-1] > window:
            clustered.append(t)
    return clustered


def filter_predictions_matching_weak(predicted_times, weak_times, tolerance):
    """
    中文註解：strong-hit 診斷中，預測若命中弱標註，不應被算成強音假陽性。
    """
    output = []
    used_weak = [False] * len(weak_times)
    for predicted in predicted_times:
        matched_weak = False
        for idx, weak in enumerate(weak_times):
            if used_weak[idx]:
                continue
            if abs(predicted - weak) <= tolerance:
                used_weak[idx] = True
                matched_weak = True
                break
        if not matched_weak:
            output.append(predicted)
    return output


def write_event_report(meta, selected, summary_rows, output_csv, tolerance):
    """
    中文註解：將 E-GMD metadata 與輸出事件做時間層級比對。
    """
    by_audio = {item['audio_path']: item for item in meta.values()}
    selected_by_audio = {case['audio_path']: case for case in selected}
    fields = [
        'name', 'layer', 'inst', 'target', 'velocity_min', 'expected',
        'predicted', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'overall',
    ]
    report_rows = []
    for row in summary_rows:
        case = selected_by_audio[row['audio']]
        item = by_audio[case.get('excerpt_source', row['audio'])]
        excerpt_start = float(case.get('excerpt_start', 0.0) or 0.0)
        excerpt_end = excerpt_start + float(case.get('excerpt_seconds', 1e9) or 1e9)
        layer_paths = {
            'raw': row['raw_ai_events'],
            'notation': row['notation_events'],
        }
        predictions = {layer: read_predicted_events(path) for layer, path in layer_paths.items()}
        for target_name, velocity_min in [('full', 0.0), ('strong', None), ('clustered_strong', None)]:
            for inst in EVENT_FIELDS:
                min_velocity = STRONG_VELOCITY[inst] if velocity_min is None else velocity_min
                expected = [
                    float(ev['time']) - excerpt_start for ev in item['events']
                    if (
                        ev.get('inst') == inst
                        and float(ev.get('velocity', 0.0)) >= min_velocity
                        and excerpt_start <= float(ev['time']) < excerpt_end
                    )
                ]
                if target_name == 'clustered_strong':
                    expected = cluster_close_events(expected, 0.035)
                weak_expected = [
                    float(ev['time']) - excerpt_start for ev in item['events']
                    if (
                        target_name == 'strong'
                        and ev.get('inst') == inst
                        and float(ev.get('velocity', 0.0)) < min_velocity
                        and excerpt_start <= float(ev['time']) < excerpt_end
                    )
                ]
                for layer in ('raw', 'notation'):
                    predicted = predictions[layer][inst]
                    if weak_expected:
                        predicted = filter_predictions_matching_weak(predicted, weak_expected, tolerance)
                    tp, fp, fn, precision, recall, f1 = match_events(expected, predicted, tolerance)
                    report_rows.append({
                        'name': row['name'],
                        'layer': layer,
                        'inst': inst,
                        'target': target_name,
                        'velocity_min': min_velocity,
                        'expected': len(expected),
                        'predicted': len(predicted),
                        'tp': tp,
                        'fp': fp,
                        'fn': fn,
                        'precision': f'{precision:.4f}',
                        'recall': f'{recall:.4f}',
                        'f1': f'{f1:.4f}',
                        'overall': 'pass' if f1 >= 0.90 else 'fail',
                    })
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report_rows)
    return report_rows


def write_gate_summary(event_rows, output_dir):
    """
    中文註解：寫出 Round4 官方驗收摘要；full-count CSV 仍保留為診斷。
    """
    strong_rows = [row for row in event_rows if row['target'] == 'strong']
    strong_passed = sum(row['overall'] == 'pass' for row in strong_rows)
    gate_row = {
        'gate': 'round4_physical_strong_event',
        'overall': 'pass' if strong_rows and strong_passed == len(strong_rows) else 'fail',
        'passed_rows': strong_passed,
        'total_rows': len(strong_rows),
        'note': 'full MIDI count reports are diagnostic only',
    }
    csv_path = os.path.join(output_dir, 'gate_summary.csv')
    json_path = os.path.join(output_dir, 'gate_summary.json')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(gate_row.keys()))
        writer.writeheader()
        writer.writerow(gate_row)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(gate_row, f, indent=2)
    return gate_row


def run_self_check():
    """
    中文註解：最小自檢，確認選樣與 expected 欄位可用。
    """
    item = {
        'audio_path': __file__.replace('.py', '_4-4_1.wav'),
        'duration': 25.0,
        'bpm': 100,
        'split': 'test',
        'events': [{'inst': 'KD'}, {'inst': 'SD'}, {'inst': 'HH'}],
    }
    assert INST_MAP['KD'] == 'expected_kick'
    assert safe_case_name('a b#c.wav') == 'a_b_c'
    assert match_events([0.1, 0.2], [0.11, 0.4], 0.05)[:3] == (1, 1, 1)
    with tempfile.TemporaryDirectory() as tmpdir:
        assert write_gate_summary([{'target': 'strong', 'overall': 'pass'}], tmpdir)['overall'] == 'pass'
    print('Self-check passed.')


def main():
    """
    中文註解：CLI 入口，選樣、轉譜、比較 raw/notation，並輸出證據。
    """
    parser = argparse.ArgumentParser(description='Run E-GMD Round4 validation.')
    parser.add_argument('--meta', default='processed_data/egmd_meta.json')
    parser.add_argument('--model', default='mixed_formal_kick375_snare18_hh12_candidate.pth')
    parser.add_argument('--output-dir', default='validation_runs/egmd_round4')
    parser.add_argument('--expected', default=None)
    parser.add_argument('--limit', type=int, default=5)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--max-unsupported-midi-events', type=int, default=0)
    parser.add_argument('--excerpt-seconds', type=float, default=0.0)
    parser.add_argument('--thresh-kick', type=float, default=None)
    parser.add_argument('--thresh-snare', type=float, default=None)
    parser.add_argument('--thresh-hihat', type=float, default=None)
    parser.add_argument('--thresh-tom', type=float, default=None)
    parser.add_argument('--thresh-crash', type=float, default=None)
    parser.add_argument('--thresh-ride', type=float, default=None)
    parser.add_argument('--event-tolerance', type=float, default=0.05)
    parser.add_argument('--architecture', default='symmetric')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()

    if args.self_check:
        run_self_check()
        return

    with open(args.meta, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    selected = select_cases(meta, args.limit, args.offset, args.max_unsupported_midi_events)
    if len(selected) < args.limit:
        raise SystemExit(f'Only selected {len(selected)} cases, expected {args.limit}.')

    os.makedirs(args.output_dir, exist_ok=True)
    selected = materialize_excerpts(selected, meta, args.output_dir, args.excerpt_seconds)
    expected_path = args.expected or os.path.join(args.output_dir, 'expected.csv')
    write_expected(selected, expected_path)

    rows = []
    thresholds = {
        '--thresh-kick': args.thresh_kick,
        '--thresh-snare': args.thresh_snare,
        '--thresh-hihat': args.thresh_hihat,
        '--thresh-tom': args.thresh_tom,
        '--thresh-crash': args.thresh_crash,
        '--thresh-ride': args.thresh_ride,
    }
    for case in selected:
        print(f"Running E-GMD Round4: {case['audio_path']}", flush=True)
        rows.append(run_one(case['audio_path'], args.model, args.output_dir, thresholds=thresholds, architecture=args.architecture))
    summary_csv, _ = write_reports(rows, args.output_dir)

    raw_rows = compare(summary_csv, expected_path, layer='raw_acoustic')
    notation_rows = compare(summary_csv, expected_path, layer='notation')
    write_rows(raw_rows, os.path.join(args.output_dir, 'raw_compare.csv'))
    write_rows(notation_rows, os.path.join(args.output_dir, 'notation_compare.csv'))
    event_rows = write_event_report(meta, selected, rows, os.path.join(args.output_dir, 'event_compare.csv'), args.event_tolerance)
    gate_row = write_gate_summary(event_rows, args.output_dir)

    for label, result_rows in [('raw', raw_rows), ('notation', notation_rows)]:
        passed = sum(row['overall'] == 'pass' for row in result_rows)
        print(f'{label}: {passed}/{len(result_rows)} pass')
        for row in result_rows:
            print(f"{label} {row['name']}: {row['overall']} {row['failures']}")
    strong_rows = [row for row in event_rows if row['target'] == 'strong']
    strong_passed = sum(row['overall'] == 'pass' for row in strong_rows)
    print(f'event strong: {strong_passed}/{len(strong_rows)} pass')
    print(f"round4 gate: {gate_row['overall']} ({gate_row['passed_rows']}/{gate_row['total_rows']})")


if __name__ == '__main__':
    main()
