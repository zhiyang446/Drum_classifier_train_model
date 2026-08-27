# -*- coding: utf-8 -*-
"""彙整 D94 已存在的五首 MIDI，不重新執行模型推論。"""
import argparse
import json
from pathlib import Path

from run_end_to_end_validation import LABELS, aggregate_rows, load_manifest, load_midi_events, metric_row, write_csv, write_json


GENERATED_MIDI = {
    'beautiful-things': 'real-song/d94_d76_six_class_baseline/generated/beautiful-things.mid',
    'beggin': 'real-song/d94_d76_six_class_baseline/generated/beggin.mid',
    'chop-suey-drums': 'real-song/d94_single_chop_suey/generated/chop-suey-drums.mid',
    'something': 'real-song/d94_single_something/generated/something.mid',
    'toxicity-drums': 'real-song/d94_single_toxicity/generated/toxicity-drums.mid',
}


def aggregate(manifest_path, output_dir, tolerance):
    """讀取五份既有輸出，寫出逐歌、逐類與整體固定匹配報告。"""
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f'refusing to overwrite output: {output_dir}')
    output_dir.mkdir(parents=True)
    details, songs = [], []
    for song in load_manifest(manifest_path):
        generated_path = Path(GENERATED_MIDI[song['name']]).resolve()
        if not generated_path.is_file():
            raise FileNotFoundError(generated_path)
        _, expected = load_midi_events(song['reference_midi'], song['reference_offset_sec'])
        _, predicted = load_midi_events(str(generated_path))
        rows = [metric_row(song['name'], label, expected[label], predicted[label], tolerance) for label in LABELS]
        details.extend(rows)
        songs.append({
            'song': song['name'],
            'macro_f1': sum(row['f1'] for row in rows) / len(LABELS),
            'generated_midi': str(generated_path),
            'reference_offset_sec': song['reference_offset_sec'],
        })
    class_summary = aggregate_rows(details, LABELS)
    macro_f1 = sum(class_summary[label]['f1'] for label in LABELS) / len(LABELS)
    reasons = []
    if macro_f1 < 0.70:
        reasons.append(f'macro_f1 {macro_f1:.4f} < 0.7000')
    for label in LABELS:
        if class_summary[label]['f1'] < 0.55:
            reasons.append(f"{label} f1 {class_summary[label]['f1']:.4f} < 0.5500")
    gate = {'status': 'fail' if reasons else 'pass', 'macro_f1': macro_f1, 'reasons': reasons}
    summary = {
        'phase': 'D94',
        'model_inference_rerun': False,
        'tolerance_sec': tolerance,
        'gate': gate,
        'classes': class_summary,
        'songs': songs,
        'generated_midi_sources': GENERATED_MIDI,
    }
    write_csv(str(output_dir / 'details.csv'), details)
    write_json(str(output_dir / 'summary.json'), summary)
    write_json(str(output_dir / 'gate_summary.json'), gate)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return gate['status'] == 'pass'


def main():
    """執行一次性 D94 彙整，fail gate 以非零狀態回傳。"""
    parser = argparse.ArgumentParser(description='Aggregate existing D94 MIDI without rerunning inference.')
    parser.add_argument('--manifest', default='real-song/d93_intake/d94_baseline_manifest.json')
    parser.add_argument('--output-dir', default='real-song/d94_d76_six_class_baseline_complete')
    parser.add_argument('--tolerance', type=float, default=0.050)
    args = parser.parse_args()
    passed = aggregate(args.manifest, args.output_dir, args.tolerance)
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
