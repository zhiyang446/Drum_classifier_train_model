# -*- coding: utf-8 -*-
"""以正式轉譜輸出的 MIDI 執行端到端商業驗收。"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile

import pretty_midi

from run_egmd_round4_validation import match_events
from run_real_audio_validation import LABELS, PITCH_TO_LABEL_IDX


HH_ARTICULATIONS = {
    "HH_CLOSED": {42},
    "HH_PEDAL": {44},
    "HH_OPEN": {46},
}


def parse_args():
    """中文註解：解析端到端驗收參數。"""
    parser = argparse.ArgumentParser(description="Validate final transcription MIDI against fixed references.")
    parser.add_argument("--manifest", help="JSON manifest containing a list of songs")
    parser.add_argument("--output-dir", help="New empty directory for generated MIDI and reports")
    parser.add_argument("--model", help="Base checkpoint passed to transcribe.py")
    parser.add_argument("--model-rare", help="Optional rare-class checkpoint passed to transcribe.py")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to launch transcribe.py")
    parser.add_argument("--transcribe-script", default="transcribe.py")
    parser.add_argument("--tolerance", type=float, default=0.050)
    parser.add_argument("--macro-min", type=float, default=0.70)
    parser.add_argument("--class-min", type=float, default=0.55)
    parser.add_argument("--articulation-min", type=float, default=0.80)
    parser.add_argument("--tempo-error-max", type=float, default=0.02)
    parser.add_argument("--adaptive-snare", action="store_true")
    parser.add_argument("--floating-bpm", action="store_true")
    parser.add_argument("--sync-audio", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def resolve_path(base_dir, value):
    """中文註解：將 manifest 相對路徑解析為正規化絕對路徑。"""
    return os.path.abspath(value if os.path.isabs(value) else os.path.join(base_dir, value))


def load_manifest(path):
    """中文註解：載入並驗證歌曲 manifest，不接受重複歌曲名稱。"""
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    songs = payload.get("songs") if isinstance(payload, dict) else payload
    if not isinstance(songs, list) or not songs:
        raise ValueError("manifest must contain a non-empty song list")

    base_dir = os.path.dirname(os.path.abspath(path))
    names = set()
    normalized = []
    for item in songs:
        if not isinstance(item, dict):
            raise ValueError("each song must be a JSON object")
        missing = [key for key in ("name", "audio", "reference_midi") if not item.get(key)]
        if missing:
            raise ValueError(f"song is missing required fields: {', '.join(missing)}")
        if item["name"] in names:
            raise ValueError(f"duplicate song name: {item['name']}")
        names.add(item["name"])
        song = dict(item)
        song["audio"] = resolve_path(base_dir, item["audio"])
        song["reference_midi"] = resolve_path(base_dir, item["reference_midi"])
        song["reference_offset_sec"] = float(item.get("reference_offset_sec", 0.0))
        for key in ("audio", "reference_midi"):
            if not os.path.isfile(song[key]):
                raise FileNotFoundError(f"missing {key}: {song[key]}")
        normalized.append(song)
    return normalized


def prepare_output_dir(path):
    """中文註解：只允許新目錄或空目錄，避免覆蓋既有驗證證據。"""
    path = os.path.abspath(path)
    if os.path.isdir(path) and os.listdir(path):
        raise FileExistsError(f"output directory is not empty: {path}")
    os.makedirs(os.path.join(path, "generated"), exist_ok=True)
    os.makedirs(os.path.join(path, "logs"), exist_ok=True)
    return path


def safe_name(value):
    """中文註解：產生跨 Windows 檔名安全的歌曲識別字。"""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    return cleaned or "song"


def run_transcription(song, output_midi, log_path, args):
    """中文註解：以正式 transcribe.py 與固定參數產生客戶最終 MIDI。"""
    command = [
        args.python,
        args.transcribe_script,
        "--input",
        song["audio"],
        "--output",
        output_midi,
        "--model",
        args.model,
    ]
    if args.model_rare:
        command.extend(["--model-rare", args.model_rare])
    if args.adaptive_snare:
        command.append("--adaptive-snare")
    if args.floating_bpm:
        command.append("--floating-bpm")
    if args.sync_audio:
        command.append("--sync-audio")

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write(result.stdout)
        if result.stderr:
            handle.write("\n[stderr]\n")
            handle.write(result.stderr)
    if result.returncode != 0 or not os.path.isfile(output_midi):
        raise RuntimeError(f"transcription failed with exit code {result.returncode}; see {log_path}")


def load_midi_events(path, offset=0.0):
    """中文註解：依六類與 Hi-Hat articulation 載入排序後的 MIDI 事件。"""
    midi = pretty_midi.PrettyMIDI(path)
    events = {label: [] for label in LABELS}
    events.update({label: [] for label in HH_ARTICULATIONS})
    for instrument in midi.instruments:
        for note in instrument.notes:
            class_index = PITCH_TO_LABEL_IDX.get(note.pitch)
            if class_index is not None:
                events[LABELS[class_index]].append(float(note.start) + offset)
            for label, pitches in HH_ARTICULATIONS.items():
                if note.pitch in pitches:
                    events[label].append(float(note.start) + offset)
                    break
    for values in events.values():
        values.sort()
    return midi, events


def metric_row(song_name, group, expected, predicted, tolerance):
    """中文註解：建立單一歌曲與事件群組的一對一匹配結果。"""
    tp, fp, fn, precision, recall, f1 = match_events(expected, predicted, tolerance)
    return {
        "song": song_name,
        "group": group,
        "expected": len(expected),
        "predicted": len(predicted),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def aggregate_rows(rows, groups):
    """中文註解：依群組加總 TP/FP/FN，避免歌曲大小差異扭曲 micro 指標。"""
    output = {}
    for group in groups:
        selected = [row for row in rows if row["group"] == group]
        tp = sum(row["tp"] for row in selected)
        fp = sum(row["fp"] for row in selected)
        fn = sum(row["fn"] for row in selected)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        output[group] = {
            "expected": tp + fn,
            "predicted": tp + fp,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return output


def first_tempo_and_meter(midi):
    """中文註解：讀取 MIDI 起始 Tempo 與拍號；無拍號事件時依 MIDI 慣例採 4/4。"""
    _, tempi = midi.get_tempo_changes()
    tempo = float(tempi[0]) if len(tempi) else 120.0
    if midi.time_signature_changes:
        signature = midi.time_signature_changes[0]
        meter = f"{signature.numerator}/{signature.denominator}"
    else:
        meter = "4/4"
    return tempo, meter


def write_csv(path, rows):
    """中文註解：以固定欄位寫出逐歌逐類 CSV。"""
    fields = ("song", "group", "expected", "predicted", "tp", "fp", "fn", "precision", "recall", "f1")
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)


def write_json(path, payload):
    """中文註解：寫出可供後續 gate 與人工審查使用的 JSON。"""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def evaluate_gate(class_summary, articulation_summary, song_summaries, errors, args):
    """中文註解：以固定門檻判定 gate，不允許空資料或類別崩潰被平均掩蓋。"""
    reasons = list(errors)
    class_f1 = [class_summary[label]["f1"] for label in LABELS]
    macro_f1 = sum(class_f1) / len(class_f1)
    if macro_f1 < args.macro_min:
        reasons.append(f"macro_f1 {macro_f1:.4f} < {args.macro_min:.4f}")
    for label in LABELS:
        if class_summary[label]["f1"] < args.class_min:
            reasons.append(f"{label} f1 {class_summary[label]['f1']:.4f} < {args.class_min:.4f}")
    for label, summary in articulation_summary.items():
        if summary["expected"] > 0 and summary["f1"] < args.articulation_min:
            reasons.append(f"{label} f1 {summary['f1']:.4f} < {args.articulation_min:.4f}")
    for song in song_summaries:
        if not song["tempo_pass"]:
            reasons.append(f"{song['song']} tempo error {song['tempo_error_ratio']:.4f}")
        if not song["meter_pass"]:
            reasons.append(f"{song['song']} meter {song['predicted_meter']} != {song['expected_meter']}")
    return {
        "status": "pass" if not reasons else "fail",
        "macro_f1": macro_f1,
        "macro_min": args.macro_min,
        "class_min": args.class_min,
        "articulation_min": args.articulation_min,
        "reasons": reasons,
    }


def validate(args):
    """中文註解：執行完整轉譜、事件比較、Tempo/拍號驗收與報表輸出。"""
    if not all((args.manifest, args.output_dir, args.model)):
        raise ValueError("--manifest, --output-dir and --model are required")
    songs = load_manifest(args.manifest)
    output_dir = prepare_output_dir(args.output_dir)
    details = []
    song_summaries = []
    errors = []

    for song in songs:
        name = safe_name(song["name"])
        generated_midi = os.path.join(output_dir, "generated", f"{name}.mid")
        log_path = os.path.join(output_dir, "logs", f"{name}.log")
        try:
            run_transcription(song, generated_midi, log_path, args)
            reference_midi, expected = load_midi_events(song["reference_midi"], song["reference_offset_sec"])
            predicted_midi, predicted = load_midi_events(generated_midi)
            for group in tuple(LABELS) + tuple(HH_ARTICULATIONS):
                details.append(metric_row(song["name"], group, expected[group], predicted[group], args.tolerance))

            reference_tempo, reference_meter = first_tempo_and_meter(reference_midi)
            predicted_tempo, predicted_meter = first_tempo_and_meter(predicted_midi)
            expected_tempo = float(song.get("expected_tempo_bpm", reference_tempo))
            expected_meter = str(song.get("expected_time_signature", reference_meter))
            tempo_error = abs(predicted_tempo - expected_tempo) / expected_tempo if expected_tempo > 0 else 1.0
            song_summaries.append({
                "song": song["name"],
                "reference_offset_sec": song["reference_offset_sec"],
                "expected_tempo_bpm": expected_tempo,
                "predicted_tempo_bpm": predicted_tempo,
                "tempo_error_ratio": tempo_error,
                "tempo_pass": tempo_error <= args.tempo_error_max,
                "expected_meter": expected_meter,
                "predicted_meter": predicted_meter,
                "meter_pass": predicted_meter == expected_meter,
            })
        except Exception as exc:
            errors.append(f"{song['name']}: {exc}")

    class_summary = aggregate_rows(details, LABELS)
    articulation_summary = aggregate_rows(details, HH_ARTICULATIONS)
    gate = evaluate_gate(class_summary, articulation_summary, song_summaries, errors, args)
    summary = {
        "gate": gate,
        "classes": class_summary,
        "hihat_articulations": articulation_summary,
        "songs": song_summaries,
    }
    write_csv(os.path.join(output_dir, "details.csv"), details)
    write_json(os.path.join(output_dir, "summary.json"), summary)
    write_json(os.path.join(output_dir, "gate_summary.json"), gate)
    print(f"End-to-end gate: {gate['status'].upper()} (macro_f1={gate['macro_f1']:.4f})")
    for reason in gate["reasons"]:
        print(f"- {reason}")
    return gate["status"] == "pass"


def run_self_check():
    """中文註解：驗證匹配、固定 offset、完整 gate 與輸出覆蓋防護。"""
    row = metric_row("demo", "KD", [1.0, 2.0], [1.04, 2.06, 3.0], 0.050)
    assert (row["tp"], row["fp"], row["fn"]) == (1, 2, 1)
    with tempfile.TemporaryDirectory() as temp_dir:
        reference_path = os.path.join(temp_dir, "reference.mid")
        predicted_path = os.path.join(temp_dir, "predicted.mid")
        pitches = (36, 38, 42, 44, 46, 47, 49, 51)
        for path, shift in ((reference_path, 0.0), (predicted_path, 0.02)):
            midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
            midi.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
            drums = pretty_midi.Instrument(program=0, is_drum=True)
            for index, pitch in enumerate(pitches, start=1):
                start = float(index) + shift
                drums.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=start + 0.1))
            midi.instruments.append(drums)
            midi.write(path)

        reference_midi, expected = load_midi_events(reference_path, offset=0.02)
        predicted_midi, predicted = load_midi_events(predicted_path)
        assert all(abs(left - right) < 0.002 for left, right in zip(expected["HH"], predicted["HH"]))
        assert all(
            abs(left - right) < 0.002
            for left, right in zip(expected["HH_CLOSED"], predicted["HH_CLOSED"])
        )
        details = [
            metric_row("demo", group, expected[group], predicted[group], 0.050)
            for group in tuple(LABELS) + tuple(HH_ARTICULATIONS)
        ]
        class_summary = aggregate_rows(details, LABELS)
        articulation_summary = aggregate_rows(details, HH_ARTICULATIONS)
        gate_args = argparse.Namespace(macro_min=0.70, class_min=0.55, articulation_min=0.80)
        song_summary = [{
            "song": "demo",
            "tempo_pass": True,
            "tempo_error_ratio": 0.0,
            "meter_pass": True,
            "predicted_meter": "4/4",
            "expected_meter": "4/4",
        }]
        assert evaluate_gate(class_summary, articulation_summary, song_summary, [], gate_args)["status"] == "pass"
        assert first_tempo_and_meter(reference_midi) == first_tempo_and_meter(predicted_midi)

        output_path = os.path.join(temp_dir, "output")
        prepare_output_dir(output_path)
        try:
            prepare_output_dir(output_path)
        except FileExistsError:
            pass
        else:
            raise AssertionError("non-empty output directory must be rejected")
    print("Self-check passed.")


def main():
    """中文註解：CLI 入口；gate 未通過時回傳非零狀態。"""
    args = parse_args()
    if args.self_check:
        run_self_check()
        return
    try:
        passed = validate(args)
    except Exception as exc:
        print(f"Validation error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
