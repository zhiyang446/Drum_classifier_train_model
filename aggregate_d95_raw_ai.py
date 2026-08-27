# -*- coding: utf-8 -*-
"""彙整 D95 五首歌曲的 Raw AI native 事件，不讀 final 事件或 MIDI。"""
import argparse
import csv
import json
import tempfile
from pathlib import Path

from run_end_to_end_validation import LABELS, aggregate_rows, metric_row, write_csv, write_json


NATIVE_COLUMNS = {
    "KD": "native_kick",
    "SD": "native_snare",
    "HH": "native_hihat",
    "TOM": "native_tom",
    "CRASH": "native_crash",
    "RIDE": "native_ride",
}


def load_reference_events(path):
    """讀取 D93 已完成時間校正的六類參考事件。"""
    events = {label: [] for label in LABELS}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["inst"] in events:
                events[row["inst"]].append(float(row["time"]))
    for values in events.values():
        values.sort()
    return events


def load_raw_events(path):
    """只讀 raw_time 與 native_*，刻意忽略 final_*、量化時間及 MIDI。"""
    events = {label: [] for label in LABELS}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = {"raw_time", *NATIVE_COLUMNS.values()} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        for row in reader:
            raw_time = float(row["raw_time"])
            for label, column in NATIVE_COLUMNS.items():
                if row[column].strip().lower() == "true":
                    events[label].append(raw_time)
    for values in events.values():
        values.sort()
    return events


def aggregate(manifest_path, raw_root, d94_summary_path, output_dir, tolerance):
    """以固定 50 ms 配對彙整五首 Raw AI 指標，並與 D94 最終 MIDI 比較。"""
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")

    manifest_path = Path(manifest_path).resolve()
    raw_root = Path(raw_root).resolve()
    with open(manifest_path, encoding="utf-8") as handle:
        items = json.load(handle)["items"]
    if len(items) != 5:
        raise ValueError(f"D95 requires exactly 5 songs, got {len(items)}")

    details, songs, raw_sources = [], [], {}
    for item in items:
        song = Path(item["audio_path"]).stem
        reference_path = (manifest_path.parent / item["reference_events_csv"]).resolve()
        raw_path = raw_root / song / f"{song}_raw_ai_events.csv"
        if not reference_path.is_file() or not raw_path.is_file():
            raise FileNotFoundError(f"missing D95 input for {song}")
        expected = load_reference_events(reference_path)
        predicted = load_raw_events(raw_path)
        rows = [metric_row(song, label, expected[label], predicted[label], tolerance) for label in LABELS]
        details.extend(rows)
        songs.append({
            "song": song,
            "split": item["split"],
            "macro_f1": sum(row["f1"] for row in rows) / len(LABELS),
            "raw_ai_events_csv": str(raw_path),
        })
        raw_sources[song] = str(raw_path)

    class_summary = aggregate_rows(details, LABELS)
    macro_f1 = sum(class_summary[label]["f1"] for label in LABELS) / len(LABELS)
    reasons = ([f"macro_f1 {macro_f1:.4f} < 0.7000"] if macro_f1 < 0.70 else [])
    reasons.extend(
        f"{label} f1 {class_summary[label]['f1']:.4f} < 0.5500"
        for label in LABELS
        if class_summary[label]["f1"] < 0.55
    )
    gate = {"status": "fail" if reasons else "pass", "macro_f1": macro_f1, "reasons": reasons}

    with open(d94_summary_path, encoding="utf-8") as handle:
        d94 = json.load(handle)
    comparison = {
        label: {
            "raw_f1": class_summary[label]["f1"],
            "final_midi_f1": d94["classes"][label]["f1"],
            "delta_raw_minus_final": class_summary[label]["f1"] - d94["classes"][label]["f1"],
        }
        for label in LABELS
    }
    comparison["macro"] = {
        "raw_f1": macro_f1,
        "final_midi_f1": d94["gate"]["macro_f1"],
        "delta_raw_minus_final": macro_f1 - d94["gate"]["macro_f1"],
    }
    summary = {
        "phase": "D95",
        "source_layer": "raw_time + native_*",
        "model_inference_rerun": True,
        "tolerance_sec": tolerance,
        "gate": gate,
        "classes": class_summary,
        "songs": songs,
        "comparison_to_d94_final_midi": comparison,
        "raw_ai_sources": raw_sources,
    }
    output_dir.mkdir(parents=True)
    write_csv(str(output_dir / "details.csv"), details)
    write_json(str(output_dir / "summary.json"), summary)
    write_json(str(output_dir / "gate_summary.json"), gate)
    assert len(details) == 5 * len(LABELS)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return gate["status"] == "pass"


def self_check():
    """確認載入器只採 native 欄位，不能把 final 欄位混入 Raw 指標。"""
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "raw.csv"
        path.write_text(
            "raw_time,native_kick,native_snare,native_hihat,native_tom,native_crash,native_ride,final_snare\n"
            "1.0,True,False,False,False,False,False,True\n",
            encoding="utf-8",
        )
        events = load_raw_events(path)
        assert events["KD"] == [1.0] and events["SD"] == []
    print("D95 self-check passed.")


def main():
    """執行 D95 Raw AI 彙整或最小自我檢查。"""
    parser = argparse.ArgumentParser(description="Aggregate existing D95 native raw events.")
    parser.add_argument("--manifest", default="real-song/d93_intake/manifest.json")
    parser.add_argument("--raw-root", default="real-song/d95_raw_ai")
    parser.add_argument("--d94-summary", default="real-song/d94_d76_six_class_baseline_complete/summary.json")
    parser.add_argument("--output-dir", default="real-song/d95_raw_ai_baseline_complete")
    parser.add_argument("--tolerance", type=float, default=0.050)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    passed = aggregate(args.manifest, args.raw_root, args.d94_summary, args.output_dir, args.tolerance)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
