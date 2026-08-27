# -*- coding: utf-8 -*-
"""建立 D96 三首 train 真實鼓的低記憶體四秒窗口索引。"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf


LABELS = ("KD", "SD", "HH", "TOM", "CRASH", "RIDE")
WINDOW_SECONDS = 4.0


def load_events(path):
    """讀取 D93 校正事件並保留既有六類、力度與來源音高。"""
    events = []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["inst"] not in LABELS:
                continue
            events.append({
                "time": float(row["time"]),
                "inst": row["inst"],
                "velocity": int(row["velocity"]),
                "pitch": int(row["midi_pitch"]),
                "review_required": row["review_required"].strip().lower() == "true",
            })
    return sorted(events, key=lambda event: (event["time"], event["inst"], event["pitch"]))


def build_windows(item_id, group_id, events, frames, sample_rate):
    """依既有 build_window 的四秒夾限規則建立含事件 anchor 索引。"""
    source_samples = int(round(WINDOW_SECONDS * sample_rate))
    duration = frames / float(sample_rate)
    rows = []
    anchor = WINDOW_SECONDS / 2.0
    index = 0
    while anchor <= duration:
        anchor_sample = int(anchor * sample_rate)
        start_sample = max(0, anchor_sample - source_samples // 2)
        start_sample = min(start_sample, max(0, frames - source_samples))
        start = start_sample / float(sample_rate)
        counts = Counter(
            event["inst"] for event in events
            if start <= event["time"] < start + WINDOW_SECONDS
        )
        if counts:
            rows.append({
                "window_id": f"{item_id}_win{index:03d}",
                "item_id": item_id,
                "group_id": group_id,
                "anchor_time": anchor,
                "start_time": start,
                "end_time": min(duration, start + WINDOW_SECONDS),
                "event_count": sum(counts.values()),
                **{label: counts[label] for label in LABELS},
            })
        anchor += WINDOW_SECONDS
        index += 1
    return rows


def write_outputs(output_dir, metadata, windows, audit):
    """只寫全新 D96 產物，拒絕覆寫既有證據。"""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite D96 output: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "metadata_d96.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fields = (
        "window_id", "item_id", "group_id", "anchor_time", "start_time",
        "end_time", "event_count", *LABELS,
    )
    with (output_dir / "windows_d96.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(windows)
    (output_dir / "audit_d96.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def build(manifest_path, output_dir):
    """只處理 D93 train，產生窗口覆蓋與 D97 就緒判定。"""
    manifest_path = Path(manifest_path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = payload["items"]
    group_splits = defaultdict(set)
    for item in items:
        group_splits[item["group_id"]].add(item["split"])
    leaks = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    train_items = [item for item in items if item["split"] == "train"]

    metadata, windows = {}, []
    groups_by_class = defaultdict(set)
    events_by_class = Counter()
    out_of_bounds = []
    for item in train_items:
        audio_path = (manifest_path.parent / item["audio_path"]).resolve()
        reference_path = (manifest_path.parent / item["reference_events_csv"]).resolve()
        if not audio_path.is_file() or not reference_path.is_file():
            raise FileNotFoundError(f"Missing D96 train input for {item['id']}")
        info = sf.info(str(audio_path))
        events = load_events(reference_path)
        duration = info.frames / float(info.samplerate)
        invalid = [event for event in events if not 0.0 <= event["time"] < duration]
        out_of_bounds.extend({"item_id": item["id"], **event} for event in invalid)
        valid_events = [event for event in events if 0.0 <= event["time"] < duration]
        item_windows = build_windows(
            item["id"], item["group_id"], valid_events, info.frames, info.samplerate
        )
        windows.extend(item_windows)
        for event in valid_events:
            events_by_class[event["inst"]] += 1
            groups_by_class[event["inst"]].add(item["group_id"])
        metadata[item["id"]] = {
            "audio_path": str(audio_path),
            "reference_midi": str((manifest_path.parent / item["reference_midi"]).resolve()),
            "reference_events_csv": str(reference_path),
            "duration": duration,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "split": "train",
            "source": "d96_real_song",
            "group_id": item["group_id"],
            "events": valid_events,
            "window_anchors": [row["anchor_time"] for row in item_windows],
        }

    windows_by_class = {
        label: sum(int(row[label] > 0) for row in windows) for label in LABELS
    }
    reasons = []
    if len(train_items) != 3:
        reasons.append(f"expected 3 train songs, got {len(train_items)}")
    if leaks:
        reasons.append(f"group_id crosses splits: {leaks}")
    if out_of_bounds:
        reasons.append(f"{len(out_of_bounds)} events are outside audio duration")
    for label in LABELS:
        if windows_by_class[label] == 0:
            reasons.append(f"{label} has zero train windows")
        if len(groups_by_class[label]) < 2:
            reasons.append(f"{label} appears in fewer than 2 train groups")
    audit = {
        "phase": "D96",
        "status": "fail" if reasons else "pass",
        "window_seconds": WINDOW_SECONDS,
        "window_storage": "anchor_index_only_on_demand_audio_read",
        "train_songs": len(train_items),
        "heldout_songs_counted_only": len(items) - len(train_items),
        "heldout_audio_read": False,
        "train_windows": len(windows),
        "events_by_class": {label: events_by_class[label] for label in LABELS},
        "windows_by_class": windows_by_class,
        "groups_by_class": {
            label: sorted(groups_by_class[label]) for label in LABELS
        },
        "group_split_leaks": leaks,
        "out_of_bounds_events": out_of_bounds,
        "unmapped_review_pitches_preserved": {
            item["id"]: item.get("review_pitches", []) for item in train_items
            if item.get("review_pitches")
        },
        "training_started": False,
        "ready_for_d97_candidate": not reasons,
        "ready_for_release": False,
        "reasons": reasons,
    }
    write_outputs(output_dir, metadata, windows, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return not reasons


def run_self_check():
    """驗證四秒夾限、空窗口排除與六類窗口計數。"""
    events = [
        {"time": 0.5, "inst": "KD"},
        {"time": 4.5, "inst": "RIDE"},
        {"time": 7.5, "inst": "CRASH"},
    ]
    rows = build_windows("demo", "group_demo", events, frames=800, sample_rate=100)
    assert len(rows) == 2
    assert rows[0]["KD"] == 1 and rows[1]["RIDE"] == 1 and rows[1]["CRASH"] == 1
    print("D96 self-check passed.")


def main():
    """執行 D96 builder 或最小自我檢查。"""
    parser = argparse.ArgumentParser(description="Build D96 train-only real-song window index.")
    parser.add_argument("--manifest", default="real-song/d93_intake/manifest.json")
    parser.add_argument("--output-dir", default="real-song/d96_train_windows")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    passed = build(args.manifest, args.output_dir)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
