# -*- coding: utf-8 -*-
"""D107：把已稽核 ENST 六類事件轉成現有 trainer metadata；不訓練。"""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


CLASSES = ("KD", "SD", "HH", "TOM", "CRASH", "RIDE")
EXPECTED_SPLITS = {"train": 97, "validation": 105, "test": 116}


def read_json(path):
    """讀取 UTF-8 JSON，讓來源格式錯誤在建立輸出前立即顯示。"""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def file_sha256(path):
    """計算 D106 證據雜湊，保留 D107 的來源追溯。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def flatten_events(entry):
    """將逐類時間陣列展平為現有 build_window 所需的 time/inst events。"""
    raw_events = entry.get("events")
    if not isinstance(raw_events, dict) or set(raw_events) != set(CLASSES):
        raise ValueError(f"Unexpected six-class event keys: {entry.get('group_id')}")
    duration = float(entry["duration_seconds"])
    events = []
    for class_index, label in enumerate(CLASSES):
        for timestamp in raw_events[label]:
            timestamp = float(timestamp)
            if not math.isfinite(timestamp) or timestamp < 0 or timestamp > duration + 0.05:
                raise ValueError(f"Out-of-bounds event: {entry.get('group_id')} {label} {timestamp}")
            events.append((timestamp, class_index, {"time": timestamp, "inst": label}))
    events.sort(key=lambda row: (row[0], row[1]))
    pairs = [(row[0], row[2]["inst"]) for row in events]
    if len(pairs) != len(set(pairs)):
        raise ValueError(f"Duplicate six-class event: {entry.get('group_id')}")
    return [row[2] for row in events]


def convert_entry(entry):
    """建立單一 training-ready item，保留來源路徑與 group 身分。"""
    audio_path = Path(entry["audio_path"])
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    return {
        "source": "d107_enst",
        "split": entry["split"],
        "group_id": entry["group_id"],
        "audio_path": str(audio_path),
        "annotation_path": entry["annotation_path"],
        "dry_mix_path": entry["dry_mix_path"],
        "input_mode": "mix",
        "events": flatten_events(entry),
    }


def class_counts(metadata):
    """計算 metadata 六類事件總量，供 D106 計數逐值核對。"""
    counts = Counter()
    for item in metadata.values():
        counts.update(event["inst"] for event in item["events"])
    return {label: counts[label] for label in CLASSES}


def build(summary_path, manifest_path):
    """轉換 train/validation 並驗證 test 不會寫入可用 metadata。"""
    summary = read_json(summary_path)
    manifest = read_json(manifest_path)
    if not summary.get("audit_pass") or summary.get("phase") != "D106":
        raise ValueError("D106 summary is not an accepted audit.")
    entries = manifest.get("entries", [])
    if len(entries) != 318:
        raise ValueError(f"Expected 318 D106 entries, got {len(entries)}.")

    split_counts = Counter(entry.get("split") for entry in entries)
    if split_counts != Counter(EXPECTED_SPLITS):
        raise ValueError(f"Unexpected D106 splits: {dict(split_counts)}")
    group_splits = {}
    for entry in entries:
        group_id = entry["group_id"]
        previous = group_splits.setdefault(group_id, entry["split"])
        if previous != entry["split"]:
            raise ValueError(f"Group crosses splits: {group_id}")
    if len(group_splits) != 318:
        raise ValueError("D106 group_id values are not unique.")

    train, validation = {}, {}
    for entry in entries:
        split = entry["split"]
        if split == "test":
            continue
        item = convert_entry(entry)
        destination = train if split == "train" else validation
        destination[entry["group_id"]] = item

    actual_counts = {
        "train": class_counts(train),
        "validation": class_counts(validation),
    }
    expected_counts = summary["split_class_events"]
    for split in actual_counts:
        if actual_counts[split] != expected_counts[split]:
            raise ValueError(f"{split} event counts differ from D106.")
        if any(actual_counts[split][label] <= 0 for label in CLASSES):
            raise ValueError(f"{split} does not cover all six classes.")
    overlap = sorted(set(train) & set(validation))
    if overlap:
        raise ValueError(f"Train/validation group overlap: {overlap[:3]}")

    audit = {
        "phase": "D107",
        "status": "metadata_ready_smoke_pending_not_training",
        "source_summary": str(summary_path.resolve()),
        "source_summary_sha256": file_sha256(summary_path),
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": file_sha256(manifest_path),
        "tracks": {
            "train_written": len(train),
            "validation_written": len(validation),
            "test_sealed_not_written": split_counts["test"],
        },
        "events": actual_counts,
        "empty_event_tracks": {
            "train": sum(not item["events"] for item in train.values()),
            "validation": sum(not item["events"] for item in validation.values()),
        },
        "group_overlap": len(overlap),
        "test_metadata_written": False,
        "velocity_supervision": False,
        "training_started": False,
        "ready_for_candidate_training": False,
    }
    return train, validation, audit


def run_window_smoke(train_metadata):
    """以現有排程與特徵入口建立六類正窗口加一個 NEG 窗口。"""
    import numpy as np

    from train_six_class_candidate import batch_from_schedule, build_schedule

    schedule = build_schedule(train_metadata, per_class=1, window_negative_from_train=True)
    if len(schedule) != 7 or [row["label"] for row in schedule] != [*CLASSES, "NEG"]:
        raise AssertionError("D107 smoke schedule is not six classes plus NEG.")
    features, onsets, velocities = batch_from_schedule(
        schedule,
        train_metadata,
        start=0,
        batch_size=len(schedule),
        use_true_superflux=True,
        input_mode="mix",
    )
    if features.shape != (7, 2, 256, 688) or onsets.shape != (7, 688, 6):
        raise AssertionError(f"Unexpected smoke shapes: {features.shape}, {onsets.shape}")
    if velocities.shape != onsets.shape or not np.isfinite(features).all():
        raise AssertionError("D107 smoke produced invalid feature or velocity arrays.")
    for index, label in enumerate(CLASSES):
        if onsets[index, :, CLASSES.index(label)].max() != 1.0:
            raise AssertionError(f"Smoke target missing selected class: {label}")
    return {
        "schedule": schedule,
        "feature_shape": list(features.shape),
        "onset_shape": list(onsets.shape),
        "velocity_shape": list(velocities.shape),
        "feature_finite": True,
        "selected_positive_classes_present": True,
    }


def write_outputs(output_dir, train, validation, audit, smoke):
    """只寫入全新 D107 目錄，拒絕覆寫任何歷史候選。"""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    final_audit = {
        **audit,
        "status": "pass_not_training",
        "window_smoke": {key: value for key, value in smoke.items() if key != "schedule"},
        "ready_for_candidate_training": True,
    }
    payloads = {
        "metadata_d107_train.json": train,
        "metadata_d107_validation.json": validation,
        "smoke_schedule.json": smoke["schedule"],
        "audit_d107.json": final_audit,
    }
    for name, payload in payloads.items():
        (output_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return final_audit


def run_self_check():
    """驗證事件展平排序、類別欄位與越界防線。"""
    entry = {
        "group_id": "demo",
        "duration_seconds": 2.0,
        "events": {
            "KD": [1.0], "SD": [0.5], "HH": [], "TOM": [],
            "CRASH": [0.5], "RIDE": [],
        },
    }
    assert flatten_events(entry) == [
        {"time": 0.5, "inst": "SD"},
        {"time": 0.5, "inst": "CRASH"},
        {"time": 1.0, "inst": "KD"},
    ]
    invalid = {**entry, "events": {**entry["events"], "KD": [3.0]}}
    try:
        flatten_events(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("Out-of-bounds events must be rejected.")
    print("D107 ENST metadata self-check passed.")


def main():
    """解析 CLI，執行 D107 self-check 或不可覆寫的實際建立。"""
    parser = argparse.ArgumentParser(description="Build ENST D107 trainer metadata without training.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("validation_runs/d106_enst_six_class_audit/summary.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("validation_runs/d106_enst_six_class_audit/manifest_candidate.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("enst_d107"))
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    train, validation, audit = build(args.summary, args.manifest)
    smoke = run_window_smoke(train)
    final_audit = write_outputs(args.output_dir, train, validation, audit, smoke)
    print(json.dumps(final_audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
