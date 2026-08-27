# -*- coding: utf-8 -*-
"""D106：唯讀稽核 ENST-Drums 標註是否可映射為專案六類。"""

import argparse
import csv
import json
import tempfile
import wave
from collections import Counter, defaultdict
from pathlib import Path


CLASSES = ("KD", "SD", "HH", "TOM", "CRASH", "RIDE")
RAW_TO_CLASS = {
    "bd": "KD",
    "sd": "SD", "sd-": "SD", "cs": "SD", "rs": "SD",
    "chh": "HH", "ohh": "HH",
    "lt": "TOM", "mt": "TOM", "lmt": "TOM", "lft": "TOM", "mtr": "TOM", "ltr": "TOM",
    "c1": "CRASH", "cr1": "CRASH", "cr2": "CRASH", "cr5": "CRASH",
    "ch1": "CRASH", "ch5": "CRASH", "spl2": "CRASH",
    "rc2": "RIDE", "rc3": "RIDE", "rc4": "RIDE", "c4": "RIDE",
}
EXCLUDED_LABELS = {"cb", "sweep", "sticks"}
DRUMMER_SPLIT = {"drummer_1": "train", "drummer_2": "validation", "drummer_3": "test"}
EXPECTED_TRACKS = {"drummer_1": 97, "drummer_2": 105, "drummer_3": 116}


def read_events(path):
    """讀取 ENST 的「秒數 標籤」文字檔，拒絕格式錯誤或時間倒退。"""
    events = []
    previous = -1.0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 2:
                raise ValueError(f"{path}:{line_number}: expected '<seconds> <label>'")
            timestamp = float(parts[0])
            if timestamp < 0 or timestamp < previous:
                raise ValueError(f"{path}:{line_number}: invalid event time {timestamp}")
            events.append((timestamp, parts[1]))
            previous = timestamp
    return events


def wav_duration(path):
    """只讀 WAV header 並回傳秒數，不載入音訊內容。"""
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        if rate <= 0:
            raise ValueError(f"Invalid WAV sample rate: {path}")
        return handle.getnframes() / float(rate)


def write_csv(path, fieldnames, rows):
    """以固定欄位寫出 UTF-8 CSV 稽核證據。"""
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    """寫出可由後續流程直接讀取的 UTF-8 JSON。"""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def audit(root, output_dir):
    """稽核配對、標籤映射、事件邊界、split 隔離與六類覆蓋。"""
    if not root.is_dir():
        raise FileNotFoundError(f"ENST root not found: {root}")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    raw_counts = Counter()
    raw_files = defaultdict(set)
    split_events = {split: Counter() for split in DRUMMER_SPLIT.values()}
    split_files = {split: Counter() for split in DRUMMER_SPLIT.values()}
    split_groups = {split: set() for split in DRUMMER_SPLIT.values()}
    pairing_rows = []
    manifest = []
    unknown_counts = Counter()
    excluded_counts = Counter()
    annotation_errors = []
    out_of_bounds = 0
    duration_mismatches = 0
    missing_channel_files = 0
    tracks_by_drummer = Counter()

    for drummer, split in DRUMMER_SPLIT.items():
        drummer_root = root / drummer
        annotation_dir = drummer_root / "annotation"
        audio_root = drummer_root / "audio"
        channels = sorted(path.name for path in audio_root.iterdir() if path.is_dir())
        annotation_paths = sorted(annotation_dir.glob("*.txt"))
        tracks_by_drummer[drummer] = len(annotation_paths)

        for annotation_path in annotation_paths:
            basename = annotation_path.stem
            group_id = f"enst_{drummer}_{basename}"
            channel_paths = {channel: audio_root / channel / f"{basename}.wav" for channel in channels}
            missing_channels = [channel for channel, path in channel_paths.items() if not path.is_file()]
            wet_path = channel_paths.get("wet_mix")
            dry_path = channel_paths.get("dry_mix")
            row_status = "pass"
            duration = None
            events = []

            try:
                events = read_events(annotation_path)
            except (OSError, ValueError) as exc:
                annotation_errors.append(str(exc))
                row_status = "fail"

            if missing_channels or wet_path is None or dry_path is None:
                missing_channel_files += len(missing_channels) + int(wet_path is None) + int(dry_path is None)
                row_status = "fail"
            elif wet_path.is_file() and dry_path.is_file():
                try:
                    duration = wav_duration(wet_path)
                    if abs(duration - wav_duration(dry_path)) > 1.0 / 44100.0:
                        duration_mismatches += 1
                        row_status = "fail"
                except (OSError, EOFError, wave.Error, ValueError) as exc:
                    annotation_errors.append(str(exc))
                    row_status = "fail"
            else:
                missing_channel_files += int(not wet_path.is_file()) + int(not dry_path.is_file())
                row_status = "fail"

            class_events = {label: [] for label in CLASSES}
            track_classes = set()
            track_unknown = 0
            track_excluded = 0
            for timestamp, raw_label in events:
                raw_counts[raw_label] += 1
                raw_files[raw_label].add(f"{drummer}/{basename}")
                target = RAW_TO_CLASS.get(raw_label)
                if target:
                    class_events[target].append(timestamp)
                    split_events[split][target] += 1
                    track_classes.add(target)
                elif raw_label in EXCLUDED_LABELS:
                    excluded_counts[raw_label] += 1
                    track_excluded += 1
                else:
                    unknown_counts[raw_label] += 1
                    track_unknown += 1
                    row_status = "fail"
                if duration is not None and timestamp > duration + 0.05:
                    out_of_bounds += 1
                    row_status = "fail"

            for target in track_classes:
                split_files[split][target] += 1
            split_groups[split].add(group_id)
            pairing_rows.append({
                "drummer": drummer,
                "basename": basename,
                "group_id": group_id,
                "split": split,
                "annotation_path": str(annotation_path),
                "wet_mix_path": str(wet_path) if wet_path else "",
                "dry_mix_path": str(dry_path) if dry_path else "",
                "channels_expected": len(channels),
                "channels_present": len(channels) - len(missing_channels),
                "duration_seconds": "" if duration is None else round(duration, 6),
                "event_count": len(events),
                "excluded_count": track_excluded,
                "unknown_count": track_unknown,
                "status": row_status,
            })
            manifest.append({
                "group_id": group_id,
                "source": "ENST-Drums",
                "drummer": drummer,
                "split": split,
                "annotation_path": str(annotation_path),
                "audio_path": str(wet_path) if wet_path else "",
                "dry_mix_path": str(dry_path) if dry_path else "",
                "duration_seconds": duration,
                "events": class_events,
            })

    group_overlap = sorted(
        (split_a, split_b, group_id)
        for split_a, groups_a in split_groups.items()
        for split_b, groups_b in split_groups.items()
        if split_a < split_b
        for group_id in groups_a & groups_b
    )
    split_coverage = {
        split: {label: split_events[split][label] for label in CLASSES}
        for split in split_events
    }
    all_split_classes_nonzero = all(
        count > 0 for coverage in split_coverage.values() for count in coverage.values()
    )
    fileparts = sum(1 for _ in root.rglob("*.filepart"))
    failed_rows = sum(row["status"] != "pass" for row in pairing_rows)
    audit_pass = (
        tracks_by_drummer == Counter(EXPECTED_TRACKS)
        and len(pairing_rows) == 318
        and failed_rows == 0
        and not annotation_errors
        and not unknown_counts
        and out_of_bounds == 0
        and duration_mismatches == 0
        and missing_channel_files == 0
        and not group_overlap
        and all_split_classes_nonzero
        and fileparts == 0
    )

    raw_rows = []
    for raw_label, count in sorted(raw_counts.items()):
        raw_rows.append({
            "raw_label": raw_label,
            "target_class": RAW_TO_CLASS.get(raw_label, ""),
            "status": "mapped" if raw_label in RAW_TO_CLASS else "excluded" if raw_label in EXCLUDED_LABELS else "unknown",
            "events": count,
            "files": len(raw_files[raw_label]),
            "sample_files": ";".join(sorted(raw_files[raw_label])[:3]),
        })
    class_rows = [
        {
            "split": split,
            "class": label,
            "events": split_events[split][label],
            "files": split_files[split][label],
        }
        for split in ("train", "validation", "test")
        for label in CLASSES
    ]
    summary = {
        "phase": "D106",
        "status": "audit_pass_not_training" if audit_pass else "audit_fail_not_training",
        "audit_pass": audit_pass,
        "root": str(root),
        "tracks": len(pairing_rows),
        "tracks_by_drummer": dict(tracks_by_drummer),
        "raw_event_count": sum(raw_counts.values()),
        "mapped_event_count": sum(sum(counter.values()) for counter in split_events.values()),
        "excluded_event_count": sum(excluded_counts.values()),
        "excluded_labels": dict(sorted(excluded_counts.items())),
        "unknown_labels": dict(sorted(unknown_counts.items())),
        "failed_pairing_rows": failed_rows,
        "missing_channel_files": missing_channel_files,
        "annotation_errors": annotation_errors,
        "out_of_bounds_events": out_of_bounds,
        "wet_dry_duration_mismatches": duration_mismatches,
        "filepart_count": fileparts,
        "group_overlap_count": len(group_overlap),
        "group_overlap": group_overlap,
        "split_class_events": split_coverage,
        "all_splits_cover_six_classes": all_split_classes_nonzero,
        "training_started": False,
        "ready_for_training_candidate": audit_pass,
        "ready_for_six_class_release": False,
    }

    write_csv(output_dir / "raw_label_mapping.csv",
              ("raw_label", "target_class", "status", "events", "files", "sample_files"), raw_rows)
    write_csv(output_dir / "class_counts.csv", ("split", "class", "events", "files"), class_rows)
    write_csv(output_dir / "pairing_audit.csv", tuple(pairing_rows[0]), pairing_rows)
    write_json(output_dir / "manifest_candidate.json", {
        "phase": "D106",
        "candidate_only_not_training": True,
        "entries": manifest,
    })
    write_json(output_dir / "summary.json", summary)
    return summary


def run_self_check():
    """驗證解析、六類映射與未知標籤邊界的最小可執行檢查。"""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "sample.txt"
        path.write_text("0.10 bd\n0.20 sd-\n0.30 c4\n0.40 cb\n", encoding="utf-8")
        events = read_events(path)
    assert [RAW_TO_CLASS.get(label) for _, label in events] == ["KD", "SD", "RIDE", None]
    assert events[-1][1] in EXCLUDED_LABELS
    assert set(RAW_TO_CLASS.values()) == set(CLASSES)
    assert set(DRUMMER_SPLIT.values()) == {"train", "validation", "test"}
    print("D106 ENST self-check passed.")


def main():
    """解析 CLI，執行 D106 唯讀稽核或最小 self-check。"""
    parser = argparse.ArgumentParser(description="Audit ENST-Drums six-class labels without training.")
    parser.add_argument("--root", type=Path,
                        default=Path(r"D:\DrumDatasets\ENST-Drums\ENST-drums-public"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("validation_runs/d106_enst_six_class_audit"))
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    summary = audit(args.root, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["audit_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
