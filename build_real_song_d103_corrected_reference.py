# -*- coding: utf-8 -*-
"""依 D102 人工決定建立不可覆寫的 D103 reference 修正版候選。"""

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import mido
import soundfile as sf

from build_real_song_d93_intake import EVENT_FIELDS
from build_real_song_d96_windows import LABELS


EXPECTED_BEFORE = {"KD": 1304, "SD": 986, "HH": 1641, "TOM": 460, "CRASH": 267, "RIDE": 218}
EXPECTED_AFTER = {"KD": 1304, "SD": 984, "HH": 1641, "TOM": 462, "CRASH": 267, "RIDE": 218}
REFERENCE_CORRECT_IDS = {
    "D101_001", "D101_002", "D101_003", "D101_004", "D101_009", "D101_010",
    "D101_011", "D101_012", "D101_013", "D101_014", "D101_015",
}
HUMAN_REVIEWS = {
    "real_song_beautiful_things": {"confirmed_low_support_classes": ["RIDE"]},
    "real_song_chop_suey_drums": {
        "reviewed_pitch_overrides": {"64": "TOM"},
        "confirmed_absent_classes": ["CRASH"],
    },
    "real_song_something": {"confirmed_low_support_classes": ["RIDE"]},
    "real_song_toxicity_drums": {
        "confirmed_absent_classes": ["RIDE"],
        "confirmed_low_support_classes": ["TOM"],
    },
}


def read_json(path):
    """讀取 UTF-8 JSON。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path):
    """計算來源檔 SHA-256，留下不可變證據。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_events(path):
    """讀取 D93 event CSV，保留固定欄位文字值。"""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_events(path, rows):
    """寫入全新 D103 event CSV。"""
    with Path(path).open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def validate_decisions(payload):
    """確認 D102 已完整結案，且修正集合沒有漂移。"""
    if (
        payload.get("status") != "complete_15_of_15"
        or payload.get("decisions_received") != 15
        or payload.get("decisions_pending") != 0
        or payload.get("ready_for_reference_correction") is not True
        or set(payload.get("reference_correct_ids", [])) != REFERENCE_CORRECT_IDS
        or set(payload.get("confirmed_error_ids", []))
        != {"D101_005", "D101_006", "D101_007", "D101_008"}
    ):
        raise ValueError("D102 decisions are incomplete or differ from the reviewed set")
    actions = payload.get("correction_actions", [])
    if len(actions) != 3 or sum(int(action.get("remove_count", 0)) for action in actions) != 2:
        raise ValueError("D102 correction action count differs from the locked set")
    pitch_action = next((row for row in actions if row["action"] == "map_source_pitch_to_class"), None)
    if (
        pitch_action is None
        or pitch_action.get("song_id") != "real_song_chop_suey_drums"
        or pitch_action.get("source_pitch") != 64
        or pitch_action.get("target_class") != "TOM"
        or pitch_action.get("corrected_audio_times_seconds") != [15.050032, 22.550048]
    ):
        raise ValueError("D102 pitch-64 action differs from the locked review")
    return actions


def pitch_override_events(midi_path, offset_seconds, action):
    """從原始 MIDI 擷取已確認 pitch，僅在本候選映射成 TOM。"""
    clock, rows = 0.0, []
    for message in mido.MidiFile(midi_path):
        clock += message.time
        if (
            message.type == "note_on"
            and message.velocity > 0
            and message.note == action["source_pitch"]
        ):
            rows.append({
                "time": f"{clock + offset_seconds:.6f}",
                "inst": action["target_class"],
                "velocity": str(message.velocity),
                "midi_pitch": str(message.note),
                "source": "d102_human_review_pitch_override",
                "review_required": "True",
            })
    actual_times = [float(row["time"]) for row in rows]
    if actual_times != action["corrected_audio_times_seconds"]:
        raise ValueError(f"Pitch-64 times differ: {actual_times}")
    return rows


def remove_one_duplicate(rows, action):
    """在指定時間只移除一筆完全重複事件，拒絕模糊比對。"""
    matches = [
        index for index, row in enumerate(rows)
        if row["inst"] == action["inst"]
        and int(row["midi_pitch"]) == action["pitch"]
        and float(row["time"]) == action["time_seconds"]
    ]
    if len(matches) != 2 or rows[matches[0]] != rows[matches[1]] or action["remove_count"] != 1:
        raise ValueError(f"Expected one exact duplicate pair at {action['time_seconds']}")
    del rows[matches[-1]]


def exact_duplicate_count(rows):
    """統計六個 CSV 欄位完全相同的多餘事件數。"""
    counts = Counter(tuple(row[field] for field in EVENT_FIELDS) for row in rows)
    return sum(count - 1 for count in counts.values() if count > 1)


def build(manifest_file, decisions_file, output_dir):
    """建立 D103 候選並驗證事件數、邊界與 split 隔離。"""
    manifest_path = Path(manifest_file).resolve()
    decisions_path = Path(decisions_file).resolve()
    output_path = Path(output_dir).resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite D103 output: {output_path}")
    source_manifest = read_json(manifest_path)
    actions = validate_decisions(read_json(decisions_path))
    pitch_action = next(row for row in actions if row["action"] == "map_source_pitch_to_class")
    remove_actions = [row for row in actions if row["action"] == "remove_one_exact_duplicate_event"]

    events_dir = output_path / "reference_events"
    events_dir.mkdir(parents=True)
    total_before, total_after = Counter(), Counter()
    song_audit, manifest_items = [], []
    group_splits = {}
    for item in source_manifest["items"]:
        source_csv = (manifest_path.parent / item["reference_events_csv"]).resolve()
        audio_path = (manifest_path.parent / item["audio_path"]).resolve()
        midi_path = (manifest_path.parent / item["reference_midi"]).resolve()
        rows = read_events(source_csv)
        before = Counter(row["inst"] for row in rows)
        total_before.update(before)
        changed = False
        if item["id"] == pitch_action["song_id"]:
            rows.extend(pitch_override_events(midi_path, float(item["reference_offset_sec"]), pitch_action))
            changed = True
        for action in remove_actions:
            if item["id"] == action["song_id"]:
                remove_one_duplicate(rows, action)
                changed = True
        rows.sort(key=lambda row: (float(row["time"]), row["inst"], int(row["midi_pitch"])))

        target_csv = events_dir / source_csv.name
        if changed:
            write_events(target_csv, rows)
        else:
            shutil.copy2(source_csv, target_csv)
        after = Counter(row["inst"] for row in rows)
        total_after.update(after)
        duration = sf.info(str(audio_path)).duration
        out_of_bounds = sum(not 0.0 <= float(row["time"]) < duration for row in rows)
        duplicates = exact_duplicate_count(rows)
        if out_of_bounds or duplicates:
            raise ValueError(f"D103 event audit failed for {item['id']}")
        reviews = HUMAN_REVIEWS.get(item["id"], {})
        manifest_items.append({
            **item,
            "reference_events_csv": str(Path("reference_events") / target_csv.name),
            "review_pitches": [],
            **reviews,
        })
        group_splits.setdefault(item["group_id"], set()).add(item["split"])
        song_audit.append({
            "id": item["id"],
            "changed": changed,
            "before": {label: before[label] for label in LABELS},
            "after": {label: after[label] for label in LABELS},
            "exact_duplicate_events": duplicates,
            "out_of_bounds_events": out_of_bounds,
        })

    before_counts = {label: total_before[label] for label in LABELS}
    after_counts = {label: total_after[label] for label in LABELS}
    group_leaks = sum(len(splits) != 1 for splits in group_splits.values())
    if before_counts != EXPECTED_BEFORE or after_counts != EXPECTED_AFTER or group_leaks:
        raise ValueError("D103 totals or group isolation differ from the locked specification")
    manifest = {
        "phase": "D103",
        "status": "corrected_reference_candidate",
        "source_manifest": str(manifest_path),
        "source_decisions": str(decisions_path),
        "items": manifest_items,
    }
    audit = {
        "phase": "D103",
        "status": "corrected_reference_candidate",
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_decisions_sha256": sha256_file(decisions_path),
        "correction_event_operations": 4,
        "total_class_counts_before": before_counts,
        "total_class_counts_after": after_counts,
        "group_split_leaks": group_leaks,
        "song_audit": song_audit,
        "source_files_modified": False,
        "training_started": False,
        "ready_for_quality_reaudit": True,
        "ready_for_training_candidate": False,
    }
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_path / "audit_d103.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def run_self_check():
    """確認完全重複只刪一筆，非目標事件保持不變。"""
    rows = [
        {"time": "1.000000", "inst": "SD", "velocity": "80", "midi_pitch": "38", "source": "x", "review_required": "False"},
        {"time": "1.000000", "inst": "SD", "velocity": "80", "midi_pitch": "38", "source": "x", "review_required": "False"},
        {"time": "2.000000", "inst": "KD", "velocity": "90", "midi_pitch": "36", "source": "x", "review_required": "False"},
    ]
    remove_one_duplicate(rows, {
        "inst": "SD", "pitch": 38, "time_seconds": 1.0, "remove_count": 1,
    })
    assert len(rows) == 2 and exact_duplicate_count(rows) == 0
    print("D103 self-check passed.")


def main():
    """D103 CLI 入口。"""
    parser = argparse.ArgumentParser(description="Build corrected D103 reference candidate without training.")
    parser.add_argument("--manifest", default="real-song/d93_intake/manifest.json")
    parser.add_argument(
        "--decisions",
        default="validation_runs/d102_reference_review_decisions/decisions_final.json",
    )
    parser.add_argument("--output-dir", default="real-song/d103_corrected_reference")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    print(json.dumps(build(args.manifest, args.decisions, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
