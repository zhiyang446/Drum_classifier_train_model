# -*- coding: utf-8 -*-
"""D100：只讀稽核五首真實鼓音訊與六類 MIDI reference 品質。"""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import librosa
import mido
import numpy as np
import soundfile as sf
from scipy.signal import correlate

from align_whack_metal_d29 import HOP_LENGTH, SAMPLE_RATE, onset_envelope
from build_real_song_d96_windows import LABELS, load_events
from run_real_audio_validation import PITCH_TO_LABEL_IDX


MAX_RESIDUAL_OFFSET_SECONDS = 0.50
OFFSET_REVIEW_SECONDS = 0.15
DRIFT_REVIEW_SECONDS = 0.25
TRANSIENT_TOLERANCE_SECONDS = 0.10
LOCAL_FRACTIONS = (0.10, 0.30, 0.50, 0.70, 0.90)


def event_impulses(events, frame_count):
    """把已校正到音訊時間的 reference events 轉成 onset 格點脈衝。"""
    frame_seconds = HOP_LENGTH / float(SAMPLE_RATE)
    pulses = np.zeros(frame_count, dtype=np.float64)
    for event in events:
        frame = int(round(float(event["time"]) / frame_seconds))
        if 0 <= frame < frame_count:
            pulses[frame] += math.sqrt(max(1, int(event.get("velocity", 127))) / 127.0)
    return pulses


def correlation_offset(envelope, pulses, max_offset_seconds=MAX_RESIDUAL_OFFSET_SECONDS):
    """以限制範圍的正規化 FFT correlation 量測 MIDI 相對音訊殘餘位移。"""
    denominator = float(np.linalg.norm(envelope) * np.linalg.norm(pulses))
    if denominator == 0.0:
        return None
    values = correlate(envelope, pulses, mode="full", method="fft") / denominator
    lags = np.arange(-len(pulses) + 1, len(envelope))
    max_frames = int(round(max_offset_seconds * SAMPLE_RATE / HOP_LENGTH))
    allowed = np.abs(lags) <= max_frames
    index = int(np.argmax(np.where(allowed, values, -np.inf)))
    return {
        "offset_seconds": float(lags[index] * HOP_LENGTH / SAMPLE_RATE),
        "score": float(values[index]),
    }


def local_alignment_profile(envelope, pulses):
    """在歌曲五個位置量測局部殘餘 offset，避免全曲分數掩蓋時間漂移。"""
    half_window = int(round(20.0 * SAMPLE_RATE / HOP_LENGTH))
    rows = []
    for fraction in LOCAL_FRACTIONS:
        center = int(round((len(envelope) - 1) * fraction))
        start = max(0, center - half_window)
        end = min(len(envelope), center + half_window)
        result = correlation_offset(envelope[start:end], pulses[start:end])
        if result is None:
            return None
        rows.append({
            "fraction": fraction,
            "center_seconds": float(center * HOP_LENGTH / SAMPLE_RATE),
            **result,
        })
    return rows


def raw_pitch_counts(midi_path):
    """讀取原始 MIDI note_on 音高，保留未知音高作人工 review 證據。"""
    counts = Counter()
    for message in mido.MidiFile(midi_path):
        if message.type == "note_on" and message.velocity > 0:
            counts[int(message.note)] += 1
    return counts


def duplicate_counts(events):
    """統計完全重複與同類 20ms 內近重複事件，不自動刪除可能的 flam。"""
    exact = Counter(
        (
            round(float(event["time"]), 6),
            event["inst"],
            int(event["pitch"]),
            int(event.get("velocity", 0)),
        )
        for event in events
    )
    exact_duplicates = sum(count - 1 for count in exact.values() if count > 1)
    exact_examples = [
        {
            "time": time,
            "inst": inst,
            "pitch": pitch,
            "velocity": velocity,
            "count": count,
        }
        for (time, inst, pitch, velocity), count in sorted(exact.items())
        if count > 1
    ][:10]
    near_examples = []
    by_class = {label: [] for label in LABELS}
    for event in events:
        by_class[event["inst"]].append(float(event["time"]))
    for label, times in by_class.items():
        near_examples.extend(
            {"inst": label, "previous_time": previous, "current_time": current, "gap_seconds": current - previous}
            for previous, current in zip(sorted(times), sorted(times)[1:])
            if 0.0 < current - previous < 0.020
        )
    return exact_duplicates, len(near_examples), exact_examples, near_examples[:10]


def transient_support(events, envelope):
    """量測每類 reference event 是否在 100ms 內有任一可偵測音訊 onset。"""
    onset_times = librosa.onset.onset_detect(
        onset_envelope=envelope,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        units="time",
        backtrack=False,
    )
    rows = {}
    for label in LABELS:
        times = [float(event["time"]) for event in events if event["inst"] == label]
        supported = sum(
            bool(len(onset_times)) and float(np.min(np.abs(onset_times - time))) <= TRANSIENT_TOLERANCE_SECONDS
            for time in times
        )
        rows[label] = {
            "events": len(times),
            "supported": supported,
            "support_rate": supported / len(times) if times else None,
        }
    total = len(events)
    supported_total = sum(row["supported"] for row in rows.values())
    return rows, supported_total / total if total else 0.0, len(onset_times)


def resolve_human_reviews(item, missing_classes, unknown_pitches, low_support_classes):
    """套用已完成的人工作答，只移除已被明確確認的 review 原因。"""
    reviewed_pitch_overrides = {
        int(pitch): label for pitch, label in item.get("reviewed_pitch_overrides", {}).items()
    }
    confirmed_absent = set(item.get("confirmed_absent_classes", []))
    confirmed_low_support = set(item.get("confirmed_low_support_classes", []))
    invalid_labels = (
        set(reviewed_pitch_overrides.values()) | confirmed_absent | confirmed_low_support
    ) - set(LABELS)
    if invalid_labels:
        raise ValueError(f"Invalid human-reviewed labels for {item['id']}: {sorted(invalid_labels)}")
    unknown_pitch_numbers = set(map(int, unknown_pitches))
    if set(reviewed_pitch_overrides) - unknown_pitch_numbers:
        raise ValueError(f"Reviewed pitch is not present in raw MIDI for {item['id']}")
    if confirmed_absent - set(missing_classes):
        raise ValueError(f"Confirmed absent class is not absent for {item['id']}")
    if confirmed_low_support - set(low_support_classes):
        raise ValueError(f"Confirmed low-support class is not low-support for {item['id']}")
    unresolved_unknown = {
        pitch: count
        for pitch, count in unknown_pitches.items()
        if int(pitch) not in reviewed_pitch_overrides
    }
    return {
        "reviewed_pitch_overrides": {
            str(pitch): label for pitch, label in sorted(reviewed_pitch_overrides.items())
        },
        "unresolved_unknown_pitches": unresolved_unknown,
        "confirmed_absent_classes": sorted(confirmed_absent),
        "unresolved_missing_classes": [
            label for label in missing_classes if label not in confirmed_absent
        ],
        "confirmed_low_support_classes": sorted(confirmed_low_support),
        "unresolved_low_transient_support_classes": [
            label for label in low_support_classes if label not in confirmed_low_support
        ],
    }


def analyse_song(manifest_path, item):
    """稽核單首音訊／MIDI／event CSV，回傳不修改來源的可序列化證據。"""
    audio_path = (manifest_path.parent / item["audio_path"]).resolve()
    midi_path = (manifest_path.parent / item["reference_midi"]).resolve()
    event_path = (manifest_path.parent / item["reference_events_csv"]).resolve()
    if not audio_path.is_file() or not midi_path.is_file() or not event_path.is_file():
        raise FileNotFoundError(f"Missing D100 pair for {item['id']}")

    info = sf.info(str(audio_path))
    duration = info.frames / float(info.samplerate)
    events = load_events(event_path)
    class_counts = Counter(event["inst"] for event in events)
    missing_classes = [label for label in LABELS if class_counts[label] == 0]
    out_of_bounds = [
        event for event in events if not 0.0 <= float(event["time"]) < duration
    ]
    exact_duplicates, near_duplicates, exact_examples, near_examples = duplicate_counts(events)

    pitch_counts = raw_pitch_counts(midi_path)
    unknown_pitches = {
        str(pitch): count
        for pitch, count in sorted(pitch_counts.items())
        if pitch not in PITCH_TO_LABEL_IDX
    }
    declared_review = sorted(int(pitch) for pitch in item.get("review_pitches", []))

    envelope = onset_envelope(audio_path)
    pulses = event_impulses(events, len(envelope))
    global_alignment = correlation_offset(envelope, pulses)
    local_profile = local_alignment_profile(envelope, pulses)
    offsets = [] if local_profile is None else [row["offset_seconds"] for row in local_profile]
    drift_span = None if not offsets else max(offsets) - min(offsets)
    per_class_support, overall_support, detected_onsets = transient_support(events, envelope)
    low_support_classes = [
        label
        for label, row in per_class_support.items()
        if row["events"] and row["support_rate"] < 0.50
    ]
    human_reviews = resolve_human_reviews(
        item, missing_classes, unknown_pitches, low_support_classes
    )
    unresolved_unknown = human_reviews["unresolved_unknown_pitches"]
    unresolved_missing = human_reviews["unresolved_missing_classes"]
    unresolved_low_support = human_reviews["unresolved_low_transient_support_classes"]
    review_pitch_mismatch = sorted(set(map(int, unresolved_unknown)) ^ set(declared_review))

    reasons = []
    if unresolved_missing:
        reasons.append("missing_classes")
    if unresolved_unknown:
        reasons.append("unknown_midi_pitches")
    if review_pitch_mismatch:
        reasons.append("review_pitch_mismatch")
    if out_of_bounds:
        reasons.append("out_of_bounds_events")
    if exact_duplicates:
        reasons.append("exact_duplicate_events")
    if global_alignment is None or local_profile is None:
        reasons.append("alignment_unavailable")
    else:
        if abs(global_alignment["offset_seconds"]) > OFFSET_REVIEW_SECONDS:
            reasons.append("residual_offset")
        if drift_span > DRIFT_REVIEW_SECONDS:
            reasons.append("local_alignment_drift")
    if overall_support < 0.50:
        reasons.append("low_transient_support")
    if unresolved_low_support:
        reasons.append("low_class_transient_support")

    return {
        "id": item["id"],
        "group_id": item["group_id"],
        "split": item["split"],
        "audio_path": str(audio_path),
        "midi_path": str(midi_path),
        "reference_events_csv": str(event_path),
        "duration_seconds": duration,
        "reference_offset_seconds": float(item["reference_offset_sec"]),
        "events": len(events),
        "class_counts": {label: class_counts[label] for label in LABELS},
        "missing_classes": missing_classes,
        "raw_pitch_counts": {str(pitch): count for pitch, count in sorted(pitch_counts.items())},
        "unknown_pitches": unknown_pitches,
        "declared_review_pitches": declared_review,
        "review_pitch_mismatch": review_pitch_mismatch,
        **human_reviews,
        "out_of_bounds_events": len(out_of_bounds),
        "exact_duplicate_events": exact_duplicates,
        "exact_duplicate_examples": exact_examples,
        "near_duplicate_events_under_20ms": near_duplicates,
        "near_duplicate_examples": near_examples,
        "global_alignment": global_alignment,
        "local_alignment": local_profile,
        "local_drift_span_seconds": drift_span,
        "detected_audio_onsets": detected_onsets,
        "transient_support": per_class_support,
        "overall_transient_support_rate": overall_support,
        "low_transient_support_classes": low_support_classes,
        "status": "needs_reference_review" if reasons else "alignment_pass",
        "review_reasons": reasons,
    }


def write_outputs(output_dir, report):
    """建立全新 D100 JSON／CSV，拒絕覆寫任何既有稽核證據。"""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite D100 output: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "audit_d100.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fields = (
        "id", "split", "events", "missing_classes", "unresolved_missing_classes",
        "unknown_pitches", "unresolved_unknown_pitches",
        "global_residual_offset_seconds", "global_alignment_score",
        "local_drift_span_seconds", "overall_transient_support_rate",
        "low_transient_support_classes", "unresolved_low_transient_support_classes",
        "exact_duplicate_events", "near_duplicate_events_under_20ms",
        "status", "review_reasons",
    )
    with (output_dir / "songs_d100.csv").open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report["songs"]:
            global_alignment = row["global_alignment"] or {}
            writer.writerow({
                "id": row["id"],
                "split": row["split"],
                "events": row["events"],
                "missing_classes": "|".join(row["missing_classes"]),
                "unresolved_missing_classes": "|".join(row["unresolved_missing_classes"]),
                "unknown_pitches": json.dumps(row["unknown_pitches"], sort_keys=True),
                "unresolved_unknown_pitches": json.dumps(
                    row["unresolved_unknown_pitches"], sort_keys=True
                ),
                "global_residual_offset_seconds": global_alignment.get("offset_seconds"),
                "global_alignment_score": global_alignment.get("score"),
                "local_drift_span_seconds": row["local_drift_span_seconds"],
                "overall_transient_support_rate": row["overall_transient_support_rate"],
                "low_transient_support_classes": "|".join(row["low_transient_support_classes"]),
                "unresolved_low_transient_support_classes": "|".join(
                    row["unresolved_low_transient_support_classes"]
                ),
                "exact_duplicate_events": row["exact_duplicate_events"],
                "near_duplicate_events_under_20ms": row["near_duplicate_events_under_20ms"],
                "status": row["status"],
                "review_reasons": "|".join(row["review_reasons"]),
            })


def audit(manifest_file, output_dir):
    """執行五首 D100 稽核並依硬性證據決定是否需要人工 reference review。"""
    manifest_path = Path(manifest_file).resolve()
    items = json.loads(manifest_path.read_text(encoding="utf-8"))["items"]
    if len(items) != 5 or len({item["group_id"] for item in items}) != 5:
        raise ValueError("D100 requires exactly five unique song groups")
    songs = [analyse_song(manifest_path, item) for item in sorted(items, key=lambda row: row["id"])]
    review_songs = [row["id"] for row in songs if row["status"] != "alignment_pass"]
    report = {
        "phase": "D100",
        "status": "needs_reference_review" if review_songs else "alignment_pass",
        "algorithm": "D29 onset envelope plus residual global/local FFT correlation",
        "thresholds": {
            "max_residual_search_seconds": MAX_RESIDUAL_OFFSET_SECONDS,
            "offset_review_seconds": OFFSET_REVIEW_SECONDS,
            "drift_review_seconds": DRIFT_REVIEW_SECONDS,
            "transient_tolerance_seconds": TRANSIENT_TOLERANCE_SECONDS,
            "low_transient_support_rate": 0.50,
        },
        "songs_count": len(songs),
        "review_songs": review_songs,
        "songs": songs,
        "source_files_modified": False,
        "training_started": False,
        "ready_for_training_candidate": False,
        "ready_for_release": False,
    }
    write_outputs(output_dir, report)
    print(json.dumps({
        "phase": report["phase"],
        "status": report["status"],
        "review_songs": review_songs,
    }, ensure_ascii=False, indent=2))


def run_self_check():
    """驗證正向位移、重複事件與人工 review 解決邏輯。"""
    pulses = np.zeros(40)
    pulses[[5, 15, 25]] = 1.0
    envelope = np.roll(pulses, 2)
    result = correlation_offset(envelope, pulses)
    expected = 2 * HOP_LENGTH / float(SAMPLE_RATE)
    assert abs(result["offset_seconds"] - expected) < 1e-9
    events = [
        {"time": 1.0, "inst": "KD", "pitch": 36},
        {"time": 1.0, "inst": "KD", "pitch": 36},
        {"time": 1.01, "inst": "SD", "pitch": 38},
    ]
    exact, near, examples, _ = duplicate_counts(events)
    assert exact == 1 and near == 0 and examples[0]["count"] == 2
    reviewed = resolve_human_reviews(
        {
            "id": "self-check",
            "reviewed_pitch_overrides": {"64": "TOM"},
            "confirmed_absent_classes": ["CRASH"],
            "confirmed_low_support_classes": ["RIDE"],
        },
        ["CRASH"],
        {"64": 2},
        ["RIDE"],
    )
    assert not reviewed["unresolved_unknown_pitches"]
    assert not reviewed["unresolved_missing_classes"]
    assert not reviewed["unresolved_low_transient_support_classes"]
    print("D100 self-check passed.")


def main():
    """D100 CLI 入口。"""
    parser = argparse.ArgumentParser(description="Audit D93 real-song reference quality.")
    parser.add_argument("--manifest", default="real-song/d93_intake/manifest.json")
    parser.add_argument("--output-dir", default="validation_runs/d100_real_song_data_audit")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    audit(args.manifest, args.output_dir)


if __name__ == "__main__":
    main()
