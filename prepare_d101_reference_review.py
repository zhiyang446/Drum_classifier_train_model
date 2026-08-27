# -*- coding: utf-8 -*-
"""D101：把 D100 可疑 reference 事件整理成不可覆寫的人工聽辨包。"""

import argparse
import csv
import json
from pathlib import Path

import librosa
import mido
import numpy as np
import soundfile as sf

from align_whack_metal_d29 import HOP_LENGTH, SAMPLE_RATE, onset_envelope
from build_real_song_d96_windows import load_events
from run_real_audio_validation import PITCH_TO_LABEL_IDX


CLIP_PRE_SECONDS = 0.75
CLIP_POST_SECONDS = 1.25
TRANSIENT_TOLERANCE_SECONDS = 0.10
REPRESENTATIVE_EVENTS = 3


def select_representative(rows, limit=REPRESENTATIVE_EVENTS):
    """從時間排序事件取前／中／後代表點，避免把同一歌曲所有疑點都複製成 clip。"""
    rows = sorted(rows, key=lambda row: row["time"])
    if len(rows) <= limit:
        return rows
    indices = sorted(set(int(round(index)) for index in np.linspace(0, len(rows) - 1, limit)))
    return [rows[index] for index in indices]


def audio_onset_times(audio_path):
    """重用 D100 onset envelope，取得同一門檻下的音訊 onset 時間。"""
    envelope = onset_envelope(audio_path)
    return librosa.onset.onset_detect(
        onset_envelope=envelope,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        units="time",
        backtrack=False,
    )


def unsupported_events(events, label, onset_times):
    """找出指定類別中距離最近音訊 onset 超過 100ms 的 reference events。"""
    rows = []
    for event in events:
        if event["inst"] != label:
            continue
        time = float(event["time"])
        distance = float(np.min(np.abs(onset_times - time))) if len(onset_times) else None
        if distance is None or distance > TRANSIENT_TOLERANCE_SECONDS:
            rows.append({
                "time": time,
                "inst": label,
                "pitch": int(event["pitch"]),
                "nearest_audio_onset_distance_seconds": distance,
            })
    return rows


def unknown_pitch_events(midi_path, offset_seconds, unknown_pitches):
    """列出原始 MIDI 未映射音高的校正音訊時間，供逐點聽辨。"""
    clock = 0.0
    rows = []
    for message in mido.MidiFile(midi_path):
        clock += message.time
        if (
            message.type == "note_on"
            and message.velocity > 0
            and int(message.note) in unknown_pitches
        ):
            rows.append({
                "time": float(clock + offset_seconds),
                "pitch": int(message.note),
                "velocity": int(message.velocity),
            })
    return rows


def clip_bounds(event_time, duration):
    """將兩秒 review clip 夾限在來源音訊邊界內。"""
    start = max(0.0, float(event_time) - CLIP_PRE_SECONDS)
    end = min(float(duration), float(event_time) + CLIP_POST_SECONDS)
    return start, end


def write_clip(source_path, output_path, event_time):
    """從來源音訊串流讀取兩秒片段並寫成 PCM WAV，不載入或改寫整首音訊。"""
    with sf.SoundFile(str(source_path)) as source:
        duration = len(source) / float(source.samplerate)
        start, end = clip_bounds(event_time, duration)
        start_frame = int(round(start * source.samplerate))
        end_frame = int(round(end * source.samplerate))
        source.seek(start_frame)
        audio = source.read(end_frame - start_frame, always_2d=True, dtype="float32")
        sf.write(str(output_path), audio, source.samplerate, subtype="PCM_16")
    return start, end, source.samplerate, len(audio)


def build_review_rows(manifest_path, d100_report):
    """由 D100 問題建立事件級與整首級 review rows，所有人工決定保持空白。"""
    manifest_items = {
        item["id"]: item
        for item in json.loads(manifest_path.read_text(encoding="utf-8"))["items"]
    }
    audit_items = {item["id"]: item for item in d100_report["songs"]}
    rows = []
    for song_id in sorted(audit_items):
        audit = audit_items[song_id]
        item = manifest_items[song_id]
        audio_path = (manifest_path.parent / item["audio_path"]).resolve()
        midi_path = (manifest_path.parent / item["reference_midi"]).resolve()
        event_path = (manifest_path.parent / item["reference_events_csv"]).resolve()
        events = load_events(event_path)

        if audit["low_transient_support_classes"]:
            onsets = audio_onset_times(audio_path)
            for label in audit["low_transient_support_classes"]:
                for event in select_representative(unsupported_events(events, label, onsets)):
                    rows.append({
                        "song_id": song_id,
                        "issue_type": "low_transient_support",
                        "target_inst": label,
                        "event_time": event["time"],
                        "source_audio_path": str(audio_path),
                        "reference_detail": (
                            f"pitch={event['pitch']}; nearest_audio_onset_distance="
                            f"{event['nearest_audio_onset_distance_seconds']}"
                        ),
                        "question": f"此時間是否真的聽到 {label}？",
                    })

        for example in audit["exact_duplicate_examples"]:
            rows.append({
                "song_id": song_id,
                "issue_type": "exact_duplicate_reference",
                "target_inst": example["inst"],
                "event_time": float(example["time"]),
                "source_audio_path": str(audio_path),
                "reference_detail": (
                    f"pitch={example['pitch']}; velocity={example['velocity']}; "
                    f"duplicate_count={example['count']}"
                ),
                "question": "此時間應保留一個事件，還是確實需要兩個完全相同事件？",
            })

        unknown_pitches = {int(pitch) for pitch in audit["unknown_pitches"]}
        for event in unknown_pitch_events(
            midi_path, float(item["reference_offset_sec"]), unknown_pitches
        ):
            rows.append({
                "song_id": song_id,
                "issue_type": "unknown_midi_pitch",
                "target_inst": "",
                "event_time": event["time"],
                "source_audio_path": str(audio_path),
                "reference_detail": f"pitch={event['pitch']}; velocity={event['velocity']}",
                "question": "此聲音屬於 TOM／CRASH／RIDE／其他，還是應忽略？",
            })

        for label in audit["missing_classes"]:
            rows.append({
                "song_id": song_id,
                "issue_type": "missing_class_whole_song",
                "target_inst": label,
                "event_time": None,
                "source_audio_path": str(audio_path),
                "reference_detail": f"D100 reference count for {label}=0",
                "question": f"整首音訊是否完全沒有 {label}？",
            })
    return sorted(
        rows,
        key=lambda row: (
            row["song_id"],
            row["issue_type"],
            -1.0 if row["event_time"] is None else row["event_time"],
        ),
    )


def write_outputs(output_dir, rows):
    """建立全新 D101 clips、manifest 與摘要，拒絕覆寫既有 review 判定。"""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite D101 output: {output_dir}")
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True)

    manifest_rows = []
    for index, row in enumerate(rows, start=1):
        review_id = f"D101_{index:03d}"
        output = {
            "review_id": review_id,
            **row,
            "clip_start": None,
            "clip_end": None,
            "clip_path": "",
            "clip_sample_rate": None,
            "clip_frames": None,
            "status": "pending_human_review",
            "user_decision": "",
        }
        if row["event_time"] is not None:
            clip_path = clips_dir / f"{review_id}_{row['song_id']}.wav"
            start, end, sample_rate, frames = write_clip(
                row["source_audio_path"], clip_path, row["event_time"]
            )
            output.update({
                "clip_start": start,
                "clip_end": end,
                "clip_path": str(clip_path.resolve()),
                "clip_sample_rate": sample_rate,
                "clip_frames": frames,
            })
        manifest_rows.append(output)

    fields = tuple(manifest_rows[0])
    with (output_dir / "review_manifest.csv").open(
        "x", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    (output_dir / "review_manifest.json").write_text(
        json.dumps(manifest_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "phase": "D101",
        "status": "pending_human_review",
        "review_items": len(manifest_rows),
        "clips": sum(bool(row["clip_path"]) for row in manifest_rows),
        "whole_song_review_items": sum(not row["clip_path"] for row in manifest_rows),
        "pending_decisions": sum(not row["user_decision"] for row in manifest_rows),
        "source_files_modified": False,
        "training_started": False,
        "ready_for_reference_correction": False,
        "ready_for_training_candidate": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def run_self_check():
    """驗證代表抽樣與音訊邊界夾限，避免產生越界或偏單一區段 clips。"""
    rows = [{"time": float(index)} for index in range(10)]
    assert [row["time"] for row in select_representative(rows)] == [0.0, 4.0, 9.0]
    assert clip_bounds(0.25, 10.0) == (0.0, 1.5)
    assert clip_bounds(9.5, 10.0) == (8.75, 10.0)
    print("D101 self-check passed.")


def main():
    """D101 CLI 入口。"""
    parser = argparse.ArgumentParser(description="Prepare D101 human reference review clips.")
    parser.add_argument("--manifest", default="real-song/d93_intake/manifest.json")
    parser.add_argument(
        "--d100-report",
        default="validation_runs/d100_real_song_data_audit_final/audit_d100.json",
    )
    parser.add_argument(
        "--output-dir", default="validation_runs/d101_reference_review_clips"
    )
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    manifest_path = Path(args.manifest).resolve()
    report = json.loads(Path(args.d100_report).read_text(encoding="utf-8"))
    rows = build_review_rows(manifest_path, report)
    summary = write_outputs(args.output_dir, rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
