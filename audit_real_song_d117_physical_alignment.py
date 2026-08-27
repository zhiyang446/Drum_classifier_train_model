# -*- coding: utf-8 -*-
"""D117：以 D116 分類 stem 建立五首真歌的高解析度對齊證據包，不修改 MIDI。"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import find_peaks

from build_real_song_d96_windows import LABELS, load_events


# 中文註解：D117 只量測對應鼓件 stem，絕不依結果回寫任何事件時間。
ROOT = Path(__file__).resolve().parent
SOURCE_MANIFEST = ROOT / "real-song" / "d103_corrected_reference" / "manifest.json"
D116_ROOT = ROOT / "real-song" / "d116_drumsep"
D116_MANIFEST = D116_ROOT / "manifest.json"
D116_AUDIT = D116_ROOT / "audit_d116.json"
OUTPUT_ROOT = ROOT / "real-song" / "d117_physical_alignment_audit"
SAMPLE_RATE = 44100
HOP_LENGTH = 64
FFT_SIZE = 1024
MATCH_RADIUS_SECONDS = 0.100
SHORT_SUPPORT_SECONDS = 0.025
MEDIUM_SUPPORT_SECONDS = 0.050
CLIP_SECONDS = 1.20
MAX_REVIEW_CLIPS = 30
LABEL_TO_STEM = {
    "KD": "kick", "SD": "snare", "HH": "hh", "TOM": "toms", "CRASH": "crash", "RIDE": "ride",
}


def sha256_file(path):
    """計算檔案雜湊，鎖定 D103 與 D116 的資料關係。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    """讀取 UTF-8 JSON 檔案。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_new(path, payload):
    """只寫入全新 JSON，保留每次稽核不可覆寫。"""
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite D117 output: {target}")
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def preflight():
    """驗證 D103 與 D116 的五首資料、雜湊及六 stem 完整性。"""
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Refusing to reuse D117 output root: {OUTPUT_ROOT}")
    source = read_json(SOURCE_MANIFEST)
    d116 = read_json(D116_MANIFEST)
    audit = read_json(D116_AUDIT)
    if audit.get("status") != "pass" or audit.get("stem_files_verified") != 30:
        raise ValueError("D116 stem audit is not a complete pass.")
    if d116.get("source_manifest_sha256") != sha256_file(SOURCE_MANIFEST):
        raise ValueError("D116 does not match the current D103 source manifest.")
    source_items = source["items"]
    stem_items = d116["items"]
    if len(source_items) != 5 or len(stem_items) != 5:
        raise ValueError("D117 requires exactly five songs.")
    source_by_group = {item["group_id"]: item for item in source_items}
    stem_by_group = {item["group_id"]: item for item in stem_items}
    if len(source_by_group) != 5 or set(source_by_group) != set(stem_by_group):
        raise ValueError("D103/D116 group isolation mismatch.")
    for item in stem_items:
        paths = item.get("drumsep_stems", {}).get("paths", {})
        if set(paths) != set(LABEL_TO_STEM.values()) or not all(Path(path).is_file() for path in paths.values()):
            raise ValueError(f"Incomplete D116 stem paths: {item['group_id']}")
    return source_by_group, stem_by_group


def stem_peak_times(path):
    """以 64-sample hop 取得單一鼓件 stem 的 onset 峰值時間與強度。"""
    waveform, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    if sample_rate != SAMPLE_RATE or waveform.shape[1] != 2:
        raise ValueError(f"D117 requires 44.1kHz stereo stem: {path}")
    mono = waveform.mean(axis=1)
    strength = librosa.onset.onset_strength(
        y=mono, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, n_fft=FFT_SIZE, center=True,
    )
    peaks, _ = find_peaks(strength, distance=max(1, int(round(0.008 * SAMPLE_RATE / HOP_LENGTH))))
    if not len(peaks):
        return mono, np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    times = librosa.frames_to_time(peaks, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, n_fft=FFT_SIZE)
    return mono, np.asarray(times, dtype=np.float64), np.asarray(strength[peaks], dtype=np.float64)


def nearest_peak(event_time, peak_times, peak_strengths):
    """搜尋指定 event 的最近同類 stem 峰值；超過 100ms 則明確標記無支援。"""
    if not len(peak_times):
        return None, None
    index = int(np.searchsorted(peak_times, event_time))
    candidates = [candidate for candidate in (index - 1, index) if 0 <= candidate < len(peak_times)]
    best = min(candidates, key=lambda candidate: abs(float(peak_times[candidate]) - event_time))
    delta = float(peak_times[best] - event_time)
    if abs(delta) > MATCH_RADIUS_SECONDS:
        return None, None
    return float(peak_times[best]), {"delta_seconds": delta, "strength": float(peak_strengths[best])}


def measure_song(source_item, stem_item):
    """量測一首歌的全部六類 events，回傳逐事件證據與可供 clip 的波形。"""
    event_path = (SOURCE_MANIFEST.parent / source_item["reference_events_csv"]).resolve()
    events = load_events(event_path)
    stem_paths = stem_item["drumsep_stems"]["paths"]
    stem_audio, peaks = {}, {}
    for label, stem_name in LABEL_TO_STEM.items():
        waveform, times, strengths = stem_peak_times(stem_paths[stem_name])
        stem_audio[stem_name] = waveform
        peaks[label] = (times, strengths)
    rows = []
    for event_index, event in enumerate(events):
        label = event["inst"]
        if label not in LABEL_TO_STEM:
            raise ValueError(f"Unexpected D103 event label: {label}")
        event_time = float(event["time"])
        peak_time, measurement = nearest_peak(event_time, *peaks[label])
        delta = None if measurement is None else measurement["delta_seconds"]
        rows.append({
            "id": source_item["id"], "group_id": source_item["group_id"], "split": source_item["split"],
            "event_index": event_index, "inst": label, "pitch": int(event["pitch"]),
            "velocity": int(event.get("velocity", 0)), "event_time_seconds": event_time,
            "nearest_peak_time_seconds": peak_time, "delta_seconds": delta,
            "abs_delta_seconds": None if delta is None else abs(delta),
            "peak_strength": None if measurement is None else measurement["strength"],
            "peak_within_25ms": bool(delta is not None and abs(delta) <= SHORT_SUPPORT_SECONDS),
            "peak_within_50ms": bool(delta is not None and abs(delta) <= MEDIUM_SUPPORT_SECONDS),
            "peak_within_100ms": measurement is not None,
        })
    return rows, stem_audio, stem_paths


def class_summary(rows):
    """彙總每歌每類的峰值鄰近率與誤差分位數，不把它當成真值成功率。"""
    summary = []
    for song_id in sorted({row["id"] for row in rows}):
        for label in LABELS:
            selected = [row for row in rows if row["id"] == song_id and row["inst"] == label]
            deltas = [row["abs_delta_seconds"] for row in selected if row["abs_delta_seconds"] is not None]
            summary.append({
                "id": song_id, "inst": label, "events": len(selected),
                "peak_within_25ms": sum(row["peak_within_25ms"] for row in selected),
                "peak_within_50ms": sum(row["peak_within_50ms"] for row in selected),
                "peak_within_100ms": sum(row["peak_within_100ms"] for row in selected),
                "median_abs_delta_seconds": None if not deltas else float(np.median(deltas)),
                "p90_abs_delta_seconds": None if not deltas else float(np.quantile(deltas, 0.90)),
            })
    return summary


def select_review_rows(rows):
    """每首每類先取一個最不確定事件，再補足到最多 30 個人工 review 樣本。"""
    def severity(row):
        return 1.0 if row["abs_delta_seconds"] is None else float(row["abs_delta_seconds"])
    selected, keys = [], set()
    for row in sorted(rows, key=lambda value: (-severity(value), value["id"], value["inst"], value["event_index"])):
        key = (row["id"], row["inst"])
        if key not in keys:
            selected.append(row)
            keys.add(key)
    for row in sorted(rows, key=lambda value: (-severity(value), value["id"], value["event_index"])):
        if len(selected) >= MAX_REVIEW_CLIPS:
            break
        if row not in selected:
            selected.append(row)
    return selected[:MAX_REVIEW_CLIPS]


def write_review_clip(destination, row, stem_audio):
    """輸出事件前後 1.2 秒的六 stem 加總浮點 WAV，供人工聆聽而非模型訓練。"""
    length = min(len(waveform) for waveform in stem_audio.values())
    center = int(round(row["event_time_seconds"] * SAMPLE_RATE))
    clip_samples = int(round(CLIP_SECONDS * SAMPLE_RATE))
    start = min(max(0, center - clip_samples // 2), max(0, length - clip_samples))
    mix = np.zeros(min(clip_samples, length), dtype=np.float32)
    for waveform in stem_audio.values():
        mix += waveform[start:start + len(mix)]
    sf.write(str(destination), mix, SAMPLE_RATE, subtype="FLOAT")
    return start / float(SAMPLE_RATE)


def write_csv(path, rows, fields):
    """建立 UTF-8-SIG CSV，方便人工在 Windows 試算表檢視。"""
    with Path(path).open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def audit():
    """執行 D117 高解析度量測並建立不可覆寫的人工 review 證據包。"""
    source_by_group, stem_by_group = preflight()
    OUTPUT_ROOT.mkdir(parents=True)
    clips_root = OUTPUT_ROOT / "review_clips"
    clips_root.mkdir()
    all_rows, audio_by_song, paths_by_song = [], {}, {}
    for group_id in sorted(source_by_group):
        rows, audio, paths = measure_song(source_by_group[group_id], stem_by_group[group_id])
        all_rows.extend(rows)
        audio_by_song[rows[0]["id"]] = audio
        paths_by_song[rows[0]["id"]] = paths
    review_rows = select_review_rows(all_rows)
    for review_number, row in enumerate(review_rows, start=1):
        clip_name = f"{review_number:02d}_{row['id']}_{row['inst']}_{row['event_index']:05d}.wav"
        row["review_clip"] = str((clips_root / clip_name).relative_to(OUTPUT_ROOT))
        row["clip_start_seconds"] = write_review_clip(clips_root / clip_name, row, audio_by_song[row["id"]])
        row["stem_path"] = paths_by_song[row["id"]][LABEL_TO_STEM[row["inst"]]]
    summary = class_summary(all_rows)
    report = {
        "phase": "D117",
        "status": "manual_review_required",
        "algorithm": "same-class D116 stem onset-strength local peaks; no offset correction",
        "measurement": {
            "sample_rate": SAMPLE_RATE, "hop_length_samples": HOP_LENGTH,
            "hop_seconds": HOP_LENGTH / SAMPLE_RATE, "fft_size": FFT_SIZE,
            "nearest_peak_radius_seconds": MATCH_RADIUS_SECONDS,
            "review_clip_seconds": CLIP_SECONDS, "review_clip_limit": MAX_REVIEW_CLIPS,
        },
        "songs": len(source_by_group), "events": len(all_rows), "stems": 30,
        "per_song_per_class": summary, "review_candidates": review_rows,
        "source_files_modified": False, "midi_events_modified": False,
        "training_started": False, "ready_for_alignment_decision": False,
        "ready_for_training_candidate": False, "ready_for_release": False,
    }
    fields = list(all_rows[0])
    write_csv(OUTPUT_ROOT / "events_d117.csv", all_rows, fields)
    review_fields = list(review_rows[0]) if review_rows else fields
    write_csv(OUTPUT_ROOT / "review_candidates_d117.csv", review_rows, review_fields)
    write_json_new(OUTPUT_ROOT / "audit_d117.json", report)
    print(json.dumps({
        "phase": "D117", "status": report["status"], "songs": report["songs"],
        "events": report["events"], "review_clips": len(review_rows),
    }, ensure_ascii=False, indent=2))


def run_self_check():
    """驗證最近峰值選擇與每歌每類 review 上限。"""
    peak_time, result = nearest_peak(1.0, np.array([0.98, 1.03]), np.array([2.0, 3.0]))
    assert peak_time == 0.98 and abs(result["delta_seconds"] + 0.02) < 1e-9
    peak_time, result = nearest_peak(1.0, np.array([1.2]), np.array([1.0]))
    assert peak_time is None and result is None
    rows = [
        {"id": "a", "inst": "KD", "event_index": 0, "abs_delta_seconds": 0.09},
        {"id": "a", "inst": "KD", "event_index": 1, "abs_delta_seconds": 0.01},
        {"id": "a", "inst": "SD", "event_index": 2, "abs_delta_seconds": None},
    ]
    assert [row["event_index"] for row in select_review_rows(rows)] == [2, 0, 1]
    print("D117 self-check passed.")


def main():
    """D117 CLI 入口。"""
    parser = argparse.ArgumentParser(description="Build D117 physical alignment review evidence.")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
    else:
        audit()


if __name__ == "__main__":
    main()
