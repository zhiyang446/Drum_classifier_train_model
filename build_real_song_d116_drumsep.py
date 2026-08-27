# -*- coding: utf-8 -*-
"""D116：為五首 D103 真歌建立不可覆寫的 MDX23C DrumSep 六 Stem 資料。"""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import librosa
import soundfile as sf

from build_d52_drumsep_batch import STEMS, file_sha256


# 中文註解：所有 D116 產物都集中在新目錄，絕不覆寫 D103/D104。
ROOT = Path(__file__).resolve().parent
SOURCE_MANIFEST = ROOT / "real-song" / "d103_corrected_reference" / "manifest.json"
D48_AUDIT = ROOT / "drumsep_d48" / "audit_d48.json"
OUTPUT_ROOT = ROOT / "real-song" / "d116_drumsep"
INPUT_ROOT = OUTPUT_ROOT / "input"
STEM_ROOT = OUTPUT_ROOT / "output"
PLAN_PATH = OUTPUT_ROOT / "plan_d116.json"
PREFLIGHT_PATH = OUTPUT_ROOT / "preflight_d116.json"
AUDIT_PATH = OUTPUT_ROOT / "audit_d116.json"
MANIFEST_PATH = OUTPUT_ROOT / "manifest.json"
MANIFEST_AUDIT_PATH = OUTPUT_ROOT / "manifest_audit_d116.json"
MIN_FREE_GIB = 8.0
DURATION_TOLERANCE_SECONDS = 0.25


def read_json(path):
    """讀取 UTF-8 JSON，所有來源與產物均由呼叫端明確指定。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new_json(path, payload):
    """只寫入尚不存在的 JSON，拒絕覆寫任何 D116 證據。"""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite D116 output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def source_sha256(path):
    """計算 manifest 雜湊，固定 D116 使用的 D103 source 版本。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source_items():
    """讀取並驗證剛好五首、group 隔離的 D103 音訊/MIDI/reference 組。"""
    manifest_path = SOURCE_MANIFEST.resolve()
    payload = read_json(manifest_path)
    items = payload.get("items", [])
    if len(items) != 5 or len({item.get("group_id") for item in items}) != 5:
        raise ValueError("D116 requires exactly five unique D103 song groups.")
    rows = []
    for item in sorted(items, key=lambda row: row["id"]):
        audio_path = (manifest_path.parent / item["audio_path"]).resolve()
        midi_path = (manifest_path.parent / item["reference_midi"]).resolve()
        events_path = (manifest_path.parent / item["reference_events_csv"]).resolve()
        if not audio_path.is_file() or not midi_path.is_file() or not events_path.is_file():
            raise FileNotFoundError(f"Missing D103 source for {item['id']}")
        info = sf.info(str(audio_path))
        rows.append({
            "id": item["id"],
            "group_id": item["group_id"],
            "split": item["split"],
            "audio_path": str(audio_path),
            "reference_midi": str(midi_path),
            "reference_events_csv": str(events_path),
            "reference_offset_sec": float(item["reference_offset_sec"]),
            "duration_seconds": info.frames / float(info.samplerate),
            "input_name": item["id"],
            "input_extension": audio_path.suffix.lower(),
        })
    return rows


def d48_model():
    """鎖定已驗證 D48 的 MDX23C checkpoint/config，拒絕配方漂移。"""
    audit = read_json(D48_AUDIT)
    model = audit["model"]
    checkpoint = (ROOT / model["checkpoint"]).resolve()
    config = (ROOT / model["config"]).resolve()
    if not checkpoint.is_file() or not config.is_file():
        raise FileNotFoundError("D48 MDX23C checkpoint or config is missing.")
    if file_sha256(checkpoint) != model["checkpoint_sha256"]:
        raise ValueError("D48 checkpoint hash differs; refusing D116 inference.")
    # D48 稽核只封存 checkpoint 雜湊；D116 另行記錄目前設定檔雜湊以保留可重現性。
    config_sha256 = file_sha256(config)
    return {
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": model["checkpoint_sha256"],
        "config": str(config.relative_to(ROOT)),
        "config_sha256": config_sha256,
        "source_revision": model["source_revision"],
        "tta": False,
        "lora": False,
        "inference_batch_size": 1,
    }


def build_plan():
    """建立五首 D116 的來源、模型、時間與輸出容量預估。"""
    rows = load_source_items()
    d48 = read_json(D48_AUDIT)
    total_seconds = sum(row["duration_seconds"] for row in rows)
    density = int(d48["output"]["total_bytes"]) / float(d48["source"]["selection"]["total_seconds"])
    estimated_bytes = int(round(total_seconds * density))
    return {
        "phase": "D116",
        "status": "preflight_complete_not_inference_started",
        "source_manifest": str(SOURCE_MANIFEST.relative_to(ROOT)),
        "source_manifest_sha256": source_sha256(SOURCE_MANIFEST),
        "model": d48_model(),
        "selection": {
            "tracks": len(rows),
            "unique_groups": len({row["group_id"] for row in rows}),
            "total_seconds": total_seconds,
            "training_started": False,
            "sealed_gate_read": False,
        },
        "expected_output": {
            "stems": list(STEMS),
            "stem_files": len(rows) * len(STEMS),
            "estimated_bytes_from_d48_density": estimated_bytes,
            "estimated_gib_from_d48_density": estimated_bytes / 1024 ** 3,
            "sample_rate": 44100,
            "channels": 2,
            "duration_tolerance_seconds": DURATION_TOLERANCE_SECONDS,
        },
        "entries": rows,
    }


def prepare(plan):
    """建立同磁碟硬連結輸入與不可變 preflight，尚不啟動分離推論。"""
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Refusing to reuse D116 output root: {OUTPUT_ROOT}")
    free_bytes = shutil.disk_usage(ROOT).free
    if free_bytes < int(MIN_FREE_GIB * 1024 ** 3):
        raise RuntimeError(f"Only {free_bytes / 1024 ** 3:.2f} GiB free; D116 needs {MIN_FREE_GIB:.0f} GiB.")
    if free_bytes < plan["expected_output"]["estimated_bytes_from_d48_density"]:
        raise RuntimeError("Estimated D116 stem output exceeds current free space.")
    OUTPUT_ROOT.mkdir(parents=True)
    INPUT_ROOT.mkdir()
    STEM_ROOT.mkdir()
    for row in plan["entries"]:
        source = Path(row["audio_path"])
        destination = INPUT_ROOT / f"{row['input_name']}{row['input_extension']}"
        os.link(source, destination)
        if not os.path.samefile(source, destination):
            raise RuntimeError(f"D116 input hard link verification failed: {destination}")
    preflight = {
        **plan,
        "free_gib_before_inference": free_bytes / 1024 ** 3,
        "input_hard_links": len(list(INPUT_ROOT.iterdir())),
    }
    if preflight["input_hard_links"] != len(plan["entries"]):
        raise RuntimeError("D116 hard-link count is incomplete.")
    write_new_json(PLAN_PATH, plan)
    write_new_json(PREFLIGHT_PATH, preflight)
    return preflight


def expected_stems(row):
    """回傳單首歌曲固定六 Stem 的官方輸出路徑。"""
    return {stem: STEM_ROOT / row["input_name"] / f"{stem}.wav" for stem in STEMS}


def audit_stems(plan):
    """檢查每首六 Stem 的檔案、格式與時長，拒絕不完整資料進入後續。"""
    incomplete, per_song, files = [], [], 0
    tolerance = float(plan["expected_output"]["duration_tolerance_seconds"])
    for row in plan["entries"]:
        paths = expected_stems(row)
        failures, durations = [], {}
        # 與 MDX23C inference.py 相同的解碼方式，避免 MP3 容器 metadata 誤判。
        decoded_mix, decoded_sr = librosa.load(row["audio_path"], sr=44100, mono=False)
        decoded_duration = decoded_mix.shape[-1] / float(decoded_sr)
        for stem, path in paths.items():
            if not path.is_file() or path.stat().st_size == 0:
                failures.append(f"missing_or_empty:{stem}")
                continue
            info = sf.info(str(path))
            duration = info.frames / float(info.samplerate)
            durations[stem] = duration
            if info.samplerate != 44100 or info.channels != 2:
                failures.append(f"format:{stem}={info.samplerate}Hz/{info.channels}ch")
            if abs(duration - decoded_duration) > tolerance:
                failures.append(f"duration:{stem}")
        if failures:
            incomplete.append(row["id"])
        else:
            files += len(STEMS)
        per_song.append({
            "id": row["id"],
            "manifest_duration_seconds": row["duration_seconds"],
            "decoded_input_duration_seconds": decoded_duration,
            "manifest_minus_decoded_seconds": float(row["duration_seconds"]) - decoded_duration,
            "stem_durations_seconds": durations,
            "failures": failures,
        })
    payload = {
        "phase": "D116",
        "status": "pass" if not incomplete else "incomplete",
        "expected_tracks": len(plan["entries"]),
        "complete_tracks": len(plan["entries"]) - len(incomplete),
        "incomplete_tracks": len(incomplete),
        "incomplete_ids": incomplete,
        "stem_files_verified": files,
        "per_song": per_song,
        "training_started": False,
        "ready_for_alignment_audit": not incomplete,
        "ready_for_training_candidate": False,
        "ready_for_release": False,
    }
    write_new_json(AUDIT_PATH, payload)
    if incomplete:
        raise AssertionError(f"D116 stem audit incomplete: {incomplete}")
    return payload


def build_manifest(plan):
    """以 D103 reference 原樣建立 D116 drumsep-mix manifest，不改事件時間。"""
    audit = read_json(AUDIT_PATH)
    if audit.get("status") != "pass":
        raise RuntimeError("D116 stems must pass audit before manifest creation.")
    source = read_json(SOURCE_MANIFEST)
    rows_by_id = {row["id"]: row for row in plan["entries"]}
    items = []
    for item in source["items"]:
        row = rows_by_id[item["id"]]
        paths = {stem: str(path.resolve()) for stem, path in expected_stems(row).items()}
        items.append({
            **item,
            "input_mode": "drumsep-mix",
            "drumsep_stems": {
                "version": "d116_mdx23c_d48_compatible",
                "mix_strategy": "sum_mono",
                "paths": paths,
            },
        })
    payload = {
        "phase": "D116",
        "status": "complete_not_training",
        "source_manifest": str(SOURCE_MANIFEST.relative_to(ROOT)),
        "source_manifest_sha256": source_sha256(SOURCE_MANIFEST),
        "model": plan["model"],
        "items": items,
    }
    group_count = len({item["group_id"] for item in items})
    if len(items) != 5 or group_count != 5:
        raise AssertionError("D116 manifest group isolation failed.")
    write_new_json(MANIFEST_PATH, payload)
    write_new_json(MANIFEST_AUDIT_PATH, {
        "phase": "D116",
        "status": "pass",
        "items": len(items),
        "unique_groups": group_count,
        "input_mode": "drumsep-mix",
        "event_times_modified": False,
        "training_started": False,
        "ready_for_alignment_audit": True,
        "ready_for_training_candidate": False,
    })
    return payload


def run_self_check():
    """驗證固定六 Stem、時間容差與輸出名稱不會意外改變。"""
    assert tuple(STEMS) == ("kick", "snare", "toms", "hh", "ride", "crash")
    assert DURATION_TOLERANCE_SECONDS == 0.25
    assert OUTPUT_ROOT.name == "d116_drumsep"
    print("D116 self-check passed.")


def main():
    """提供 D116 的 prepare、audit、manifest 與最小自檢入口。"""
    parser = argparse.ArgumentParser(description="Build audited D116 DrumSep stems for five D103 songs.")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--build-manifest", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    if sum((args.prepare, args.audit, args.build_manifest)) != 1:
        parser.error("Choose exactly one action.")
    plan = build_plan() if args.prepare else read_json(PLAN_PATH)
    if args.prepare:
        result = prepare(plan)
    elif args.audit:
        result = audit_stems(plan)
    else:
        result = build_manifest(plan)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
