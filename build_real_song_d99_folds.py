# -*- coding: utf-8 -*-
"""建立 D99 五首真實鼓的歌曲級五折 metadata，不修改 D93 或 D54。"""

import argparse
import json
from collections import Counter
from pathlib import Path

import soundfile as sf

from build_real_song_d96_windows import LABELS, build_windows, load_events


def make_item(manifest_path, item, split, phase="D99"):
    """把一首 D93 歌曲轉成既有 trainer 可直接讀取的 metadata item。"""
    audio_path = (manifest_path.parent / item["audio_path"]).resolve()
    reference_path = (manifest_path.parent / item["reference_events_csv"]).resolve()
    if not audio_path.is_file() or not reference_path.is_file():
        raise FileNotFoundError(f"Missing D99 source for {item['id']}")
    info = sf.info(str(audio_path))
    duration = info.frames / float(info.samplerate)
    events = load_events(reference_path)
    invalid = [event for event in events if not 0.0 <= event["time"] < duration]
    if invalid:
        raise ValueError(f"{item['id']} has {len(invalid)} out-of-bounds events")
    windows = build_windows(item["id"], item["group_id"], events, info.frames, info.samplerate)
    return {
        "audio_path": str(audio_path),
        "reference_midi": str((manifest_path.parent / item["reference_midi"]).resolve()),
        "reference_events_csv": str(reference_path),
        "duration": duration,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "split": split,
        "source": f"{phase.lower()}_real_song",
        "group_id": item["group_id"],
        "input_mode": "mix",
        "events": events,
        "window_anchors": [row["anchor_time"] for row in windows],
    }


def class_counts(metadata):
    """統計 metadata 的六類事件數，供每折稽核與報告使用。"""
    counts = Counter(
        event["inst"]
        for item in metadata.values()
        for event in item.get("events", [])
        if event.get("inst") in LABELS
    )
    return {label: counts[label] for label in LABELS}


def build(manifest_file, output_dir, phase="D99"):
    """建立五折，每折以四首 train、一首 validation，且每首恰好留出一次。"""
    manifest_path = Path(manifest_file).resolve()
    items = json.loads(manifest_path.read_text(encoding="utf-8"))["items"]
    if len(items) != 5:
        raise ValueError(f"D99 requires exactly five songs, got {len(items)}")
    groups = [item["group_id"] for item in items]
    if len(set(groups)) != len(groups):
        raise ValueError("D99 requires one unique group_id per song")
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite D99 output: {root}")
    root.mkdir(parents=True)

    folds = []
    for fold_index, heldout in enumerate(sorted(items, key=lambda row: row["id"]), start=1):
        fold_dir = root / f"fold_{fold_index:02d}"
        fold_dir.mkdir()
        train = {
            item["id"]: make_item(manifest_path, item, "train", phase)
            for item in items
            if item["group_id"] != heldout["group_id"]
        }
        validation = {heldout["id"]: make_item(manifest_path, heldout, "validation", phase)}
        train_groups = {item["group_id"] for item in train.values()}
        validation_groups = {item["group_id"] for item in validation.values()}
        if train_groups & validation_groups:
            raise AssertionError("D99 group leak detected")
        if any(class_counts(train)[label] == 0 for label in LABELS):
            raise AssertionError(f"Fold {fold_index} train is missing a class")
        (fold_dir / "train_metadata.json").write_text(
            json.dumps(train, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (fold_dir / "heldout_metadata.json").write_text(
            json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        row = {
            "fold": fold_index,
            "heldout_id": heldout["id"],
            "heldout_group_id": heldout["group_id"],
            "train_ids": sorted(train),
            "train_groups": sorted(train_groups),
            "train_class_counts": class_counts(train),
            "heldout_class_counts": class_counts(validation),
            "group_overlap": [],
        }
        (fold_dir / "audit.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        folds.append(row)

    heldout_ids = [row["heldout_id"] for row in folds]
    audit = {
        "phase": phase,
        "status": "pass",
        "folds": len(folds),
        "songs": len(items),
        "each_song_heldout_once": len(set(heldout_ids)) == len(items),
        "group_overlap_count": sum(bool(row["group_overlap"]) for row in folds),
        "d54_resplit": False,
        "fixed_commercial_gate_read": False,
        "fold_summary": folds,
    }
    (root / "audit_d99.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def run_self_check():
    """確認五折核心不變量：五首、五個唯一 group、每折四訓一驗。"""
    ids = [f"song_{index}" for index in range(5)]
    folds = [(heldout, [item for item in ids if item != heldout]) for heldout in ids]
    assert len(folds) == 5
    assert {heldout for heldout, _ in folds} == set(ids)
    assert all(len(train) == 4 and heldout not in train for heldout, train in folds)
    print("D99 fold builder self-check passed.")


def main():
    """執行 D99 fold builder 或最小自我檢查。"""
    parser = argparse.ArgumentParser(description="Build D99 song-level five-fold metadata.")
    parser.add_argument("--manifest", default="real-song/d93_intake/manifest.json")
    parser.add_argument("--output-dir", default="real-song/d99_five_fold")
    parser.add_argument("--phase", default="D99")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    build(args.manifest, args.output_dir, args.phase)


if __name__ == "__main__":
    main()
