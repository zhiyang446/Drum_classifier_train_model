# -*- coding: utf-8 -*-
"""D110：稽核 ENST 全歌曲覆蓋排程、音訊對齊、target 與 D89 梯度路徑。"""

import argparse
import csv
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from align_whack_metal_d29 import onset_envelope
from audit_real_song_d100 import correlation_offset, event_impulses, transient_support
from run_six_class_smoke import CHUNK_FRAMES, HOP_LENGTH, LABEL_INDEX, LABELS, SR, build_window
from train_d77_fused_lora import fused_logits, load_frozen_lora_model, load_parent_adapter
from train_six_class_candidate import batch_from_schedule, build_schedule, gaussian_smooth_targets

# 中文註解：D110B 只在多項證據共同支持時把相關峰視為真實時間錯位。
OFFSET_LIMIT_SECONDS = 0.15
EXPANDED_SEARCH_SECONDS = 1.0
SEARCH_STABILITY_SECONDS = 0.05
ALIAS_SUPPORT_RATE = 0.70
MIN_SHIFT_SUPPORT_GAIN = 0.05


def read_json(path):
    """中文註解：讀取固定 metadata 或既有報告。"""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def assign_tracks_to_labels(available, per_class):
    """中文註解：以容量化二分匹配讓每首歌先取得一個正樣本類別。"""
    slots = [(label, index) for index in range(per_class) for label in LABELS]
    slot_to_track = {}

    def match(track, seen):
        """中文註解：尋找或重排一個可用類別 slot。"""
        for slot in slots:
            if slot[0] not in available[track] or slot in seen:
                continue
            seen.add(slot)
            if slot not in slot_to_track or match(slot_to_track[slot], seen):
                slot_to_track[slot] = track
                return True
        return False

    for track in sorted(available, key=lambda key: (len(available[key]), key)):
        if not match(track, set()):
            raise ValueError(f"Cannot assign full-coverage label slot for {track}")
    return {track: slot[0] for slot, track in slot_to_track.items()}


def evenly_spaced(rows, count):
    """中文註解：固定均勻抽取剩餘候選，不加入隨機狀態。"""
    return [rows[index * len(rows) // count] for index in range(count)]


def build_full_coverage_schedule(metadata, old_schedule, per_class=24):
    """中文註解：保持逐類配額與總窗口不變，先覆蓋所有歌曲再補足類別。"""
    candidates = {label: [] for label in LABELS}
    available = {}
    for key, item in metadata.items():
        by_label = {
            label: sorted({float(event["time"]) for event in item["events"] if event.get("inst") == label})
            for label in LABELS
        }
        available[key] = {label for label, times in by_label.items() if times}
        for label, times in by_label.items():
            candidates[label].extend((key, time) for time in times)

    positive_available = {key: labels for key, labels in available.items() if labels}
    negative_only = sorted(set(available) - set(positive_available))
    assignment = assign_tracks_to_labels(positive_available, per_class)
    selected = {label: [] for label in LABELS}
    for key, label in sorted(assignment.items()):
        times = sorted(
            float(event["time"]) for event in metadata[key]["events"] if event.get("inst") == label
        )
        selected[label].append((key, times[len(times) // 2]))

    for label in LABELS:
        used = set(selected[label])
        remaining = sorted(set(candidates[label]) - used)
        needed = per_class - len(selected[label])
        if needed < 0 or len(remaining) < needed:
            raise ValueError(f"Cannot fill {label} to {per_class} windows")
        selected[label].extend(evenly_spaced(remaining, needed))
        selected[label].sort()

    negatives = [{
        "label": "NEG",
        "key": key,
        "anchor": float(sf.info(metadata[key]["audio_path"]).duration / 2.0),
    } for key in negative_only]
    old_negatives = [row for row in old_schedule if row["label"] == "NEG"]
    negatives.extend(dict(row) for row in old_negatives[:per_class - len(negatives)])
    if len(negatives) != per_class:
        raise ValueError(f"Expected {per_class} combined NEG rows, got {len(negatives)}")
    rows = []
    for index in range(per_class):
        rows.extend(
            {"label": label, "key": selected[label][index][0], "anchor": selected[label][index][1]}
            for label in LABELS
        )
        rows.append(dict(negatives[index]))
    return rows


def audit_windows(schedule, metadata):
    """中文註解：逐窗口驗證實際 clamp 後的 anchor target 與特徵有限性。"""
    failures = []
    edge_clamped = 0
    for row in schedule:
        item = metadata[row["key"]]
        features, onsets, _, start_sec = build_window(
            item,
            row["anchor"],
            use_true_superflux=True,
            input_mode=item.get("input_mode", "mix"),
        )
        if not np.isfinite(features).all() or not np.isfinite(onsets).all():
            failures.append(f'non_finite:{row["key"]}:{row["anchor"]}')
        if row["label"] == "NEG":
            if onsets[:, [LABEL_INDEX[label] for label in ("TOM", "CRASH", "RIDE")]].sum() != 0:
                failures.append(f'rare_in_neg:{row["key"]}:{row["anchor"]}')
            continue
        frame = int(round((row["anchor"] - start_sec) * SR / HOP_LENGTH))
        if not 0 <= frame < CHUNK_FRAMES or onsets[frame, LABEL_INDEX[row["label"]]] != 1.0:
            failures.append(f'anchor_target_missing:{row["key"]}:{row["label"]}:{row["anchor"]}')
        if abs((row["anchor"] - start_sec) - 2.0) > HOP_LENGTH / SR:
            edge_clamped += 1
    return {"windows": len(schedule), "edge_clamped_windows": edge_clamped, "failures": failures}


def classify_offset(original_offset, expanded_offset, local_span, support_rate, shifted_support_rate):
    """中文註解：以搜尋穩定性、局部一致性與平移收益裁決相關峰。"""
    if abs(original_offset) <= OFFSET_LIMIT_SECONDS:
        return "within_tolerance"
    search_unstable = (
        expanded_offset is not None
        and abs(expanded_offset - original_offset) > SEARCH_STABILITY_SECONDS
    )
    local_unstable = local_span is not None and local_span > OFFSET_LIMIT_SECONDS
    support_gain = shifted_support_rate - support_rate
    if (
        support_rate >= ALIAS_SUPPORT_RATE
        and support_gain < MIN_SHIFT_SUPPORT_GAIN
        and (search_unstable or local_unstable)
    ):
        return "periodic_correlation_alias"
    return "correction_required"


def adjudicate_offset(envelope, events, alignment, support_rate):
    """中文註解：只評估候選平移，不修改來源事件或建立校正 metadata。"""
    original_offset = None if alignment is None else float(alignment["offset_seconds"])
    result = {
        "expanded_offset_seconds": None,
        "search_offset_delta_seconds": None,
        "local_offset_span_seconds": None,
        "shifted_transient_support_rate": None,
        "shift_support_delta": None,
        "offset_gate_status": "alignment_unavailable" if original_offset is None else "within_tolerance",
        "correction_applied": False,
    }
    if original_offset is None or abs(original_offset) <= OFFSET_LIMIT_SECONDS:
        return result

    pulses = event_impulses(events, len(envelope))
    expanded = correlation_offset(envelope, pulses, max_offset_seconds=EXPANDED_SEARCH_SECONDS)
    local_offsets = []
    for index in range(3):
        start = index * len(envelope) // 3
        end = (index + 1) * len(envelope) // 3
        local = correlation_offset(
            envelope[start:end],
            pulses[start:end],
            max_offset_seconds=EXPANDED_SEARCH_SECONDS,
        )
        if local is not None:
            local_offsets.append(float(local["offset_seconds"]))
    local_span = max(local_offsets) - min(local_offsets) if len(local_offsets) >= 2 else None

    shifted_events = [
        {**event, "time": float(event["time"]) + original_offset}
        for event in events
    ]
    assert len(shifted_events) == len(events)
    _, shifted_support, _ = transient_support(shifted_events, envelope)
    expanded_offset = None if expanded is None else float(expanded["offset_seconds"])
    status = classify_offset(
        original_offset,
        expanded_offset,
        local_span,
        support_rate,
        float(shifted_support),
    )
    result.update({
        "expanded_offset_seconds": expanded_offset,
        "search_offset_delta_seconds": (
            None if expanded_offset is None else expanded_offset - original_offset
        ),
        "local_offset_span_seconds": local_span,
        "shifted_transient_support_rate": float(shifted_support),
        "shift_support_delta": float(shifted_support - support_rate),
        "offset_gate_status": status,
    })
    return result


def audit_alignments(metadata):
    """中文註解：重用 D100 onset 方法量測 97 首音訊與標註的殘餘位移及瞬態支持。"""
    rows = []
    for key, item in sorted(metadata.items()):
        envelope = onset_envelope(Path(item["audio_path"]))
        if not item["events"]:
            rows.append({
                "key": key,
                "group_id": item["group_id"],
                "alignment_applicable": False,
                "offset_seconds": None,
                "alignment_score": None,
                "transient_support_rate": None,
                "detected_audio_onsets": None,
                "onset_envelope_max": float(np.max(envelope)),
                "expanded_offset_seconds": None,
                "search_offset_delta_seconds": None,
                "local_offset_span_seconds": None,
                "shifted_transient_support_rate": None,
                "shift_support_delta": None,
                "offset_gate_status": "not_applicable",
                "correction_applied": False,
            })
            continue
        pulses = event_impulses(item["events"], len(envelope))
        alignment = correlation_offset(envelope, pulses)
        _, support_rate, detected = transient_support(item["events"], envelope)
        adjudication = adjudicate_offset(envelope, item["events"], alignment, support_rate)
        rows.append({
            "key": key,
            "group_id": item["group_id"],
            "alignment_applicable": True,
            "offset_seconds": None if alignment is None else float(alignment["offset_seconds"]),
            "alignment_score": None if alignment is None else float(alignment["score"]),
            "transient_support_rate": float(support_rate),
            "detected_audio_onsets": int(detected),
            "onset_envelope_max": float(np.max(envelope)),
            **adjudication,
        })
    return rows


def gradient_smoke(schedule, metadata, args):
    """中文註解：只做一次 D89 forward/backward，確認兩個 LoRA 分支可收梯度但不更新權重。"""
    smoke_rows = [next(row for row in schedule if row["label"] == label) for label in LABELS]
    features, onsets, _ = batch_from_schedule(
        smoke_rows, metadata, 0, len(smoke_rows), use_true_superflux=True, input_mode="mix",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d76_model = load_frozen_lora_model(args.d76_checkpoint, device, args.rank, args.alpha)
    d64_model = load_frozen_lora_model(args.d64_checkpoint, device, args.rank, args.alpha)
    load_parent_adapter(args.d89_adapter, args, d76_model, d64_model)
    feature_tensor = torch.from_numpy(features).float().to(device)
    target_tensor = gaussian_smooth_targets(torch.from_numpy(onsets).float().to(device))
    loss = F.binary_cross_entropy_with_logits(
        fused_logits(d76_model, d64_model, feature_tensor), target_tensor,
    )
    loss.backward()

    norms = {}
    for name, model in (("d76", d76_model), ("d64", d64_model)):
        gradients = [
            parameter.grad.detach()
            for parameter in model.onset_head.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        norms[name] = float(math.sqrt(sum(float((gradient ** 2).sum().cpu()) for gradient in gradients)))
    result = {
        "device": str(device),
        "batch_rows": smoke_rows,
        "loss": float(loss.detach().cpu()),
        "gradient_norms": norms,
        "gradients_finite": all(math.isfinite(value) and value > 0.0 for value in norms.values()),
        "optimizer_created": False,
        "optimizer_step": False,
        "checkpoint_written": False,
    }
    del d76_model, d64_model, feature_tensor, target_tensor, loss
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def write_outputs(output_dir, summary, schedule, alignments):
    """中文註解：只新建 D110 JSON／CSV 證據，拒絕覆寫。"""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite D110 output: {output}")
    output.mkdir(parents=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    (output / "proposed_schedule.json").write_text(
        json.dumps(schedule, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    (output / "offset_adjudication.json").write_text(
        json.dumps(
            [row for row in alignments if row["offset_gate_status"] not in ("within_tolerance", "not_applicable")],
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    with (output / "track_alignment.csv").open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=alignments[0].keys())
        writer.writeheader()
        writer.writerows(alignments)


def run_self_check():
    """中文註解：驗證歌曲優先匹配能保持逐類配額並覆蓋全部歌曲。"""
    metadata = {
        "sd_only": {"events": [{"inst": "SD", "time": 1.0}]},
        "tom_only": {"events": [{"inst": "TOM", "time": 1.0}]},
        "ride_only": {"events": [{"inst": "RIDE", "time": 1.0}]},
        "mixed_a": {"events": [{"inst": label, "time": 2.0} for label in LABELS]},
        "mixed_b": {"events": [{"inst": label, "time": 3.0} for label in LABELS]},
    }
    old = [{"label": "NEG", "key": "mixed_a", "anchor": float(index)} for index in range(2)]
    schedule = build_full_coverage_schedule(metadata, old, per_class=2)
    counts = Counter(row["label"] for row in schedule)
    assert len(schedule) == 14 and counts == Counter({label: 2 for label in [*LABELS, "NEG"]})
    assert set(metadata) <= {row["key"] for row in schedule if row["label"] != "NEG"}
    assert classify_offset(0.10, 0.10, 0.0, 0.90, 0.90) == "within_tolerance"
    assert classify_offset(-0.40, 0.60, 0.50, 0.80, 0.79) == "periodic_correlation_alias"
    assert classify_offset(-0.40, -0.40, 0.02, 0.40, 0.70) == "correction_required"
    print("D110 self-check passed.")


def main():
    """中文註解：執行 D110 唯讀根因稽核並判定是否允許另立 D111。"""
    parser = argparse.ArgumentParser(description="Audit ENST full-coverage continual-training path.")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--metadata", default="enst_d107/metadata_d107_train.json")
    parser.add_argument("--d89-adapter", default="validation_runs/d89_d82_tim_gm_lora_retry/d89_d82_tim_gm_lora_retry_adapter.pth")
    parser.add_argument("--d76-checkpoint", default="validation_runs/d76_crash_kd_retry_candidate/d76_crash_kd_retry_candidate.pth")
    parser.add_argument("--d64-checkpoint", default="validation_runs/d64_tom_competitor_candidate/d64_tom_competitor_candidate.pth")
    parser.add_argument("--output-dir", default="validation_runs/d110_enst_training_path_audit")
    parser.add_argument("--phase", choices=("D110", "D110B"), default="D110")
    parser.add_argument("--per-class", type=int, default=24)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=8.0)
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    if args.per_class != 24:
        raise ValueError("D110 locks per-class to 24.")
    for path in (args.metadata, args.d89_adapter, args.d76_checkpoint, args.d64_checkpoint):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    if os.path.exists(args.output_dir):
        raise FileExistsError(f"Refusing to overwrite D110 output: {args.output_dir}")

    metadata = read_json(args.metadata)
    if len(metadata) != 97 or any(item.get("split") != "train" for item in metadata.values()):
        raise ValueError("D110 requires exactly 97 train-only ENST tracks.")
    if any("drummer_1" not in f'{key} {item.get("audio_path", "")}'.lower() for key, item in metadata.items()):
        raise ValueError("D110 accepts drummer_1 train paths only.")

    old_schedule = build_schedule(metadata, args.per_class, window_negative_from_train=True)
    proposed = build_full_coverage_schedule(metadata, old_schedule, args.per_class)
    old_counts = Counter(row["label"] for row in old_schedule)
    proposed_counts = Counter(row["label"] for row in proposed)
    old_tracks = {row["key"] for row in old_schedule}
    proposed_tracks = {row["key"] for row in proposed}
    window_audit = audit_windows(proposed, metadata)
    alignments = audit_alignments(metadata)
    gradient = gradient_smoke(proposed, metadata, args)

    positive_alignments = [row for row in alignments if row["alignment_applicable"]]
    negative_alignments = [row for row in alignments if not row["alignment_applicable"]]
    unavailable = [row["key"] for row in positive_alignments if row["offset_seconds"] is None]
    offset_failures = [row["key"] for row in alignments if row["offset_seconds"] is not None and abs(row["offset_seconds"]) > OFFSET_LIMIT_SECONDS]
    periodic_aliases = [row["key"] for row in positive_alignments if row["offset_gate_status"] == "periodic_correlation_alias"]
    correction_required = [row["key"] for row in positive_alignments if row["offset_gate_status"] == "correction_required"]
    support_failures = [row["key"] for row in positive_alignments if row["transient_support_rate"] < 0.50]
    silent_negatives = [row["key"] for row in negative_alignments if row["onset_envelope_max"] <= 0.0]
    offsets = [abs(row["offset_seconds"]) for row in positive_alignments if row["offset_seconds"] is not None]
    blockers = []
    if len(proposed) != 168 or proposed_counts != Counter({label: 24 for label in [*LABELS, "NEG"]}):
        blockers.append("schedule_shape")
    if len(proposed_tracks) != len(metadata):
        blockers.append("track_coverage")
    if window_audit["failures"]:
        blockers.append("window_target_or_feature")
    if unavailable or correction_required:
        blockers.append("audio_reference_offset")
    if support_failures:
        blockers.append("transient_support")
    if silent_negatives:
        blockers.append("silent_negative_audio")
    if not gradient["gradients_finite"]:
        blockers.append("gradient_path")

    summary = {
        "phase": args.phase,
        "status": "ready_for_d111" if not blockers else "blocked",
        "training_started": False,
        "optimizer_step": False,
        "checkpoint_written": False,
        "sealed_validation_or_test_read": False,
        "metadata_tracks": len(metadata),
        "old_schedule": {
            "windows": len(old_schedule),
            "per_label": dict(old_counts),
            "unique_tracks": len(old_tracks),
        },
        "proposed_schedule": {
            "windows": len(proposed),
            "per_label": dict(proposed_counts),
            "unique_tracks": len(proposed_tracks),
            "newly_covered_tracks": sorted(proposed_tracks - old_tracks),
        },
        "window_audit": window_audit,
        "alignment": {
            "tracks": len(alignments),
            "positive_alignment_tracks": len(positive_alignments),
            "six_class_negative_tracks": len(negative_alignments),
            "unavailable": unavailable,
            "offset_failures_over_0_15s": offset_failures,
            "periodic_correlation_aliases": periodic_aliases,
            "correction_required": correction_required,
            "corrections_applied": False,
            "support_failures_under_0_50": support_failures,
            "silent_negative_audio": silent_negatives,
            "median_abs_offset_seconds": float(np.median(offsets)) if offsets else None,
            "p95_abs_offset_seconds": float(np.percentile(offsets, 95)) if offsets else None,
            "mean_transient_support_rate": float(np.mean([row["transient_support_rate"] for row in positive_alignments])),
        },
        "gradient_smoke": gradient,
        "blockers": blockers,
        "ready_for_d111": not blockers,
    }
    write_outputs(args.output_dir, summary, proposed, alignments)
    print(json.dumps({
        "status": summary["status"],
        "old_unique_tracks": len(old_tracks),
        "proposed_unique_tracks": len(proposed_tracks),
        "blockers": blockers,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
