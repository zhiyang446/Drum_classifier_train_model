# -*- coding: utf-8 -*-
"""比較 D89 parent 與五個 D99 fold adapter 的歌曲級留出結果。"""

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from run_egmd_round4_validation import match_events
from run_six_class_smoke import CHUNK_FRAMES, HOP_LENGTH, LABELS, SR, build_window
from run_six_class_validation import expected_events, local_maxima
from train_d77_fused_lora import fused_logits, load_frozen_lora_model, load_parent_adapter


TOLERANCE = 0.050
WINDOW_SECONDS = CHUNK_FRAMES * HOP_LENGTH / float(SR)


def load_models(adapter_path, args, device):
    """載入 frozen D76/D64 並嚴格套用一份完整 adapter 狀態。"""
    d76 = load_frozen_lora_model(args.d76_checkpoint, device, args.rank, args.alpha)
    d64 = load_frozen_lora_model(args.d64_checkpoint, device, args.rank, args.alpha)
    adapter_args = SimpleNamespace(
        d76_checkpoint=args.d76_checkpoint,
        d64_checkpoint=args.d64_checkpoint,
        rank=args.rank,
        alpha=args.alpha,
    )
    load_parent_adapter(adapter_path, adapter_args, d76, d64)
    return d76, d64


def evaluate_metadata(d76_model, d64_model, metadata):
    """以不重疊完整四秒窗口評估 metadata，回傳逐類真值與預測時間。"""
    device = next(d76_model.parameters()).device
    aggregate = {label: ([], []) for label in LABELS}
    selections = []
    window_index = 0
    with torch.no_grad():
        for key, item in sorted(metadata.items()):
            start = 0.0
            while start + WINDOW_SECONDS <= float(item["duration"]) + 1e-9:
                anchor = start + WINDOW_SECONDS / 2.0
                features, _, _, actual_start = build_window(
                    item, anchor, use_true_superflux=True, use_multi_log_mel=False, input_mode="mix"
                )
                tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)
                probabilities = torch.sigmoid(
                    fused_logits(d76_model, d64_model, tensor)
                ).squeeze(0).cpu().numpy()
                predicted = local_maxima(probabilities)
                expected = expected_events(item, actual_start)
                offset = window_index * (WINDOW_SECONDS + 1.0)
                for label in LABELS:
                    aggregate[label][0].extend(time + offset for time in expected[label])
                    aggregate[label][1].extend(time + offset for time in predicted[label])
                selections.append({"key": key, "anchor": anchor, "window_start": actual_start})
                window_index += 1
                start += WINDOW_SECONDS
    return aggregate, selections


def metrics_from_events(aggregate):
    """以既有 50ms 一對一 matcher 將事件集合轉成逐類可加總指標。"""
    rows = []
    for label in LABELS:
        expected, predicted = aggregate[label]
        tp, fp, fn, precision, recall, f1 = match_events(expected, predicted, TOLERANCE)
        rows.append({
            "inst": label,
            "expected": len(expected),
            "predicted": len(predicted),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    return rows


def aggregate_rows(fold_rows):
    """加總五折 TP／FP／FN，再由總數重算 precision、recall 與 F1。"""
    output = []
    for label in LABELS:
        selected = [row for rows in fold_rows for row in rows if row["inst"] == label]
        tp = sum(row["tp"] for row in selected)
        fp = sum(row["fp"] for row in selected)
        fn = sum(row["fn"] for row in selected)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        output.append({
            "inst": label,
            "expected": sum(row["expected"] for row in selected),
            "predicted": sum(row["predicted"] for row in selected),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    return output


def write_csv(path, rows):
    """以固定欄位寫出逐類結果，保留完整數值供後續報告重算。"""
    with Path(path).open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    """逐折比較 parent/candidate，最後寫出全五首合併指標與判定。"""
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite D99 evaluation: {output_dir}")
    output_dir.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parent_fold_rows, candidate_fold_rows, fold_reports = [], [], []

    for fold_index in range(1, 6):
        fold_name = f"fold_{fold_index:02d}"
        metadata_path = Path(args.fold_root) / fold_name / "heldout_metadata.json"
        candidate_path = Path(args.training_root) / fold_name / args.candidate_name
        train_report_path = Path(args.training_root) / fold_name / "train_report.json"
        if not metadata_path.is_file() or not candidate_path.is_file() or not train_report_path.is_file():
            raise FileNotFoundError(f"Missing D99 fold input: {fold_name}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        train_report = json.loads(train_report_path.read_text(encoding="utf-8"))

        d76, d64 = load_models(args.parent_adapter, args, device)
        parent_events, selections = evaluate_metadata(d76, d64, metadata)
        parent_rows = metrics_from_events(parent_events)
        del d76, d64
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        d76, d64 = load_models(str(candidate_path), args, device)
        candidate_events, candidate_selections = evaluate_metadata(d76, d64, metadata)
        candidate_rows = metrics_from_events(candidate_events)
        if selections != candidate_selections:
            raise AssertionError(f"Window mismatch in {fold_name}")
        del d76, d64
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        parent_fold_rows.append(parent_rows)
        candidate_fold_rows.append(candidate_rows)
        fold_dir = output_dir / fold_name
        fold_dir.mkdir()
        write_csv(fold_dir / "parent.csv", parent_rows)
        write_csv(fold_dir / "candidate.csv", candidate_rows)
        (fold_dir / "selected_windows.json").write_text(
            json.dumps(selections, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        fold_reports.append({
            "fold": fold_index,
            "heldout_id": next(iter(metadata)),
            "windows": len(selections),
            "parent_macro_f1": float(np.mean([row["f1"] for row in parent_rows])),
            "candidate_macro_f1": float(np.mean([row["f1"] for row in candidate_rows])),
            "fixed_gate_promotes_parent": bool(train_report["epochs"][0]["promotes_parent"]),
        })

    parent = aggregate_rows(parent_fold_rows)
    candidate = aggregate_rows(candidate_fold_rows)
    write_csv(output_dir / "parent_aggregate.csv", parent)
    write_csv(output_dir / "candidate_aggregate.csv", candidate)
    parent_macro = float(np.mean([row["f1"] for row in parent]))
    candidate_macro = float(np.mean([row["f1"] for row in candidate]))
    heldout_no_class_regression = all(
        candidate_row["f1"] >= parent_row["f1"]
        for parent_row, candidate_row in zip(parent, candidate)
    )
    all_d56_parent_gates_pass = all(row["fixed_gate_promotes_parent"] for row in fold_reports)
    research_candidate = (
        candidate_macro > parent_macro
        and heldout_no_class_regression
        and all_d56_parent_gates_pass
    )
    summary = {
        "phase": args.phase,
        "status": "research_candidate" if research_candidate else "rejected",
        "device": str(device),
        "threshold": 0.50,
        "tolerance_seconds": TOLERANCE,
        "window_seconds": WINDOW_SECONDS,
        "parent_macro_f1": parent_macro,
        "candidate_macro_f1": candidate_macro,
        "delta": candidate_macro - parent_macro,
        "heldout_macro_improved": candidate_macro > parent_macro,
        "heldout_no_class_regression": heldout_no_class_regression,
        "all_d56_parent_gates_pass": all_d56_parent_gates_pass,
        "folds": fold_reports,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def run_self_check():
    """確認跨折加總會由 TP／FP／FN重算，而不是平均不相容的 F1。"""
    rows = [[{
        "inst": label, "expected": 1, "predicted": 1,
        "tp": 1, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0,
    } for label in LABELS]]
    aggregate = aggregate_rows(rows)
    assert all(row["f1"] == 1.0 for row in aggregate)
    print("D99 evaluator self-check passed.")


def main():
    """D99 五折留出評估 CLI。"""
    parser = argparse.ArgumentParser(description="Evaluate D99 five-fold held-out songs.")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--fold-root", default="real-song/d99_five_fold")
    parser.add_argument("--training-root", default="validation_runs/d99_five_fold")
    parser.add_argument("--output-dir", default="validation_runs/d99_five_fold_evaluation")
    parser.add_argument("--candidate-name", default="d99_fold_adapter_epoch1.pth")
    parser.add_argument("--phase", default="D99")
    parser.add_argument("--parent-adapter", default="validation_runs/d89_d82_tim_gm_lora_retry/d89_d82_tim_gm_lora_retry_adapter.pth")
    parser.add_argument("--d76-checkpoint", default="validation_runs/d76_crash_kd_retry_candidate/d76_crash_kd_retry_candidate.pth")
    parser.add_argument("--d64-checkpoint", default="validation_runs/d64_tom_competitor_candidate/d64_tom_competitor_candidate.pth")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=8.0)
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    run(args)


if __name__ == "__main__":
    main()
