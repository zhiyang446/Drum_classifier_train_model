# -*- coding: utf-8 -*-
"""
Release Validation: Tests the A_opt threshold configuration against A0 (baseline)
on unseen STAR test split data and expanded E-GMD Round 4 dataset.
"""
import os
import sys
import json
import subprocess
import numpy as np
import torch

sys.path.append(os.getcwd())

from train_six_class_candidate import create_model
from run_six_class_validation import select_windows, expected_events
from run_six_class_smoke import build_window, load_accompaniment, CHUNK_FRAMES, HOP_LENGTH, LABELS, LABEL_INDEX, SR, TARGET_SAMPLES
from run_egmd_round4_validation import match_events

MODEL_PATH = "validation_runs/six_class_candidate_d7_d4d_earlystop20/six_class_candidate_d7_best.pth"
META_PATH = "processed_data/star_egmd_six_class_d4d.json"
ACCOMPANIMENT_PATH = r"C:\Users\zhiya\Documents\MyProject\Drum_classifier_train_model\accompaniment\queen_no_drums.wav"
ACCOMPANIMENT_GAIN = 0.17
TOLERANCE = 0.050

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. Cache predictions on STAR "test" split (completely unseen)
print("Loading model and caching STAR test split predictions...")
model, transferred = create_model("dcnn-tcn-conformer", MODEL_PATH, device)
model.eval()

metadata = json.load(open(META_PATH, encoding='utf-8'))
# Load split='test' instead of 'validation'! This is completely unseen.
selected_windows = select_windows(metadata, split='test', per_class=8)
print(f"Selected {len(selected_windows)} windows for STAR test split validation.")

accompaniment = load_accompaniment(ACCOMPANIMENT_PATH)
cached_data = []
window_seconds = CHUNK_FRAMES * HOP_LENGTH / float(SR)

for window_index, selected in enumerate(selected_windows):
    accompaniment_offset = window_index * TARGET_SAMPLES
    features, _, _, start_sec = build_window(
        selected['item'], selected['anchor'], accompaniment=accompaniment,
        accompaniment_gain=ACCOMPANIMENT_GAIN, accompaniment_offset=accompaniment_offset,
        use_true_superflux=False, use_multi_log_mel=False,
    )
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(features).float().unsqueeze(0).to(device))
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    
    expected = expected_events(selected['item'], start_sec)
    aggregate_offset = window_index * (window_seconds + 1.0)
    
    cached_data.append({
        'probs': probs,
        'expected': expected,
        'aggregate_offset': aggregate_offset
    })

def get_peaks_per_class(probs, thresholds):
    events = {label: [] for label in LABELS}
    for label, index in LABEL_INDEX.items():
        thresh = thresholds[label]
        values = probs[:, index]
        for frame in range(1, len(values) - 1):
            if values[frame] >= thresh and values[frame] >= values[frame - 1] and values[frame] > values[frame + 1]:
                events[label].append(frame * HOP_LENGTH / float(SR))
    return events

def evaluate_test_split(thresholds_dict):
    aggregate = {label: ([], []) for label in LABELS}
    for item in cached_data:
        probs = item['probs']
        expected = item['expected']
        aggregate_offset = item['aggregate_offset']
        
        predicted = get_peaks_per_class(probs, thresholds_dict)
        for label in LABELS:
            aggregate[label][0].extend(time + aggregate_offset for time in expected[label])
            aggregate[label][1].extend(time + aggregate_offset for time in predicted[label])
            
    f1_dict = {}
    for label in LABELS:
        expected_list, predicted_list = aggregate[label]
        _, _, _, _, _, f1 = match_events(expected_list, predicted_list, TOLERANCE)
        f1_dict[label] = f1
    macro_f1 = np.mean(list(f1_dict.values()))
    return macro_f1, f1_dict

# Load config
t_a0 = {"KD": 0.50, "SD": 0.50, "HH": 0.50, "TOM": 0.50, "CRASH": 0.50, "RIDE": 0.50}
t_opt = {"KD": 0.50, "SD": 0.60, "HH": 0.60, "TOM": 0.60, "CRASH": 0.45, "RIDE": 0.55}

# Part 1: Evaluate STAR test split
print("\n--- [Part 1] Evaluating STAR test split (Unseen) ---")
a0_macro, a0_f1s = evaluate_test_split(t_a0)
opt_macro, opt_f1s = evaluate_test_split(t_opt)

print(f"A0 (Baseline) Macro F1: {a0_macro:.4f}")
print(f"A_opt (Optimized) Macro F1: {opt_macro:.4f} (diff: {opt_macro - a0_macro:+.4f})")
print("Per-class F1 comparisons (A0 vs A_opt):")
for label in LABELS:
    print(f"  {label}: {a0_f1s[label]:.4f} -> {opt_f1s[label]:.4f} ({opt_f1s[label] - a0_f1s[label]:+.4f})")

# Part 2: Evaluate Expanded E-GMD Round 4 validation (offset=0, limit=6)
print("\n--- [Part 2] Evaluating Expanded Round 4 (limit=6, 36 strong hit checks) ---")

def run_r4(name, t_dict):
    out_dir = f"validation_runs/release_validation/{name}_round4"
    cmd = [
        sys.executable, "run_egmd_round4_validation.py",
        "--model", MODEL_PATH,
        "--output-dir", out_dir,
        "--thresh-kick", str(t_dict["KD"]),
        "--thresh-snare", str(t_dict["SD"]),
        "--thresh-hihat", str(t_dict["HH"]),
        "--thresh-tom", str(t_dict["TOM"]),
        "--thresh-crash", str(t_dict["CRASH"]),
        "--thresh-ride", str(t_dict["RIDE"]),
        "--architecture", "dcnn-tcn-conformer",
        "--limit", "6" # E-GMD test set only has 6 valid 4/4 cases
    ]
    subprocess.run(cmd, check=True)
    with open(os.path.join(out_dir, "gate_summary.json"), "r", encoding="utf-8") as f:
        gate_summary = json.load(f)
    return f"{gate_summary['passed_rows']}/{gate_summary['total_rows']}"

print("Running Round 4 A0 baseline...")
a0_r4_pass = run_r4("A0", t_a0)
print("Running Round 4 A_opt candidate...")
opt_r4_pass = run_r4("A_opt", t_opt)

print(f"Round 4 (6 tracks) A0 Baseline Pass: {a0_r4_pass}")
print(f"Round 4 (6 tracks) A_opt Candidate Pass: {opt_r4_pass}")

# Final validation report check
print("\n================== Release Validation Summary ==================")
print(f"STAR test split Macro F1: {a0_macro:.4f} -> {opt_macro:.4f} (PASSED)" if opt_macro > a0_macro else "FAILED")
print(f"STAR test split KD F1: {a0_f1s['KD']:.4f} -> {opt_f1s['KD']:.4f} (PASSED)" if opt_f1s['KD'] >= a0_f1s['KD'] - 0.0001 else "FAILED")
print(f"Round 4 strong hit rate: {a0_r4_pass} -> {opt_r4_pass} (PASSED)" if int(opt_r4_pass.split('/')[0]) >= int(a0_r4_pass.split('/')[0]) else "FAILED")
print("================================================================")
