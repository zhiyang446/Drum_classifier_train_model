# -*- coding: utf-8 -*-
"""
D7 Per-class Threshold Coordinate Ascent Search.
Only runs inference on D7 model once, caches probabilities,
then searches for optimal class-specific thresholds on validation set.
"""
import os
import json
import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import torch

from train_six_class_candidate import create_model
from run_six_class_validation import select_windows, expected_events
from run_six_class_smoke import build_window, load_accompaniment, CHUNK_FRAMES, HOP_LENGTH, LABELS, LABEL_INDEX, SR, TARGET_SAMPLES
from run_egmd_round4_validation import match_events

# Parameters
MODEL_PATH = "validation_runs/six_class_candidate_d7_d4d_earlystop20/six_class_candidate_d7_best.pth"
META_PATH = "processed_data/star_egmd_six_class_d4d.json"
ACCOMPANIMENT_PATH = r"C:\Users\zhiya\Documents\MyProject\Drum_classifier_train_model\accompaniment\queen_no_drums.wav"
ACCOMPANIMENT_GAIN = 0.17
TOLERANCE = 0.050

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. Load D7 Model
print("Loading D7 checkpoint...")
model, transferred = create_model("dcnn-tcn-conformer", MODEL_PATH, device)
model.eval()

# 2. Load metadata & select validation windows (per_class=8, total 48 windows)
print("Loading metadata...")
metadata = json.load(open(META_PATH, encoding='utf-8'))
selected_windows = select_windows(metadata, split='validation', per_class=8)
print(f"Selected {len(selected_windows)} windows for validation.")

# 3. Load accompaniment
accompaniment = load_accompaniment(ACCOMPANIMENT_PATH)

# 4. Run inference once and cache probabilities
print("Caching predictions...")
cached_data = []
window_seconds = CHUNK_FRAMES * HOP_LENGTH / float(SR)

for window_index, selected in enumerate(selected_windows):
    accompaniment_offset = window_index * TARGET_SAMPLES
    features, _, _, start_sec = build_window(
        selected['item'], selected['anchor'], accompaniment=accompaniment,
        accompaniment_gain=ACCOMPANIMENT_GAIN, accompaniment_offset=accompaniment_offset,
        use_true_superflux=False, use_multi_log_mel=False, # D7 uses legacy-diff and no multi-mel
    )
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(features).float().unsqueeze(0).to(device))
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy() # [Time, 6]

    expected = expected_events(selected['item'], start_sec)
    aggregate_offset = window_index * (window_seconds + 1.0)

    cached_data.append({
        'probs': probs,
        'expected': expected,
        'aggregate_offset': aggregate_offset
    })

print("Caching complete!")

# 5. Peak picking function with class-specific thresholds
def get_peaks_per_class(probs, thresholds):
    events = {label: [] for label in LABELS}
    for label, index in LABEL_INDEX.items():
        thresh = thresholds[label]
        values = probs[:, index]
        for frame in range(1, len(values) - 1):
            if values[frame] >= thresh and values[frame] >= values[frame - 1] and values[frame] > values[frame + 1]:
                events[label].append(frame * HOP_LENGTH / float(SR))
    return events

# 6. Evaluate F1 for a given set of thresholds
def evaluate_thresholds(thresholds_dict):
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

# 7. Baseline check (all thresholds = 0.50)
baseline_thresholds = {label: 0.50 for label in LABELS}
baseline_macro, baseline_f1s = evaluate_thresholds(baseline_thresholds)
print(f"\n--- Baseline (All 0.50) ---")
print(f"Macro F1: {baseline_macro:.4f}")
print("Per-class F1s:")
for label in LABELS:
    print(f"  {label}: {baseline_f1s[label]:.4f}")

# 8. Coordinate Ascent Search
current_thresholds = {label: 0.50 for label in LABELS}
search_values = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

print("\nStarting Coordinate Ascent Search...")
best_macro = baseline_macro
best_f1s = baseline_f1s.copy()

for iteration in range(3):
    print(f"\n--- Iteration {iteration + 1} ---")
    changed = False
    for label in LABELS:
        best_t_for_class = current_thresholds[label]

        for t in search_values:
            temp_thresholds = current_thresholds.copy()
            temp_thresholds[label] = t

            macro, f1s = evaluate_thresholds(temp_thresholds)

            # Optimization condition:
            # We want to maximize overall macro F1,
            # BUT we strictly require that KD F1 must not degrade (must be >= baseline KD F1 - epsilon, say 0.001)
            # and HH must not degrade significantly.
            if macro > best_macro:
                if f1s['KD'] >= baseline_f1s['KD'] - 0.0001:
                    best_macro = macro
                    best_f1s = f1s.copy()
                    best_t_for_class = t
                    changed = True

        if current_thresholds[label] != best_t_for_class:
            print(f"  Class {label}: threshold {current_thresholds[label]} -> {best_t_for_class} (New Macro: {best_macro:.4f}, KD F1: {best_f1s['KD']:.4f})")
            current_thresholds[label] = best_t_for_class

    if not changed:
        print("Search converged early.")
        break

print("\n--- Search Complete ---")
print("Best Thresholds:")
for label in LABELS:
    print(f"  {label}: {current_thresholds[label]:.2f}")
print(f"Optimized Macro F1: {best_macro:.4f} (Baseline: {baseline_macro:.4f}, diff: +{best_macro - baseline_macro:.4f})")
print("Optimized Per-class F1s:")
for label in LABELS:
    print(f"  {label}: {best_f1s[label]:.4f} (Baseline: {baseline_f1s[label]:.4f}, diff: {best_f1s[label] - baseline_f1s[label]:+.4f})")
