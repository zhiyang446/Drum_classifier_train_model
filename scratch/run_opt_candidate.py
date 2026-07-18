# -*- coding: utf-8 -*-
"""
Evaluate the combined 5-class optimized thresholds (A_opt) with KD maintained at 0.50.
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

# Load model & cache predictions
print("Loading model and caching validation...")
model, transferred = create_model("dcnn-tcn-conformer", MODEL_PATH, device)
model.eval()

metadata = json.load(open(META_PATH, encoding='utf-8'))
selected_windows = select_windows(metadata, split='validation', per_class=8)
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

def evaluate_validation(thresholds_dict):
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

# Run configurations
t_dict = {"KD": 0.50, "SD": 0.60, "HH": 0.60, "TOM": 0.60, "CRASH": 0.45, "RIDE": 0.55}

print("\n--- Evaluating A_opt Combined Config ---")
val_macro, val_f1s = evaluate_validation(t_dict)
print(f"Validation Macro F1: {val_macro:.4f}")
print("Per-class F1s:")
for label in LABELS:
    print(f"  {label}: {val_f1s[label]:.4f}")

# Run Round 4 validation
r4_out_dir = "validation_runs/ablation_study/A_opt_round4"
r4_cmd = [
    sys.executable, "run_egmd_round4_validation.py",
    "--model", MODEL_PATH,
    "--output-dir", r4_out_dir,
    "--thresh-kick", str(t_dict["KD"]),
    "--thresh-snare", str(t_dict["SD"]),
    "--thresh-hihat", str(t_dict["HH"]),
    "--thresh-tom", str(t_dict["TOM"]),
    "--thresh-crash", str(t_dict["CRASH"]),
    "--thresh-ride", str(t_dict["RIDE"]),
    "--architecture", "dcnn-tcn-conformer",
    "--limit", "5"
]
print(f"Running Round 4 subprocess...")
subprocess.run(r4_cmd, check=True)

gate_summary_path = os.path.join(r4_out_dir, "gate_summary.json")
with open(gate_summary_path, "r", encoding="utf-8") as f:
    gate_summary = json.load(f)
r4_pass = f"{gate_summary['passed_rows']}/{gate_summary['total_rows']}"
print(f"Round 4 Pass rate: {r4_pass}")

# Run Blind Tests
blind_out_dir = "validation_runs/ablation_study/A_opt_blind"
blind_cmd = [
    sys.executable, "run_blind_test.py",
    "--input", "blind_user_tests",
    "--model", MODEL_PATH,
    "--output-dir", blind_out_dir,
    "--thresh-kick", str(t_dict["KD"]),
    "--thresh-snare", str(t_dict["SD"]),
    "--thresh-hihat", str(t_dict["HH"]),
    "--thresh-tom", str(t_dict["TOM"]),
    "--thresh-crash", str(t_dict["CRASH"]),
    "--thresh-ride", str(t_dict["RIDE"]),
    "--architecture", "dcnn-tcn-conformer"
]
print(f"Running Blind test subprocess...")
subprocess.run(blind_cmd, check=True)

blind_summary_path = os.path.join(blind_out_dir, "summary.json")
with open(blind_summary_path, "r", encoding="utf-8") as f:
    blind_rows = json.load(f)

total_kick = sum(int(row.get('raw_kick', 0)) for row in blind_rows)
total_snare = sum(int(row.get('raw_snare', 0)) for row in blind_rows)
total_hihat = sum(int(row.get('raw_hihat', 0)) for row in blind_rows)
blind_events_str = f"KD:{total_kick}/SD:{total_snare}/HH:{total_hihat}"
print(f"Blind Prediction counts: {blind_events_str}")
