# -*- coding: utf-8 -*-
"""
Ablation Study for Phase D13: 7 configurations (A0 to A6).
Runs validation cache, then invokes Round 4 and Blind Tests validation via subprocess.
"""
import os
import sys
import json
import subprocess
import numpy as np
import torch

# Fix import path
sys.path.append(os.getcwd())

from train_six_class_candidate import create_model
from run_six_class_validation import select_windows, expected_events
from run_six_class_smoke import build_window, load_accompaniment, CHUNK_FRAMES, HOP_LENGTH, LABELS, LABEL_INDEX, SR, TARGET_SAMPLES
from run_egmd_round4_validation import match_events

# Paths
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

# 2. Load validation windows and cache predictions
print("Loading metadata...")
metadata = json.load(open(META_PATH, encoding='utf-8'))
selected_windows = select_windows(metadata, split='validation', per_class=8)
print(f"Selected {len(selected_windows)} windows for validation.")

print("Caching predictions...")
accompaniment = load_accompaniment(ACCOMPANIMENT_PATH)
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

# Peak picking with class-specific thresholds
def get_peaks_per_class(probs, thresholds):
    events = {label: [] for label in LABELS}
    for label, index in LABEL_INDEX.items():
        thresh = thresholds[label]
        values = probs[:, index]
        for frame in range(1, len(values) - 1):
            if values[frame] >= thresh and values[frame] >= values[frame - 1] and values[frame] > values[frame + 1]:
                events[label].append(frame * HOP_LENGTH / float(SR))
    return events

# Evaluate F1 for a given set of thresholds
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

# 7 Configurations definition
configs = {
    "A0": {"KD": 0.50, "SD": 0.50, "HH": 0.50, "TOM": 0.50, "CRASH": 0.50, "RIDE": 0.50},
    "A1": {"KD": 0.60, "SD": 0.50, "HH": 0.50, "TOM": 0.50, "CRASH": 0.50, "RIDE": 0.50},
    "A2": {"KD": 0.50, "SD": 0.60, "HH": 0.50, "TOM": 0.50, "CRASH": 0.50, "RIDE": 0.50},
    "A3": {"KD": 0.50, "SD": 0.50, "HH": 0.60, "TOM": 0.50, "CRASH": 0.50, "RIDE": 0.50},
    "A4": {"KD": 0.50, "SD": 0.50, "HH": 0.50, "TOM": 0.60, "CRASH": 0.50, "RIDE": 0.50},
    "A5": {"KD": 0.50, "SD": 0.50, "HH": 0.50, "TOM": 0.50, "CRASH": 0.45, "RIDE": 0.50},
    "A6": {"KD": 0.50, "SD": 0.50, "HH": 0.50, "TOM": 0.50, "CRASH": 0.50, "RIDE": 0.55},
}

results = []

for name, t_dict in configs.items():
    print(f"\n================ Running Configuration: {name} ================")
    # 1. Validation F1
    val_macro, val_f1s = evaluate_validation(t_dict)
    f1_str = f"KD:{val_f1s['KD']:.4f}/SD:{val_f1s['SD']:.4f}/HH:{val_f1s['HH']:.4f}/TOM:{val_f1s['TOM']:.4f}/CRASH:{val_f1s['CRASH']:.4f}/RIDE:{val_f1s['RIDE']:.4f}"
    print(f"[{name} Val] Macro F1: {val_macro:.4f}")

    # 2. Run Round 4 validation via subprocess
    r4_out_dir = f"validation_runs/ablation_study/{name}_round4"
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
    print(f"Running Round 4 subprocess: {' '.join(r4_cmd)}")
    subprocess.run(r4_cmd, check=True)

    # Read Round 4 pass/30
    gate_summary_path = os.path.join(r4_out_dir, "gate_summary.json")
    with open(gate_summary_path, "r", encoding="utf-8") as f:
        gate_summary = json.load(f)
    r4_pass = f"{gate_summary['passed_rows']}/{gate_summary['total_rows']}"
    print(f"[{name} R4] Pass: {r4_pass}")

    # 3. Run Blind Tests via subprocess
    blind_out_dir = f"validation_runs/ablation_study/{name}_blind"
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
    print(f"Running Blind test subprocess: {' '.join(blind_cmd)}")
    subprocess.run(blind_cmd, check=True)

    # Read Blind prediction events count
    blind_summary_path = os.path.join(blind_out_dir, "summary.json")
    with open(blind_summary_path, "r", encoding="utf-8") as f:
        blind_rows = json.load(f)

    total_kick = sum(int(row.get('raw_kick', 0)) for row in blind_rows)
    total_snare = sum(int(row.get('raw_snare', 0)) for row in blind_rows)
    total_hihat = sum(int(row.get('raw_hihat', 0)) for row in blind_rows)
    blind_events_str = f"KD:{total_kick}/SD:{total_snare}/HH:{total_hihat}"
    print(f"[{name} Blind] Total physical event count: {blind_events_str}")

    results.append({
        "Config": name,
        "F1s": f1_str,
        "Macro F1": f"{val_macro:.4f}",
        "Round 4 Pass": r4_pass,
        "Blind Events": blind_events_str
    })

# 4. Output summary report table
print("\n=================== Ablation Study Summary Matrix ===================")
header = f"{'Config':<8} | {'Per-class F1 (KD/SD/HH/TOM/CRASH/RIDE)':<45} | {'Macro F1':<8} | {'Round 4 Pass':<12} | {'Blind Events (KD/SD/HH)':<25}"
print(header)
print("-" * len(header))
for res in results:
    print(f"{res['Config']:<8} | {res['F1s']:<45} | {res['Macro F1']:<8} | {res['Round 4 Pass']:<12} | {res['Blind Events']:<25}")
print("=====================================================================")
