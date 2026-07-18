# -*- coding: utf-8 -*-
"""
Independent Shadow Run: Direct MIDI-to-MIDI F1 validation.
Evaluates A_opt against A0 on unseen E-GMD tracks by comparing generated
MIDI files with ground-truth MIDI files.
"""
import os
import sys
import pretty_midi
import numpy as np

# Onset matching algorithm identical to evaluation rules
def match_onsets(gt_times, pred_times, tolerance=0.050):
    gt = sorted(gt_times)
    pred = sorted(pred_times)
    
    tps = 0
    matched_gt = set()
    
    for p in pred:
        best_match = -1
        min_dist = tolerance
        for idx, g in enumerate(gt):
            if idx in matched_gt:
                continue
            dist = abs(p - g)
            if dist < min_dist:
                min_dist = dist
                best_match = idx
        if best_match != -1:
            tps += 1
            matched_gt.add(best_match)
            
    fps = len(pred) - tps
    fns = len(gt) - len(matched_gt)
    
    precision = tps / len(pred) if len(pred) > 0 else 0.0
    recall = tps / len(gt) if len(gt) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1, len(gt), len(pred)

# Load MIDI note onset times by instrument pitch
def load_midi_onsets(midi_path):
    pm = pretty_midi.PrettyMIDI(midi_path)
    pitch_map = {
        "Kick": [35, 36],
        "Snare": [38],
        "Hi-Hat": [42, 44, 46]
    }
    onsets = {k: [] for k in pitch_map}
    for instrument in pm.instruments:
        for note in instrument.notes:
            for k, pitches in pitch_map.items():
                if note.pitch in pitches:
                    onsets[k].append(note.start)
                    break
    return onsets

# Run paths
MODEL_PATH = "validation_runs/six_class_candidate_d7_d4d_earlystop20/six_class_candidate_d7_best.pth"
TEST_TRACKS = [
    {
        "wav": "e-gmd-v1.0.0/drummer5/eval_session/2_funk-groove2_105_beat_4-4_1.wav",
        "midi": "e-gmd-v1.0.0/drummer5/eval_session/2_funk-groove2_105_beat_4-4_1.midi"
    },
    {
        "wav": "e-gmd-v1.0.0/drummer1/eval_session/9_soul-groove9_105_beat_4-4_1.wav",
        "midi": "e-gmd-v1.0.0/drummer1/eval_session/9_soul-groove9_105_beat_4-4_1.midi"
    }
]

import subprocess

results = []

for track in TEST_TRACKS:
    wav_path = track["wav"]
    gt_midi_path = track["midi"]
    song_name = os.path.basename(wav_path)
    
    print(f"\n================== Processing {song_name} ==================")
    
    # 1. Generate A_opt MIDI
    opt_midi = f"validation_runs/shadow_run/A_opt_{song_name.replace('.wav', '.mid')}"
    cmd_opt = [
        sys.executable, "transcribe.py",
        "--model", MODEL_PATH,
        "--input", wav_path,
        "--output", opt_midi,
        "--architecture", "dcnn-tcn-conformer"
    ]
    print(f"Running A_opt transcription...")
    subprocess.run(cmd_opt, check=True, stdout=subprocess.DEVNULL)
    
    # 2. Generate A0 (Rollback) MIDI
    a0_midi = f"validation_runs/shadow_run/A0_{song_name.replace('.wav', '.mid')}"
    cmd_a0 = [
        sys.executable, "transcribe.py",
        "--model", MODEL_PATH,
        "--input", wav_path,
        "--output", a0_midi,
        "--architecture", "dcnn-tcn-conformer",
        "--rollback-baseline"
    ]
    print(f"Running A0 transcription...")
    subprocess.run(cmd_a0, check=True, stdout=subprocess.DEVNULL)
    
    # 3. Load onsets & Match
    onsets_gt = load_midi_onsets(gt_midi_path)
    onsets_opt = load_midi_onsets(opt_midi)
    onsets_a0 = load_midi_onsets(a0_midi)
    
    res_track = {"track": song_name, "A0": {}, "A_opt": {}}
    
    for inst in ["Kick", "Snare", "Hi-Hat"]:
        p_a0, r_a0, f_a0, gt_c, pred_a0_c = match_onsets(onsets_gt[inst], onsets_a0[inst])
        p_opt, r_opt, f_opt, _, pred_opt_c = match_onsets(onsets_gt[inst], onsets_opt[inst])
        
        res_track["A0"][inst] = {"Precision": p_a0, "Recall": r_a0, "F1": f_a0, "GT": gt_c, "Pred": pred_a0_c}
        res_track["A_opt"][inst] = {"Precision": p_opt, "Recall": r_opt, "F1": f_opt, "GT": gt_c, "Pred": pred_opt_c}
        
    results.append(res_track)

# Output final report table
print("\n================== Shadow Run Evaluation Results ==================")
for res in results:
    print(f"\nTrack: {res['track']}")
    print(f"  {'Instrument':<12} | {'A0 F1 (GT/Pred)':<18} | {'A_opt F1 (GT/Pred)':<20} | {'Improvement':<12}")
    print("  " + "-"*65)
    a0 = res["A0"]
    opt = res["A_opt"]
    for inst in ["Kick", "Snare", "Hi-Hat"]:
        m_a0 = a0[inst]
        m_opt = opt[inst]
        diff = m_opt["F1"] - m_a0["F1"]
        print(f"  {inst:<12} | {m_a0['F1']:<6.2%} ({m_a0['GT']}/{m_a0['Pred']}) | {m_opt['F1']:<6.2%} ({m_opt['GT']}/{m_opt['Pred']}) | {diff:+.2%}")
print("===================================================================")
