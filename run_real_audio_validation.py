# -*- coding: utf-8 -*-
"""
Real-audio validation runner for evaluating six-class ADT candidates on complete tracks.
"""
import os
import json
import argparse
import librosa
import numpy as np
import torch
import pretty_midi

from train_phase2 import SymmetricDrumTCN
from dsp_utils import extract_features
from run_egmd_round4_validation import match_events

LABELS = ('KD', 'SD', 'HH', 'TOM', 'CRASH', 'RIDE')
LABEL_INDEX = {label: i for i, label in enumerate(LABELS)}

# 標準 General MIDI 鼓件 Pitch 映射到六類別索引
PITCH_TO_LABEL_IDX = {
    35: 0, 36: 0,                       # KD: Acoustic/Electric Bass Drum
    37: 1, 38: 1, 40: 1,                 # SD: Side Stick, Acoustic/Electric Snare
    42: 2, 44: 2, 46: 2,                 # HH: Closed, Pedal, Open Hi-Hat
    41: 3, 43: 3, 45: 3, 47: 3, 48: 3, 50: 3, # TOM: Low/Mid/High Toms
    49: 4, 52: 4, 55: 4, 57: 4,          # CRASH: Crash 1, China, Splash, Crash 2
    51: 5, 53: 5, 59: 5                  # RIDE: Ride 1, Ride Bell, Ride 2
}

def local_maxima(probabilities, threshold=0.50, sr=44100, hop_length=256):
    """
    中文註解：使用局部峰值與設定門檻擷取 onset 時間點。
    """
    events = {label: [] for label in LABELS}
    for label, index in LABEL_INDEX.items():
        values = probabilities[:, index]
        for frame in range(2, len(values) - 2):
            if (values[frame] >= threshold and 
                values[frame] >= values[frame - 1] and 
                values[frame] > values[frame + 1] and 
                values[frame] >= values[frame - 2] and 
                values[frame] > values[frame + 2]):
                events[label].append(frame * hop_length / float(sr))
    return events

def load_midi_reference(midi_path, offset=0.020):
    """
    中文註解：載入真值 MIDI 的 notes，依據 pitch 映射到六個類別，並加上固定時間偏移。
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    events = {label: [] for label in LABELS}
    for instrument in pm.instruments:
        for note in instrument.notes:
            pitch = note.pitch
            if pitch in PITCH_TO_LABEL_IDX:
                idx = PITCH_TO_LABEL_IDX[pitch]
                label = LABELS[idx]
                events[label].append(note.start + offset)
    # 排序確保匹配正確
    for label in LABELS:
        events[label].sort()
    return events

def main():
    parser = argparse.ArgumentParser(description="Evaluate a six-class ADT checkpoint on complete real audio tracks.")
    parser.add_argument('--audio', type=str, required=True, help="Path to complete track WAV")
    parser.add_argument('--midi', type=str, required=True, help="Path to reference MIDI file")
    parser.add_argument('--model', type=str, required=True, help="Path to six-class candidate checkpoint")
    parser.add_argument('--threshold', type=float, default=0.50, help="Onset activation threshold (default: 0.50)")
    parser.add_argument('--offset', type=float, default=0.020, help="Alignment offset added to MIDI reference (default: +0.020s)")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. 載入模型
    model = SymmetricDrumTCN(num_classes=6).to(device)
    state = torch.load(args.model, map_location=device, weights_only=False)
    if 'backbone.legacy_slot_proj.weight' in state:
        model.backbone.use_legacy_proj = True
    elif 'backbone.slot_proj.weight' in state and state['backbone.slot_proj.weight'].shape == torch.Size([64, 1024, 1, 1]):
        model.backbone.use_legacy_proj = True
        state = dict(state)
        state['backbone.legacy_slot_proj.weight'] = state.pop('backbone.slot_proj.weight')
        state['backbone.legacy_slot_proj.bias'] = state.pop('backbone.slot_proj.bias')
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"Loaded candidate model successfully: {args.model}")

    # 2. 載入音訊與特徵提取
    print(f"Loading audio: {args.audio}")
    y, sr = librosa.load(args.audio, sr=44100, mono=True)
    features = extract_features(y, sr=sr, hop_length=256, n_mels=256)
    features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)

    # 3. 前向推理
    print("Running sequence TCN inference...")
    with torch.no_grad():
        onset_logits, _ = model(features_tensor)
        onset_preds = torch.sigmoid(onset_logits).squeeze(0).cpu().numpy()

    # 4. 偵測 onsets
    predicted_events = local_maxima(onset_preds, threshold=args.threshold, sr=sr, hop_length=256)

    # 5. 載入 Reference MIDI
    print(f"Loading reference MIDI: {args.midi} (offset={args.offset}s)")
    ref_events = load_midi_reference(args.midi, offset=args.offset)

    # 6. 比對並計算 F1-Score
    print("\n" + "="*50)
    print(f"Evaluation Report: {os.path.basename(args.audio)}")
    print("="*50)
    
    rows = []
    macro_f1 = 0.0
    for label in LABELS:
        expected = ref_events[label]
        predicted = predicted_events[label]
        tp, fp, fn, precision, recall, f1 = match_events(expected, predicted, tolerance=0.050)
        
        macro_f1 += f1
        rows.append({
            'inst': label,
            'expected': len(expected),
            'predicted': len(predicted),
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': f"{precision:.4f}",
            'recall': f"{recall:.4f}",
            'f1': f"{f1:.4f}"
        })
        
    macro_f1 /= len(LABELS)
    
    # 列印表格
    print(f"{'Inst':<6} | {'Expected':<8} | {'Predicted':<9} | {'TP':<4} | {'FP':<4} | {'FN':<4} | {'Precision':<9} | {'Recall':<6} | {'F1':<6}")
    print("-"*85)
    for r in rows:
        print(f"{r['inst']:<6} | {r['expected']:<8} | {r['predicted']:<9} | {r['tp']:<4} | {r['fp']:<4} | {r['fn']:<4} | {r['precision']:<9} | {r['recall']:<6} | {r['f1']:<6}")
    print("-"*85)
    print(f"Macro F1-Score (6 classes): {macro_f1:.4f}\n")

if __name__ == '__main__':
    main()
