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

def local_maxima(probabilities, threshold=0.50, sr=44100, hop_length=256, y=None, adaptive_snare=False):
    """
    中文註解：使用局部峰值與設定門檻擷取 onset 時間點。
    """
    events = {label: [] for label in LABELS}
    n_frames = len(probabilities)
    thresh_arrays = {}
    
    if y is not None and adaptive_snare:
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
        if len(rms) < n_frames:
            rms = np.pad(rms, (0, n_frames - len(rms)), mode='edge')
        elif len(rms) > n_frames:
            rms = rms[:n_frames]
        rms_db = 20 * np.log10(rms + 1e-5)
        max_db = np.max(rms_db)
        min_db = np.max([np.min(rms_db), max_db - 40.0])
        rms_db_norm = np.clip((rms_db - min_db) / (max_db - min_db + 1e-6), 0.0, 1.0)
        
        thresh_arrays[0] = np.clip(threshold + (0.08 - 0.16 * rms_db_norm), 0.25, 0.75)  # KD
        thresh_arrays[1] = np.clip(threshold - 0.12 + 0.16 * rms_db_norm, 0.26, 0.45)  # SD (動態翻轉)
        thresh_arrays[2] = np.clip(threshold + (0.10 - 0.25 * rms_db_norm), 0.25, 0.75)  # HH
        
        if probabilities.shape[1] == 6:
            thresh_arrays[3] = np.clip(0.50 + (0.10 - 0.25 * rms_db_norm), 0.25, 0.75)  # TOM
            thresh_arrays[4] = np.clip(0.50 + (0.10 - 0.25 * rms_db_norm), 0.25, 0.75)  # CRASH
            thresh_arrays[5] = np.clip(0.50 + (0.10 - 0.25 * rms_db_norm), 0.25, 0.75)  # RIDE
    else:
        for c in range(probabilities.shape[1]):
            thresh_arrays[c] = np.ones(n_frames) * (threshold if c < 3 else 0.50)
            
    for label, index in LABEL_INDEX.items():
        values = probabilities[:, index]
        thresh_t = thresh_arrays[index]
        for frame in range(2, len(values) - 2):
            if (values[frame] >= thresh_t[frame] and 
                values[frame] >= values[frame - 1] and 
                values[frame] > values[frame + 1] and 
                values[frame] >= values[frame - 2] and 
                values[frame] > values[frame + 2]):
                events[label].append(frame)
                
    # 執行 AME 互斥過濾
    if probabilities.shape[1] == 6:
        # AME Snare vs Tom
        cleaned_tom = []
        for t_f in events['TOM']:
            is_crosstalk = False
            # 只有當 TOM 的機率小於 0.48 且與強 SD 重合時，才視為串音
            if probabilities[t_f, 3] < 0.48:
                for s_f in events['SD']:
                    if abs(t_f - s_f) <= 2 and probabilities[s_f, 1] >= 0.80:
                        is_crosstalk = True
                        break
            if not is_crosstalk:
                cleaned_tom.append(t_f)
        events['TOM'] = cleaned_tom
        
        # AME Kick vs Tom
        cleaned_tom = []
        for t_f in events['TOM']:
            is_crosstalk = False
            # 只有當 TOM 的機率小於 0.48 且與強 KD 重合時，才視為串音
            if probabilities[t_f, 3] < 0.48:
                for k_f in events['KD']:
                    if abs(t_f - k_f) <= 2 and probabilities[k_f, 0] >= 0.80:
                        is_crosstalk = True
                        break
            if not is_crosstalk:
                cleaned_tom.append(t_f)
        events['TOM'] = cleaned_tom
        
        # AME HH vs Ride
        cleaned_ride = []
        for r_f in events['RIDE']:
            is_crosstalk = False
            # 只有當 RIDE 機率小於 0.45 且與強 HH 重合時，才視為串音
            if probabilities[r_f, 5] < 0.45:
                for h_f in events['HH']:
                    if abs(r_f - h_f) <= 2 and probabilities[h_f, 2] >= 0.75:
                        is_crosstalk = True
                        break
            if not is_crosstalk:
                cleaned_ride.append(r_f)
        events['RIDE'] = cleaned_ride
        
        # AME Snare vs Crash
        cleaned_crash = []
        for c_f in events['CRASH']:
            is_crosstalk = False
            # 只有當 CRASH 機率小於 0.45 且與強 SD 重合時，才視為串音
            if probabilities[c_f, 4] < 0.45:
                for s_f in events['SD']:
                    if abs(c_f - s_f) <= 2 and probabilities[s_f, 1] >= 0.80:
                        is_crosstalk = True
                        break
            if not is_crosstalk:
                cleaned_crash.append(c_f)
        events['CRASH'] = cleaned_crash
        
    time_events = {label: [] for label in LABELS}
    for label in LABELS:
        time_events[label] = [f * hop_length / float(sr) for f in events[label]]
    return time_events

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

def init_and_load_model(ckpt_path, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model file not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    n_classes = 3
    if 'onset_head.weight' in checkpoint:
        n_classes = checkpoint['onset_head.weight'].shape[0]
    net = SymmetricDrumTCN(num_classes=n_classes).to(device)
    if 'backbone.legacy_slot_proj.weight' in checkpoint:
        net.backbone.use_legacy_proj = True
    elif 'backbone.slot_proj.weight' in checkpoint and checkpoint['backbone.slot_proj.weight'].shape == torch.Size([64, 1024, 1, 1]):
        net.backbone.use_legacy_proj = True
        checkpoint['backbone.legacy_slot_proj.weight'] = checkpoint.pop('backbone.slot_proj.weight')
        checkpoint['backbone.legacy_slot_proj.bias'] = checkpoint.pop('backbone.slot_proj.bias')
    net.load_state_dict(checkpoint, strict=True)
    net.eval()
    return net, n_classes

def main():
    parser = argparse.ArgumentParser(description="Evaluate a six-class ADT checkpoint on complete real audio tracks.")
    parser.add_argument('--audio', type=str, required=True, help="Path to complete track WAV")
    parser.add_argument('--midi', type=str, required=True, help="Path to reference MIDI file")
    parser.add_argument('--model', type=str, required=True, help="Path to six-class candidate checkpoint")
    parser.add_argument('--model-rare', type=str, default=None, help="Path to optional rare drum classes extension model")
    parser.add_argument('--threshold', type=float, default=0.50, help="Onset activation threshold (default: 0.50)")
    parser.add_argument('--offset', type=float, default=0.020, help="Alignment offset added to MIDI reference (default: +0.020s)")
    parser.add_argument('--adaptive-snare', action='store_true', help="Enable dynamic Snare thresholding")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. 載入模型
    model, num_classes = init_and_load_model(args.model, device)
    print(f"Loaded base model successfully: {args.model} (classes={num_classes})")
    
    model_rare = None
    num_classes_rare = 0
    if args.model_rare:
        model_rare, num_classes_rare = init_and_load_model(args.model_rare, device)
        print(f"Loaded rare model successfully: {args.model_rare} (classes={num_classes_rare})")

    # 2. 載入音訊與特徵提取
    print(f"Loading audio: {args.audio}")
    y, sr = librosa.load(args.audio, sr=44100, mono=True)
    features = extract_features(y, sr=sr, hop_length=256, n_mels=256)
    features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)

    # 3. 前向推理與機率融合
    print("Running sequence TCN inference...")
    with torch.no_grad():
        if model_rare is not None:
            onset_logits_base, _ = model(features_tensor)
            onset_logits_rare, _ = model_rare(features_tensor)
            
            base_p = torch.sigmoid(onset_logits_base).squeeze(0).cpu().numpy()
            rare_p = torch.sigmoid(onset_logits_rare).squeeze(0).cpu().numpy()
            
            onset_preds = np.zeros((base_p.shape[0], 6), dtype=np.float32)
            onset_preds[:, :3] = base_p[:, :3]
            onset_preds[:, 3:6] = rare_p[:, 3:6]
        else:
            onset_logits, _ = model(features_tensor)
            preds = torch.sigmoid(onset_logits).squeeze(0).cpu().numpy()
            if num_classes == 6:
                onset_preds = preds
            else:
                onset_preds = np.zeros((preds.shape[0], 6), dtype=np.float32)
                onset_preds[:, :3] = preds[:, :3]

    # 4. 偵測 onsets
    predicted_events = local_maxima(onset_preds, threshold=args.threshold, sr=sr, hop_length=256, y=y, adaptive_snare=args.adaptive_snare)

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
