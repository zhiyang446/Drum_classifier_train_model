# -*- coding: utf-8 -*-
"""
Batch Evaluation script for 5 real-world songs under double tower mode.
Includes an Auto-Aligner based on First-Kick Coarse Alignment + Local Fine Search.
"""
import subprocess
import os
import re
import numpy as np
import pretty_midi
import torch
import librosa
import run_real_audio_validation as rrav

songs = [
    {
        "name": "Blue (Yung Kai)",
        "audio": "test_real_audio/blue-yung-kai.wav",
        "midi": "test_real_audio/blue-yung-kai-drum-sheet-music.mid"
    },
    {
        "name": "Counting Stars (OneRepublic)",
        "audio": "test_real_audio/counting-stars.wav",
        "midi": "test_real_audio/counting-stars.mid"
    },
    {
        "name": "Payphone (Maroon 5)",
        "audio": "test_real_audio/payphone.wav",
        "midi": "test_real_audio/payphone.mid"
    },
    {
        "name": "Rolling In The Deep (Adele)",
        "audio": "test_real_audio/rolling-in-the-deep.wav",
        "midi": "test_real_audio/rolling-in-the-deep-adele-drum-sheet-music.mid"
    },
    {
        "name": "Toto - Rosanna",
        "audio": "test_real_audio/toto-rosanna.wav",
        "midi": "test_real_audio/toto-rosanna.mid"
    }
]

base_model = "mixed_formal_kick375_snare18_hh12_candidate.pth"
rare_model = "six_class_tower_b_specialized.pth"
python_exec = r".\.venv\Scripts\python.exe"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 載入模型以便進行快速推理對齊
model, num_classes = rrav.init_and_load_model(base_model)
if os.path.exists(rare_model):
    model_rare, num_classes_rare = rrav.init_and_load_model(rare_model)
else:
    model_rare = None

def get_predicted_kicks(audio_path):
    """
    中文註解：對音訊進行極速推理，僅取得大鼓 (KD) 的預測時間點。
    """
    y, sr = librosa.load(audio_path, sr=44100, mono=True)
    features = rrav.extract_features(y, sr=sr, hop_length=256, n_mels=256)
    features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)
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
            onset_preds = np.zeros((preds.shape[0], 6), dtype=np.float32)
            onset_preds[:, :3] = preds[:, :3]
            
    # 只取大鼓 KD
    predicted_events = rrav.local_maxima(onset_preds, threshold=0.50, sr=sr, hop_length=256)
    return predicted_events['KD']

def find_best_offset(ref_kicks, pred_kicks):
    """
    中文註解：使用 First-Kick Coarse + Fine Grid Search 尋找大鼓匹配率最高的最佳時間偏移。
    """
    if not ref_kicks or not pred_kicks:
        return 0.020 # 預設偏移
        
    # 1. 粗對齊：計算前三個大鼓的平均偏移候選
    coarse_candidates = []
    for i in range(min(3, len(ref_kicks), len(pred_kicks))):
        coarse_candidates.append(pred_kicks[i] - ref_kicks[i])
        
    best_offset = 0.020
    max_tps = -1
    
    # 2. 細搜尋：在每個粗候選附近 +-0.3 秒內以 5ms 步長進行精細搜尋
    for coarse in coarse_candidates:
        search_range = np.arange(coarse - 0.3, coarse + 0.3, 0.005)
        for offset in search_range:
            # 加上偏移後的真值
            shifted_ref = [t + offset for t in ref_kicks]
            # 統計 50ms 內的匹配 TP 數
            tp, _, _, _, _, _ = rrav.match_events(shifted_ref, pred_kicks, tolerance=0.050)
            if tp > max_tps:
                max_tps = tp
                best_offset = offset
            elif tp == max_tps:
                # 若 TP 相同，選擇絕對值較小（偏移較小）的 offset
                if abs(offset) < abs(best_offset):
                    best_offset = offset
                    
    return float(best_offset)

print("=" * 80)
print("Starting Batch Real-Audio Evaluation with Auto-Aligner...")
print("=" * 80)

all_results = {}

for song in songs:
    print(f"\n>>> Analyzing: {song['name']}")
    if not os.path.exists(song['audio']) or not os.path.exists(song['midi']):
        print(f"Error: WAV or MIDI file not found for {song['name']}")
        continue
        
    # 1. 載入原始無偏移的 MIDI 大鼓時間點
    pm = pretty_midi.PrettyMIDI(song['midi'])
    ref_kicks = []
    for instrument in pm.instruments:
        for note in instrument.notes:
            if note.pitch in (35, 36): # GM Kick pitches
                ref_kicks.append(note.start)
    ref_kicks.sort()
    
    # 2. 獲取模型預測大鼓時間點
    pred_kicks = get_predicted_kicks(song['audio'])
    
    # 3. 自動搜尋黃金偏移
    best_offset = find_best_offset(ref_kicks, pred_kicks)
    print(f"Auto-Aligner calculated optimal offset: {best_offset:.4f}s (Matches: {len(pred_kicks)} predicted, {len(ref_kicks)} expected)")
    
    # 4. 執行帶有黃金偏移的完整評估
    cmd = [
        python_exec, "run_real_audio_validation.py",
        "--audio", song['audio'],
        "--midi", song['midi'],
        "--model", base_model,
        "--model-rare", rare_model,
        "--offset", f"{best_offset:.4f}",
        "--adaptive-snare"
    ]
    
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        stdout = res.stdout
        
        # 解析每首歌的類別數據
        song_data = {}
        labels = ["KD", "SD", "HH", "TOM", "CRASH", "RIDE"]
        for lbl in labels:
            match = re.search(rf"{lbl}\s*\|\s*(\d+)\s*\|\s*\d+\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*[\d\s\.]*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)", stdout)
            if match:
                expected = int(match.group(1))
                tp = int(match.group(2))
                fp = int(match.group(3))
                prec = float(match.group(4))
                recall = float(match.group(5))
                f1 = float(match.group(6))
                song_data[lbl] = {"expected": expected, "tp": tp, "fp": fp, "prec": prec, "recall": recall, "f1": f1}
            else:
                song_data[lbl] = {"expected": 0, "tp": 0, "fp": 0, "prec": 0.0, "recall": 0.0, "f1": 0.0}
                
        macro_match = re.search(r"Macro F1-Score \(6 classes\):\s*([\d\.]+)", stdout)
        song_data["macro_f1"] = float(macro_match.group(1)) if macro_match else 0.0
        song_data["best_offset"] = best_offset
        
        all_results[song['name']] = song_data
        
        # 印出簡要結果
        print(f"--- Results for {song['name']} ---")
        print(f"Macro F1-Score: {song_data['macro_f1']:.4f}")
        for lbl in labels:
            d = song_data[lbl]
            print(f"  {lbl:<5} : Expected={d['expected']:<4} | TP={d['tp']:<4} | FP={d['fp']:<4} | Recall={d['recall']:.4f} | F1={d['f1']:.4f}")
            
    except Exception as e:
        print(f"Failed to evaluate {song['name']}: {e}")

# 計算 5 首歌曲的宏平均與微平均指標
print("\n" + "=" * 80)
print("BATCH EVALUATION REPORT SUMMARY (AUTO-ALIGNED)")
print("=" * 80)

print("| Song Name | Offset (s) | KD F1 | SD F1 | HH F1 | TOM Recall (FP) | RIDE Recall (FP) | CRASH Recall (FP) | Macro F1 |")
print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
for name, data in all_results.items():
    print(f"| {name} | {data['best_offset']:.4f} | {data['KD']['f1']:.4f} | {data['SD']['f1']:.4f} | {data['HH']['f1']:.4f} | {data['TOM']['recall']:.4f} ({data['TOM']['fp']}) | {data['RIDE']['recall']:.4f} ({data['RIDE']['fp']}) | {data['CRASH']['recall']:.4f} ({data['CRASH']['fp']}) | {data['macro_f1']:.4f} |")

# 計算平均
avg_kd = sum(all_results[s]['KD']['f1'] for s in all_results) / len(all_results)
avg_sd = sum(all_results[s]['SD']['f1'] for s in all_results) / len(all_results)
avg_hh = sum(all_results[s]['HH']['f1'] for s in all_results) / len(all_results)
avg_tom_rec = sum(all_results[s]['TOM']['recall'] for s in all_results) / len(all_results)
avg_ride_rec = sum(all_results[s]['RIDE']['recall'] for s in all_results) / len(all_results)
avg_crash_rec = sum(all_results[s]['CRASH']['recall'] for s in all_results) / len(all_results)
avg_macro = sum(all_results[s]['macro_f1'] for s in all_results) / len(all_results)

print("| **Average** | **-** | **{:.4f}** | **{:.4f}** | **{:.4f}** | **{:.4f}** | **{:.4f}** | **{:.4f}** | **{:.4f}** |".format(
    avg_kd, avg_sd, avg_hh, avg_tom_rec, avg_ride_rec, avg_crash_rec, avg_macro
))
print("=" * 80)
