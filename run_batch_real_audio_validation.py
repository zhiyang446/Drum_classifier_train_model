# -*- coding: utf-8 -*-
"""
Batch Evaluation script for 5 real-world songs under double tower mode.
"""
import subprocess
import os
import re

songs = [
    {
        "name": "Blue (Yung Kai)",
        "audio": "test_real_audio/blue-yung-kai.wav",
        "midi": "test_real_audio/blue-yung-kai-drum-sheet-music.mid",
        "offset": 1.000
    },
    {
        "name": "Counting Stars (OneRepublic)",
        "audio": "test_real_audio/counting-stars.wav",
        "midi": "test_real_audio/counting-stars.mid",
        "offset": 0.100
    },
    {
        "name": "Payphone (Maroon 5)",
        "audio": "test_real_audio/payphone.wav",
        "midi": "test_real_audio/payphone.mid",
        "offset": -2.100
    },
    {
        "name": "Rolling In The Deep (Adele)",
        "audio": "test_real_audio/rolling-in-the-deep.wav",
        "midi": "test_real_audio/rolling-in-the-deep-adele-drum-sheet-music.mid",
        "offset": 0.020
    },
    {
        "name": "Toto - Rosanna",
        "audio": "test_real_audio/toto-rosanna.wav",
        "midi": "test_real_audio/toto-rosanna.mid",
        "offset": 0.020
    }
]

base_model = "mixed_formal_kick375_snare18_hh12_candidate.pth"
rare_model = "six_class_tower_b_specialized.pth"
python_exec = r".\.venv\Scripts\python.exe"

print("=" * 80)
print("Starting Batch Real-Audio Evaluation for 5 Songs...")
print(f"Base Model: {base_model}")
print(f"Specialized Model B: {rare_model}")
print("=" * 80)

all_results = {}

for song in songs:
    print(f"\n>>> Running evaluation on: {song['name']} (Offset: {song['offset']}s)")
    if not os.path.exists(song['audio']) or not os.path.exists(song['midi']):
        print(f"Error: WAV or MIDI file not found for {song['name']}")
        continue
        
    cmd = [
        python_exec, "run_real_audio_validation.py",
        "--audio", song['audio'],
        "--midi", song['midi'],
        "--model", base_model,
        "--model-rare", rare_model,
        "--offset", str(song['offset'])
    ]
    
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        stdout = res.stdout
        
        # 解析每首歌的類別數據
        song_data = {}
        labels = ["KD", "SD", "HH", "TOM", "CRASH", "RIDE"]
        for lbl in labels:
            match = re.search(rf"{lbl}\s*\|\s*(\d+)\s*\|\s*\d+\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*\d+\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)", stdout)
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
print("BATCH EVALUATION REPORT SUMMARY")
print("=" * 80)

# 列印 Markdown 格式表格以利導出 Walkthrough
print("| Song Name | KD F1 | SD F1 | HH F1 | TOM Recall (FP) | RIDE Recall (FP) | CRASH Recall (FP) | Macro F1 |")
print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
for name, data in all_results.items():
    print(f"| {name} | {data['KD']['f1']:.4f} | {data['SD']['f1']:.4f} | {data['HH']['f1']:.4f} | {data['TOM']['recall']:.4f} ({data['TOM']['fp']}) | {data['RIDE']['recall']:.4f} ({data['RIDE']['fp']}) | {data['CRASH']['recall']:.4f} ({data['CRASH']['fp']}) | {data['macro_f1']:.4f} |")

# 計算平均
avg_kd = sum(all_results[s]['KD']['f1'] for s in all_results) / len(all_results)
avg_sd = sum(all_results[s]['SD']['f1'] for s in all_results) / len(all_results)
avg_hh = sum(all_results[s]['HH']['f1'] for s in all_results) / len(all_results)
avg_tom_rec = sum(all_results[s]['TOM']['recall'] for s in all_results) / len(all_results)
avg_ride_rec = sum(all_results[s]['RIDE']['recall'] for s in all_results) / len(all_results)
avg_crash_rec = sum(all_results[s]['CRASH']['recall'] for s in all_results) / len(all_results)
avg_macro = sum(all_results[s]['macro_f1'] for s in all_results) / len(all_results)

print("| **Average** | **{:.4f}** | **{:.4f}** | **{:.4f}** | **{:.4f}** | **{:.4f}** | **{:.4f}** | **{:.4f}** |".format(
    avg_kd, avg_sd, avg_hh, avg_tom_rec, avg_ride_rec, avg_crash_rec, avg_macro
))
print("=" * 80)
