# -*- coding: utf-8 -*-
"""
Build human-verifiable onset annotation CSV templates for user blind files.
"""
import argparse
import csv
import os

import librosa
import soundfile as sf


INSTS = {
    'kick': ('KD', 'expected_kick', 'final_kick', 'prob_kick'),
    'snare': ('SD', 'expected_snare', 'final_snare', 'prob_snare'),
    'hihat': ('HH', 'expected_hihat', 'final_hihat', 'prob_hihat'),
}


def duration(path):
    """
    中文註解：取得音訊秒數，避免候選點超出檔案尾端。
    """
    with sf.SoundFile(path) as f:
        return f.frames / float(f.samplerate)


def load_expected(path):
    """
    中文註解：讀取使用者提供的每首歌 KD/SD/HH 目標數量。
    """
    return {row['name']: row for row in csv.DictReader(open(path, 'r', encoding='utf-8'))}


def load_raw_candidates(raw_csv):
    """
    中文註解：從 raw AI CSV 取已觸發的事件，保留模型定位到的 onset 時間與機率。
    """
    rows = []
    if not raw_csv or not os.path.exists(raw_csv):
        return rows
    with open(raw_csv, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            raw_time = float(row.get('raw_time') or row.get('quantized_time') or 0.0)
            for _, (inst, _, hit_col, prob_col) in INSTS.items():
                if row.get(hit_col) == 'True':
                    rows.append({
                        'time': raw_time,
                        'inst': inst,
                        'velocity': 100,
                        'source': 'raw_ai',
                        'confirmed': 'False',
                        'probability': f"{float(row.get(prob_col) or 0.0):.4f}",
                    })
    return rows


def audio_onsets(audio_path):
    """
    中文註解：偵測一般音訊 onset 候選，供 grid fill 對齊到更接近真實敲擊的位置。
    """
    y, sr = librosa.load(audio_path, sr=44100, mono=True)
    frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=256, backtrack=True)
    return list(librosa.frames_to_time(frames, sr=sr, hop_length=256))


def nearest_onset(target, onsets, max_dist=0.055):
    """
    中文註解：把網格候選吸附到附近音訊 onset；找不到就保留原網格。
    """
    if not onsets:
        return target, 'grid_fill'
    nearest = min(onsets, key=lambda value: abs(value - target))
    if abs(nearest - target) <= max_dist:
        return nearest, 'grid_fill+audio_onset'
    return target, 'grid_fill'


def add_grid_candidates(rows, audio_path, expected_row):
    """
    中文註解：補足每個鼓組的候選點數量，讓人工只需要確認/調整，不用從零建立。
    """
    dur = duration(audio_path)
    onsets = audio_onsets(audio_path)
    existing = {
        inst: sorted(float(row['time']) for row in rows if row['inst'] == inst)
        for inst, _, _, _ in INSTS.values()
    }
    start = min((float(row['time']) for row in rows), default=0.30)
    end = max(start + 0.1, dur - 0.05)

    for _, (inst, expected_col, _, _) in INSTS.items():
        target_count = int(expected_row[expected_col])
        times = list(existing[inst])
        interval = (end - start) / max(target_count - 1, 1)
        idx = 0
        while len(times) < target_count:
            candidate = start + idx * interval
            idx += 1
            if candidate >= dur:
                break
            snapped, source = nearest_onset(candidate, onsets)
            if all(abs(snapped - value) > 0.035 for value in times):
                times.append(snapped)
                rows.append({
                    'time': snapped,
                    'inst': inst,
                    'velocity': 90,
                    'source': source,
                    'confirmed': 'False',
                    'probability': '',
                })
    return rows


def write_template(rows, output_csv):
    """
    中文註解：輸出人工確認用 CSV，依時間排序並去除極近重複候選。
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    deduped = []
    for row in sorted(rows, key=lambda item: (float(item['time']), item['inst'], item['source'])):
        if any(row['inst'] == old['inst'] and abs(float(row['time']) - float(old['time'])) < 0.005 for old in deduped):
            continue
        deduped.append(row)
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['time', 'inst', 'velocity', 'source', 'confirmed', 'probability'])
        writer.writeheader()
        for row in deduped:
            row = dict(row)
            row['time'] = f"{float(row['time']):.6f}"
            writer.writerow(row)
    return len(deduped)


def run_self_check():
    """
    中文註解：最小自檢，確認 onset 吸附不會偏離過遠。
    """
    assert nearest_onset(1.0, [0.97])[0] == 0.97
    assert nearest_onset(1.0, [0.80])[0] == 1.0
    print('Self-check passed.')


def main():
    """
    中文註解：CLI 入口，批次建立 5 首 user blind annotation templates。
    """
    parser = argparse.ArgumentParser(description='Build user onset annotation templates.')
    parser.add_argument('--input-dir', default='blind_user_tests')
    parser.add_argument('--expected', default='blind_user_tests_expected.csv')
    parser.add_argument('--raw-root', default='validation_runs/blind_test_user_first_batch')
    parser.add_argument('--output-dir', default='annotations/user_blind_precise')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()

    if args.self_check:
        run_self_check()
        return

    expected = load_expected(args.expected)
    for name, row in expected.items():
        audio_path = os.path.join(args.input_dir, f'{name}.wav')
        raw_csv = os.path.join(args.raw_root, name, f'{name}_raw_ai_events.csv')
        rows = add_grid_candidates(load_raw_candidates(raw_csv), audio_path, row)
        output_csv = os.path.join(args.output_dir, f'{name}_annotations.csv')
        count = write_template(rows, output_csv)
        print(f'{name}: wrote {count} candidates -> {output_csv}')


if __name__ == '__main__':
    main()
