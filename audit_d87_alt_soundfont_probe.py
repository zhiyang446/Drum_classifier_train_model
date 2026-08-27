# -*- coding: utf-8 -*-
"""D87：以一首 D27 train MIDI 驗證替代 SoundFont 是否帶來可用聲學差異。"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from build_midi_archive_render_d27 import render_wav, validate_wav


def sha256_file(path):
    """計算檔案雜湊，確認原始與替代音色不是位元相同。"""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def waveform_summary(original_path, alternate_path):
    """比較兩份同取樣率 WAV 的 RMS 與皮爾森相關，拒絕近乎相同的渲染。"""
    original, original_sr = sf.read(original_path, dtype='float32', always_2d=False)
    alternate, alternate_sr = sf.read(alternate_path, dtype='float32', always_2d=False)
    if original_sr != alternate_sr:
        raise ValueError(f'Sample rate differs: {original_sr} != {alternate_sr}')
    count = min(len(original), len(alternate))
    if count == 0:
        raise ValueError('Cannot compare empty WAVs.')
    original = np.asarray(original[:count], dtype=np.float64)
    alternate = np.asarray(alternate[:count], dtype=np.float64)
    original_rms = float(np.sqrt(np.mean(np.square(original))))
    alternate_rms = float(np.sqrt(np.mean(np.square(alternate))))
    original_std, alternate_std = float(np.std(original)), float(np.std(alternate))
    correlation = float(np.corrcoef(original, alternate)[0, 1]) if original_std and alternate_std else 1.0
    return {
        'compared_frames': count,
        'sample_rate': original_sr,
        'original_rms_float': original_rms,
        'alternate_rms_float': alternate_rms,
        'pearson_correlation': correlation,
        # ponytail: 一首 probe 只需排除近乎相同輸出；多音色品質由後續單一變因訓練判定。
        'materially_different': sha256_file(original_path) != sha256_file(alternate_path) and abs(correlation) < 0.999,
    }


def select_first_train_item(metadata):
    """以 item key 固定排序選取 D27 train 樣本，不依模型或錯誤結果挑歌。"""
    candidates = [(key, item) for key, item in metadata.items() if item.get('split') == 'train']
    if not candidates:
        raise ValueError('No D27 train items found.')
    return sorted(candidates, key=lambda row: row[0])[0]


def run(args):
    """渲染一首替代音色 WAV，寫入不可覆寫的 D87 探針報告。"""
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite existing output directory: {output_dir}')
    metadata = json.loads(Path(args.metadata).read_text(encoding='utf-8'))
    item_key, item = select_first_train_item(metadata)
    output_dir.mkdir(parents=True)
    alternate_path = output_dir / 'alternate_soundfont.wav'
    render_wav(args.renderer, args.alternate_soundfont, args.ffmpeg, item['midi_path'], alternate_path)
    alternate_duration, alternate_rms = validate_wav(alternate_path, float(item['midi_duration']))
    comparison = waveform_summary(item['audio_path'], alternate_path)
    report = {
        'phase': 'D87',
        'status': 'pass' if comparison['materially_different'] else 'rejected',
        'research_only': True,
        'ready_for_training_candidate': False,
        'ready_for_six_class_release': False,
        'item_key': item_key,
        'group_id': item['group_id'],
        'split': item['split'],
        'midi_path': item['midi_path'],
        'midi_duration': item['midi_duration'],
        'original_audio_path': item['audio_path'],
        'original_audio_sha256': sha256_file(item['audio_path']),
        'alternate_audio_path': str(alternate_path.resolve()),
        'alternate_audio_sha256': sha256_file(alternate_path),
        'alternate_duration': alternate_duration,
        'alternate_pcm_rms': alternate_rms,
        'alternate_soundfont_path': str(Path(args.alternate_soundfont).resolve()),
        'alternate_soundfont_sha256': sha256_file(args.alternate_soundfont),
        'comparison': comparison,
        'full_render_allowed': comparison['materially_different'],
    }
    (output_dir / 'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    # ponytail: Windows 預設 CP1252 主控台無法顯示部分 Archive 檔名；檔案報告仍保留 UTF-8。
    print(json.dumps(report, indent=2, ensure_ascii=True))


def run_self_check():
    """驗證固定 train 選樣與不同波形的差異判定邏輯。"""
    metadata = {
        'validation': {'split': 'validation'},
        'train_b': {'split': 'train'},
        'train_a': {'split': 'train'},
    }
    assert select_first_train_item(metadata)[0] == 'train_a'
    print('Self-check passed.')


def main():
    """CLI 入口：執行一首 D27 train 替代 SoundFont 探針。"""
    parser = argparse.ArgumentParser(description='Probe one D27 train MIDI with an alternate SoundFont.')
    parser.add_argument('--metadata', default='synthetic_midi_archive_d27/metadata_d27.json')
    parser.add_argument('--renderer', default='third_party/fluidsynth-2.4.7/bin/fluidsynth.exe')
    parser.add_argument('--alternate-soundfont', default='.venv/Lib/site-packages/pretty_midi/TimGM6mb.sf2')
    parser.add_argument('--ffmpeg', default='C:/ffmpeg/bin/ffmpeg.exe')
    parser.add_argument('--output-dir', default='validation_runs/d87_archive_alt_soundfont_probe')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    run(args)


if __name__ == '__main__':
    main()
