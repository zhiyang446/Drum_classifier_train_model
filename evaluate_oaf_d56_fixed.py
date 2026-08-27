# -*- coding: utf-8 -*-
"""以 OaF 官方權重重放 D56 封存窗口，僅建立零訓練比較證據。"""

import argparse
import json
import os
import subprocess
from collections import Counter

import numpy as np
import pretty_midi
import soundfile as sf

from build_egmd_pitch_weighted_meta import LABELS, PITCH_TO_INST
from run_six_class_smoke import SR, TARGET_SAMPLES, read_mono_window
from run_six_class_validation import (
    TOLERANCE,
    expected_events,
    load_fixed_windows,
    write_outputs,
)


def parse_args():
    """解析固定 D56 OaF baseline 的唯讀評估參數。"""
    parser = argparse.ArgumentParser(description='Evaluate official OaF on fixed D56 drumsep-mix windows.')
    parser.add_argument('--metadata', default='mixed_d54_stem/metadata_d54.json')
    parser.add_argument(
        '--selected-windows',
        default='validation_runs/d82_d77_fused_lora_candidate/epoch_05_fixed_validation/selected_windows.json',
    )
    parser.add_argument('--output-dir', default='validation_runs/oaf_d56_fixed_baseline')
    parser.add_argument('--conda-exe', default=r'C:\Users\zhiya\miniconda3\Scripts\conda.exe')
    parser.add_argument('--conda-env', default='oaf_compat_py37')
    parser.add_argument('--model-dir', default='validation_runs/oaf_compat_probe/checkpoint')
    parser.add_argument('--self-check', action='store_true')
    return parser.parse_args()


def ensure_new_output_dir(path):
    """建立全新輸出目錄，拒絕覆寫既有驗收證據。"""
    if os.path.exists(path):
        raise FileExistsError(f'Output directory already exists: {path}')
    os.makedirs(path)
    return os.path.abspath(path)


def load_frozen_windows(metadata_path, selected_path):
    """讀取 D54 metadata 與 D82 的封存 48-window selection，驗證兩者逐列一致。"""
    with open(metadata_path, encoding='utf-8') as handle:
        metadata = json.load(handle)
    with open(selected_path, encoding='utf-8') as handle:
        stored_rows = json.load(handle)
    windows = load_fixed_windows(metadata, selected_path, split='validation', per_class=8)
    if len(stored_rows) != 48 or len(windows) != 48:
        raise ValueError('D56 fixed selection must contain exactly 48 windows.')
    for stored, window in zip(stored_rows, windows):
        if stored['label'] != window['label'] or stored['key'] != window['key']:
            raise ValueError('Fixed selection order or identity changed.')
        if abs(float(stored['anchor']) - float(window['anchor'])) > 1e-9:
            raise ValueError('Fixed selection anchor changed.')
        if stored.get('split') != 'validation':
            raise ValueError('Fixed selection leaves validation split.')
        if stored.get('input_mode') != 'drumsep-mix':
            raise ValueError('OaF baseline must use the D56 drumsep-mix input.')
    return windows, stored_rows


def drumsep_waveform(item, start_sec):
    """以 D56 相同的六 stem 時域相加重建指定四秒窗口，絕不讀取原始 mixed 音訊。"""
    stems = item.get('drumsep_stems', {}).get('paths', {})
    required = {'kick', 'snare', 'toms', 'hh', 'ride', 'crash'}
    if set(stems) != required:
        raise ValueError('D56 item is missing a versioned six-stem set.')
    seconds = TARGET_SAMPLES / float(SR)
    waveform = sum(
        (read_mono_window(stems[name], start_sec, seconds) for name in sorted(required)),
        np.zeros(TARGET_SAMPLES, dtype=np.float32),
    )
    if len(waveform) != TARGET_SAMPLES or not np.isfinite(waveform).all():
        raise ValueError('Invalid reconstructed drumsep-mix waveform.')
    return np.asarray(waveform, dtype=np.float32)


def write_clips(output_dir, windows, stored_rows):
    """只把封存窗口轉為新建 16-bit PCM clips，回傳比較所需的窗口描述。"""
    clips_dir = os.path.join(output_dir, 'clips')
    os.makedirs(clips_dir)
    rows = []
    for index, (window, stored) in enumerate(zip(windows, stored_rows)):
        start_sec = float(stored['window_start'])
        waveform = drumsep_waveform(window['item'], start_sec)
        clip_path = os.path.join(clips_dir, f'window_{index:02d}.wav')
        sf.write(clip_path, waveform, SR, subtype='PCM_16')
        info = sf.info(clip_path)
        if info.samplerate != SR or info.frames != TARGET_SAMPLES or info.subtype != 'PCM_16':
            raise ValueError(f'Invalid OaF clip format: {clip_path}')
        rows.append({'window': window, 'stored': stored, 'start_sec': start_sec, 'clip_path': clip_path})
    return rows


def run_oaf(args, output_dir, clip_rows):
    """以一次官方 OaF CLI 批次處理所有固定 clips，避免逐窗重載 checkpoint。"""
    model_dir = os.path.abspath(args.model_dir)
    if not os.path.isfile(os.path.join(model_dir, 'checkpoint')):
        raise FileNotFoundError(f'OaF model directory is invalid: {model_dir}')
    command = [
        args.conda_exe,
        'run', '--no-capture-output', '-n', args.conda_env,
        'onsets_frames_transcription_transcribe',
        f'--model_dir={model_dir}', '--config=drums',
        *(row['clip_path'] for row in clip_rows),
    ]
    environment = dict(os.environ)
    environment['NUMBA_CACHE_DIR'] = os.path.join(output_dir, 'numba_cache')
    result = subprocess.run(command, capture_output=True, text=True, env=environment, check=False)
    with open(os.path.join(output_dir, 'oaf_stdout.log'), 'w', encoding='utf-8') as handle:
        handle.write(result.stdout)
    with open(os.path.join(output_dir, 'oaf_stderr.log'), 'w', encoding='utf-8') as handle:
        handle.write(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f'OaF exited {result.returncode}; see oaf_stdout.log and oaf_stderr.log')
    for row in clip_rows:
        row['midi_path'] = row['clip_path'] + '.midi'
        if not os.path.isfile(row['midi_path']):
            raise FileNotFoundError(f'OaF MIDI is missing: {row["midi_path"]}')


def midi_events(path):
    """解析 OaF MIDI 為現有六類事件，同時保留未映射音高稽核。"""
    events = {label: [] for label in LABELS}
    unknown = Counter()
    midi = pretty_midi.PrettyMIDI(path)
    for instrument in midi.instruments:
        for note in instrument.notes:
            label = PITCH_TO_INST.get(int(note.pitch))
            if label is None:
                unknown[int(note.pitch)] += 1
                continue
            events[label].append(float(note.start))
    for values in events.values():
        values.sort()
    return events, unknown


def evaluate_windows(output_dir, clip_rows, selected_path):
    """以 D56 的預期事件、聚合偏移與 .05 秒容差輸出可比的逐類統計。"""
    aggregate = {label: ([], []) for label in LABELS}
    selected_rows = []
    unknown = Counter()
    window_seconds = TARGET_SAMPLES / float(SR)
    for index, row in enumerate(clip_rows):
        window = row['window']
        expected = expected_events(window['item'], row['start_sec'])
        predicted, unknown_pitches = midi_events(row['midi_path'])
        unknown.update(unknown_pitches)
        aggregate_offset = index * (window_seconds + 1.0)
        for label in LABELS:
            aggregate[label][0].extend(time + aggregate_offset for time in expected[label])
            aggregate[label][1].extend(time + aggregate_offset for time in predicted[label])
        selected_rows.append({
            'label': window['label'], 'key': window['key'], 'anchor': window['anchor'],
            'window_start': row['start_sec'], 'audio_path': window['item']['audio_path'],
            'split': 'validation', 'aggregate_offset': aggregate_offset,
            'architecture': 'official-oaf-egmd', 'feature_mode': 'official-oaf',
            'input_mode': 'drumsep-mix', 'fixed_windows_source': selected_path,
            'expected_counts': {label: len(expected[label]) for label in LABELS},
            'clip_path': row['clip_path'], 'midi_path': row['midi_path'],
        })
    _, gate = write_outputs(selected_rows, aggregate, output_dir)
    if int(gate.get('selected_windows', 0)) != len(selected_rows):
        raise ValueError('OaF report did not preserve the frozen D56 window count.')
    audit = {
        'status': 'pass', 'selection_count': len(selected_rows), 'split': 'validation',
        'tolerance_seconds': TOLERANCE, 'unknown_pitch_counts': dict(sorted(unknown.items())),
        'all_oaf_events_mapped': not unknown, 'gate': gate,
        'decision': 'baseline_evidence_only_not_training_or_release',
    }
    with open(os.path.join(output_dir, 'oaf_audit.json'), 'w', encoding='utf-8') as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
    return audit


def run_self_check():
    """驗證六類映射與 D56 固定窗口數量的最低防線。"""
    assert PITCH_TO_INST[36] == 'KD'
    assert PITCH_TO_INST[38] == 'SD'
    assert PITCH_TO_INST[46] == 'HH'
    assert PITCH_TO_INST[48] == 'TOM'
    assert PITCH_TO_INST[49] == 'CRASH'
    assert PITCH_TO_INST[53] == 'RIDE'
    assert len(LABELS) == 6
    print('Self-check passed.')


def main():
    """CLI 入口：建立 OaF D56 唯讀 baseline，不把 F1 結果當成發布決策。"""
    args = parse_args()
    if args.self_check:
        run_self_check()
        return
    windows, stored_rows = load_frozen_windows(args.metadata, args.selected_windows)
    output_dir = ensure_new_output_dir(args.output_dir)
    clip_rows = write_clips(output_dir, windows, stored_rows)
    run_oaf(args, output_dir, clip_rows)
    audit = evaluate_windows(output_dir, clip_rows, args.selected_windows)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
