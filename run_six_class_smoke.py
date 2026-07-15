# -*- coding: utf-8 -*-
"""六類 STAR metadata 與模型路徑的單窗口 smoke training。"""
import argparse
import json
import os

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from dsp_utils import extract_features
from train_phase2 import SymmetricDrumTCN


LABELS = ('KD', 'SD', 'HH', 'TOM', 'CRASH', 'RIDE')
LABEL_INDEX = {label: index for index, label in enumerate(LABELS)}
HEAD_SOURCE_LABELS = {'KD': 'KD', 'SD': 'SD', 'HH': 'HH', 'TOM': 'SD', 'CRASH': 'HH', 'RIDE': 'HH'}
SR = 44100
HOP_LENGTH = 256
N_MELS = 256
CHUNK_FRAMES = 688
TARGET_SAMPLES = CHUNK_FRAMES * HOP_LENGTH


def transfer_semantic_head_rows(target, source):
    """中文註解：把三類輸出頭依鼓件語義複製到六類 head 的對應列。"""
    source_index = {'KD': 0, 'SD': 1, 'HH': 2}
    copied = 0
    for head_name in ('onset_head', 'velocity_head'):
        for parameter_name in ('weight', 'bias'):
            target_name = f'{head_name}.{parameter_name}'
            if target_name not in source or target_name not in target:
                continue
            for label, source_label in HEAD_SOURCE_LABELS.items():
                target[target_name][LABEL_INDEX[label]].copy_(source[target_name][source_index[source_label]])
                copied += 1
    return copied


def configure_legacy_projection(model, state):
    """中文註解：依 checkpoint 的投影權重選擇與訓練一致的 legacy 分支。"""
    if 'backbone.legacy_slot_proj.weight' in state:
        model.backbone.use_legacy_proj = True
    elif 'backbone.slot_proj.weight' in state and state['backbone.slot_proj.weight'].shape == torch.Size([64, 1024, 1, 1]):
        model.backbone.use_legacy_proj = True
        state = dict(state)
        state['backbone.legacy_slot_proj.weight'] = state.pop('backbone.slot_proj.weight')
        state['backbone.legacy_slot_proj.bias'] = state.pop('backbone.slot_proj.bias')
    return state


def load_compatible_weights(model, checkpoint_path, device):
    """中文註解：轉移相容骨幹並保留 KD/SD/HH head，為新增鼓件提供語義初始值。"""
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = configure_legacy_projection(model, state)
    target = model.state_dict()
    if 'onset_head.weight' in state and state['onset_head.weight'].shape == target['onset_head.weight'].shape:
        model.load_state_dict(state)
        return len(state)
    compatible = {name: value for name, value in state.items() if name in target and value.shape == target[name].shape}
    target.update(compatible)
    semantic_head_rows = transfer_semantic_head_rows(target, state)
    model.load_state_dict(target)
    return len(compatible) + semantic_head_rows


def load_six_class_checkpoint(model, checkpoint_path, device):
    """中文註解：載入六類候選並還原 legacy 投影分支狀態。"""
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = configure_legacy_projection(model, state)
    model.load_state_dict(state)


def select_train_item(metadata):
    """中文註解：選擇含最多六類標籤的固定 train item，讓 smoke 可重現。"""
    candidates = []
    for key, item in metadata.items():
        if item.get('split') != 'train':
            continue
        labels = {event.get('inst') for event in item.get('events', [])}
        if labels.intersection(LABEL_INDEX):
            candidates.append((len(labels.intersection(LABEL_INDEX)), key, item))
    if not candidates:
        raise ValueError('No six-class STAR train item found.')
    return max(candidates, key=lambda row: (row[0], row[1]))


def load_accompaniment(path):
    """中文註解：以模型採樣率載入單聲道伴奏，供六類域增強重用。"""
    waveform, _ = librosa.load(path, sr=SR, mono=True)
    return np.asarray(waveform, dtype=np.float32)


def mix_accompaniment(waveform, accompaniment, gain, offset=0):
    """中文註解：沿用 Phase 3 peak-relative 公式，把固定長度伴奏片段混入鼓聲。"""
    if gain < 0.0:
        raise ValueError('accompaniment gain must be non-negative')
    start = min(max(0, int(offset)), max(0, len(accompaniment) - len(waveform)))
    segment = accompaniment[start:start + len(waveform)]
    if len(segment) < len(waveform):
        segment = np.pad(segment, (0, len(waveform) - len(segment)))
    drum_peak = float(np.max(np.abs(waveform))) + 1e-6
    accompaniment_peak = float(np.max(np.abs(segment))) + 1e-6
    mixed = waveform + (segment / accompaniment_peak) * drum_peak * gain
    mixed_peak = float(np.max(np.abs(mixed))) + 1e-6
    if mixed_peak > 1.0:
        mixed = mixed / mixed_peak
    return np.asarray(mixed, dtype=np.float32)


def build_window(
    item, anchor=None, accompaniment=None, accompaniment_gain=0.17,
    accompaniment_offset=0, use_true_superflux=False,
):
    """中文註解：讀取一個實體四秒音訊窗口，並建立六類 onset/velocity target。"""
    events = item['events']
    if anchor is None:
        anchor = sorted(float(event['time']) for event in events)[len(events) // 2]
    with sf.SoundFile(item['audio_path']) as audio:
        source_sr = audio.samplerate
        # 中文註解：來源採樣率可能不是 44.1 kHz，先讀取等長的實體四秒再重採樣。
        source_window_samples = int(round(TARGET_SAMPLES * source_sr / SR))
        start_sample = max(0, int(anchor * source_sr) - source_window_samples // 2)
        start_sample = min(start_sample, max(0, audio.frames - source_window_samples))
        audio.seek(start_sample)
        waveform = audio.read(source_window_samples, dtype='float32')
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)
    if source_sr != SR:
        waveform = librosa.resample(waveform, orig_sr=source_sr, target_sr=SR)
    waveform = np.pad(waveform[:TARGET_SAMPLES], (0, max(0, TARGET_SAMPLES - len(waveform))))
    if accompaniment is not None:
        waveform = mix_accompaniment(waveform, accompaniment, accompaniment_gain, accompaniment_offset)
    start_sec = start_sample / float(source_sr)
    onset = np.zeros((CHUNK_FRAMES, len(LABELS)), dtype=np.float32)
    velocity = np.zeros_like(onset)
    window_seconds = TARGET_SAMPLES / float(SR)
    for event in events:
        label = event.get('inst')
        time_sec = float(event['time'])
        if label not in LABEL_INDEX or not start_sec <= time_sec < start_sec + window_seconds:
            continue
        frame = int(round((time_sec - start_sec) * SR / HOP_LENGTH))
        if 0 <= frame < CHUNK_FRAMES:
            index = LABEL_INDEX[label]
            onset[frame, index] = 1.0
            velocity[frame, index] = float(event.get('velocity', 100.0)) / 127.0
    features = extract_features(
        waveform, sr=SR, hop_length=HOP_LENGTH, n_mels=N_MELS,
        use_hybrid=False, use_true_superflux=use_true_superflux,
    )
    features = features[:, :, :CHUNK_FRAMES]
    if features.shape[2] < CHUNK_FRAMES:
        features = np.pad(features, ((0, 0), (0, 0), (0, CHUNK_FRAMES - features.shape[2])))
    return features, onset, velocity, start_sec


def run_self_check():
    """中文註解：確認六類 head 列會從正確的三類語義來源複製。"""
    source, target = {}, {}
    for head_name in ('onset_head', 'velocity_head'):
        source[f'{head_name}.weight'] = torch.arange(3, dtype=torch.float32).view(3, 1, 1)
        source[f'{head_name}.bias'] = torch.arange(3, dtype=torch.float32)
        target[f'{head_name}.weight'] = torch.zeros(6, 1, 1)
        target[f'{head_name}.bias'] = torch.zeros(6)
    assert transfer_semantic_head_rows(target, source) == 24
    assert target['onset_head.weight'][:, 0, 0].tolist() == [0.0, 1.0, 2.0, 1.0, 2.0, 2.0]
    assert target['velocity_head.bias'].tolist() == [0.0, 1.0, 2.0, 1.0, 2.0, 2.0]
    mixed = mix_accompaniment(np.ones(8, dtype=np.float32) * 0.2, np.arange(16, dtype=np.float32), 0.2, 4)
    assert mixed.shape == (8,) and np.isfinite(mixed).all()
    model = SymmetricDrumTCN(num_classes=6)
    configure_legacy_projection(model, {'backbone.legacy_slot_proj.weight': torch.zeros(64, 1024, 1, 1)})
    assert model.backbone.use_legacy_proj
    print('Self-check passed.')


def main():
    """中文註解：執行一個六類 STAR 訓練步，並驗證候選 checkpoint 可重新載入。"""
    parser = argparse.ArgumentParser(description='Run one isolated six-class STAR smoke update.')
    parser.add_argument('--meta')
    parser.add_argument('--checkpoint', default='mixed_formal_kick375_snare18_hh12_candidate.pth')
    parser.add_argument('--output-dir', default='validation_runs/six_class_smoke')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    if not args.meta:
        parser.error('--meta is required unless --self-check is used')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.meta, encoding='utf-8') as handle:
        metadata = json.load(handle)
    coverage = sorted({event.get('inst') for item in metadata.values() for event in item.get('events', []) if event.get('inst') in LABEL_INDEX})
    if coverage != sorted(LABELS):
        raise AssertionError(f'Incomplete six-class coverage: {coverage}')
    class_count, key, item = select_train_item(metadata)
    features, onset, velocity, _ = build_window(item)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SymmetricDrumTCN(num_classes=len(LABELS)).to(device)
    transferred = load_compatible_weights(model, args.checkpoint, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    x = torch.from_numpy(features).float().unsqueeze(0).to(device)
    onset_target = torch.from_numpy(onset).float().unsqueeze(0).to(device)
    velocity_target = torch.from_numpy(velocity).float().unsqueeze(0).to(device)
    model.train()
    onset_logits, velocity_logits = model(x)
    onset_loss = F.binary_cross_entropy_with_logits(onset_logits, onset_target)
    active = onset_target.sum().clamp_min(1.0)
    velocity_loss = ((torch.sigmoid(velocity_logits) - velocity_target).pow(2) * onset_target).sum() / active
    loss = onset_loss + velocity_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if not torch.isfinite(loss):
        raise AssertionError('Smoke loss is not finite.')
    candidate_path = os.path.join(args.output_dir, 'six_class_smoke_candidate.pth')
    torch.save(model.state_dict(), candidate_path)
    reloaded = SymmetricDrumTCN(num_classes=len(LABELS)).to(device)
    load_six_class_checkpoint(reloaded, candidate_path, device)
    reloaded.eval()
    with torch.no_grad():
        onset_out, velocity_out = reloaded(x)
    expected_shape = (1, CHUNK_FRAMES, len(LABELS))
    if tuple(onset_out.shape) != expected_shape or tuple(velocity_out.shape) != expected_shape:
        raise AssertionError(f'Unexpected output shapes: {tuple(onset_out.shape)}, {tuple(velocity_out.shape)}')
    report = {
        'status': 'pass', 'labels': LABELS, 'coverage': coverage, 'selected_item': key,
        'selected_item_label_count': class_count, 'transferred_compatible_tensors': transferred,
        'loss': float(loss.item()), 'candidate': os.path.abspath(candidate_path),
        'output_shape': list(expected_shape), 'source_checkpoint': os.path.abspath(args.checkpoint),
    }
    report_path = os.path.join(args.output_dir, 'smoke_report.json')
    with open(report_path, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
