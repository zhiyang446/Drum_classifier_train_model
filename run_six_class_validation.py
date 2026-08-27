# -*- coding: utf-8 -*-
"""使用 STAR test split 驗證獨立六類候選模型。"""
import argparse
import csv
import json
import os

import numpy as np
import torch

from model_dcnn import DCNNDrumTCN, ResidualDCNNDrumTCN, load_dcnn_checkpoint, load_residual_dcnn_checkpoint
from model_conformer import ResidualDCNNDrumConformer, ResidualDCNNDrumHybridConformer, load_conformer_checkpoint, load_hybrid_conformer_checkpoint
from run_egmd_round4_validation import match_events
from run_six_class_smoke import CHUNK_FRAMES, HOP_LENGTH, LABELS, LABEL_INDEX, SR, TARGET_SAMPLES, build_window, load_accompaniment, load_six_class_checkpoint
from train_phase2 import SymmetricDrumTCN


THRESHOLD = 0.50
TOLERANCE = 0.050
MACRO_F1_MIN = 0.70
PER_CLASS_F1_MIN = 0.55


def physical_window(item, anchor):
    """中文註解：以 metadata 時長重建 build_window 的四秒物理邊界，供重疊去重。"""
    window_seconds = CHUNK_FRAMES * HOP_LENGTH / float(SR)
    latest_start = max(0.0, float(item.get('duration', window_seconds)) - window_seconds)
    start = min(max(0.0, float(anchor) - window_seconds / 2.0), latest_start)
    return start, start + window_seconds


def overlaps_existing(audio_path, start, end, occupied):
    """中文註解：同一音訊只要物理窗口相交就視為重複，避免事件被重複計分。"""
    return any(path == audio_path and start < old_end and old_start < end for path, old_start, old_end in occupied)


def select_windows(metadata, split='validation', per_class=8):
    """中文註解：從指定 split 為每類輪替歌曲群組選取固定且不重疊的標註窗口。"""
    if per_class <= 0:
        raise ValueError('per_class must be positive.')
    selected = []
    occupied = []
    for label in LABELS:
        candidates = []
        for key, item in metadata.items():
            if item.get('split') != split:
                continue
            for event in item.get('events', []):
                if event.get('inst') == label:
                    candidates.append((key, float(event['time']), item))
        if not candidates:
            raise ValueError(f'No STAR {split} event for {label}.')
        # ponytail: 每輪每 group 只取一個；需要權重取樣時才另加策略。
        by_group = {}
        for key, anchor, item in sorted(candidates, key=lambda row: (row[0], row[1])):
            group_id = str(item.get('group_id') or key)
            by_group.setdefault(group_id, []).append((key, anchor, item))
        positions = {group_id: 0 for group_id in by_group}
        class_count = 0
        while class_count < per_class:
            progressed = False
            for group_id in sorted(by_group):
                rows = by_group[group_id]
                while positions[group_id] < len(rows):
                    key, anchor, item = rows[positions[group_id]]
                    positions[group_id] += 1
                    start, end = physical_window(item, anchor)
                    audio_path = item['audio_path']
                    if overlaps_existing(audio_path, start, end, occupied):
                        continue
                    selected.append({'label': label, 'key': key, 'anchor': anchor, 'item': item})
                    occupied.append((audio_path, start, end))
                    class_count += 1
                    progressed = True
                    break
                if class_count >= per_class:
                    break
            if not progressed:
                break
        if class_count < per_class:
            raise ValueError(f'Only {class_count} non-overlapping STAR {split} windows for {label}; need {per_class}.')
    return selected


def fixed_windows(metadata, stored_rows, split='validation', per_class=8):
    """中文註解：以封存 key／anchor 重建同一批物理窗口，避免重選造成比較偏差。"""
    if len(stored_rows) != per_class * len(LABELS):
        raise ValueError('Fixed selection has an unexpected window count.')
    selected, occupied, label_counts = [], [], {label: 0 for label in LABELS}
    for row in stored_rows:
        label = row.get('label')
        key = row.get('key')
        anchor = float(row.get('anchor', float('nan')))
        if label not in LABELS or key not in metadata or not np.isfinite(anchor):
            raise ValueError('Fixed selection contains an invalid label, key, or anchor.')
        item = metadata[key]
        if item.get('split') != split:
            raise ValueError(f'Fixed selection leaves the {split} split: {key}')
        start, end = physical_window(item, anchor)
        if overlaps_existing(item['audio_path'], start, end, occupied):
            raise ValueError(f'Fixed selection has overlapping physical windows: {key}')
        occupied.append((item['audio_path'], start, end))
        label_counts[label] += 1
        selected.append({'label': label, 'key': key, 'anchor': anchor, 'item': item})
    if any(count != per_class for count in label_counts.values()):
        raise ValueError(f'Fixed selection class counts are invalid: {label_counts}')
    return selected


def load_fixed_windows(metadata, selected_windows_path, split='validation', per_class=8):
    """中文註解：讀取封存選窗檔後交由共用驗證函式做完整防護。"""
    with open(selected_windows_path, encoding='utf-8') as handle:
        return fixed_windows(metadata, json.load(handle), split, per_class)


def local_maxima(probabilities):
    """中文註解：以固定門檻擷取六類 onset 局部峰值，避免依個案調整。"""
    events = {label: [] for label in LABELS}
    for label, index in LABEL_INDEX.items():
        values = probabilities[:, index]
        for frame in range(1, len(values) - 1):
            if values[frame] >= THRESHOLD and values[frame] >= values[frame - 1] and values[frame] > values[frame + 1]:
                events[label].append(frame * HOP_LENGTH / float(SR))
    return events


def fused_probabilities(base_logits, rare_logits=None):
    """中文註解：保留基礎塔前三類，並以 rare 塔後三類建立固定六類機率。"""
    base_probabilities = torch.sigmoid(base_logits)
    if rare_logits is None:
        return base_probabilities
    rare_probabilities = torch.sigmoid(rare_logits)
    if base_probabilities.shape[-1] < 3 or rare_probabilities.shape[-1] < 6:
        raise ValueError('Dual-tower fusion requires Model A >= 3 classes and Model B >= 6 classes.')
    # ponytail: 僅重用既有雙塔的固定 3+3 拼接；有跨架構需求時再擴充。
    output = torch.zeros(*rare_probabilities.shape[:-1], 6, device=rare_probabilities.device, dtype=rare_probabilities.dtype)
    output[..., :3] = base_probabilities[..., :3]
    output[..., 3:6] = rare_probabilities[..., 3:6]
    return output


def checkpoint_class_count(state):
    """中文註解：從 Symmetric checkpoint 的 onset head 列數取得原始類別數。"""
    if 'onset_head.weight' not in state:
        raise ValueError('Checkpoint is missing onset_head.weight.')
    return int(state['onset_head.weight'].shape[0])


def load_symmetric_model(checkpoint_path, device):
    """中文註解：依 checkpoint 原始類別數建立並嚴格載入 Symmetric 模型。"""
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # 重要變數：class_count 保留歷史三類 Model A 與六類 Model B 的原始 head 形狀。
    class_count = checkpoint_class_count(state)
    model = SymmetricDrumTCN(num_classes=class_count).to(device)
    load_six_class_checkpoint(model, checkpoint_path, device)
    model.eval()
    return model


def expected_events(item, start_sec):
    """中文註解：轉換指定物理窗口內的六類標註為窗口相對時間。"""
    end_sec = start_sec + CHUNK_FRAMES * HOP_LENGTH / float(SR)
    output = {label: [] for label in LABELS}
    for event in item.get('events', []):
        label = event.get('inst')
        time_sec = float(event['time'])
        if label in output and start_sec <= time_sec < end_sec:
            output[label].append(time_sec - start_sec)
    return output


def write_outputs(selected_rows, aggregate, output_dir):
    """中文註解：寫出可追溯的窗口選樣、逐類事件統計與固定門檻摘要。"""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'selected_windows.json'), 'w', encoding='utf-8') as handle:
        json.dump(selected_rows, handle, indent=2)
    fields = ['inst', 'expected', 'predicted', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'overall']
    rows = []
    for label in LABELS:
        expected, predicted = aggregate[label]
        tp, fp, fn, precision, recall, f1 = match_events(expected, predicted, TOLERANCE)
        rows.append({
            'inst': label, 'expected': len(expected), 'predicted': len(predicted), 'tp': tp, 'fp': fp, 'fn': fn,
            'precision': f'{precision:.4f}', 'recall': f'{recall:.4f}', 'f1': f'{f1:.4f}',
            'overall': 'pass' if f1 >= PER_CLASS_F1_MIN else 'fail',
        })
    with open(os.path.join(output_dir, 'event_compare.csv'), 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    macro_f1 = float(np.mean([float(row['f1']) for row in rows]))
    gate = {
        'gate': 'six_class_star_test_event',
        'overall': 'pass' if macro_f1 >= MACRO_F1_MIN and all(row['overall'] == 'pass' for row in rows) else 'fail',
        'macro_f1': round(macro_f1, 4),
        'macro_f1_min': MACRO_F1_MIN,
        'per_class_f1_min': PER_CLASS_F1_MIN,
        'threshold': THRESHOLD,
        'tolerance_seconds': TOLERANCE,
        'selected_windows': len(selected_rows),
    }
    with open(os.path.join(output_dir, 'gate_summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(gate, handle, indent=2)
    return rows, gate


def run_self_check():
    """中文註解：確認固定峰值與六類選樣邏輯可用。"""
    assert local_maxima(np.array([[0.0] * 6, [0.6] + [0.0] * 5, [0.2] + [0.0] * 5], dtype=np.float32))['KD'] == [HOP_LENGTH / float(SR)]
    test_meta = {
        f'case_{label}': {
            'split': 'validation', 'duration': 8.0, 'audio_path': f'{label}.wav',
            'events': [{'inst': label, 'time': 1.0}],
        }
        for label in LABELS
    }
    assert [row['label'] for row in select_windows(test_meta, 'validation', 1)] == list(LABELS)
    grouped_meta = {
        f'{label}_{group_id}': {
            'split': 'validation', 'duration': 8.0, 'audio_path': f'{label}_{group_id}.wav',
            'group_id': group_id, 'events': [{'inst': label, 'time': 1.0}],
        }
        for label in LABELS for group_id in ('group_a', 'group_b')
    }
    grouped_rows = select_windows(grouped_meta, 'validation', 2)
    assert [row['item']['group_id'] for row in grouped_rows if row['label'] == 'KD'] == ['group_a', 'group_b']
    fixed_meta = {
        f'fixed_{label}': {
            'split': 'validation', 'duration': 8.0, 'audio_path': f'fixed_{label}.wav',
            'events': [{'inst': label, 'time': 1.0}],
        }
        for label in LABELS
    }
    fixed_rows = [
        {'label': label, 'key': f'fixed_{label}', 'anchor': 1.0}
        for label in LABELS
    ]
    assert [row['key'] for row in fixed_windows(fixed_meta, fixed_rows, 'validation', 1)] == [
        f'fixed_{label}' for label in LABELS
    ]
    assert overlaps_existing('same.wav', 1.0, 5.0, [('same.wav', 4.0, 8.0)])
    assert not overlaps_existing('same.wav', 0.0, 4.0, [('same.wav', 4.0, 8.0)])
    base_logits = torch.tensor([[[0.0, 1.0, 2.0]]])
    rare_logits = torch.tensor([[[3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]])
    fused = fused_probabilities(base_logits, rare_logits)
    assert fused.shape == (1, 1, 6)
    assert torch.equal(fused[..., :3], torch.sigmoid(base_logits))
    assert torch.equal(fused[..., 3:6], torch.sigmoid(rare_logits[..., 3:6]))
    assert checkpoint_class_count({'onset_head.weight': torch.zeros(3, 64, 1)}) == 3
    assert checkpoint_class_count({'onset_head.weight': torch.zeros(6, 64, 1)}) == 6
    print('Self-check passed.')


def evaluate_model(
    model, metadata, output_dir, split='validation', per_class=8,
    accompaniment=None, accompaniment_path=None, accompaniment_gain=0.17,
    architecture='symmetric', feature_mode='legacy-diff', device=None,
    use_multi_log_mel=False, rare_model=None, rare_model_path=None,
    selected_windows_path=None, input_mode='mix',
):
    """中文註解：以已載入模型執行共用六類驗證，供 CLI 與逐 epoch 訓練共用。"""
    device = device or next(model.parameters()).device
    model.eval()
    aggregate = {label: ([], []) for label in LABELS}
    selected_rows = []
    window_seconds = CHUNK_FRAMES * HOP_LENGTH / float(SR)
    # ponytail: 固定選窗只供可追溯重評；一般訓練與驗證仍走既有選窗。
    windows = load_fixed_windows(metadata, selected_windows_path, split, per_class) if selected_windows_path else select_windows(metadata, split, per_class)
    for window_index, selected in enumerate(windows):
        accompaniment_offset = window_index * TARGET_SAMPLES
        features, _, _, start_sec = build_window(
            selected['item'], selected['anchor'], accompaniment=accompaniment,
            accompaniment_gain=accompaniment_gain, accompaniment_offset=accompaniment_offset,
            use_true_superflux=feature_mode == 'true-superflux',
            use_multi_log_mel=use_multi_log_mel,
            input_mode=input_mode,
        )
        feature_tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _ = model(feature_tensor)
            rare_logits = rare_model(feature_tensor)[0] if rare_model else None
        predicted = local_maxima(fused_probabilities(logits, rare_logits).squeeze(0).cpu().numpy())
        expected = expected_events(selected['item'], start_sec)
        aggregate_offset = window_index * (window_seconds + 1.0)
        for label in LABELS:
            aggregate[label][0].extend(time + aggregate_offset for time in expected[label])
            aggregate[label][1].extend(time + aggregate_offset for time in predicted[label])
        selected_rows.append({
            'label': selected['label'], 'key': selected['key'], 'anchor': selected['anchor'],
            'window_start': start_sec, 'audio_path': selected['item']['audio_path'],
            'split': split, 'aggregate_offset': aggregate_offset,
            'accompaniment': accompaniment_path, 'accompaniment_gain': accompaniment_gain,
            'architecture': architecture, 'feature_mode': feature_mode,
            'input_mode': input_mode,
            'model_rare': rare_model_path,
            'fixed_windows_source': selected_windows_path,
            'expected_counts': {label: len(expected[label]) for label in LABELS},
        })
    return write_outputs(selected_rows, aggregate, output_dir)


def resolve_feature_mode(architecture, feature_mode):
    """將驗證特徵明確綁定到報告，而不是由架構隱式決定。"""
    if feature_mode:
        return feature_mode
    return 'true-superflux' if architecture == 'dcnn-tcn' else 'legacy-diff'


def main():
    """中文註解：對六類候選執行 STAR test 物理事件驗收。"""
    parser = argparse.ArgumentParser(description='Validate an isolated six-class candidate on STAR test data.')
    parser.add_argument('--meta')
    parser.add_argument('--model')
    parser.add_argument('--model-rare', help='Optional historic six-class rare tower; only supported with symmetric architecture.')
    parser.add_argument('--output-dir')
    parser.add_argument('--split', default='validation', choices=('validation', 'test'))
    parser.add_argument('--per-class', type=int, default=8)
    parser.add_argument('--accompaniment')
    parser.add_argument('--accompaniment-gain', type=float, default=0.17)
    parser.add_argument('--architecture', choices=('symmetric', 'dcnn-tcn', 'dcnn-residual-tcn', 'dcnn-conformer', 'dcnn-tcn-conformer'), default='symmetric')
    parser.add_argument('--feature-mode', choices=('legacy-diff', 'true-superflux'))
    parser.add_argument('--selected-windows', help='Use an existing selection JSON for fixed-window re-evaluation.')
    parser.add_argument('--input-mode', choices=('mix', 'drumsep-mix'), default='mix')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    if not args.meta or not args.model or not args.output_dir:
        parser.error('--meta, --model, and --output-dir are required unless --self-check is used')
    args.feature_mode = resolve_feature_mode(args.architecture, args.feature_mode)
    if args.model_rare and args.architecture != 'symmetric':
        parser.error('--model-rare only supports the historic symmetric dual-tower comparison.')
    with open(args.meta, encoding='utf-8') as handle:
        metadata = json.load(handle)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.architecture == 'dcnn-tcn':
        model = DCNNDrumTCN(num_classes=len(LABELS)).to(device)
        load_dcnn_checkpoint(model, args.model, device)
    elif args.architecture == 'dcnn-residual-tcn':
        model = ResidualDCNNDrumTCN(num_classes=len(LABELS)).to(device)
        load_residual_dcnn_checkpoint(model, args.model, device)
    elif args.architecture == 'dcnn-conformer':
        model = ResidualDCNNDrumConformer(num_classes=len(LABELS)).to(device)
        load_conformer_checkpoint(model, args.model, device)
    elif args.architecture == 'dcnn-tcn-conformer':
        model = ResidualDCNNDrumHybridConformer(num_classes=len(LABELS)).to(device)
        load_hybrid_conformer_checkpoint(model, args.model, device)
    else:
        model = load_symmetric_model(args.model, device)
    rare_model = None
    if args.model_rare:
        rare_model = load_symmetric_model(args.model_rare, device)
    accompaniment = load_accompaniment(args.accompaniment) if args.accompaniment else None
    rows, gate = evaluate_model(
        model, metadata, args.output_dir, split=args.split, per_class=args.per_class,
        accompaniment=accompaniment, accompaniment_path=args.accompaniment,
        accompaniment_gain=args.accompaniment_gain, architecture=args.architecture,
        feature_mode=args.feature_mode, device=device,
        rare_model=rare_model, rare_model_path=args.model_rare,
        selected_windows_path=args.selected_windows,
        input_mode=args.input_mode,
    )
    print(json.dumps({'gate': gate, 'rows': rows}, indent=2))


if __name__ == '__main__':
    main()
