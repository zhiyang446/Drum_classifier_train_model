# -*- coding: utf-8 -*-
"""D67：固定 D61/D64 checkpoint 的 TOM 類別專家離線融合審計。"""

import argparse
import json
import os

import numpy as np
import torch

from model_conformer import ResidualDCNNDrumHybridConformer, load_hybrid_conformer_checkpoint
from run_six_class_smoke import build_window
from run_six_class_validation import (
    CHUNK_FRAMES,
    HOP_LENGTH,
    LABELS,
    LABEL_INDEX,
    SR,
    expected_events,
    load_fixed_windows,
    local_maxima,
    write_outputs,
)


TOM_INDEX = LABEL_INDEX['TOM']


def read_json(path):
    """中文註解：讀取 UTF-8 JSON，讓輸入檔案錯誤在審計開始前立即顯示。"""
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def selection_identity(rows):
    """中文註解：抽出會決定封存物理窗口的不可變欄位與原始順序。"""
    fields = ('label', 'key', 'anchor', 'window_start', 'audio_path', 'split')
    return [tuple(row.get(field) for field in fields) for row in rows]


def assert_same_selection(d61_rows, d64_rows):
    """中文註解：拒絕不同選窗，避免把資料差異誤當成 TOM 融合改善。"""
    if selection_identity(d61_rows) != selection_identity(d64_rows):
        raise ValueError('D61/D64 selected_windows are not identical; fusion audit is invalid.')


def fuse_tom_probabilities(d61_probabilities, d64_probabilities):
    """中文註解：只以 D64 TOM 機率覆蓋 D61 TOM，五個非 TOM 類別逐值保留。"""
    if d61_probabilities.shape != d64_probabilities.shape:
        raise ValueError('D61/D64 probability shapes differ.')
    if d61_probabilities.shape[-1] != len(LABELS):
        raise ValueError('Fusion expects exactly six class probabilities.')
    # ponytail: 不做平均或權重搜尋；唯一可驗證的變因就是完整 TOM 欄位替換。
    output = d61_probabilities.copy()
    output[..., TOM_INDEX] = d64_probabilities[..., TOM_INDEX]
    return output


def load_model(checkpoint_path, device):
    """中文註解：以既有 hybrid loader 嚴格還原凍結 checkpoint。"""
    model = ResidualDCNNDrumHybridConformer(num_classes=len(LABELS)).to(device)
    load_hybrid_conformer_checkpoint(model, checkpoint_path, device)
    model.eval()
    return model


def run_self_check():
    """中文註解：確認 TOM 唯一替換，並確認選窗差異必定被拒絕。"""
    d61 = np.zeros((2, len(LABELS)), dtype=np.float32)
    d64 = np.ones((2, len(LABELS)), dtype=np.float32)
    fused = fuse_tom_probabilities(d61, d64)
    assert np.array_equal(fused[..., :TOM_INDEX], d61[..., :TOM_INDEX])
    assert np.array_equal(fused[..., TOM_INDEX + 1:], d61[..., TOM_INDEX + 1:])
    assert np.array_equal(fused[..., TOM_INDEX], d64[..., TOM_INDEX])
    identical = [{'label': 'KD', 'key': 'a', 'anchor': 1.0, 'window_start': 0.0, 'audio_path': 'a.wav', 'split': 'validation'}]
    assert_same_selection(identical, list(identical))
    try:
        assert_same_selection(identical, [{**identical[0], 'anchor': 2.0}])
    except ValueError:
        pass
    else:
        raise AssertionError('Selection mismatch must be rejected.')
    print('Self-check passed.')


def audit(d61_checkpoint, d64_checkpoint, metadata_path, d61_selection_path, d64_selection_path,
          output_dir, phase, base_label, base_macro_f1):
    """中文註解：在封存窗口推論兩個模型、融合 TOM，並寫出可追溯的固定門檻證據。"""
    if os.path.exists(output_dir):
        raise FileExistsError(f'Output directory already exists: {output_dir}')
    d61_selection = read_json(d61_selection_path)
    d64_selection = read_json(d64_selection_path)
    assert_same_selection(d61_selection, d64_selection)
    metadata = read_json(metadata_path)
    # 重要變數：windows 只能由已驗證相同的 D61 封存 selection 重建。
    windows = load_fixed_windows(metadata, d61_selection_path, split='validation', per_class=8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d61_model = load_model(d61_checkpoint, device)
    d64_model = load_model(d64_checkpoint, device)
    aggregate = {label: ([], []) for label in LABELS}
    selected_rows = []
    window_seconds = CHUNK_FRAMES * HOP_LENGTH / float(SR)
    for window_index, selected in enumerate(windows):
        features, _, _, start_sec = build_window(
            selected['item'], selected['anchor'], use_true_superflux=True,
            use_multi_log_mel=False, input_mode='drumsep-mix',
        )
        feature_tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)
        with torch.no_grad():
            d61_probabilities = torch.sigmoid(d61_model(feature_tensor)[0]).squeeze(0).cpu().numpy()
            d64_probabilities = torch.sigmoid(d64_model(feature_tensor)[0]).squeeze(0).cpu().numpy()
        predicted = local_maxima(fuse_tom_probabilities(d61_probabilities, d64_probabilities))
        expected = expected_events(selected['item'], start_sec)
        aggregate_offset = window_index * (window_seconds + 1.0)
        for label in LABELS:
            aggregate[label][0].extend(time + aggregate_offset for time in expected[label])
            aggregate[label][1].extend(time + aggregate_offset for time in predicted[label])
        selected_rows.append({
            'label': selected['label'], 'key': selected['key'], 'anchor': selected['anchor'],
            'window_start': start_sec, 'audio_path': selected['item']['audio_path'], 'split': 'validation',
            'aggregate_offset': aggregate_offset, 'architecture': 'dcnn-tcn-conformer',
            'feature_mode': 'true-superflux', 'input_mode': 'drumsep-mix',
            'fusion_recipe': f'{base_label} KD/SD/HH/CRASH/RIDE + D64 TOM',
            'd61_checkpoint': d61_checkpoint, 'd64_checkpoint': d64_checkpoint,
            'd61_selection_source': d61_selection_path, 'd64_selection_source': d64_selection_path,
        })
    rows, gate = write_outputs(selected_rows, aggregate, output_dir)
    macro_f1 = float(gate['macro_f1'])
    summary = {
        'phase': phase,
        'recipe': f'{base_label} KD/SD/HH/CRASH/RIDE + D64 TOM',
        'd61_checkpoint': d61_checkpoint,
        'd64_checkpoint': d64_checkpoint,
        'selected_windows': len(selected_rows),
        'selection_identical': True,
        'fixed_threshold': gate['threshold'],
        'tolerance_seconds': gate['tolerance_seconds'],
        'macro_f1': macro_f1,
        'base_label': base_label,
        'base_macro_f1': base_macro_f1,
        'improves_base_research_baseline': macro_f1 > base_macro_f1,
        'release_gate': gate,
        'research_status': 'research_baseline_only' if macro_f1 > base_macro_f1 else 'rejected',
        'ready_for_six_class_release': False,
        'per_class': {row['inst']: float(row['f1']) for row in rows},
    }
    with open(os.path.join(output_dir, 'fusion_summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main():
    """中文註解：處理 CLI 參數，執行 D67 離線融合或最小自檢。"""
    parser = argparse.ArgumentParser(description='Audit fixed five-class/TOM expert fusion without training.')
    parser.add_argument('--d61-checkpoint', default='validation_runs/d61_kd_negative_candidate/d61_kd_negative_candidate.pth')
    parser.add_argument('--d64-checkpoint', default='validation_runs/d64_tom_competitor_candidate/d64_tom_competitor_candidate.pth')
    parser.add_argument('--metadata', default='mixed_d54_stem/metadata_d54.json')
    parser.add_argument('--d61-selection', default='validation_runs/d61_kd_negative_candidate/independent_validation/selected_windows.json')
    parser.add_argument('--d64-selection', default='validation_runs/d64_tom_competitor_candidate/independent_validation/selected_windows.json')
    parser.add_argument('--output-dir', default='validation_runs/d67_d61_d64_tom_fusion')
    parser.add_argument('--phase', default='D67')
    parser.add_argument('--base-label', default='D61')
    parser.add_argument('--base-macro-f1', type=float, default=0.5267)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    summary = audit(
        args.d61_checkpoint, args.d64_checkpoint, args.metadata,
        args.d61_selection, args.d64_selection, args.output_dir,
        args.phase, args.base_label, args.base_macro_f1,
    )
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
