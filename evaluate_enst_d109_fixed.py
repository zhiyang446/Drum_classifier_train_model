import argparse
import json
import os

import torch

from run_six_class_validation import LABELS, select_windows
from train_d77_fused_lora import (
    evaluate_fixed_fusion,
    file_sha256,
    load_frozen_lora_model,
    load_parent_adapter,
)


def read_json(path):
    """中文註解：讀取固定 metadata、既有報告或選窗檔。"""
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def load_adapter_pair(adapter_path, args, device):
    """中文註解：獨立載入一組凍結 base 與指定 adapter，避免兩次評估互相污染。"""
    d76_model = load_frozen_lora_model(args.d76_checkpoint, device, args.rank, args.alpha)
    d64_model = load_frozen_lora_model(args.d64_checkpoint, device, args.rank, args.alpha)
    payload = load_parent_adapter(adapter_path, args, d76_model, d64_model)
    return d76_model, d64_model, payload


def diagnose(parent_macro, candidate_macro, d56_parent, d56_candidate):
    """中文註解：用固定 ENST 與既有 D56 方向判斷未學到或域別衝突。"""
    if candidate_macro <= parent_macro:
        return 'candidate_did_not_improve_enst'
    if d56_candidate < d56_parent:
        return 'domain_tradeoff_confirmed'
    return 'enst_improved_without_d56_regression'


def run_self_check():
    """中文註解：鎖定三種診斷分支，避免報告把退步誤判成提升。"""
    assert diagnose(0.50, 0.49, 0.55, 0.54) == 'candidate_did_not_improve_enst'
    assert diagnose(0.50, 0.51, 0.55, 0.54) == 'domain_tradeoff_confirmed'
    assert diagnose(0.50, 0.51, 0.55, 0.56) == 'enst_improved_without_d56_regression'
    print('D109 self-check passed.')


def main():
    """中文註解：零訓練比較 D89 與指定候選在相同 ENST validation 視窗的結果。"""
    parser = argparse.ArgumentParser(description='Compare D89 and one candidate on fixed ENST validation windows.')
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--phase', default='D109')
    parser.add_argument('--candidate-label', default='d108_epoch1')
    parser.add_argument('--metadata', default='enst_d107/metadata_d107_validation.json')
    parser.add_argument('--d89-adapter', default='validation_runs/d89_d82_tim_gm_lora_retry/d89_d82_tim_gm_lora_retry_adapter.pth')
    parser.add_argument('--candidate-adapter', '--d108-adapter', dest='candidate_adapter', default='validation_runs/d108_d89_enst_lora_candidate/d108_d89_enst_lora_adapter_epoch1.pth')
    parser.add_argument('--candidate-report', '--d108-report', dest='candidate_report', default='validation_runs/d108_d89_enst_lora_candidate/train_report.json')
    parser.add_argument('--d76-checkpoint', default='validation_runs/d76_crash_kd_retry_candidate/d76_crash_kd_retry_candidate.pth')
    parser.add_argument('--d64-checkpoint', default='validation_runs/d64_tom_competitor_candidate/d64_tom_competitor_candidate.pth')
    parser.add_argument('--output-dir', default='validation_runs/d109_enst_fixed_validation')
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--alpha', type=float, default=8.0)
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    if not args.candidate_label.replace('_', '').isalnum():
        raise ValueError('candidate-label must contain only letters, numbers or underscores.')
    if os.path.exists(args.output_dir):
        raise FileExistsError(f'Output directory already exists: {args.output_dir}')
    for path in (
        args.metadata, args.d89_adapter, args.candidate_adapter, args.candidate_report,
        args.d76_checkpoint, args.d64_checkpoint,
    ):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    metadata = read_json(args.metadata)
    if any(item.get('split') != 'validation' for item in metadata.values()):
        raise ValueError('ENST evaluator accepts validation metadata only.')
    if any('drummer_3' in f'{key} {item.get("audio_path", "")}'.lower() for key, item in metadata.items()):
        raise ValueError('Sealed ENST drummer_3 data is forbidden.')
    selected = select_windows(metadata, split='validation', per_class=8)
    counts = {label: sum(row['label'] == label for row in selected) for label in LABELS}
    groups = {str(row['item'].get('group_id') or row['key']) for row in selected}
    if len(selected) != 48 or counts != {label: 8 for label in LABELS} or len(groups) != 48:
        raise AssertionError(f'Unexpected fixed selection: windows={len(selected)}, counts={counts}, groups={len(groups)}')

    os.makedirs(args.output_dir)
    selection_path = os.path.join(args.output_dir, 'selected_windows.json')
    with open(selection_path, 'w', encoding='utf-8') as handle:
        json.dump(
            [{key: row[key] for key in ('label', 'key', 'anchor')} for row in selected],
            handle, indent=2,
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    for name, adapter_path in (('d89_parent', args.d89_adapter), (args.candidate_label, args.candidate_adapter)):
        d76_model, d64_model, _ = load_adapter_pair(adapter_path, args, device)
        macro, per_class, gate = evaluate_fixed_fusion(
            d76_model, d64_model, metadata, selection_path, os.path.join(args.output_dir, name),
        )
        results[name] = {'macro_f1': macro, 'per_class_f1': per_class, 'gate': gate}
        del d76_model, d64_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    candidate_report = read_json(args.candidate_report)
    d56_parent = float(candidate_report['parent_fixed_validation_macro_f1'])
    d56_candidate = float(candidate_report['epochs'][0]['fixed_validation_macro_f1'])
    parent = results['d89_parent']
    candidate = results[args.candidate_label]
    summary = {
        'phase': args.phase,
        'candidate_label': args.candidate_label,
        'device': str(device),
        'training_started': False,
        'sealed_test_read': False,
        'selection': {'windows': len(selected), 'per_class': counts, 'unique_groups': len(groups)},
        'adapters': {
            'd89_parent': {'path': os.path.abspath(args.d89_adapter), 'sha256': file_sha256(args.d89_adapter)},
            args.candidate_label: {'path': os.path.abspath(args.candidate_adapter), 'sha256': file_sha256(args.candidate_adapter)},
        },
        'enst_validation': results,
        'enst_delta': {
            'macro_f1': candidate['macro_f1'] - parent['macro_f1'],
            'per_class_f1': {
                label: candidate['per_class_f1'][label] - parent['per_class_f1'][label]
                for label in LABELS
            },
        },
        'd56_reference': {
            'd89_parent_macro_f1': d56_parent,
            f'{args.candidate_label}_macro_f1': d56_candidate,
            'delta': d56_candidate - d56_parent,
        },
        'diagnosis': diagnose(parent['macro_f1'], candidate['macro_f1'], d56_parent, d56_candidate),
        'promotion_allowed': False,
    }
    with open(os.path.join(args.output_dir, 'summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
