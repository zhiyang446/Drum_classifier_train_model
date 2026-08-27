"""建立 D54 全量 stem manifest；保留 D50 標註、split 與 auxiliary mask。"""

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import soundfile as sf

from build_d52_drumsep_batch import STEMS, read_json, write_json


ROOT = Path(__file__).resolve().parent
D50_META = ROOT / 'mixed_d50_stem_candidate' / 'metadata_d50.json'
D52_PLAN = ROOT / 'drumsep_d52' / 'key_map_d52.json'
D53_PLAN = ROOT / 'drumsep_d53' / 'key_map_d53.json'
OUTPUT_ROOT = ROOT / 'mixed_d54_stem'
META_PATH = OUTPUT_ROOT / 'metadata_d54.json'
AUDIT_PATH = OUTPUT_ROOT / 'audit_d54.json'


def stem_paths(root, input_name):
    """由官方推論固定輸出結構建立六個 stem 路徑。"""
    return {stem: str(root / input_name / f'{stem}.wav') for stem in STEMS}


def planned_paths(plan_path, output_dir):
    """將 D52/D53 key map 轉成可驗證的 key→stem path 映射。"""
    plan = read_json(plan_path)
    return {
        row['key']: stem_paths(output_dir, row['input_name'])
        for row in plan['entries']
    }


def build_manifest():
    """複製 D50，為每筆資料加入同版六 stem，而不改任何既有欄位。"""
    source = read_json(D50_META)
    d52_paths = planned_paths(D52_PLAN, ROOT / 'drumsep_d52' / 'output')
    d53_paths = planned_paths(D53_PLAN, ROOT / 'drumsep_d53' / 'output')
    candidate = copy.deepcopy(source)
    for key, item in candidate.items():
        if key in d52_paths:
            paths = d52_paths[key]
        elif key in d53_paths:
            paths = d53_paths[key]
        else:
            auxiliary = item.get('drumsep_stem_auxiliary')
            if not auxiliary:
                raise ValueError(f'D50 item has no D48/D52/D53 stem mapping: {key}')
            paths = auxiliary['stem_paths']
        item['drumsep_stems'] = {
            'version': 'drumsep_d47_d48_d52_d53',
            'mix_strategy': 'sum_mono',
            'paths': paths,
        }
    return source, candidate


def audit(source, candidate):
    """驗證完整 stem、split 隔離及 D50 原始欄位完全不變。"""
    if source.keys() != candidate.keys() or len(candidate) != 1460:
        raise ValueError('D54 must preserve exactly 1,460 D50 keys.')
    train, validation, group_splits, files = 0, 0, {}, 0
    for key, original in source.items():
        item = candidate[key]
        stems = item.pop('drumsep_stems')
        if item != original:
            raise ValueError(f'D54 changed a D50 field for {key}')
        candidate[key]['drumsep_stems'] = stems
        paths = stems['paths']
        if set(paths) != set(STEMS):
            raise ValueError(f'D54 stem names mismatch for {key}')
        for path_text in paths.values():
            path = Path(path_text)
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f'Missing D54 stem: {path}')
            info = sf.info(path)
            if info.samplerate != 44100 or info.channels != 2:
                raise ValueError(f'Unexpected D54 stem format: {path}')
            files += 1
        split = original['split']
        train += split == 'train'
        validation += split == 'validation'
        group_splits.setdefault(original['group_id'], set()).add(split)
    leaks = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    if train != 1452 or validation != 8 or leaks:
        raise ValueError(f'D54 split failure: train={train}, validation={validation}, leaks={leaks}')
    mask_counts = Counter()
    for item in candidate.values():
        for event in item.get('drumsep_stem_auxiliary', {}).get('ignored_events', []):
            mask_counts[event['inst']] += 1
    if mask_counts != Counter({'RIDE': 2}):
        raise ValueError(f'D50 auxiliary mask changed: {dict(mask_counts)}')
    return {
        'phase': 'D54',
        'status': 'full_stem_manifest_complete_not_training',
        'items': len(candidate),
        'train_items': train,
        'validation_items': validation,
        'stem_files_verified': files,
        'group_split_leaks': len(leaks),
        'auxiliary_ignored_event_counts': dict(mask_counts),
        'ready_for_training_candidate': False,
        'ready_for_six_class_release': False,
    }


def run_self_check():
    """確認固定六 stem 結構與版本名不會意外改變。"""
    assert set(STEMS) == {'kick', 'snare', 'toms', 'hh', 'ride', 'crash'}
    assert Path(stem_paths(Path('out'), 'key')['kick']) == Path('out') / 'key' / 'kick.wav'
    print('D54 self-check passed.')


def main():
    """建立 D54 全量 manifest 與稽核。"""
    parser = argparse.ArgumentParser(description='Build D54 full train/validation stem manifest.')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    source, candidate = build_manifest()
    summary = audit(source, candidate)
    write_json(META_PATH, candidate)
    write_json(AUDIT_PATH, summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
