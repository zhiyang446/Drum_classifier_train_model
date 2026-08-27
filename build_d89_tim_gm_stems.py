"""建立 D89 TimGM Archive 的 DrumSep stems 與替代 manifest。"""

import argparse
import copy
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import soundfile as sf

from build_d52_drumsep_batch import STEMS, file_sha256, input_name


ROOT = Path(__file__).resolve().parent
D88_META = ROOT / 'synthetic_midi_archive_d88_tim_gm' / 'metadata_d88.json'
D54_META = ROOT / 'mixed_d54_stem' / 'metadata_d54.json'
D89_ROOT = ROOT / 'drumsep_d89_tim_gm'
INPUT_ROOT = D89_ROOT / 'input'
OUTPUT_ROOT = D89_ROOT / 'output'
PLAN_PATH = D89_ROOT / 'key_map_d89.json'
PREFLIGHT_PATH = D89_ROOT / 'preflight_d89.json'
AUDIT_PATH = D89_ROOT / 'audit_d89.json'
MANIFEST_ROOT = ROOT / 'mixed_d89_tim_gm_stem'
META_PATH = MANIFEST_ROOT / 'metadata_d89.json'
MANIFEST_AUDIT_PATH = MANIFEST_ROOT / 'audit_d89.json'
EXPECTED_ARCHIVE_TRAIN = 1382
MIN_FREE_GIB = 20.0


def read_json(path):
    """讀取 UTF-8 JSON，讓 D89 所有來源可追溯。"""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new_json(path, payload):
    """只寫入新的 D89 JSON，拒絕覆寫任何既有候選紀錄。"""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def build_plan():
    """將 D88 train 項目一對一映射到 D54 Archive train key 與安全輸入檔名。"""
    d88 = read_json(D88_META)
    d54 = read_json(D54_META)
    train_items = {item_id: item for item_id, item in d88.items() if item['split'] == 'train'}
    if len(train_items) != EXPECTED_ARCHIVE_TRAIN or len(train_items) != len(d88):
        raise ValueError(f'D88 must contain exactly {EXPECTED_ARCHIVE_TRAIN} train-only items.')
    archive_keys = {
        key for key, item in d54.items()
        if item.get('split') == 'train' and item.get('source') == 'd36_archive_synthetic'
    }
    entries, seconds = [], 0.0
    for item_id, item in sorted(train_items.items()):
        d54_key = f'd36_archive_synthetic:{item_id}'
        if d54_key not in archive_keys:
            raise KeyError(f'D54 Archive train key missing: {d54_key}')
        audio_path = Path(item['audio_path'])
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        name = input_name('d89_tim_gm', item_id)
        entries.append({
            'd54_key': d54_key, 'item_id': item_id, 'audio_path': str(audio_path.resolve()),
            'input_name': name, 'duration_seconds': float(item['duration']),
        })
        seconds += float(item['duration'])
    if {row['d54_key'] for row in entries} != archive_keys:
        raise AssertionError('D88 and D54 Archive train keys differ.')
    if len({row['input_name'] for row in entries}) != len(entries):
        raise AssertionError('D89 input names are not unique.')
    d52_plan = read_json(ROOT / 'drumsep_d52' / 'key_map_d52.json')
    d52_seconds = sum(float(row['duration_seconds']) for row in d52_plan['entries'])
    d52_bytes = sum(path.stat().st_size for path in (ROOT / 'drumsep_d52' / 'output').rglob('*.wav'))
    return {
        'phase': 'D89', 'status': 'preflight_complete_not_inference_started',
        'selection': {'tracks': len(entries), 'total_seconds': seconds, 'validation_or_test_read': False},
        'model': {
            'checkpoint': 'Drumsep/MDX23C-DrumSep-aufr33-jarredou.ckpt',
            'checkpoint_sha256': file_sha256(ROOT / 'Drumsep' / 'MDX23C-DrumSep-aufr33-jarredou.ckpt'),
            'config': 'Drumsep/config_drumsep_mdx23c.yaml',
            'config_sha256': file_sha256(ROOT / 'Drumsep' / 'config_drumsep_mdx23c.yaml'),
            'tta': False, 'lora': False,
        },
        'expected_output': {
            'stems_per_track': len(STEMS), 'stem_files': len(entries) * len(STEMS),
            'estimated_bytes_from_d52_density': int(round(seconds * d52_bytes / d52_seconds)),
        },
        'entries': entries,
    }


def prepare(plan):
    """建立全新 D89 hard-link 輸入並拒絕不足空間或既有輸出。"""
    if D89_ROOT.exists():
        raise FileExistsError(f'Refusing to reuse existing D89 root: {D89_ROOT}')
    free_bytes = shutil.disk_usage(ROOT).free
    if free_bytes < int(MIN_FREE_GIB * 1024 ** 3):
        raise RuntimeError(f'Only {free_bytes / 1024 ** 3:.2f} GiB free; D89 requires {MIN_FREE_GIB:.0f} GiB.')
    D89_ROOT.mkdir()
    INPUT_ROOT.mkdir()
    for row in plan['entries']:
        source = Path(row['audio_path'])
        destination = INPUT_ROOT / f"{row['input_name']}.wav"
        os.link(source, destination)
    if len(list(INPUT_ROOT.glob('*.wav'))) != len(plan['entries']):
        raise AssertionError('D89 hard-link count is incomplete.')
    preflight = {**plan, 'free_gib_before_inference': free_bytes / 1024 ** 3}
    write_new_json(PLAN_PATH, plan)
    write_new_json(PREFLIGHT_PATH, preflight)
    return preflight


def audit_stems(plan):
    """驗證官方分離結果完整且每個 D89 stem 符合既有格式。"""
    incomplete, files = [], 0
    for row in plan['entries']:
        paths = [OUTPUT_ROOT / row['input_name'] / f'{stem}.wav' for stem in STEMS]
        valid = all(path.is_file() and path.stat().st_size > 0 and sf.info(path).samplerate == 44100 and sf.info(path).channels == 2 for path in paths)
        if valid:
            files += len(paths)
        else:
            incomplete.append(row['d54_key'])
    payload = {
        'phase': 'D89', 'status': 'pass' if not incomplete else 'incomplete',
        'expected_tracks': len(plan['entries']), 'complete_tracks': len(plan['entries']) - len(incomplete),
        'incomplete_tracks': len(incomplete), 'incomplete_keys': incomplete, 'stem_files_verified': files,
        'validation_or_test_read': False, 'training_started': False,
    }
    write_new_json(AUDIT_PATH, payload)
    if incomplete:
        raise AssertionError(f'D89 stems incomplete: {incomplete[:3]}')
    return payload


def build_manifest(plan):
    """複製 D54，且只把 Archive train 指向 D88 音訊與 D89 stems。"""
    if META_PATH.exists() or MANIFEST_AUDIT_PATH.exists():
        raise FileExistsError(f'Refusing to overwrite D89 manifest: {MANIFEST_ROOT}')
    d54 = read_json(D54_META)
    d88 = read_json(D88_META)
    candidate = copy.deepcopy(d54)
    replaced = set()
    for row in plan['entries']:
        item = candidate[row['d54_key']]
        d88_item = d88[row['item_id']]
        paths = {stem: str((OUTPUT_ROOT / row['input_name'] / f'{stem}.wav').resolve()) for stem in STEMS}
        item['audio_path'] = d88_item['audio_path']
        item['duration'] = d88_item['duration']
        item['rms'] = d88_item['rms']
        item['drumsep_stems'] = {'version': 'drumsep_d89_tim_gm', 'mix_strategy': 'sum_mono', 'paths': paths}
        item['d89_tim_gm_source'] = {'original_audio_path': d54[row['d54_key']]['audio_path'], 'd88_item_id': row['item_id']}
        replaced.add(row['d54_key'])
    if len(candidate) != len(d54) or len(replaced) != EXPECTED_ARCHIVE_TRAIN:
        raise AssertionError('D89 manifest item replacement count failed.')
    if any(candidate[key] != item for key, item in d54.items() if item['split'] != 'train'):
        raise AssertionError('D89 changed a held-out item.')
    group_splits = {}
    stem_files = 0
    for item in candidate.values():
        group_splits.setdefault(item['group_id'], set()).add(item['split'])
        paths = item['drumsep_stems']['paths']
        if set(paths) != set(STEMS):
            raise AssertionError('D89 stem names mismatch.')
        for path in paths.values():
            if not Path(path).is_file():
                raise FileNotFoundError(path)
            stem_files += 1
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise AssertionError('D89 group_id crosses splits.')
    audit = {
        'phase': 'D89', 'status': 'pass', 'items': len(candidate), 'archive_train_replaced': len(replaced),
        'validation_items_unchanged': sum(item['split'] == 'validation' for item in d54.values()),
        'stem_files_verified': stem_files, 'group_split_leaks': 0,
        'sources': dict(Counter(item['source'] for item in candidate.values() if item['split'] == 'train')),
        'ready_for_training_candidate': True, 'ready_for_six_class_release': False,
    }
    write_new_json(META_PATH, candidate)
    write_new_json(MANIFEST_AUDIT_PATH, audit)
    return audit


def run_self_check():
    """驗證 D89 固定規模與 stem 名稱，且不依賴實體資料。"""
    assert EXPECTED_ARCHIVE_TRAIN == 1382
    assert set(STEMS) == {'kick', 'snare', 'toms', 'hh', 'ride', 'crash'}
    assert input_name('d89_tim_gm', 'midi_archive_d27_demo').startswith('d89_tim_gm__')
    print('D89 stem builder self-check passed.')


def main():
    """提供 D89 的 prepare、audit 與 manifest 三個不可覆寫步驟。"""
    parser = argparse.ArgumentParser(description='Build D89 TimGM stems and D54 replacement manifest.')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--build-manifest', action='store_true')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    if sum((args.prepare, args.audit, args.build_manifest)) != 1:
        parser.error('Choose exactly one action.')
    plan = build_plan() if args.prepare else read_json(PLAN_PATH)
    if args.prepare:
        result = prepare(plan)
    elif args.audit:
        result = audit_stems(plan)
    else:
        if read_json(AUDIT_PATH)['status'] != 'pass':
            raise RuntimeError('D89 stems must pass audit before manifest creation.')
        result = build_manifest(plan)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
