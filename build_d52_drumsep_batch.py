"""建立與稽核 D52 的 DrumSep 訓練資料批次；不啟動模型推論。"""

import argparse
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path

import soundfile as sf


ROOT = Path(__file__).resolve().parent
D50_META = ROOT / 'mixed_d50_stem_candidate' / 'metadata_d50.json'
D48_AUDIT = ROOT / 'drumsep_d48' / 'audit_d48.json'
D52_ROOT = ROOT / 'drumsep_d52'
INPUT_ROOT = D52_ROOT / 'input'
OUTPUT_ROOT = D52_ROOT / 'output'
KEY_MAP_PATH = D52_ROOT / 'key_map_d52.json'
PREFLIGHT_PATH = D52_ROOT / 'preflight_d52.json'
AUDIT_PATH = D52_ROOT / 'audit_d52.json'
STEMS = ('kick', 'snare', 'toms', 'hh', 'ride', 'crash')
TARGET_SOURCES = {'d36_archive_synthetic': 1382, 'd36_breakdown_real': 42}
MIN_FREE_GIB = 40.0


def read_json(path):
    """讀取 UTF-8 JSON，讓資料來源與 audit 皆可重複檢查。"""
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def write_json(path, payload):
    """以原子替換寫入新的 D52 紀錄，避免留下半份 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    temporary.replace(path)


def file_sha256(path):
    """計算固定模型與 YAML 的雜湊，避免不小心換了分離配方。"""
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def input_name(source, key):
    """把 metadata key 轉成安全且可逆的 Windows 檔名。"""
    # ponytail: 輸入檔名就是 source/key 的安全化版本；不建立第二份名稱資料庫。
    return re.sub(r'[^A-Za-z0-9._-]+', '_', f'{source}__{key}')


def select_missing_train(metadata):
    """只選取尚未有 D50 stem 的 Archive/Breakdown train，嚴禁碰 held-out。"""
    selected = []
    for key, item in sorted(metadata.items()):
        if item.get('split') != 'train' or 'drumsep_stem_auxiliary' in item:
            continue
        source = item.get('source')
        if source not in TARGET_SOURCES:
            continue
        audio_path = Path(item['audio_path'])
        if not audio_path.is_file():
            raise FileNotFoundError(f'Input audio missing for {key}: {audio_path}')
        selected.append((key, source, audio_path))
    source_counts = Counter(source for _, source, _ in selected)
    if source_counts != Counter(TARGET_SOURCES) or len(selected) != 1424:
        raise ValueError(f'Unexpected D52 selection: {dict(source_counts)}, tracks={len(selected)}')
    names = [input_name(source, key) for key, source, _ in selected]
    if len(names) != len(set(names)):
        raise ValueError('Sanitized D52 input names are not unique.')
    return selected


def build_plan():
    """建立 D52 key map 與時長／空間預估，過程只讀原始音訊 header。"""
    metadata = read_json(D50_META)
    d48 = read_json(D48_AUDIT)
    selected = select_missing_train(metadata)
    rows, total_seconds = [], 0.0
    for key, source, audio_path in selected:
        info = sf.info(audio_path)
        seconds = info.frames / float(info.samplerate)
        total_seconds += seconds
        rows.append({
            'key': key,
            'source': source,
            'audio_path': str(audio_path),
            'input_name': input_name(source, key),
            'input_extension': audio_path.suffix.lower(),
            'duration_seconds': seconds,
        })
    d48_seconds = float(d48['source']['selection']['total_seconds'])
    d48_bytes = int(d48['output']['total_bytes'])
    expected_bytes = int(round(total_seconds * d48_bytes / d48_seconds))
    checkpoint = ROOT / d48['model']['checkpoint']
    config = ROOT / d48['model']['config']
    if not checkpoint.is_file() or not config.is_file():
        raise FileNotFoundError('D47/D48 DrumSep checkpoint or YAML is missing.')
    if file_sha256(checkpoint) != d48['model']['checkpoint_sha256']:
        raise ValueError('DrumSep checkpoint hash differs from D48.')
    return {
        'phase': 'D52',
        'status': 'preflight_complete_not_inference_started',
        'selection': {
            'tracks': len(rows),
            'sources': dict(sorted(Counter(row['source'] for row in rows).items())),
            'total_seconds': total_seconds,
            'validation_or_test_read': False,
        },
        'model': {
            'checkpoint': str(checkpoint.relative_to(ROOT)),
            'checkpoint_sha256': file_sha256(checkpoint),
            'config': str(config.relative_to(ROOT)),
            'config_sha256': file_sha256(config),
            'source_revision': d48['model']['source_revision'],
            'tta': False,
            'lora': False,
        },
        'expected_output': {
            'stems_per_track': len(STEMS),
            'stem_files': len(rows) * len(STEMS),
            'estimated_bytes_from_d48_density': expected_bytes,
            'estimated_gib_from_d48_density': expected_bytes / 1024 ** 3,
        },
        'entries': rows,
    }


def prepare(plan):
    """建立不可覆寫的 hard-link 輸入，並確認磁碟空間足以容納 D52。"""
    free_bytes = shutil.disk_usage(ROOT).free
    if free_bytes < int(MIN_FREE_GIB * 1024 ** 3):
        raise RuntimeError(f'Only {free_bytes / 1024 ** 3:.2f} GiB free; D52 requires at least {MIN_FREE_GIB:.0f} GiB.')
    D52_ROOT.mkdir(exist_ok=True)
    if KEY_MAP_PATH.exists():
        existing = read_json(KEY_MAP_PATH)
        if existing['entries'] != plan['entries']:
            raise RuntimeError('Existing D52 key map differs; refusing to overwrite candidate inputs.')
    else:
        write_json(KEY_MAP_PATH, plan)
    INPUT_ROOT.mkdir(exist_ok=True)
    for row in plan['entries']:
        destination = INPUT_ROOT / f"{row['input_name']}{row['input_extension']}"
        source = Path(row['audio_path'])
        if destination.exists():
            if not os.path.samefile(source, destination):
                raise RuntimeError(f'Existing D52 input is not the expected hard link: {destination}')
            continue
        os.link(source, destination)
    preflight = dict(plan)
    preflight['free_gib_before_inference'] = free_bytes / 1024 ** 3
    preflight['input_hard_links'] = sum(
        (INPUT_ROOT / f"{row['input_name']}{row['input_extension']}").is_file()
        for row in plan['entries']
    )
    if preflight['input_hard_links'] != len(plan['entries']):
        raise RuntimeError('D52 input hard-link count is incomplete.')
    write_json(PREFLIGHT_PATH, preflight)
    return preflight


def audit(plan):
    """稽核官方推論輸出是否完整；不修改任何音訊或既有資料。"""
    complete, incomplete, files = [], [], 0
    for row in plan['entries']:
        stem_paths = {stem: OUTPUT_ROOT / row['input_name'] / f'{stem}.wav' for stem in STEMS}
        valid = True
        for path in stem_paths.values():
            if not path.is_file() or path.stat().st_size == 0:
                valid = False
                continue
            info = sf.info(path)
            valid = valid and info.samplerate == 44100 and info.channels == 2
        if valid:
            complete.append(row['key'])
            files += len(STEMS)
        else:
            incomplete.append(row['key'])
    payload = {
        'phase': 'D52',
        'status': 'complete_six_stem_batch_not_training' if not incomplete else 'incomplete_resume_only',
        'expected_tracks': len(plan['entries']),
        'complete_tracks': len(complete),
        'incomplete_tracks': len(incomplete),
        'stem_files_verified': files,
        'incomplete_keys': incomplete,
        'validation_or_test_read': False,
        'training_started': False,
        'lora_started': False,
        'ready_for_training_candidate': False,
        'ready_for_six_class_release': False,
    }
    write_json(AUDIT_PATH, payload)
    return payload


def run_self_check():
    """驗證 D52 的檔名映射與完整 stem 判定基本不變。"""
    assert input_name('d36_archive_synthetic', 'archive:demo/01') == 'd36_archive_synthetic__archive_demo_01'
    assert f"{input_name('d36_breakdown_real', 'demo')}.mp3".endswith('.mp3')
    assert len(STEMS) == 6 and len(set(STEMS)) == 6
    assert Counter(TARGET_SOURCES) == Counter({'d36_archive_synthetic': 1382, 'd36_breakdown_real': 42})
    print('D52 self-check passed.')


def main():
    """解析模式並執行 preflight、hard-link 建置或唯讀完成稽核。"""
    parser = argparse.ArgumentParser(description='Build or audit the D52 DrumSep train-only batch.')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    if args.prepare == args.audit:
        parser.error('Choose exactly one of --prepare or --audit.')
    plan = build_plan() if args.prepare else read_json(KEY_MAP_PATH)
    result = prepare(plan) if args.prepare else audit(plan)
    if args.audit:
        display = {key: result[key] for key in (
            'phase', 'status', 'expected_tracks', 'complete_tracks',
            'incomplete_tracks', 'stem_files_verified',
        )}
    else:
        display = {
            'phase': result['phase'], 'tracks': result['selection']['tracks'],
            'input_hard_links': result['input_hard_links'],
            'free_gib_before_inference': round(result['free_gib_before_inference'], 3),
            'estimated_gib': round(result['expected_output']['estimated_gib_from_d48_density'], 3),
        }
    print(json.dumps(display, ensure_ascii=False))


if __name__ == '__main__':
    main()
