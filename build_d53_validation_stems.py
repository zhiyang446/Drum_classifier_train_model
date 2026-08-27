"""建立與稽核 D53 封存 validation 的 DrumSep stem；不讀取事件標註或訓練。"""

import argparse
import json
import os
import shutil
from pathlib import Path

import soundfile as sf

from build_d52_drumsep_batch import STEMS, file_sha256, input_name, read_json, write_json


ROOT = Path(__file__).resolve().parent
META_PATH = ROOT / 'mixed_d50_stem_candidate' / 'metadata_d50.json'
D48_AUDIT = ROOT / 'drumsep_d48' / 'audit_d48.json'
D53_ROOT = ROOT / 'drumsep_d53'
INPUT_ROOT = D53_ROOT / 'input'
OUTPUT_ROOT = D53_ROOT / 'output'
PLAN_PATH = D53_ROOT / 'key_map_d53.json'
PREFLIGHT_PATH = D53_ROOT / 'preflight_d53.json'
AUDIT_PATH = D53_ROOT / 'audit_d53.json'


def build_plan():
    """只依 split 選取八首封存 validation，不檢視或使用其事件欄位。"""
    metadata = read_json(META_PATH)
    d48 = read_json(D48_AUDIT)
    entries = []
    for key, item in sorted(metadata.items()):
        if item.get('split') != 'validation':
            continue
        audio_path = Path(item['audio_path'])
        if not audio_path.is_file():
            raise FileNotFoundError(f'Validation audio missing for {key}: {audio_path}')
        entries.append({
            'key': key,
            'source': item.get('source'),
            'audio_path': str(audio_path),
            'input_name': input_name(item.get('source'), key),
            'input_extension': audio_path.suffix.lower(),
        })
    if len(entries) != 8 or len({row['input_name'] for row in entries}) != 8:
        raise ValueError(f'D53 requires exactly eight unique validation inputs, got {len(entries)}.')
    checkpoint = ROOT / d48['model']['checkpoint']
    config = ROOT / d48['model']['config']
    if file_sha256(checkpoint) != d48['model']['checkpoint_sha256']:
        raise ValueError('DrumSep checkpoint hash differs from D48.')
    return {
        'phase': 'D53',
        'status': 'preflight_complete_not_inference_started',
        'selection': {'tracks': 8, 'split': 'validation', 'event_labels_used': False},
        'model': {
            'checkpoint': str(checkpoint.relative_to(ROOT)),
            'checkpoint_sha256': file_sha256(checkpoint),
            'config': str(config.relative_to(ROOT)),
            'config_sha256': file_sha256(config),
            'source_revision': d48['model']['source_revision'],
        },
        'entries': entries,
    }


def prepare(plan):
    """以 hard link 建立隔離輸入，避免複製 held-out 原始音訊。"""
    D53_ROOT.mkdir(exist_ok=True)
    if PLAN_PATH.exists():
        if read_json(PLAN_PATH)['entries'] != plan['entries']:
            raise RuntimeError('Existing D53 mapping differs; refusing to overwrite held-out candidate.')
    else:
        write_json(PLAN_PATH, plan)
    INPUT_ROOT.mkdir(exist_ok=True)
    for row in plan['entries']:
        destination = INPUT_ROOT / f"{row['input_name']}{row['input_extension']}"
        source = Path(row['audio_path'])
        if destination.exists():
            if not os.path.samefile(source, destination):
                raise RuntimeError(f'Unexpected existing D53 input: {destination}')
        else:
            os.link(source, destination)
    preflight = dict(plan)
    preflight['input_hard_links'] = sum(
        (INPUT_ROOT / f"{row['input_name']}{row['input_extension']}").is_file()
        for row in plan['entries']
    )
    if preflight['input_hard_links'] != 8:
        raise RuntimeError('D53 validation hard links are incomplete.')
    write_json(PREFLIGHT_PATH, preflight)
    return preflight


def audit(plan):
    """只驗證分離檔格式與路徑，不讀 validation events 或計算品質分數。"""
    complete, incomplete, files = [], [], 0
    for row in plan['entries']:
        valid = True
        for stem in STEMS:
            path = OUTPUT_ROOT / row['input_name'] / f'{stem}.wav'
            if not path.is_file() or path.stat().st_size == 0:
                valid = False
                continue
            info = sf.info(path)
            valid = valid and info.samplerate == 44100 and info.channels == 2
        (complete if valid else incomplete).append(row['key'])
        files += len(STEMS) if valid else 0
    result = {
        'phase': 'D53',
        'status': 'complete_isolated_validation_stems' if not incomplete else 'incomplete_resume_only',
        'expected_tracks': 8,
        'complete_tracks': len(complete),
        'incomplete_tracks': len(incomplete),
        'stem_files_verified': files,
        'incomplete_keys': incomplete,
        'event_labels_used': False,
        'training_started': False,
    }
    write_json(AUDIT_PATH, result)
    return result


def main():
    """執行 D53 preflight 或格式稽核。"""
    parser = argparse.ArgumentParser(description='Build or audit isolated D53 validation stems.')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    if args.prepare == args.audit:
        parser.error('Choose exactly one of --prepare or --audit.')
    plan = build_plan() if args.prepare else read_json(PLAN_PATH)
    result = prepare(plan) if args.prepare else audit(plan)
    if args.audit:
        result = {key: result[key] for key in ('phase', 'status', 'expected_tracks', 'complete_tracks', 'incomplete_tracks', 'stem_files_verified')}
    else:
        result = {'phase': 'D53', 'input_hard_links': result['input_hard_links'], 'event_labels_used': False}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
