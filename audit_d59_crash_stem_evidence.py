# -*- coding: utf-8 -*-
"""D59：以既有 DrumSep stem 為 D58 unannotated CRASH 提供聲學證據排序。"""
import argparse
import csv
import json
import math
import os

import numpy as np
import soundfile as sf


STEMS = ('kick', 'snare', 'toms', 'hh', 'ride', 'crash')


def prepare_output_dir(path):
    """只允許新的空輸出目錄，避免覆寫先前的審計證據。"""
    if os.path.exists(path) and os.listdir(path):
        raise FileExistsError(f'Output directory is not empty: {path}')
    os.makedirs(path, exist_ok=True)
    return path


def read_power(path, center_seconds, window_seconds=0.1):
    """讀取事件附近固定窗口的單聲道平均功率與取樣率。"""
    with sf.SoundFile(path) as audio:
        sample_rate = int(audio.samplerate)
        half_window = max(1, int(round(window_seconds * sample_rate / 2.0)))
        center = int(round(center_seconds * sample_rate))
        start = max(0, center - half_window)
        audio.seek(start)
        samples = audio.read(half_window * 2, dtype='float32', always_2d=True).mean(axis=1)
    return float(np.mean(np.square(samples))) if len(samples) else 0.0, sample_rate


def describe_energy(powers):
    """將 six-stem power 轉為可比較的 CRASH share、相對 dB 與描述性分組。"""
    crash_power = float(powers['crash'])
    other_power = max(float(value) for stem, value in powers.items() if stem != 'crash')
    total_power = sum(float(value) for value in powers.values())
    crash_share = crash_power / total_power if total_power else 0.0
    relative_db = 10.0 * math.log10((crash_power + 1e-12) / (other_power + 1e-12))
    # ponytail: 只做可讀排序；要修改標註時必須有獨立人工或真值證據。
    if crash_share >= 0.50 and relative_db >= -3.0:
        bucket = 'crash_dominant'
    elif crash_share >= 0.20:
        bucket = 'mixed_energy'
    else:
        bucket = 'other_stem_dominant'
    return crash_share, relative_db, bucket


def count_by(rows, field):
    """依欄位值統計輸出列，保持穩定排序便於交接比較。"""
    counts = {}
    for row in rows:
        value = str(row[field])
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def audit_rows(error_rows, metadata):
    """對 D58 unannotated CRASH 列讀取對應 six stem，建立能量證據。"""
    output = []
    for row in error_rows:
        if row.get('cause') != 'unannotated':
            continue
        item = metadata[row['key']]
        paths = item.get('drumsep_stems', {}).get('paths', {})
        if set(paths) != set(STEMS):
            raise ValueError(f"Missing six DrumSep stems for {row['key']}")
        event_time = float(row['audio_time'])
        powers, sample_rates = {}, set()
        for stem in STEMS:
            power, sample_rate = read_power(paths[stem], event_time)
            powers[stem] = power
            sample_rates.add(sample_rate)
        if len(sample_rates) != 1:
            raise ValueError(f"Inconsistent stem sample rates for {row['key']}")
        share, relative_db, bucket = describe_energy(powers)
        output.append({
            'key': row['key'], 'group_id': row['group_id'], 'audio_time': f'{event_time:.6f}',
            'model_probability': row['probability'], 'sample_rate': sample_rates.pop(),
            'crash_power_share': f'{share:.6f}', 'crash_vs_max_other_db': f'{relative_db:.4f}',
            'evidence_bucket': bucket,
            **{f'{stem}_power': f'{powers[stem]:.10f}' for stem in STEMS},
        })
    return sorted(output, key=lambda item: float(item['crash_power_share']), reverse=True)


def write_csv(path, rows):
    """寫出固定欄位 CSV，空結果亦可被後續流程安全讀取。"""
    fields = ['key', 'group_id', 'audio_time', 'model_probability', 'sample_rate', 'crash_power_share', 'crash_vs_max_other_db', 'evidence_bucket'] + [f'{stem}_power' for stem in STEMS]
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    """執行 D59 聲學證據稽核，不更動任何來源或標註。"""
    output_dir = prepare_output_dir(args.output_dir)
    with open(args.errors, newline='', encoding='utf-8') as handle:
        error_rows = list(csv.DictReader(handle))
    with open(args.meta, encoding='utf-8') as handle:
        metadata = json.load(handle)
    rows = audit_rows(error_rows, metadata)
    write_csv(os.path.join(output_dir, 'unannotated_crash_stem_evidence.csv'), rows)
    summary = {
        'phase': 'D59', 'status': 'complete_read_only_stem_evidence',
        'unannotated_crash_events': len(rows), 'window_ms': 100,
        'bucket_counts': count_by(rows, 'evidence_bucket'),
        'interpretation': 'descriptive audio evidence only; not a label-correction decision',
    }
    with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def run_self_check():
    """驗證能量分組邊界不會把非 CRASH 主導事件誤列為主導。"""
    dominant = describe_energy({'kick': 1.0, 'snare': 1.0, 'toms': 1.0, 'hh': 1.0, 'ride': 1.0, 'crash': 8.0})
    weak = describe_energy({'kick': 8.0, 'snare': 1.0, 'toms': 1.0, 'hh': 1.0, 'ride': 1.0, 'crash': 0.1})
    assert dominant[2] == 'crash_dominant'
    assert weak[2] == 'other_stem_dominant'
    print('Self-check passed.')


def main():
    """解析 D59 參數並保留可獨立執行的最小自檢入口。"""
    parser = argparse.ArgumentParser(description='Rank D58 unannotated CRASH errors by DrumSep stem energy evidence.')
    parser.add_argument('--errors')
    parser.add_argument('--meta')
    parser.add_argument('--output-dir')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    if not all((args.errors, args.meta, args.output_dir)):
        parser.error('--errors, --meta, and --output-dir are required unless --self-check is used')
    run(args)


if __name__ == '__main__':
    main()
