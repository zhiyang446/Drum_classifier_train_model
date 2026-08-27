"""D49：唯讀稽核 D48 DrumSep stem 與既有 D46 MIDI event 的對應品質。"""

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


SAMPLE_RATE = 44_100
CHANNELS = 2
ENVELOPE_HOP_SECONDS = 0.020
EVENT_RADIUS_SECONDS = 0.050
SILENCE_RMS = 1e-5
LABEL_TO_STEM = {
    'KD': 'kick',
    'SD': 'snare',
    'TOM': 'toms',
    'HH': 'hh',
    'RIDE': 'ride',
    'CRASH': 'crash',
}
EXPECTED_STEMS = tuple(LABEL_TO_STEM.values())


def load_json(path):
    """以 UTF-8 載入既有 JSON，整個 D49 流程不會改寫來源資料。"""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, payload):
    """只寫入新的 D49 audit，拒絕覆寫既有稽核結果。"""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite existing audit: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


def stem_envelope(path):
    """串流計算單一 stem 的 20ms RMS envelope 與基礎音訊品質指標。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing stem: {path}')
    hop_frames = int(round(ENVELOPE_HOP_SECONDS * SAMPLE_RATE))
    block_frames = hop_frames * 1000
    envelope_parts = []
    square_sum = 0.0
    mono_count = 0
    peak = 0.0
    clipped_count = 0
    sample_count = 0
    with sf.SoundFile(str(path)) as handle:
        if handle.samplerate != SAMPLE_RATE or handle.channels != CHANNELS:
            raise ValueError(f'Unexpected stem format for {path}: {handle.samplerate}Hz/{handle.channels}ch')
        while True:
            block = handle.read(block_frames, dtype='float32', always_2d=True)
            if len(block) == 0:
                break
            mono = block.mean(axis=1)
            square_sum += float(np.dot(mono, mono))
            mono_count += int(len(mono))
            peak = max(peak, float(np.max(np.abs(block))))
            clipped_count += int(np.count_nonzero(np.abs(block) >= 0.999))
            sample_count += int(block.size)
            full_frames = len(mono) // hop_frames
            if full_frames:
                chunk = mono[:full_frames * hop_frames].reshape(full_frames, hop_frames)
                envelope_parts.append(np.sqrt(np.mean(np.square(chunk), axis=1)))
            if len(mono) % hop_frames:
                tail = mono[full_frames * hop_frames:]
                envelope_parts.append(np.array([math.sqrt(float(np.mean(np.square(tail))))], dtype=np.float32))
    if mono_count == 0:
        raise ValueError(f'Empty stem: {path}')
    envelope = np.concatenate(envelope_parts).astype(np.float32, copy=False)
    return {
        'envelope': envelope,
        'duration_seconds': float(mono_count / SAMPLE_RATE),
        'rms': math.sqrt(square_sum / mono_count),
        'peak': peak,
        'clip_fraction': float(clipped_count / sample_count),
    }


def event_energy_metrics(envelope, event_times, duration):
    """比較既有 MIDI event 附近與同 stem 全曲背景的 RMS 能量。"""
    if not event_times:
        return {
            'event_count': 0,
            'valid_event_count': 0,
            'not_assessable': True,
            'review_required': False,
            'reason': 'not_assessable_no_events',
        }
    background = float(np.median(envelope))
    radius = int(math.ceil(EVENT_RADIUS_SECONDS / ENVELOPE_HOP_SECONDS))
    values = []
    for event_time in event_times:
        if not 0.0 <= event_time <= duration:
            continue
        center = int(round(event_time / ENVELOPE_HOP_SECONDS))
        start = max(0, center - radius)
        end = min(len(envelope), center + radius + 1)
        if start < end:
            # ponytail: 固定 ±50ms 只作快速稽核；若 D49 指出問題才升級為類別專屬 onset 偵測。
            values.append(float(np.max(envelope[start:end])))
    if not values:
        return {
            'event_count': len(event_times),
            'valid_event_count': 0,
            'background_rms': background,
            'review_required': True,
            'reason': 'events_outside_stem_duration',
        }
    local_median = float(np.median(values))
    ratio_db = float(20.0 * math.log10((local_median + 1e-12) / (background + 1e-12)))
    return {
        'event_count': len(event_times),
        'valid_event_count': len(values),
        'background_rms': background,
        'event_local_median_rms': local_median,
        'event_to_background_db': ratio_db,
        'event_coverage_above_background': float(np.mean(np.asarray(values) > background)),
        'review_required': bool(local_median <= background),
        'reason': 'event_energy_not_above_background' if local_median <= background else None,
    }


def envelope_leakage_proxy(envelopes):
    """以 RMS envelope 相關性產生 stem 間可能洩漏的代理指標，而非真實 source-separation 分數。"""
    pairs = []
    for left, right in combinations(EXPECTED_STEMS, 2):
        count = min(len(envelopes[left]), len(envelopes[right]))
        if count < 2:
            correlation = 0.0
        else:
            first = envelopes[left][:count]
            second = envelopes[right][:count]
            denominator = float(np.linalg.norm(first - first.mean()) * np.linalg.norm(second - second.mean()))
            correlation = 0.0 if denominator == 0.0 else float(np.corrcoef(first, second)[0, 1])
        pairs.append({'stems': [left, right], 'envelope_correlation': correlation})
    pairs.sort(key=lambda row: abs(row['envelope_correlation']), reverse=True)
    return {'max_abs_correlation': abs(pairs[0]['envelope_correlation']), 'pairs': pairs}


def reconstruction_metrics(mix_path, stem_paths):
    """把六 stem 相加後與原混音比較，量測重組相關性與相對殘差。"""
    mix, sample_rate = librosa.load(str(mix_path), sr=SAMPLE_RATE, mono=False, dtype=np.float32)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f'Unexpected resample rate for mix: {sample_rate}')
    if mix.ndim == 1:
        mix = np.repeat(mix[np.newaxis, :], CHANNELS, axis=0)
    if mix.shape[0] != CHANNELS:
        raise ValueError(f'Unexpected mix channels for {mix_path}: {mix.shape[0]}')
    # ponytail: 每首只保留一個重組陣列；若記憶體不夠才改為分塊重取樣與累加。
    reconstruction = np.zeros_like(mix, dtype=np.float32)
    usable_frames = mix.shape[1]
    for stem_name in EXPECTED_STEMS:
        samples, stem_rate = sf.read(str(stem_paths[stem_name]), dtype='float32', always_2d=True)
        if stem_rate != SAMPLE_RATE or samples.shape[1] != CHANNELS:
            raise ValueError(f'Unexpected reconstruction stem format: {stem_paths[stem_name]}')
        frames = min(reconstruction.shape[1], len(samples))
        usable_frames = min(usable_frames, frames)
        reconstruction[:, :frames] += samples[:frames].T
    mix = mix[:, :usable_frames]
    reconstruction = reconstruction[:, :usable_frames]
    mix_flat = mix.reshape(-1)
    reconstruction_flat = reconstruction.reshape(-1)
    mix_norm = float(np.linalg.norm(mix_flat))
    reconstruction_norm = float(np.linalg.norm(reconstruction_flat))
    denominator = mix_norm * reconstruction_norm
    correlation = 0.0 if denominator == 0.0 else float(np.dot(mix_flat, reconstruction_flat) / denominator)
    residual = mix_flat - reconstruction_flat
    return {
        'comparison_frames': int(usable_frames),
        'mix_rms': math.sqrt(float(np.mean(np.square(mix_flat)))),
        'reconstruction_rms': math.sqrt(float(np.mean(np.square(reconstruction_flat)))),
        'mix_reconstruction_correlation': correlation,
        'normalized_residual_rms': math.sqrt(float(np.mean(np.square(residual)))) / max(math.sqrt(float(np.mean(np.square(mix_flat)))), 1e-12),
    }


def audit_item(key, item, output_root):
    """稽核一首 D46 Whack train 的六 stem 與既有對齊 event。"""
    output_key = key.replace(':', '_')
    stem_dir = Path(output_root) / output_key
    stem_paths = {stem: stem_dir / f'{stem}.wav' for stem in EXPECTED_STEMS}
    missing = [stem for stem, path in stem_paths.items() if not path.is_file()]
    if missing:
        return {'key': key, 'group_id': item['group_id'], 'review_required': True, 'error': f'missing_stems:{missing}'}
    events_by_label = {label: [] for label in LABEL_TO_STEM}
    for event in item['events']:
        if event['inst'] in events_by_label:
            events_by_label[event['inst']].append(float(event['time']))
    stem_reports = {}
    envelopes = {}
    review_reasons = []
    for label, stem in LABEL_TO_STEM.items():
        report = stem_envelope(stem_paths[stem])
        envelope = report.pop('envelope')
        energy = event_energy_metrics(envelope, events_by_label[label], report['duration_seconds'])
        report['event_alignment'] = energy
        report['non_silent'] = bool(report['rms'] > SILENCE_RMS)
        if not report['non_silent']:
            review_reasons.append(f'{stem}:silent_or_near_silent')
        if energy['review_required']:
            review_reasons.append(f"{stem}:{energy['reason']}")
        stem_reports[stem] = report
        envelopes[stem] = envelope
    reconstruction = reconstruction_metrics(item['audio_path'], stem_paths)
    return {
        'key': key,
        'group_id': item['group_id'],
        'source': item['source'],
        'split': item['split'],
        'audio_path': item['audio_path'],
        'duration_seconds': float(item['duration']),
        'stem_reports': stem_reports,
        'leakage_proxy': envelope_leakage_proxy(envelopes),
        'reconstruction': reconstruction,
        'review_required': bool(review_reasons),
        'review_reasons': review_reasons,
    }


def build_audit(metadata_path, output_root, output_path):
    """建立完整 D49 唯讀 audit，將個別異常保留為 review 而不改資料。"""
    metadata = load_json(metadata_path)
    selected = [
        (key, item) for key, item in metadata.items()
        if item['source'] == 'd36_whack_real' and item['split'] == 'train'
    ]
    selected.sort(key=lambda row: row[0])
    if len(selected) != 28 or len({item['group_id'] for _, item in selected}) != 28:
        raise AssertionError(f'Expected 28 unique D46 Whack train groups, got {len(selected)}')
    results = []
    failures = []
    for index, (key, item) in enumerate(selected, start=1):
        try:
            results.append(audit_item(key, item, output_root))
        except Exception as error:
            failures.append({'key': key, 'group_id': item['group_id'], 'error': str(error)})
        print(f'Audited {index}/{len(selected)} D49 tracks.', flush=True)
    passed_format = sum('error' not in row for row in results)
    review_count = sum(row.get('review_required', True) for row in results) + len(failures)
    payload = {
        'phase': 'D49',
        'status': 'quality_alignment_audit_complete_not_training_ready',
        'inputs': {
            'metadata': str(Path(metadata_path)),
            'output_root': str(Path(output_root)),
            'selection': {'source': 'd36_whack_real', 'split': 'train', 'tracks': len(selected)},
        },
        'algorithm': {
            'envelope_hop_seconds': ENVELOPE_HOP_SECONDS,
            'event_radius_seconds': EVENT_RADIUS_SECONDS,
            'event_energy': 'median_of_event_local_rms_max_vs_full_stem_median_rms',
            'leakage_proxy': 'pairwise_rms_envelope_correlation',
            'reconstruction': 'sum_of_six_stems_vs_librosa_resampled_original_mix',
        },
        'summary': {
            'expected_tracks': 28,
            'audited_tracks': len(results),
            'audit_failures': len(failures),
            'format_complete_tracks': passed_format,
            'review_required_tracks': review_count,
        },
        'results': results,
        'failures': failures,
        'ready_for_training_candidate': False,
        'ready_for_six_class_release': False,
    }
    write_json(output_path, payload)
    return payload


def run_self_check():
    """驗證六類 stem 對映與 event-local 能量的最小正反例。"""
    assert set(LABEL_TO_STEM) == {'KD', 'SD', 'TOM', 'HH', 'RIDE', 'CRASH'}
    assert set(EXPECTED_STEMS) == {'kick', 'snare', 'toms', 'hh', 'ride', 'crash'}
    envelope = np.array([0.01, 0.01, 0.50, 0.01, 0.01], dtype=np.float32)
    metrics = event_energy_metrics(envelope, [0.04], 0.10)
    assert metrics['valid_event_count'] == 1 and metrics['event_to_background_db'] > 0.0
    print('D49 self-check passed.')


def main():
    """提供 D49 的自檢與一次性唯讀 audit CLI 入口。"""
    parser = argparse.ArgumentParser(description='Audit D48 DrumSep stem quality and D46 MIDI-event alignment.')
    parser.add_argument('--metadata', default='mixed_d46/metadata_d46.json')
    parser.add_argument('--output-root', default='drumsep_d48/output')
    parser.add_argument('--output', default='drumsep_d49/audit_d49.json')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    payload = build_audit(args.metadata, args.output_root, args.output)
    print(f"D49 audit complete: {payload['summary']}")


if __name__ == '__main__':
    main()
