"""D29：以音訊 onset 與 MIDI tick 自動稽核 Whack Metal 對齊候選。"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import correlate

from build_whack_metal_meta_d28 import midi_tick_events, split_for_group, timed_events


HOP_LENGTH = 512
SAMPLE_RATE = 11025
MAX_OFFSET_SECONDS = 4.0
REFERENCE_LIMIT = 8
BOUNDARY_OFFSET_LIMIT_SECONDS = 3.8
RECOVERY_DRIFT_LIMIT_SECONDS = 0.25
LOCAL_FRACTIONS = (0.25, 0.50, 0.75)


def load_json(path):
    """以 UTF-8 讀取既有 D28 JSON，避免改寫任何原始資料。"""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def onset_envelope(audio_path):
    """把單首 WAV 轉為低採樣率 onset 強度序列。"""
    audio, sample_rate = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    envelope = librosa.onset.onset_strength(
        y=audio,
        sr=sample_rate,
        hop_length=HOP_LENGTH,
    )
    return np.maximum(envelope - np.median(envelope), 0.0)


def midi_impulses(events, ticks_per_beat, bpm, frame_count):
    """依候選 BPM 將 MIDI tick 轉成與 onset envelope 同一格點的脈衝。"""
    if bpm <= 0 or not math.isfinite(bpm):
        raise ValueError(f'Invalid BPM: {bpm}')
    frame_seconds = HOP_LENGTH / SAMPLE_RATE
    pulses = np.zeros(frame_count, dtype=np.float64)
    for event in events:
        seconds = event['tick'] * 60.0 / (ticks_per_beat * bpm)
        frame = int(round(seconds / frame_seconds))
        if 0 <= frame < frame_count:
            # ponytail: 同格疊擊只保留簡單的 velocity 根號權重；需要音色模型時才升級。
            pulses[frame] += math.sqrt(event['velocity'] / 127.0)
    return pulses


def candidate_alignment(envelope, events, ticks_per_beat, bpm):
    """在固定 BPM 下，以 FFT 相關性找出限制範圍內最佳 MIDI 時間位移。"""
    pulses = midi_impulses(events, ticks_per_beat, bpm, len(envelope))
    denominator = float(np.linalg.norm(envelope) * np.linalg.norm(pulses))
    if denominator == 0.0:
        return {'bpm': bpm, 'offset_seconds': 0.0, 'score': 0.0}
    values = correlate(envelope, pulses, mode='full', method='fft') / denominator
    lags = np.arange(-len(pulses) + 1, len(envelope))
    max_frames = int(round(MAX_OFFSET_SECONDS * SAMPLE_RATE / HOP_LENGTH))
    allowed = np.abs(lags) <= max_frames
    best_index = int(np.argmax(np.where(allowed, values, -np.inf)))
    return {
        'bpm': float(bpm),
        'offset_seconds': float(lags[best_index] * HOP_LENGTH / SAMPLE_RATE),
        'score': float(values[best_index]),
    }


def search_alignment(envelope, events, ticks_per_beat, initial_bpm):
    """以原 BPM 正負 10% 的固定格點搜尋 BPM 與 offset 候選。"""
    if initial_bpm <= 0 or not math.isfinite(initial_bpm):
        raise ValueError(f'Invalid initial BPM: {initial_bpm}')
    candidates = [
        candidate_alignment(envelope, events, ticks_per_beat, bpm)
        for bpm in np.linspace(initial_bpm * 0.90, initial_bpm * 1.10, 21)
    ]
    candidates.sort(key=lambda row: row['score'], reverse=True)
    best = candidates[0]
    runner_up = candidates[1]
    best['margin'] = float(best['score'] - runner_up['score'])
    return best


def resolve_targets(metadata, audit):
    """合併 D28 的 review 與超界歌曲，並保留唯一 group_id。"""
    by_group = {item['group_id']: item for item in metadata.values()}
    targets = []
    for group_id in audit['review_required_groups']:
        item = by_group[group_id]
        targets.append({
            'group_id': group_id,
            'audio_path': item['audio_path'],
            'midi_path': item['midi_path'],
            'initial_bpm': float(item['bpm']),
            'source_status': 'review_required',
        })
    for item in audit['excluded_outside_audio']:
        targets.append({
            'group_id': item['group_id'],
            'audio_path': item['audio_path'],
            'midi_path': item['midi_path'],
            'initial_bpm': float(item['bpm']),
            'source_status': 'excluded_outside_audio',
        })
    unique = {item['group_id']: item for item in targets}
    return [unique[group_id] for group_id in sorted(unique)]


def reference_targets(metadata):
    """挑選固定檔名 BPM、沒有 review 的歌曲作為門檻校準參考。"""
    rows = [
        item for item in metadata.values()
        if item['bpm_source'] == 'filename_bpm' and not item['review_required']
    ]
    return sorted(rows, key=lambda item: item['group_id'])[:REFERENCE_LIMIT]


def filename_bpm_targets(metadata):
    """列出 D30 要量測的固定檔名 BPM 歌曲，保留 D28 原始資料不變。"""
    rows = [
        {
            'group_id': item['group_id'],
            'audio_path': item['audio_path'],
            'midi_path': item['midi_path'],
            'initial_bpm': float(item['bpm']),
            'source_status': 'filename_bpm',
        }
        for item in metadata.values()
        if item['bpm_source'] == 'filename_bpm' and not item['review_required']
    ]
    return sorted(rows, key=lambda item: item['group_id'])


def analyse_track(target, restrict_bpm):
    """載入單首 WAV/MIDI 並回傳可序列化的對齊結果。"""
    audio_path = Path(target['audio_path'])
    midi_path = Path(target['midi_path'])
    if not audio_path.is_file() or not midi_path.is_file():
        raise FileNotFoundError(f'Missing pair: {audio_path} / {midi_path}')
    ticks_per_beat, events, _, _, _ = midi_tick_events(midi_path)
    envelope = onset_envelope(audio_path)
    if restrict_bpm:
        result = candidate_alignment(envelope, events, ticks_per_beat, target['initial_bpm'])
        result['margin'] = None
    else:
        result = search_alignment(envelope, events, ticks_per_beat, target['initial_bpm'])
    return result


def build_report(metadata_path, audit_path, output_path):
    """建立 D29 唯讀候選報告，絕不覆寫 D28 的 metadata 或 audit。"""
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {output_path}')
    metadata = load_json(metadata_path)
    audit = load_json(audit_path)
    references = reference_targets(metadata)
    if len(references) != REFERENCE_LIMIT:
        raise AssertionError(f'Expected {REFERENCE_LIMIT} references, got {len(references)}')

    reference_scores = []
    for item in references:
        result = analyse_track({
            'audio_path': item['audio_path'],
            'midi_path': item['midi_path'],
            'initial_bpm': float(item['bpm']),
        }, restrict_bpm=True)
        reference_scores.append(result['score'])
    score_threshold = max(0.02, float(np.percentile(reference_scores, 10)) * 0.50)
    margin_threshold = 0.002

    results = []
    for target in resolve_targets(metadata, audit):
        result = analyse_track(target, restrict_bpm=False)
        accepted = result['score'] >= score_threshold and result['margin'] >= margin_threshold
        results.append({
            **target,
            **result,
            'accepted': accepted,
            'reason': 'score_and_margin_pass' if accepted else 'needs_consolidation_review',
        })
    payload = {
        'phase': 'D29',
        'status': 'audit_complete_not_training_ready',
        'algorithm': 'librosa_onset_fft_correlation',
        'search': {
            'bpm_range': 'initial_bpm ±10%, 21 fixed points',
            'max_offset_seconds': MAX_OFFSET_SECONDS,
            'sample_rate': SAMPLE_RATE,
            'hop_length': HOP_LENGTH,
        },
        'thresholds': {
            'reference_group_ids': [item['group_id'] for item in references],
            'reference_scores': reference_scores,
            'score_threshold': score_threshold,
            'margin_threshold': margin_threshold,
        },
        'target_count': len(results),
        'accepted_count': sum(row['accepted'] for row in results),
        'rejected_count': sum(not row['accepted'] for row in results),
        'results': results,
        'd28_ready_for_training_candidate_unchanged': audit['ready_for_training_candidate'],
        'ready_for_training_candidate': False,
        'ready_for_six_class_release': False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    return payload


def build_filename_bpm_report(metadata_path, audit_path, d29_report_path, output_path):
    """量測所有固定檔名 BPM 歌曲，建立 D30 唯讀一致性 audit。"""
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {output_path}')
    metadata = load_json(metadata_path)
    audit = load_json(audit_path)
    d29_report = load_json(d29_report_path)
    score_threshold = float(d29_report['thresholds']['score_threshold'])
    if d29_report['phase'] != 'D29' or not math.isfinite(score_threshold):
        raise ValueError('D29 score threshold is unavailable.')
    targets = filename_bpm_targets(metadata)
    if len(targets) != 85:
        raise AssertionError(f'Expected 85 filename BPM targets, got {len(targets)}')

    offset_review_seconds = 0.25
    results = []
    for target in targets:
        result = analyse_track(target, restrict_bpm=True)
        results.append({
            **target,
            **result,
            'score_pass': result['score'] >= score_threshold,
            # ponytail: 0.25 秒只作稽核旗標；需要校正時才另建 metadata consolidation。
            'requires_offset_consolidation': abs(result['offset_seconds']) > offset_review_seconds,
        })
    payload = {
        'phase': 'D30',
        'status': 'audit_complete_not_training_ready',
        'algorithm': 'librosa_onset_fft_correlation_fixed_bpm',
        'score_threshold_from_d29': score_threshold,
        'offset_review_seconds': offset_review_seconds,
        'target_count': len(results),
        'score_pass_count': sum(row['score_pass'] for row in results),
        'offset_consolidation_count': sum(row['requires_offset_consolidation'] for row in results),
        'results': results,
        'd28_ready_for_training_candidate_unchanged': audit['ready_for_training_candidate'],
        'ready_for_training_candidate': False,
        'ready_for_six_class_release': False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    return payload


def shift_events(events, offset_seconds, duration):
    """套用候選 offset，並明確丟棄落在實際音訊邊界外的事件。"""
    kept, dropped_before, dropped_after = [], 0, 0
    for event in events:
        shifted_time = float(event['time']) + offset_seconds
        if shifted_time < 0.0:
            dropped_before += 1
            continue
        if shifted_time > duration:
            dropped_after += 1
            continue
        kept.append({**event, 'time': shifted_time})
    return kept, dropped_before, dropped_after


def build_d31_metadata(metadata_path, audit_path, d29_report_path, d30_report_path, output_dir):
    """將已通過自動對齊的歌曲寫成 D31 獨立候選 metadata 與 audit。"""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite existing output directory: {output_dir}')
    metadata = load_json(metadata_path)
    audit = load_json(audit_path)
    d29_report = load_json(d29_report_path)
    d30_report = load_json(d30_report_path)
    if d29_report['phase'] != 'D29' or d30_report['phase'] != 'D30':
        raise ValueError('D29/D30 reports are unavailable.')
    by_group = {item['group_id']: item for item in metadata.values()}
    candidates = {}

    def add_candidate(group_id, source_phase, audio_path, midi_path, duration, split, bpm, offset, score, events):
        """集中處理 group 唯一性、邊界裁切與候選資料欄位。"""
        if group_id in candidates:
            raise AssertionError(f'Duplicate D31 group: {group_id}')
        shifted, dropped_before, dropped_after = shift_events(events, offset, duration)
        if not shifted:
            raise ValueError(f'No in-bounds events after alignment: {group_id}')
        candidates[group_id] = {
            'audio_path': str(audio_path),
            'midi_path': str(midi_path),
            'duration': float(duration),
            'bpm': float(bpm),
            'alignment_offset_seconds': float(offset),
            'alignment_score': float(score),
            'alignment_source_phase': source_phase,
            'split': split,
            'group_id': group_id,
            'source': 'whack_studio_metal_d31_auto_alignment',
            'alignment_status': 'candidate_not_training_ready',
            'dropped_before_audio_events': dropped_before,
            'dropped_after_audio_events': dropped_after,
            'events': shifted,
        }

    for row in d30_report['results']:
        if not row['score_pass'] or abs(row['offset_seconds']) >= BOUNDARY_OFFSET_LIMIT_SECONDS:
            continue
        item = by_group[row['group_id']]
        add_candidate(
            row['group_id'], 'D30', item['audio_path'], item['midi_path'], item['duration'],
            item['split'], row['bpm'], row['offset_seconds'], row['score'], item['events'],
        )
    for row in d29_report['results']:
        if not row['accepted']:
            continue
        item = by_group.get(row['group_id'])
        audio_path = Path(row['audio_path'])
        duration = float(item['duration']) if item else float(sf.info(str(audio_path)).duration)
        split = item['split'] if item else split_for_group(row['group_id'])
        ticks_per_beat, raw_events, _, _, _ = midi_tick_events(row['midi_path'])
        rebuilt_events = timed_events(raw_events, ticks_per_beat, row['bpm'])
        add_candidate(
            row['group_id'], 'D29', row['audio_path'], row['midi_path'], duration, split,
            row['bpm'], row['offset_seconds'], row['score'], rebuilt_events,
        )

    all_groups = set(by_group) | {row['group_id'] for row in audit['excluded_outside_audio']}
    selected_groups = set(candidates)
    excluded_groups = sorted(all_groups - selected_groups)
    if len(candidates) != 95 or len(excluded_groups) != 15:
        raise AssertionError(f'Expected 95 selected and 15 excluded groups, got {len(candidates)} / {len(excluded_groups)}')
    split_counts = Counter(item['split'] for item in candidates.values())
    event_counts = {split: Counter() for split in ('train', 'validation', 'test')}
    boundary_drops = {}
    for item in candidates.values():
        event_counts[item['split']].update(event['inst'] for event in item['events'])
        if item['dropped_before_audio_events'] or item['dropped_after_audio_events']:
            boundary_drops[item['group_id']] = {
                'before_audio': item['dropped_before_audio_events'],
                'after_audio': item['dropped_after_audio_events'],
            }
    missing_labels = {
        split: [label for label in ('KD', 'SD', 'HH', 'TOM', 'CRASH', 'RIDE') if event_counts[split][label] == 0]
        for split in event_counts
    }
    if any(missing_labels.values()):
        raise AssertionError(f'D31 split missing labels: {missing_labels}')
    ordered = {
        f'whack_metal_d31_{index:03d}': candidates[group_id]
        for index, group_id in enumerate(sorted(candidates), start=1)
    }
    payload = {
        'phase': 'D31',
        'status': 'candidate_metadata_complete_not_training_ready',
        'selected_groups': len(ordered),
        'excluded_groups': excluded_groups,
        'source_counts': dict(Counter(item['alignment_source_phase'] for item in ordered.values())),
        'splits': dict(split_counts),
        'events': {split: dict(event_counts[split]) for split in event_counts},
        'missing_labels': missing_labels,
        'boundary_drops': boundary_drops,
        'group_split_leaks': 0,
        'd28_ready_for_training_candidate_unchanged': audit['ready_for_training_candidate'],
        'ready_for_training_candidate': False,
        'ready_for_six_class_release': False,
    }
    output_dir.mkdir(parents=True)
    (output_dir / 'metadata_d31.json').write_text(json.dumps(ordered, indent=2, ensure_ascii=False), encoding='utf-8')
    (output_dir / 'audit_d31.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    return ordered, payload


def local_offset_profile(envelope, pulses, fractions=LOCAL_FRACTIONS):
    """在指定位置量測局部 offset 與相關分數，僅供唯讀漂移稽核使用。"""
    if not fractions or any(not 0.0 < fraction < 1.0 for fraction in fractions):
        raise ValueError('Local profile fractions must be inside (0, 1).')
    max_frames = int(round(MAX_OFFSET_SECONDS * SAMPLE_RATE / HOP_LENGTH))
    half_window = int(round(20.0 * SAMPLE_RATE / HOP_LENGTH))
    rows = []
    for fraction in fractions:
        center = int(round((len(envelope) - 1) * fraction))
        audio_start = max(0, center - half_window)
        audio_end = min(len(envelope), center + half_window)
        pulse_start = max(0, audio_start - max_frames)
        pulse_end = min(len(pulses), audio_end + max_frames)
        audio_part = envelope[audio_start:audio_end]
        pulse_part = pulses[pulse_start:pulse_end]
        denominator = float(np.linalg.norm(audio_part) * np.linalg.norm(pulse_part))
        if denominator == 0.0:
            return None
        values = correlate(audio_part, pulse_part, mode='full', method='fft') / denominator
        lags = np.arange(-len(pulse_part) + 1, len(audio_part)) + audio_start - pulse_start
        allowed = np.abs(lags) <= max_frames
        best_index = int(np.argmax(np.where(allowed, values, -np.inf)))
        rows.append({
            'fraction': float(fraction),
            'center_seconds': float(center * HOP_LENGTH / SAMPLE_RATE),
            'offset_seconds': float(lags[best_index] * HOP_LENGTH / SAMPLE_RATE),
            'score': float(values[best_index]),
        })
    return rows


def local_offsets(envelope, pulses):
    """保留既有前中後三段 offset 介面，避免改變既有 D32/D42 行為。"""
    profile = local_offset_profile(envelope, pulses)
    if profile is None:
        return None
    return [row['offset_seconds'] for row in profile]


def collect_recovery_targets(d29_report, d30_report, d31_metadata, d31_audit):
    """合併三份 audit 的疑慮群組，避免同一首歌重複處理。"""
    d31_by_group = {item['group_id']: item for item in d31_metadata.values()}
    targets = {}

    def add_row(group_id, audio_path, midi_path, initial_bpm, reason):
        """集中保留單一 target 與所有觸發修復的理由。"""
        target = targets.setdefault(group_id, {
            'group_id': group_id,
            'audio_path': audio_path,
            'midi_path': midi_path,
            'initial_bpm': float(initial_bpm),
            'reasons': [],
        })
        target['reasons'].append(reason)

    for row in d29_report['results']:
        if not row['accepted']:
            add_row(row['group_id'], row['audio_path'], row['midi_path'], row['initial_bpm'], 'd29_rejected')
    for row in d30_report['results']:
        if abs(row['offset_seconds']) >= BOUNDARY_OFFSET_LIMIT_SECONDS:
            add_row(row['group_id'], row['audio_path'], row['midi_path'], row['initial_bpm'], 'd30_boundary')
    for group_id in d31_audit['boundary_drops']:
        item = d31_by_group[group_id]
        add_row(group_id, item['audio_path'], item['midi_path'], item['bpm'], 'd31_boundary_drop')
    if len(targets) != 38:
        raise AssertionError(f'Expected 38 recovery targets, got {len(targets)}')
    return [targets[group_id] for group_id in sorted(targets)]


def search_recovery(target, score_threshold):
    """以較寬 BPM 搜尋和三段 offset，僅把時間穩定的候選標為 resolved。"""
    audio_path = Path(target['audio_path'])
    midi_path = Path(target['midi_path'])
    if not audio_path.is_file() or not midi_path.is_file():
        raise FileNotFoundError(f'Missing recovery pair: {audio_path} / {midi_path}')
    envelope = onset_envelope(audio_path)
    ticks_per_beat, events, _, _, _ = midi_tick_events(midi_path)
    candidates = []
    for bpm in np.linspace(target['initial_bpm'] * 0.85, target['initial_bpm'] * 1.15, 31):
        global_result = candidate_alignment(envelope, events, ticks_per_beat, bpm)
        offsets = local_offsets(envelope, midi_impulses(events, ticks_per_beat, bpm, len(envelope)))
        drift = None if offsets is None else max(offsets) - min(offsets)
        candidates.append({**global_result, 'local_offsets_seconds': offsets, 'drift_seconds': drift})
    stable = [
        row for row in candidates
        if row['local_offsets_seconds'] is not None
        and row['score'] >= score_threshold
        and row['drift_seconds'] <= RECOVERY_DRIFT_LIMIT_SECONDS
        and abs(row['offset_seconds']) < BOUNDARY_OFFSET_LIMIT_SECONDS
    ]
    best = max(stable or candidates, key=lambda row: row['score'])
    return {
        **target,
        **best,
        'resolved': bool(stable),
        'reason': 'stable_local_alignment' if stable else 'no_stable_local_alignment',
    }


def build_d32_report(d29_report_path, d30_report_path, d31_metadata_path, d31_audit_path, output_path):
    """一次批次稽核所有問題歌曲，輸出可追溯的 resolved/unresolved 結果。"""
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {output_path}')
    d29_report = load_json(d29_report_path)
    d30_report = load_json(d30_report_path)
    d31_metadata = load_json(d31_metadata_path)
    d31_audit = load_json(d31_audit_path)
    if (d29_report['phase'], d30_report['phase'], d31_audit['phase']) != ('D29', 'D30', 'D31'):
        raise ValueError('D29/D30/D31 reports are unavailable.')
    score_threshold = float(d29_report['thresholds']['score_threshold'])
    results = [search_recovery(target, score_threshold) for target in collect_recovery_targets(
        d29_report, d30_report, d31_metadata, d31_audit,
    )]
    payload = {
        'phase': 'D32',
        'status': 'recovery_audit_complete_not_training_ready',
        'search': {
            'bpm_range': 'initial_bpm ±15%, 31 fixed points',
            'local_fractions': list(LOCAL_FRACTIONS),
            'local_drift_limit_seconds': RECOVERY_DRIFT_LIMIT_SECONDS,
        },
        'score_threshold_from_d29': score_threshold,
        'target_count': len(results),
        'resolved_count': sum(row['resolved'] for row in results),
        'unresolved_count': sum(not row['resolved'] for row in results),
        'results': results,
        # ponytail: D32 僅修復證據，不自動併入 metadata；確認候選品質後才建立下一版資料。
        'ready_for_training_candidate': False,
        'ready_for_six_class_release': False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    return payload


def build_d33_metadata(d31_metadata_path, d31_audit_path, d32_report_path, output_dir):
    """建立只含無裁切事件的 D33 安全候選，不改寫 D31 或 D32。"""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite existing output directory: {output_dir}')
    d31_metadata = load_json(d31_metadata_path)
    d31_audit = load_json(d31_audit_path)
    d32_report = load_json(d32_report_path)
    if d31_audit['phase'] != 'D31' or d32_report['phase'] != 'D32':
        raise ValueError('D31/D32 reports are unavailable.')
    by_group = {item['group_id']: item for item in d31_metadata.values()}
    boundary_groups = set(d31_audit['boundary_drops'])
    candidates = {}
    for group_id, item in by_group.items():
        if group_id in boundary_groups:
            continue
        candidates[group_id] = {
            **item,
            'source': 'whack_studio_metal_d33_safe_subset',
            'alignment_status': 'safe_candidate_not_training_ready',
        }
    if len(candidates) != 72:
        raise AssertionError(f'Expected 72 D31 no-drop candidates, got {len(candidates)}')

    recovered_added = []
    recovered_rejected = []
    for row in d32_report['results']:
        if not row['resolved']:
            continue
        item = by_group.get(row['group_id'])
        if item is None or row['group_id'] in candidates:
            continue
        ticks_per_beat, raw_events, _, _, _ = midi_tick_events(row['midi_path'])
        rebuilt = timed_events(raw_events, ticks_per_beat, row['bpm'])
        events, dropped_before, dropped_after = shift_events(
            rebuilt, row['offset_seconds'], item['duration'],
        )
        if dropped_before or dropped_after or not events:
            recovered_rejected.append({
                'group_id': row['group_id'],
                'dropped_before_audio_events': dropped_before,
                'dropped_after_audio_events': dropped_after,
            })
            continue
        candidates[row['group_id']] = {
            **item,
            'bpm': float(row['bpm']),
            'alignment_offset_seconds': float(row['offset_seconds']),
            'alignment_score': float(row['score']),
            'alignment_source_phase': 'D32',
            'source': 'whack_studio_metal_d33_safe_subset',
            'alignment_status': 'safe_recovered_candidate_not_training_ready',
            'dropped_before_audio_events': 0,
            'dropped_after_audio_events': 0,
            'events': events,
        }
        recovered_added.append(row['group_id'])

    all_groups = set(by_group) | set(d31_audit['excluded_groups'])
    excluded_groups = sorted(all_groups - set(candidates))
    split_counts = Counter(item['split'] for item in candidates.values())
    event_counts = {split: Counter() for split in ('train', 'validation', 'test')}
    for item in candidates.values():
        if any(event['time'] < 0.0 or event['time'] > item['duration'] for event in item['events']):
            raise AssertionError(f'Out-of-bounds D33 event: {item["group_id"]}')
        event_counts[item['split']].update(event['inst'] for event in item['events'])
    missing_labels = {
        split: [label for label in ('KD', 'SD', 'HH', 'TOM', 'CRASH', 'RIDE') if event_counts[split][label] == 0]
        for split in event_counts
    }
    if any(missing_labels.values()):
        raise AssertionError(f'D33 split missing labels: {missing_labels}')
    ordered = {
        f'whack_metal_d33_{index:03d}': candidates[group_id]
        for index, group_id in enumerate(sorted(candidates), start=1)
    }
    audit = {
        'phase': 'D33',
        'status': 'safe_subset_complete_not_training_ready',
        'base_no_drop_candidates': 72,
        'd32_resolved_added': recovered_added,
        'd32_resolved_rejected_boundary': recovered_rejected,
        'selected_groups': len(ordered),
        'excluded_groups': excluded_groups,
        'splits': dict(split_counts),
        'events': {split: dict(event_counts[split]) for split in event_counts},
        'missing_labels': missing_labels,
        'group_split_leaks': 0,
        # ponytail: 安全集只做零裁切篩選；其餘歌曲等資料不足時再研究分段校正。
        'ready_for_training_candidate': False,
        'ready_for_six_class_release': False,
    }
    output_dir.mkdir(parents=True)
    (output_dir / 'metadata_d33.json').write_text(json.dumps(ordered, indent=2, ensure_ascii=False), encoding='utf-8')
    (output_dir / 'audit_d33.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')
    return ordered, audit


def run_self_check():
    """以人工脈衝確認固定 BPM 的正向位移可被相關性找回。"""
    events = [
        {'tick': 0, 'velocity': 127.0},
        {'tick': 100, 'velocity': 127.0},
        {'tick': 200, 'velocity': 127.0},
    ]
    original_hop, original_rate = HOP_LENGTH, SAMPLE_RATE
    try:
        globals()['HOP_LENGTH'], globals()['SAMPLE_RATE'] = 1, 10
        pulses = midi_impulses(events, 100, 120.0, 40)
        envelope = np.roll(pulses, 2)
        result = candidate_alignment(envelope, events, 100, 120.0)
        assert abs(result['offset_seconds'] - 0.2) < 0.01
        assert result['score'] > 0.99
        shifted, dropped_before, dropped_after = shift_events(
            [{'time': 0.0, 'inst': 'KD'}, {'time': 1.0, 'inst': 'SD'}], -0.5, 0.6,
        )
        assert shifted == [{'time': 0.5, 'inst': 'SD'}]
        assert (dropped_before, dropped_after) == (1, 0)
        local = local_offsets(envelope, pulses)
        assert local is not None and max(local) - min(local) < 0.01
    finally:
        globals()['HOP_LENGTH'], globals()['SAMPLE_RATE'] = original_hop, original_rate
    print('Self-check passed.')


def main():
    """CLI 入口：產生 D29、D30 或 D31 的不可覆寫對齊資料。"""
    parser = argparse.ArgumentParser(description='Audit Whack Metal MIDI/WAV alignment candidates.')
    parser.add_argument('--metadata', default='whack_studio_metal_d28/metadata_d28.json')
    parser.add_argument('--audit', default='whack_studio_metal_d28/audit_d28.json')
    parser.add_argument('--d29-report', default='whack_studio_metal_d29/alignment_d29.json')
    parser.add_argument('--d30-report', default='whack_studio_metal_d30/filename_bpm_audit_d30.json')
    parser.add_argument('--d31-metadata', default='whack_studio_metal_d31/metadata_d31.json')
    parser.add_argument('--d31-report', default='whack_studio_metal_d31/audit_d31.json')
    parser.add_argument('--d32-output', default='whack_studio_metal_d32/recovery_d32.json')
    parser.add_argument('--d33-output-dir', default='whack_studio_metal_d33')
    parser.add_argument('--output')
    parser.add_argument('--output-dir', default='whack_studio_metal_d31')
    parser.add_argument('--all-filename-bpm', action='store_true')
    parser.add_argument('--build-d31', action='store_true')
    parser.add_argument('--recover-d32', action='store_true')
    parser.add_argument('--build-d33', action='store_true')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    if args.all_filename_bpm:
        output = args.output or 'whack_studio_metal_d30/filename_bpm_audit_d30.json'
        report = build_filename_bpm_report(args.metadata, args.audit, args.d29_report, output)
        print(f"Wrote {output}; score-pass {report['score_pass_count']}/{report['target_count']} targets.")
        return
    if args.build_d31:
        _, audit = build_d31_metadata(
            args.metadata, args.audit, args.d29_report, args.d30_report, args.output_dir,
        )
        print(f"Wrote {args.output_dir}; selected {audit['selected_groups']} candidate groups.")
        return
    if args.recover_d32:
        report = build_d32_report(
            args.d29_report, args.d30_report, args.d31_metadata, args.d31_report, args.d32_output,
        )
        print(f"Wrote {args.d32_output}; resolved {report['resolved_count']}/{report['target_count']} targets.")
        return
    if args.build_d33:
        _, audit = build_d33_metadata(
            args.d31_metadata, args.d31_report, args.d32_output, args.d33_output_dir,
        )
        print(f"Wrote {args.d33_output_dir}; selected {audit['selected_groups']} safe candidates.")
        return
    output = args.output or 'whack_studio_metal_d29/alignment_d29.json'
    report = build_report(args.metadata, args.audit, output)
    print(f"Wrote {output}; accepted {report['accepted_count']}/{report['target_count']} targets.")


if __name__ == '__main__':
    main()
