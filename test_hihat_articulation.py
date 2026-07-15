# -*- coding: utf-8 -*-
"""Hi-Hat 原始高頻衰減判定的最小可重複檢查。"""

import numpy as np

from transcribe import classify_hihat_articulation, compute_hihat_hf_power


def run_self_check():
    """驗證開放、閉合、密集 fallback 與高頻包絡的基本行為。"""
    closed_envelope = np.full(400, 1e-6, dtype=np.float32)
    closed_envelope[98:104] = 1.0
    closed_envelope[107:128] = 0.01

    open_envelope = closed_envelope.copy()
    open_envelope[107:128] = 0.25

    closed_pitch, closed_decay = classify_hihat_articulation(closed_envelope, 100, None)
    open_pitch, open_decay = classify_hihat_articulation(open_envelope, 100, None)
    dense_pitch, dense_decay = classify_hihat_articulation(open_envelope, 100, 108)

    assert closed_pitch == 42 and closed_decay is not None
    assert open_pitch == 46 and open_decay is not None
    assert dense_pitch == 42 and dense_decay is None

    sr = 44100
    samples = np.arange(sr, dtype=np.float32) / sr
    low_tone = np.sin(2.0 * np.pi * 200.0 * samples).astype(np.float32)
    high_tone = np.sin(2.0 * np.pi * 8000.0 * samples).astype(np.float32)
    assert np.mean(compute_hihat_hf_power(high_tone, sr=sr)) > 1000.0 * np.mean(
        compute_hihat_hf_power(low_tone, sr=sr)
    )
    print('Hi-Hat articulation self-check passed.')


if __name__ == '__main__':
    run_self_check()
