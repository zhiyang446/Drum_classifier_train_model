# -*- coding: utf-8 -*-
"""True SuperFlux 特徵的最小可重複檢查。"""

import numpy as np

from dsp_utils import extract_features, superflux_difference


def run_self_check():
    """驗證 shape、靜態輸入、頻率漂移抑制、寬頻瞬態與 opt-in 相容性。"""
    stationary = np.ones((5, 6), dtype=np.float32)
    assert np.array_equal(superflux_difference(stationary), np.zeros_like(stationary))

    drifting = np.zeros((5, 4), dtype=np.float32)
    drifting[2, 0] = 4.0
    drifting[3, 2] = 4.0
    assert superflux_difference(drifting)[3, 2] == 0.0

    transient = np.zeros((5, 4), dtype=np.float32)
    transient[:, 2] = 5.0
    assert np.all(superflux_difference(transient)[:, 2] > 0.0)

    for kwargs in ({'lag': 0}, {'max_size': 2}):
        try:
            superflux_difference(stationary, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f'expected ValueError for {kwargs}')

    waveform = np.zeros(44100 // 2, dtype=np.float32)
    waveform[4000:4010] = 1.0
    legacy = extract_features(waveform)
    explicit_legacy = extract_features(waveform, use_true_superflux=False)
    true_superflux = extract_features(waveform, use_true_superflux=True)
    assert np.array_equal(legacy, explicit_legacy)
    assert true_superflux.shape == legacy.shape and np.isfinite(true_superflux).all()
    assert not np.array_equal(true_superflux[1], legacy[1])
    print('True SuperFlux self-check passed.')


if __name__ == '__main__':
    run_self_check()
