# -*- coding: utf-8 -*-
"""
ADT DSP Utilities - Custom Linear-Log Hybrid Filterbank and Feature Extraction Pipeline
"""
import numpy as np
import librosa

SR = 44100
HOP_LENGTH = 256
N_MELS = 256

def custom_hybrid_filterbank(sr, n_fft, n_filters=256, fmin=0.0, fmax=None, split_freq=5000.0, high_ratio=0.40):
    """
    Generates a custom Linear-Log Hybrid Filterbank.
    Allocates high_ratio (e.g. 40%) of filters above split_freq (5kHz) up to fmax.
    This preserves hi-hat and high-frequency cymbal transient signatures.
    """
    if fmax is None:
        fmax = sr / 2.0
    
    n_high = int(round(n_filters * high_ratio))
    n_low = n_filters - n_high
    
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)
    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
        
    mel_min = hz_to_mel(fmin)
    mel_split = hz_to_mel(split_freq)
    
    # Generate centers for low range (excluding fmin, including split_freq)
    low_centers = mel_to_hz(np.linspace(mel_min, mel_split, n_low + 1)[1:])
    
    # Generate centers for high range (excluding split_freq, including fmax)
    high_centers = np.geomspace(split_freq, fmax, n_high + 1)[1:]
    
    centers = np.concatenate([low_centers, high_centers])
    
    n_bins = n_fft // 2 + 1
    bin_freqs = np.linspace(0, sr / 2, n_bins)
    
    fb = np.zeros((n_filters, n_bins))
    for i in range(n_filters):
        f_center = centers[i]
        
        # Left boundary
        if i == 0:
            f_left = fmin
        else:
            f_left = centers[i-1]
            
        # Right boundary
        if i == n_filters - 1:
            # Extrapolate in log scale for high range
            f_right = f_center * (f_center / centers[i-1])
        else:
            f_right = centers[i+1]
            
        rising = (bin_freqs - f_left) / (f_center - f_left + 1e-6)
        falling = (f_right - bin_freqs) / (f_right - f_center + 1e-6)
        weights = np.maximum(0.0, np.minimum(rising, falling))
        
        # Slaney-style normalization (constant energy per band)
        enorm = 2.0 / (f_right - f_left + 1e-6)
        fb[i, :] = weights * enorm
        
    return fb

def extract_features(y, sr=SR, n_fft=2048, hop_length=HOP_LENGTH, n_mels=N_MELS, use_hybrid=False):
    """
    Extracts 2-channel ADT features:
    Channel 1: Log-Mel Spectrogram (standard Mel if use_hybrid=False, otherwise hybrid filterbank)
    Channel 2: Mel-domain Superflux (先差分，后对数)
    """
    # 1. Compute linear spectrogram
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    spec = np.abs(stft) ** 2
    
    if use_hybrid:
        # Apply custom hybrid filterbank
        fb = custom_hybrid_filterbank(sr, n_fft, n_filters=n_mels)
        mel = np.dot(fb, spec)
    else:
        # Standard librosa mel spectrogram using precomputed power spectrogram
        mel = librosa.feature.melspectrogram(S=spec, sr=sr, n_mels=n_mels)
        
    # Channel 1: Log-Mel
    log_mel = librosa.power_to_db(mel, ref=np.max)
    
    # Channel 2: Superflux (先差分，后对数)
    diff_mel = np.diff(mel, axis=1)
    diff_mel = np.maximum(0.0, diff_mel)
    diff_mel = np.pad(diff_mel, ((0, 0), (1, 0)), mode='constant')
    log_diff_mel = np.log1p(diff_mel * 1000.0)
    
    # Independent Channel Z-Score Normalization
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    log_diff_mel = (log_diff_mel - log_diff_mel.mean()) / (log_diff_mel.std() + 1e-6)
    
    # Stack [2, N_MELS, N_FRAMES]
    features = np.stack([log_mel, log_diff_mel], axis=0)
    return features
