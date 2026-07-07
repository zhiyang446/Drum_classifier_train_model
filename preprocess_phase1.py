# -*- coding: utf-8 -*-
import os
import glob
import xml.etree.ElementTree as ET
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Parameters
SR = 44100
HOP_LENGTH = 256
N_MELS = 256

audio_dir = 'audio'
xml_dir = 'annotation_xml'
output_dir = 'processed_data'
os.makedirs(output_dir, exist_ok=True)

inst_indices = {'KD': 0, 'SD': 1, 'HH': 2}

def get_velocity_from_clean(clean_y, onset_sec, sr, window_sec=0.06):
    start_sample = max(0, int((onset_sec - 0.01) * sr))
    end_sample = min(len(clean_y), int((onset_sec + window_sec) * sr))
    if start_sample >= end_sample:
        return 0.0
    chunk = clean_y[start_sample:end_sample]
    peak = np.max(np.abs(chunk))
    return peak

def process_file(mix_path):
    filename = os.path.basename(mix_path)
    prefix = filename.split('#')[0]
    xml_path = os.path.join(xml_dir, f"{prefix}#MIX.xml")
    
    # Load Mix WAV
    print(f"Processing: {filename}")
    y_mix, _ = librosa.load(mix_path, sr=SR, mono=True)
    n_frames = int(np.ceil(len(y_mix) / HOP_LENGTH))
    
    # Load Clean WAVs for velocity estimation
    clean_tracks = {}
    for inst in inst_indices.keys():
        clean_path = os.path.join(audio_dir, f"{prefix}#{inst}#train.wav")
        if os.path.exists(clean_path):
            y_clean, _ = librosa.load(clean_path, sr=SR, mono=True)
            clean_tracks[inst] = y_clean
        else:
            # Fallback if clean track doesn't exist: use MIX
            clean_tracks[inst] = y_mix
            
    # Parse XML
    events = []
    if os.path.exists(xml_path):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for event in root.findall('.//event'):
                inst = event.find('instrument').text
                if inst in inst_indices:
                    onset_sec = float(event.find('onsetSec').text)
                    events.append({'time': onset_sec, 'inst': inst})
        except Exception as e:
            print(f"Error parsing XML for {prefix}: {e}")
            
    # Form targets: onset (binary) and velocity (0-127)
    onset_target = np.zeros((n_frames, 3), dtype=np.float32)
    velocity_target = np.zeros((n_frames, 3), dtype=np.float32)
    
    for ev in events:
        t = ev['time']
        inst = ev['inst']
        idx = inst_indices[inst]
        frame = int(round(t * SR / HOP_LENGTH))
        if 0 <= frame < n_frames:
            # Estimate velocity
            clean_y = clean_tracks[inst]
            peak = get_velocity_from_clean(clean_y, t, SR)
            # Normalize peak to a realistic velocity (assume max amplitude is 1.0)
            velocity = np.clip(peak * 127.0, 1.0, 127.0)
            
            onset_target[frame, idx] = 1.0
            velocity_target[frame, idx] = velocity

    # Feature Extraction
    # Channel 1: Log-Mel
    mel = librosa.feature.melspectrogram(y=y_mix, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    
    # Channel 2: Linear Difference -> Log (Superflux)
    diff_mel = np.diff(mel, axis=1)
    diff_mel = np.maximum(0, diff_mel)
    diff_mel = np.pad(diff_mel, ((0, 0), (1, 0)), mode='constant')
    log_diff_mel = np.log1p(diff_mel * 1000.0) # Scale up before log to preserve resolution
    
    # Independent Z-Score Normalization
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    log_diff_mel = (log_diff_mel - log_diff_mel.mean()) / (log_diff_mel.std() + 1e-6)
    
    # Stack features: [2, N_MELS, N_FRAMES]
    features = np.stack([log_mel, log_diff_mel], axis=0)
    
    # Ensure shape alignment
    # Spectrogram might have slightly different frame count due to rounding
    n_feat_frames = features.shape[2]
    if n_feat_frames != n_frames:
        min_frames = min(n_frames, n_feat_frames)
        features = features[:, :, :min_frames]
        onset_target = onset_target[:min_frames, :]
        velocity_target = velocity_target[:min_frames, :]
        
    return features, onset_target, velocity_target

def main():
    mix_files = glob.glob(os.path.join(audio_dir, '*#MIX.wav'))
    print(f"Found {len(mix_files)} MIX files to process.")
    
    all_features = []
    all_onsets = []
    all_velocities = []
    song_names = []
    
    for mix_path in mix_files:
        feat, onset, vel = process_file(mix_path)
        all_features.append(feat)
        all_onsets.append(onset)
        all_velocities.append(vel)
        song_names.append(os.path.basename(mix_path))
        
    # Plot 5 random 4-second segments for verification
    print("Generating validation plots for Phase 1...")
    fig_dir = 'phase1_plots'
    os.makedirs(fig_dir, exist_ok=True)
    
    # Select 5 random files/positions
    np.random.seed(42)
    for i in range(5):
        song_idx = np.random.randint(0, len(all_features))
        feat = all_features[song_idx]
        onset = all_onsets[song_idx]
        
        n_frames = feat.shape[2]
        segment_len_frames = int(4.0 * SR / HOP_LENGTH) # 4 seconds
        
        if n_frames > segment_len_frames:
            start_f = np.random.randint(0, n_frames - segment_len_frames)
        else:
            start_f = 0
            segment_len_frames = n_frames
            
        end_f = start_f + segment_len_frames
        
        feat_seg = feat[:, :, start_f:end_f]
        onset_seg = onset[start_f:end_f, :]
        
        # Plot
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        
        # Channel 1: Log-Mel
        axes[0].imshow(feat_seg[0], aspect='auto', origin='lower', cmap='coolwarm')
        axes[0].set_title("Channel 1: Normalized Log-Mel Spectrogram")
        axes[0].set_ylabel("Mel Bins")
        
        # Channel 2: Log-Mel Difference (Superflux)
        axes[1].imshow(feat_seg[1], aspect='auto', origin='lower', cmap='magma')
        axes[1].set_title("Channel 2: Normalized Mel-domain Superflux Difference")
        axes[1].set_ylabel("Mel Bins")
        
        # Ground Truth Onsets
        time_axis = np.arange(segment_len_frames) * HOP_LENGTH / SR
        axes[2].plot(time_axis, onset_seg[:, 0], label='Kick (KD)', color='blue', alpha=0.7)
        axes[2].plot(time_axis, onset_seg[:, 1], label='Snare (SD)', color='red', alpha=0.7)
        axes[2].plot(time_axis, onset_seg[:, 2], label='Hi-Hat (HH)', color='green', alpha=0.7)
        axes[2].set_title("Ground Truth Drum Onsets")
        axes[2].set_xlabel("Time (seconds)")
        axes[2].set_ylabel("Activity")
        axes[2].legend(loc='upper right')
        
        plt.tight_layout()
        plot_path = os.path.join(fig_dir, f"phase1_verify_seg{i+1}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved validation plot: {plot_path}")
        
    # Save the processed data for Phase 2 training
    # Storing as object arrays containing variable length arrays for each song
    features_arr = np.empty(len(all_features), dtype=object)
    for idx, f in enumerate(all_features):
        features_arr[idx] = f
        
    onsets_arr = np.empty(len(all_onsets), dtype=object)
    for idx, o in enumerate(all_onsets):
        onsets_arr[idx] = o
        
    velocities_arr = np.empty(len(all_velocities), dtype=object)
    for idx, v in enumerate(all_velocities):
        velocities_arr[idx] = v
        
    np.savez_compressed(
        os.path.join(output_dir, 'phase1_dataset.npz'),
        features=features_arr,
        onsets=onsets_arr,
        velocities=velocities_arr,
        song_names=np.array(song_names, dtype=object)
    )
    print(f"Phase 1 complete! Saved processed data to {output_dir}/phase1_dataset.npz")
    
if __name__ == '__main__':
    main()

