# -*- coding: utf-8 -*-
import os
import glob
import json
import xml.etree.ElementTree as ET
import numpy as np
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Settings
SR = 44100
HOP_LENGTH = 256
N_MELS = 256
CHUNK_FRAMES = 688 # ~4 seconds
TARGET_SAMPLES = CHUNK_FRAMES * HOP_LENGTH # 176128 samples
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
EPOCHS = 35 # Fine-tuning/adaptation epochs in Phase 3
CHECKPOINT_PATH = 'checkpoint_gmd_phase3.pth'

audio_dir = 'audio'
xml_dir = 'annotation_xml'
accomp_dir = 'accompaniment'
gmd_meta_json = 'processed_data/gmd_meta.json'

inst_indices = {'KD': 0, 'SD': 1, 'HH': 2}

# Load accompaniment pool into memory to avoid I/O bottlenecks
print("Loading accompaniment pool into memory...")
accomp_files = glob.glob(os.path.join(accomp_dir, '*.wav'))
accomp_pool = []
for path in accomp_files:
    y_accomp, _ = librosa.load(path, sr=SR, mono=True)
    accomp_pool.append(y_accomp)
print(f"Loaded {len(accomp_pool)} accompaniment stems.")



def load_npy_slice(npy_path, start_sample, num_samples):
    """
    Loads a specific slice of a float32 array from disk using memory mapping.
    """
    try:
        arr = np.load(npy_path, mmap_mode='r')
        total_samples = len(arr)
        
        start = max(0, min(start_sample, total_samples - 1))
        end = min(start + num_samples, total_samples)
        
        slice_data = arr[start:end]
        
        if len(slice_data) < num_samples:
            slice_data = np.pad(slice_data, (0, num_samples - len(slice_data)), mode='constant')
        return np.array(slice_data, dtype=np.float32)
    except Exception as e:
        print(f"Error loading NPY slice from {npy_path}: {e}")
        return np.zeros(num_samples, dtype=np.float32)


from dsp_utils import extract_features

class UniversalMixDrumDataset(Dataset):
    """
    Dataset for Phase 3: Online accompaniment mixing and speed/tempo augmentation.
    Uses memory-mapped array slicing for constant RAM utilization.
    """
    def __init__(self, is_training=True):
        self.is_training = is_training
        
        # 1. Load GMD Metadata
        self.gmd_songs = {}
        if os.path.exists(gmd_meta_json):
            print(f"Loading GMD metadata from {gmd_meta_json}...")
            with open(gmd_meta_json, 'r', encoding='utf-8') as f:
                all_gmd = json.load(f)
            
            for k, v in all_gmd.items():
                if self.is_training and v['split'] == 'train':
                    self.gmd_songs[k] = v
                elif not self.is_training and v['split'] in ['validation', 'test']:
                    self.gmd_songs[k] = v
            print(f"Loaded {len(self.gmd_songs)} GMD songs for {'Training' if self.is_training else 'Validation'}.")
        else:
            print("Warning: GMD metadata not found.")
            
        self.gmd_keys = list(self.gmd_songs.keys())

        # 2. Load IDMT Metadata
        idmt_meta_json = 'processed_data/idmt_meta.json'
        self.idmt_songs = {}
        if os.path.exists(idmt_meta_json):
            print(f"Loading IDMT metadata from {idmt_meta_json}...")
            with open(idmt_meta_json, 'r', encoding='utf-8') as f:
                all_idmt = json.load(f)
            
            # Standard random partition matching original code split
            mix_files = glob.glob(os.path.join(audio_dir, '*#MIX.wav'))
            song_prefixes = [os.path.basename(f).split('#')[0] for f in mix_files]
            np.random.seed(42)
            np.random.shuffle(song_prefixes)
            
            idmt_prefixes = song_prefixes[:80] if self.is_training else song_prefixes[80:]
            self.idmt_keys = idmt_prefixes
            for prefix in idmt_prefixes:
                if prefix in all_idmt:
                    self.idmt_songs[prefix] = all_idmt[prefix]
            print(f"Loaded {len(self.idmt_songs)} IDMT songs for {'Training' if self.is_training else 'Validation'}.")
        else:
            print("Warning: IDMT metadata not found.")
            self.idmt_keys = []

    def __len__(self):
        if self.is_training:
            return 800
        else:
            return len(self.idmt_keys) * 2 + min(100, len(self.gmd_keys))

    def __getitem__(self, idx):
        is_gmd = False
        if len(self.gmd_keys) > 0:
            if self.is_training:
                is_gmd = np.random.rand() > 0.5
            else:
                is_gmd = idx >= (len(self.idmt_keys) * 2)

        if is_gmd:
            if self.is_training:
                key = np.random.choice(self.gmd_keys)
            else:
                gmd_idx = idx - (len(self.idmt_keys) * 2)
                key = self.gmd_keys[gmd_idx % len(self.gmd_keys)]
                
            song = self.gmd_songs[key]
            wave_npy_path = song['wave_npy_path']
            
            arr_temp = np.load(wave_npy_path, mmap_mode='r')
            total_samples = len(arr_temp)
            
            speed_rate = np.random.uniform(0.6, 1.5) if self.is_training else 1.0
            required_samples = int(TARGET_SAMPLES * speed_rate)
            
            if total_samples > required_samples:
                start_sample = np.random.randint(0, total_samples - required_samples) if self.is_training else 0
            else:
                start_sample = 0
            
            drum_slice = load_npy_slice(wave_npy_path, start_sample, required_samples)
            
            if speed_rate != 1.0:
                x_orig = np.linspace(0, required_samples - 1, required_samples)
                x_new = np.linspace(0, required_samples - 1, TARGET_SAMPLES)
                drum_resampled = np.interp(x_new, x_orig, drum_slice)
            else:
                drum_resampled = drum_slice
                
            # Form target labels
            onset_target = np.zeros((CHUNK_FRAMES, 3), dtype=np.float32)
            velocity_target = np.zeros((CHUNK_FRAMES, 3), dtype=np.float32)
            
            start_sec = start_sample / SR
            end_sec = (start_sample + required_samples) / SR
            
            for ev in song['events']:
                t_sec = ev['time']
                if start_sec <= t_sec < end_sec:
                    t_rel = t_sec - start_sec
                    t_scaled = t_rel / speed_rate
                    frame = int(round(t_scaled * SR / HOP_LENGTH))
                    if 0 <= frame < CHUNK_FRAMES:
                        idx_inst = inst_indices[ev['inst']]
                        onset_target[frame, idx_inst] = 1.0
                        velocity_target[frame, idx_inst] = ev['velocity'] / 127.0
        else:
            if self.is_training:
                key = np.random.choice(self.idmt_keys)
            else:
                key = self.idmt_keys[(idx // 2) % len(self.idmt_keys)]
                
            song = self.idmt_songs[key]
            wave_npy_path = song['wave_npy_path']
            
            arr_temp = np.load(wave_npy_path, mmap_mode='r')
            total_samples = len(arr_temp)
            
            speed_rate = np.random.uniform(0.6, 1.5) if self.is_training else 1.0
            required_samples = int(TARGET_SAMPLES * speed_rate)
            
            if self.is_training:
                start_sample = np.random.randint(0, total_samples - required_samples) if total_samples > required_samples else 0
            else:
                part = idx % 2
                start_sample = 0 if part == 0 else max(0, total_samples - TARGET_SAMPLES)
                required_samples = TARGET_SAMPLES
                speed_rate = 1.0
                
            drum_slice = load_npy_slice(wave_npy_path, start_sample, required_samples)
            
            if speed_rate != 1.0:
                x_orig = np.linspace(0, required_samples - 1, required_samples)
                x_new = np.linspace(0, required_samples - 1, TARGET_SAMPLES)
                drum_resampled = np.interp(x_new, x_orig, drum_slice)
            else:
                drum_resampled = drum_slice
                
            onset_target = np.zeros((CHUNK_FRAMES, 3), dtype=np.float32)
            velocity_target = np.zeros((CHUNK_FRAMES, 3), dtype=np.float32)
            
            start_sec = start_sample / SR
            end_sec = (start_sample + required_samples) / SR
            
            for ev in song['events']:
                t_sec = ev['time']
                if start_sec <= t_sec < end_sec:
                    t_rel = t_sec - start_sec
                    t_scaled = t_rel / speed_rate
                    frame = int(round(t_scaled * SR / HOP_LENGTH))
                    if 0 <= frame < CHUNK_FRAMES:
                        inst = ev['inst']
                        idx_inst = inst_indices[inst]
                        
                        clean_npy_path = song['clean_tracks_npy_paths'][inst]
                        start_c = max(0, int((t_sec - 0.01) * SR))
                        end_c = min(total_samples, int((t_sec + 0.06) * SR))
                        clean_slice = load_npy_slice(clean_npy_path, start_c, end_c - start_c)
                        peak = np.max(np.abs(clean_slice)) if len(clean_slice) > 0 else 0.0
                        velocity = np.clip(peak * 127.0, 1.0, 127.0)
                        
                        onset_target[frame, idx_inst] = 1.0
                        velocity_target[frame, idx_inst] = velocity / 127.0

        # --- Accompaniment mixing (online augmentation) ---
        if self.is_training:
            accomp_idx = np.random.randint(0, len(accomp_pool))
            y_accomp = accomp_pool[accomp_idx]
            if len(y_accomp) > TARGET_SAMPLES:
                start_a = np.random.randint(0, len(y_accomp) - TARGET_SAMPLES)
            else:
                start_a = 0
            accomp_slice = y_accomp[start_a:start_a + TARGET_SAMPLES]
            if len(accomp_slice) < TARGET_SAMPLES:
                accomp_slice = np.pad(accomp_slice, (0, TARGET_SAMPLES - len(accomp_slice)), mode='constant')
                
            drum_peak = np.max(np.abs(drum_resampled)) + 1e-6
            accomp_peak = np.max(np.abs(accomp_slice)) + 1e-6
            accomp_gain = np.random.uniform(0.1, 0.3) # -20dB to -10dB gain
            accomp_scaled = (accomp_slice / accomp_peak) * drum_peak * accomp_gain
            
            y_mix = drum_resampled + accomp_scaled
            mix_peak = np.max(np.abs(y_mix)) + 1e-6
            if mix_peak > 1.0:
                y_mix = y_mix / mix_peak
        else:
            # Deterministic validation mix
            song_idx = idx % len(accomp_pool)
            y_accomp = accomp_pool[song_idx]
            start_a = min(song_idx * TARGET_SAMPLES, max(0, len(y_accomp) - TARGET_SAMPLES))
            accomp_slice = y_accomp[start_a:start_a + TARGET_SAMPLES]
            if len(accomp_slice) < TARGET_SAMPLES:
                accomp_slice = np.pad(accomp_slice, (0, TARGET_SAMPLES - len(accomp_slice)), mode='constant')
                
            drum_peak = np.max(np.abs(drum_resampled)) + 1e-6
            accomp_peak = np.max(np.abs(accomp_slice)) + 1e-6
            accomp_scaled = (accomp_slice / accomp_peak) * drum_peak * 0.17 # ~ -15dB gain
            
            y_mix = drum_resampled + accomp_scaled
            mix_peak = np.max(np.abs(y_mix)) + 1e-6
            if mix_peak > 1.0:
                y_mix = y_mix / mix_peak

        # --- Feature Extraction using dsp_utils ---
        features = extract_features(y_mix, sr=SR, n_fft=2048, hop_length=HOP_LENGTH, n_mels=N_MELS)
        
        n_feat_frames = features.shape[2]
        if n_feat_frames != CHUNK_FRAMES:
            min_frames = min(CHUNK_FRAMES, n_feat_frames)
            features = features[:, :, :min_frames]
            if n_feat_frames < CHUNK_FRAMES:
                pad_w = CHUNK_FRAMES - n_feat_frames
                features = np.pad(features, ((0, 0), (0, 0), (0, pad_w)), mode='constant')
                
        return torch.from_numpy(features).float(), torch.from_numpy(onset_target).float(), torch.from_numpy(velocity_target).float()


from train_phase2 import SymmetricDrumTCN, gaussian_smooth_targets, propagate_velocity_targets, calculate_metrics

def main():
    print(f"Using device: {DEVICE}")
    print("Initializing Universal Mixed Drum Datasets...")
    train_dataset = UniversalMixDrumDataset(is_training=True)
    val_dataset = UniversalMixDrumDataset(is_training=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = SymmetricDrumTCN().to(DEVICE)
    
    # Load Phase 2 clean model weights as start checkpoint for Phase 3
    if os.path.exists('best_drum_model_phase2.pth'):
        print("Loading Phase 2 pre-trained weights from best_drum_model_phase2.pth...")
        model.load_state_dict(torch.load('best_drum_model_phase2.pth', map_location=DEVICE, weights_only=False))
    elif os.path.exists('best_drum_model.pth'):
        print("Loading pre-trained weights from best_drum_model.pth...")
        model.load_state_dict(torch.load('best_drum_model.pth', map_location=DEVICE, weights_only=False))
    else:
        print("Warning: No pre-trained weights found! Starting Phase 3 training from scratch.")
        
    optimizer = optim.Adam(model.parameters(), lr=0.0005) # Lower LR for fine-tuning
    
    start_epoch = 0
    best_val_f1 = -1.0
    best_val_rmse = 999.0
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Found active checkpoint at {CHECKPOINT_PATH}. Resuming...")
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_f1 = checkpoint['best_val_f1']
            best_val_rmse = checkpoint['best_val_rmse']
            print(f"Resumed from Epoch {start_epoch}. Best Val F1: {best_val_f1:.3f}")
        except Exception as e:
            print(f"Failed to load checkpoint ({e}). Starting Phase 3 GMD training from scratch.")
            
    print("\nStarting Universal Phase 3: Mixture & Speed Fine-Tuning...")
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss = 0.0
        train_onset_loss = 0.0
        train_vel_loss = 0.0
        
        beta = 20.0 if epoch < 10 else 10.0 # Loss weight scheduling
        
        for batch_idx, (feats, onset_hard, vel_hard) in enumerate(train_loader):
            feats = feats.to(DEVICE)
            onset_hard = onset_hard.to(DEVICE)
            vel_hard = vel_hard.to(DEVICE)
            
            onset_smoothed = gaussian_smooth_targets(onset_hard, DEVICE)
            vel_propagated = propagate_velocity_targets(vel_hard)
            
            optimizer.zero_grad()
            pred_onset_logits, pred_vel_logits = model(feats)
            
            pred_onset = torch.sigmoid(pred_onset_logits)
            bce = nn.functional.binary_cross_entropy(pred_onset, onset_smoothed, reduction='none')
            
            weight_mask = torch.ones_like(onset_smoothed)
            is_weak = (onset_smoothed > 0.0) & (vel_propagated > 0.0) & (vel_propagated < 40.0 / 127.0)
            weight_mask[is_weak] = 5.0
            loss_onset = (bce * weight_mask).mean()
            
            pred_vel = torch.sigmoid(pred_vel_logits)
            loss_vel_active = onset_smoothed * (pred_vel - vel_propagated) ** 2
            loss_vel_silent = 0.1 * (1.0 - onset_smoothed) * (pred_vel - 0.0) ** 2
            loss_vel = (loss_vel_active + loss_vel_silent).mean()
            
            loss_total = loss_onset + beta * loss_vel
            
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            train_loss += loss_total.item()
            train_onset_loss += loss_onset.item()
            train_vel_loss += loss_vel.item()
            
            if (batch_idx + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss_total.item():.4f}")
                
        # Validation Loop
        model.eval()
        val_onset_preds = []
        val_vel_preds = []
        val_onset_hard = []
        val_vel_hard = []
        
        with torch.no_grad():
            for feats, onset_hard, vel_hard in val_loader:
                feats = feats.to(DEVICE)
                pred_onset_logits, pred_vel_logits = model(feats)
                pred_onset = torch.sigmoid(pred_onset_logits).cpu().numpy()
                pred_vel = torch.sigmoid(pred_vel_logits).cpu().numpy()
                
                val_onset_preds.append(pred_onset)
                val_vel_preds.append(pred_vel)
                val_onset_hard.append(onset_hard.numpy())
                val_vel_hard.append(vel_hard.numpy())
                
        val_onset_preds = np.concatenate(val_onset_preds, axis=0)
        val_vel_preds = np.concatenate(val_vel_preds, axis=0)
        val_onset_hard = np.concatenate(val_onset_hard, axis=0)
        val_vel_hard = np.concatenate(val_vel_hard, axis=0)
        
        f1s, rmse = calculate_metrics(val_onset_preds, val_onset_hard, val_vel_preds, val_vel_hard)
        mean_f1 = np.mean(f1s)
        
        print(f"\n--- Epoch {epoch+1:02d} Validation Results (Mixed) ---")
        print(f"Val F1: [KD: {f1s[0]:.3f}, SD: {f1s[1]:.3f}, HH: {f1s[2]:.3f}] | Mean F1: {mean_f1:.3f}")
        print(f"Val Vel RMSE: {rmse:.2f}")
        
        # Save best model to final weights
        if mean_f1 > best_val_f1 or (abs(mean_f1 - best_val_f1) < 0.01 and rmse < best_val_rmse):
            best_val_f1 = mean_f1
            best_val_rmse = rmse
            torch.save(model.state_dict(), 'best_drum_model.pth')
            print(f"  --> Saved new best mixture model: best_drum_model.pth (Mean F1: {mean_f1:.3f}, RMSE: {rmse:.2f})")
            
        # Save checkpoint
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_f1': best_val_f1,
                'best_val_rmse': best_val_rmse
            }
            torch.save(checkpoint, CHECKPOINT_PATH)
        except Exception as e:
            print(f"Warning: Failed to save checkpoint ({e})")
            
    if os.path.exists(CHECKPOINT_PATH):
        try:
            os.remove(CHECKPOINT_PATH)
        except:
            pass
            
    print(f"\nPhase 3 Mixture Fine-Tuning finished. Best Model saved to best_drum_model.pth")

if __name__ == '__main__':
    main()
