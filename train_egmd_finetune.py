# -*- coding: utf-8 -*-
"""
ADT E-GMD Fine-tuning Script with Weighted Loss
Trains the SymmetricDrumTCN model on the E-GMD dataset with channel-specific onset loss weighting
to boost Kick and Hi-Hat detection confidence at the 0.50 threshold.
"""
import os
import glob
import json
import numpy as np
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
EPOCHS = 10
STARTING_CHECKPOINT = 'best_drum_model_backup.pth'
OUTPUT_CHECKPOINT = 'best_drum_model.pth'

accomp_dir = 'accompaniment'
egmd_meta_json = 'processed_data/egmd_meta.json'

inst_indices = {'KD': 0, 'SD': 1, 'HH': 2}

# Load accompaniment pool into memory
print("Loading accompaniment pool...")
accomp_files = glob.glob(os.path.join(accomp_dir, '*.wav'))
accomp_pool = []
for path in accomp_files:
    try:
        y_accomp, _ = sf.read(path, dtype='float32')
        # Ensure mono
        if len(y_accomp.shape) > 1:
            y_accomp = np.mean(y_accomp, axis=1)
        accomp_pool.append(y_accomp)
    except Exception as e:
        print(f"Error loading accompaniment {path}: {e}")
print(f"Loaded {len(accomp_pool)} accompaniment stems.")

class EGMDDataset(Dataset):
    def __init__(self, is_training=True):
        self.is_training = is_training
        
        if not os.path.exists(egmd_meta_json):
            raise FileNotFoundError(f"E-GMD metadata not found: {egmd_meta_json}")
            
        print(f"Loading E-GMD metadata from {egmd_meta_json}...")
        with open(egmd_meta_json, 'r', encoding='utf-8') as f:
            all_songs = json.load(f)
            
        self.songs = {}
        for k, v in all_songs.items():
            if self.is_training and v['split'] == 'train':
                self.songs[k] = v
            elif not self.is_training and v['split'] in ['validation', 'test']:
                self.songs[k] = v
                
        self.keys = list(self.songs.keys())
        print(f"Loaded {len(self.keys)} E-GMD songs for {'Training' if self.is_training else 'Validation'}.")

    def __len__(self):
        # We sample a fixed number of segments per epoch to keep epochs fast
        return 2000 if self.is_training else 300

    def __getitem__(self, idx):
        if self.is_training:
            key = np.random.choice(self.keys)
        else:
            key = self.keys[idx % len(self.keys)]
            
        song = self.songs[key]
        audio_path = song['audio_path']
        
        with sf.SoundFile(audio_path) as f:
            total_samples = f.frames
            
            speed_rate = np.random.uniform(0.6, 1.5) if self.is_training else 1.0
            required_samples = int(TARGET_SAMPLES * speed_rate)
            
            if total_samples > required_samples:
                start_sample = np.random.randint(0, total_samples - required_samples) if self.is_training else 0
            else:
                start_sample = 0
                
            f.seek(start_sample)
            drum_slice = f.read(required_samples, dtype='float32')
            
            if len(drum_slice) < required_samples:
                drum_slice = np.pad(drum_slice, (0, required_samples - len(drum_slice)), mode='constant')
                
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
                    idx_inst = inst_indices[ev['inst']]
                    onset_target[frame, idx_inst] = 1.0
                    velocity_target[frame, idx_inst] = ev['velocity'] / 127.0
                    
        # Mix online accompaniment
        if self.is_training and len(accomp_pool) > 0:
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
            accomp_gain = np.random.uniform(0.1, 0.3)
            accomp_scaled = (accomp_slice / accomp_peak) * drum_peak * accomp_gain
            
            y_mix = drum_resampled + accomp_scaled
            mix_peak = np.max(np.abs(y_mix)) + 1e-6
            if mix_peak > 1.0:
                y_mix = y_mix / mix_peak
        else:
            if len(accomp_pool) > 0:
                song_idx = idx % len(accomp_pool)
                y_accomp = accomp_pool[song_idx]
                start_a = min(song_idx * TARGET_SAMPLES, max(0, len(y_accomp) - TARGET_SAMPLES))
                accomp_slice = y_accomp[start_a:start_a + TARGET_SAMPLES]
                if len(accomp_slice) < TARGET_SAMPLES:
                    accomp_slice = np.pad(accomp_slice, (0, TARGET_SAMPLES - len(accomp_slice)), mode='constant')
                
                drum_peak = np.max(np.abs(drum_resampled)) + 1e-6
                accomp_peak = np.max(np.abs(accomp_slice)) + 1e-6
                accomp_scaled = (accomp_slice / accomp_peak) * drum_peak * 0.17
                y_mix = drum_resampled + accomp_scaled
                mix_peak = np.max(np.abs(y_mix)) + 1e-6
                if mix_peak > 1.0:
                    y_mix = y_mix / mix_peak
            else:
                y_mix = drum_resampled
                
        # Feature Extraction using dsp_utils (use_hybrid=False, Standard Mel!)
        from dsp_utils import extract_features
        features = extract_features(y_mix, sr=SR, n_fft=2048, hop_length=HOP_LENGTH, n_mels=N_MELS, use_hybrid=False)
        
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
    train_dataset = EGMDDataset(is_training=True)
    val_dataset = EGMDDataset(is_training=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = SymmetricDrumTCN().to(DEVICE)
    
    # Load starting checkpoint weights
    if os.path.exists(STARTING_CHECKPOINT):
        print(f"Loading pre-trained weights from {STARTING_CHECKPOINT}...")
        checkpoint = torch.load(STARTING_CHECKPOINT, map_location=DEVICE, weights_only=False)
        if 'backbone.slot_proj.weight' in checkpoint and checkpoint['backbone.slot_proj.weight'].shape == torch.Size([64, 1024, 1, 1]):
            model.backbone.use_legacy_proj = True
            checkpoint['backbone.legacy_slot_proj.weight'] = checkpoint.pop('backbone.slot_proj.weight')
            checkpoint['backbone.legacy_slot_proj.bias'] = checkpoint.pop('backbone.slot_proj.bias')
        elif 'backbone.legacy_slot_proj.weight' in checkpoint:
            model.backbone.use_legacy_proj = True
            
        model.load_state_dict(checkpoint, strict=False)
    else:
        raise FileNotFoundError(f"Starting checkpoint not found: {STARTING_CHECKPOINT}")
        
    optimizer = optim.Adam(model.parameters(), lr=5e-5) # Low learning rate for fine-tuning
    criterion_onset = nn.BCELoss(reduction='none') # Compute element-wise loss for custom weighting
    
    best_val_f1 = 0.0
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        # Velocity loss balancing schedule
        beta = 10.0
        
        for batch_idx, (feats, onset_targets, vel_targets) in enumerate(train_loader):
            feats = feats.to(DEVICE)
            onset_targets = onset_targets.to(DEVICE)
            vel_targets = vel_targets.to(DEVICE)
            
            optimizer.zero_grad()
            
            onset_logits, vel_logits = model(feats)
            
            pred_onset = torch.sigmoid(onset_logits)
            pred_vel = torch.sigmoid(vel_logits)
            
            onset_targets_smoothed = gaussian_smooth_targets(onset_targets, DEVICE)
            vel_targets_propagated = propagate_velocity_targets(vel_targets)
            
            # Weighted Onset Loss per channel: Kick=1.2, Snare=1.0, Hi-Hat=2.5
            raw_loss_onset = criterion_onset(pred_onset, onset_targets_smoothed) # [B, T, 3]
            loss_onset_KD = raw_loss_onset[:, :, 0].mean()
            loss_onset_SD = raw_loss_onset[:, :, 1].mean()
            loss_onset_HH = raw_loss_onset[:, :, 2].mean()
            loss_onset = 1.2 * loss_onset_KD + 1.0 * loss_onset_SD + 2.5 * loss_onset_HH
            
            # Velocity Loss (Asymmetric)
            loss_vel_active = onset_targets_smoothed * (pred_vel - vel_targets_propagated) ** 2
            loss_vel_silent = 0.1 * (1.0 - onset_targets_smoothed) * (pred_vel - 0.0) ** 2
            loss_vel = (loss_vel_active + loss_vel_silent).mean()
            
            # Joint Loss
            loss = loss_onset + beta * loss_vel
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % 20 == 0:
                print(f"Epoch {epoch}/{EPOCHS} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        # Validation Loop
        model.eval()
        val_onset_preds = []
        val_vel_preds = []
        val_onset_hard = []
        val_vel_hard = []
        
        with torch.no_grad():
            for feats, onset_targets, vel_targets in val_loader:
                feats = feats.to(DEVICE)
                pred_onset_logits, pred_vel_logits = model(feats)
                pred_onset = torch.sigmoid(pred_onset_logits).cpu().numpy()
                pred_vel = torch.sigmoid(pred_vel_logits).cpu().numpy()
                
                val_onset_preds.append(pred_onset)
                val_vel_preds.append(pred_vel)
                val_onset_hard.append(onset_targets.numpy())
                val_vel_hard.append(vel_targets.numpy())
                
        val_onset_preds = np.concatenate(val_onset_preds, axis=0)
        val_vel_preds = np.concatenate(val_vel_preds, axis=0)
        val_onset_hard = np.concatenate(val_onset_hard, axis=0)
        val_vel_hard = np.concatenate(val_vel_hard, axis=0)
        
        f1s, rmse = calculate_metrics(val_onset_preds, val_onset_hard, val_vel_preds, val_vel_hard, threshold=0.50)
        mean_f1 = np.mean(f1s)
        
        print(f"\n--- Epoch {epoch:02d} Validation Results (Mixed Standard Mel) ---")
        print(f"Val F1: [KD: {f1s[0]:.3f}, SD: {f1s[1]:.3f}, HH: {f1s[2]:.3f}] | Mean F1: {mean_f1:.3f}")
        print(f"Val Vel RMSE: {rmse:.2f}")
        
        # Save best model
        if mean_f1 > best_val_f1:
            best_val_f1 = mean_f1
            torch.save(model.state_dict(), OUTPUT_CHECKPOINT)
            print(f"  --> Saved new best fine-tuned model: {OUTPUT_CHECKPOINT} (Mean F1: {mean_f1:.3f}, RMSE: {rmse:.2f})")
        print()

if __name__ == '__main__':
    main()
