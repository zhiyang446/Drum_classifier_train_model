# -*- coding: utf-8 -*-
"""
ADT GMD + IDMT Fine-tuning Script with Weighted Loss
Trains the SymmetricDrumTCN model on the GMD and IDMT datasets (acoustic domain) with channel-specific onset loss weighting
to boost Kick and Hi-Hat detection confidence at the 0.50 threshold.
"""
import os
import glob
import xml.etree.ElementTree as ET
import numpy as np
import librosa
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

audio_dir = 'audio'
xml_dir = 'annotation_xml'
accomp_dir = 'accompaniment'

inst_indices = {'KD': 0, 'SD': 1, 'HH': 2}

# Load accompaniment pool into memory
print("Loading accompaniment pool into memory...")
accomp_files = glob.glob(os.path.join(accomp_dir, '*.wav'))
accomp_pool = []
for path in accomp_files:
    try:
        y_accomp, _ = librosa.load(path, sr=SR, mono=True)
        accomp_pool.append(y_accomp)
    except Exception as e:
        print(f"Error loading accompaniment {path}: {e}")
print(f"Loaded {len(accomp_pool)} accompaniment stems.")

class DrumSeqMixDataset(Dataset):
    def __init__(self, song_prefixes, is_training=True):
        self.song_prefixes = song_prefixes
        self.is_training = is_training
        
        self.song_data = {}
        for prefix in self.song_prefixes:
            xml_path = os.path.join(xml_dir, f"{prefix}#MIX.xml")
            mix_path = os.path.join(audio_dir, f"{prefix}#MIX.wav")
            
            events = []
            if os.path.exists(xml_path):
                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    for event in root.findall('.//event'):
                        inst = event.find('instrument').text
                        if inst in inst_indices:
                            onset_sec = float(event.find('onsetSec').text)
                            events.append((onset_sec, inst))
                except Exception as e:
                    pass
            
            y_drum, _ = librosa.load(mix_path, sr=SR, mono=True)
            
            clean_tracks = {}
            for inst in inst_indices.keys():
                clean_path = os.path.join(audio_dir, f"{prefix}#{inst}#train.wav")
                if os.path.exists(clean_path):
                    y_clean, _ = librosa.load(clean_path, sr=SR, mono=True)
                    clean_tracks[inst] = y_clean
                else:
                    clean_tracks[inst] = y_drum
                    
            self.song_data[prefix] = {
                'y_drum': y_drum,
                'events': events,
                'clean_tracks': clean_tracks
            }
            
    def __len__(self):
        if self.is_training:
            return 500
        else:
            return len(self.song_prefixes) * 2
            
    def __getitem__(self, idx):
        if self.is_training:
            prefix = np.random.choice(self.song_prefixes)
            song = self.song_data[prefix]
            
            speed_rate = np.random.uniform(0.6, 1.5)
            required_samples = int(TARGET_SAMPLES * speed_rate)
            
            y_drum = song['y_drum']
            n_samples = len(y_drum)
            
            if n_samples > required_samples:
                start_s = np.random.randint(0, n_samples - required_samples)
            else:
                start_s = 0
            end_s = start_s + required_samples
            
            drum_slice = y_drum[start_s:end_s]
            if len(drum_slice) < required_samples:
                drum_slice = np.pad(drum_slice, (0, required_samples - len(drum_slice)), mode='constant')
                
            x_orig = np.linspace(0, required_samples - 1, required_samples)
            x_new = np.linspace(0, required_samples - 1, TARGET_SAMPLES)
            drum_resampled = np.interp(x_new, x_orig, drum_slice)
            
            onset_target = np.zeros((CHUNK_FRAMES, 3), dtype=np.float32)
            velocity_target = np.zeros((CHUNK_FRAMES, 3), dtype=np.float32)
            
            start_sec = start_s / SR
            end_sec = end_s / SR
            
            for t_sec, inst in song['events']:
                if start_sec <= t_sec < end_sec:
                    t_rel = t_sec - start_sec
                    t_scaled = t_rel / speed_rate
                    frame = int(round(t_scaled * SR / HOP_LENGTH))
                    if 0 <= frame < CHUNK_FRAMES:
                        clean_y = song['clean_tracks'][inst]
                        start_c = max(0, int((t_sec - 0.01) * SR))
                        end_c = min(len(clean_y), int((t_sec + 0.06) * SR))
                        peak = np.max(np.abs(clean_y[start_c:end_c])) if start_c < end_c else 0.0
                        velocity = np.clip(peak * 127.0, 1.0, 127.0)
                        
                        idx_inst = inst_indices[inst]
                        onset_target[frame, idx_inst] = 1.0
                        velocity_target[frame, idx_inst] = velocity / 127.0
                        
            accomp_idx = np.random.randint(0, len(accomp_pool))
            y_accomp = accomp_pool[accomp_idx]
            if len(y_accomp) > TARGET_SAMPLES:
                start_a = np.random.randint(0, len(y_accomp) - TARGET_SAMPLES)
            else:
                start_a = 0
            accomp_slice = y_accomp[start_a:start_a+TARGET_SAMPLES]
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
            song_idx = (idx // 2) % len(self.song_prefixes)
            prefix = self.song_prefixes[song_idx]
            song = self.song_data[prefix]
            
            y_drum = song['y_drum']
            n_samples = len(y_drum)
            
            part = idx % 2
            if part == 0:
                start_s = 0
            else:
                start_s = max(0, n_samples - TARGET_SAMPLES)
            end_s = start_s + TARGET_SAMPLES
            
            drum_resampled = y_drum[start_s:end_s]
            if len(drum_resampled) < TARGET_SAMPLES:
                drum_resampled = np.pad(drum_resampled, (0, TARGET_SAMPLES - len(drum_resampled)), mode='constant')
                
            start_sec = start_s / SR
            end_sec = end_s / SR
            
            onset_target = np.zeros((CHUNK_FRAMES, 3), dtype=np.float32)
            velocity_target = np.zeros((CHUNK_FRAMES, 3), dtype=np.float32)
            
            for t_sec, inst in song['events']:
                if start_sec <= t_sec < end_sec:
                    t_rel = t_sec - start_sec
                    frame = int(round(t_rel * SR / HOP_LENGTH))
                    if 0 <= frame < CHUNK_FRAMES:
                        clean_y = song['clean_tracks'][inst]
                        start_c = max(0, int((t_sec - 0.01) * SR))
                        end_c = min(len(clean_y), int((t_sec + 0.06) * SR))
                        peak = np.max(np.abs(clean_y[start_c:end_c])) if start_c < end_c else 0.0
                        velocity = np.clip(peak * 127.0, 1.0, 127.0)
                        
                        idx_inst = inst_indices[inst]
                        onset_target[frame, idx_inst] = 1.0
                        velocity_target[frame, idx_inst] = velocity / 127.0
                        
            accomp_idx = song_idx % len(accomp_pool)
            y_accomp = accomp_pool[accomp_idx]
            start_a = min(part * TARGET_SAMPLES, max(0, len(y_accomp) - TARGET_SAMPLES))
            accomp_slice = y_accomp[start_a:start_a+TARGET_SAMPLES]
            if len(accomp_slice) < TARGET_SAMPLES:
                accomp_slice = np.pad(accomp_slice, (0, TARGET_SAMPLES - len(accomp_slice)), mode='constant')
                
            drum_peak = np.max(np.abs(drum_resampled)) + 1e-6
            accomp_peak = np.max(np.abs(accomp_slice)) + 1e-6
            accomp_scaled = (accomp_slice / accomp_peak) * drum_peak * 0.17
            
            y_mix = drum_resampled + accomp_scaled
            mix_peak = np.max(np.abs(y_mix)) + 1e-6
            if mix_peak > 1.0:
                y_mix = y_mix / mix_peak
                
        # Feature Extraction
        mel = librosa.feature.melspectrogram(y=y_mix, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        
        diff_mel = np.diff(mel, axis=1)
        diff_mel = np.maximum(0, diff_mel)
        diff_mel = np.pad(diff_mel, ((0, 0), (1, 0)), mode='constant')
        log_diff_mel = np.log1p(diff_mel * 1000.0)
        
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
        log_diff_mel = (log_diff_mel - log_diff_mel.mean()) / (log_diff_mel.std() + 1e-6)
        
        features = np.stack([log_mel, log_diff_mel], axis=0)
        
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
    
    # Load raw file prefixes
    mix_files = glob.glob(os.path.join(audio_dir, '*#MIX.wav'))
    song_prefixes = [os.path.basename(f).split('#')[0] for f in mix_files]
    
    # Deterministic Song-level Split matching Phase 3
    np.random.seed(42)
    np.random.shuffle(song_prefixes)
    
    train_prefixes = song_prefixes[:80]
    val_prefixes = song_prefixes[80:]
    
    print(f"Dataset split: {len(train_prefixes)} songs for Training, {len(val_prefixes)} songs for Validation.")
    
    train_dataset = DrumSeqMixDataset(train_prefixes, is_training=True)
    val_dataset = DrumSeqMixDataset(val_prefixes, is_training=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize Model and load starting weights
    model = SymmetricDrumTCN().to(DEVICE)
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
        
    optimizer = optim.Adam(model.parameters(), lr=2e-4) # Fine-tuning learning rate
    
    best_val_f1 = -1.0
    best_val_rmse = 999.0
    
    print("\nStarting Weighted Onset Fine-tuning on GMD + IDMT...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_onset_loss = 0.0
        train_vel_loss = 0.0
        
        beta = 10.0
        
        for batch_idx, (feats, onset_hard, vel_hard) in enumerate(train_loader):
            feats = feats.float().to(DEVICE)
            onset_hard = onset_hard.float().to(DEVICE)
            vel_hard = vel_hard.float().to(DEVICE)
            
            onset_smoothed = gaussian_smooth_targets(onset_hard, DEVICE)
            vel_propagated = propagate_velocity_targets(vel_hard)
            
            optimizer.zero_grad()
            pred_onset_logits, pred_vel_logits = model(feats)
            
            pred_onset = torch.sigmoid(pred_onset_logits)
            bce = nn.functional.binary_cross_entropy(pred_onset, onset_smoothed, reduction='none')
            
            # Base weight mask
            weight_mask = torch.ones_like(onset_smoothed)
            
            # Asymmetric weighting: scale positive frames up and negative frames down
            weight_mask[:, :, 0] = torch.where(onset_smoothed[:, :, 0] > 0.1, 5.0, 0.5)
            weight_mask[:, :, 1] = torch.where(onset_smoothed[:, :, 1] > 0.1, 1.0, 1.0)
            weight_mask[:, :, 2] = torch.where(onset_smoothed[:, :, 2] > 0.1, 150.0, 0.1)
            
            # Velocity-dependent scaling (Option C):
            # Strong notes (>40 velocity) are fully penalized (multiplier 1.0)
            # Very weak ghost notes (<15 velocity) have their loss penalty reduced (multiplier 0.1)
            # Notes in between scale linearly
            vel_val = vel_propagated * 127.0
            w_vel = torch.clamp((vel_val - 15.0) / (40.0 - 15.0), 0.0, 1.0)
            w_vel = 0.1 + 0.9 * w_vel
            
            active_mask = onset_smoothed > 0.1
            weight_mask = torch.where(active_mask, weight_mask * w_vel, weight_mask)
            
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
            
        # Validation
        model.eval()
        val_onset_preds = []
        val_vel_preds = []
        val_onset_hard = []
        val_vel_hard = []
        
        with torch.no_grad():
            for feats, onset_hard, vel_hard in val_loader:
                feats = feats.float().to(DEVICE)
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
        
        print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f} (Onset: {train_onset_loss/len(train_loader):.4f}, Vel: {train_vel_loss/len(train_loader):.4f}) | Val F1: [KD: {f1s[0]:.3f}, SD: {f1s[1]:.3f}, HH: {f1s[2]:.3f}] Mean F1: {mean_f1:.3f} | Val Vel RMSE: {rmse:.2f}")
        
        # Save model at the end of each epoch
        torch.save(model.state_dict(), OUTPUT_CHECKPOINT)
        print(f"  --> Saved model checkpoint: {OUTPUT_CHECKPOINT}")
        if mean_f1 > best_val_f1:
            best_val_f1 = mean_f1
            
    print(f"\nFine-tuning finished. Best Validation Mean F1: {best_val_f1:.3f}")

if __name__ == '__main__':
    main()
