# -*- coding: utf-8 -*-
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
EPOCHS = 35 # Fine-tuning in Phase 3
CHECKPOINT_PATH = 'checkpoint_phase3.pth'

audio_dir = 'audio'
xml_dir = 'annotation_xml'
accomp_dir = 'accompaniment'

inst_indices = {'KD': 0, 'SD': 1, 'HH': 2}

# Load accompaniment pool into memory to avoid I/O bottlenecks
print("Loading accompaniment pool into memory...")
accomp_files = glob.glob(os.path.join(accomp_dir, '*.wav'))
accomp_pool = []
for path in accomp_files:
    y_accomp, _ = librosa.load(path, sr=SR, mono=True)
    accomp_pool.append(y_accomp)
print(f"Loaded {len(accomp_pool)} accompaniment stems.")

class DrumSeqMixDataset(Dataset):
    def __init__(self, song_prefixes, is_training=True):
        self.song_prefixes = song_prefixes
        self.is_training = is_training
        
        # Pre-parse XML files and cache clean audio tracks metadata to avoid I/O overhead
        self.song_data = {}
        for prefix in self.song_prefixes:
            xml_path = os.path.join(xml_dir, f"{prefix}#MIX.xml")
            mix_path = os.path.join(audio_dir, f"{prefix}#MIX.wav")
            
            # Parse events
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
            
            # Load full drum mix track (which represents clean drums in IDMT-SMT-DRUMS)
            y_drum, _ = librosa.load(mix_path, sr=SR, mono=True)
            
            # Load KD, SD, HH clean tracks for velocity estimation
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
            return 500 # 500 augmented chunks per epoch
        else:
            return len(self.song_prefixes) * 2 # Deterministic validation chunks
            
    def __getitem__(self, idx):
        if self.is_training:
            # 1. Select random song
            prefix = np.random.choice(self.song_prefixes)
            song = self.song_data[prefix]
            
            # 2. Select speed rate for tempo augmentation (0.6x to 1.5x)
            speed_rate = np.random.uniform(0.6, 1.5)
            required_samples = int(TARGET_SAMPLES * speed_rate)
            
            y_drum = song['y_drum']
            n_samples = len(y_drum)
            
            # Select random start position
            if n_samples > required_samples:
                start_s = np.random.randint(0, n_samples - required_samples)
            else:
                start_s = 0
            end_s = start_s + required_samples
            
            drum_slice = y_drum[start_s:end_s]
            if len(drum_slice) < required_samples:
                drum_slice = np.pad(drum_slice, (0, required_samples - len(drum_slice)), mode='constant')
                
            # Filter events in this window
            start_sec = start_s / SR
            end_sec = end_s / SR
            
            # Resample drum chunk using linear interpolation
            x_orig = np.linspace(0, required_samples - 1, required_samples)
            x_new = np.linspace(0, required_samples - 1, TARGET_SAMPLES)
            drum_resampled = np.interp(x_new, x_orig, drum_slice)
            
            # Form target frames
            onset_target = np.zeros((CHUNK_FRAMES, 3), dtype=np.float32)
            velocity_target = np.zeros((CHUNK_FRAMES, 3), dtype=np.float32)
            
            for t_sec, inst in song['events']:
                if start_sec <= t_sec < end_sec:
                    # Offset relative to start of window
                    t_rel = t_sec - start_sec
                    # Scale by speed rate
                    t_scaled = t_rel / speed_rate
                    frame = int(round(t_scaled * SR / HOP_LENGTH))
                    if 0 <= frame < CHUNK_FRAMES:
                        # Estimate velocity from clean track
                        clean_y = song['clean_tracks'][inst]
                        # Window around onset in original timeline
                        start_c = max(0, int((t_sec - 0.01) * SR))
                        end_c = min(len(clean_y), int((t_sec + 0.06) * SR))
                        peak = np.max(np.abs(clean_y[start_c:end_c])) if start_c < end_c else 0.0
                        velocity = np.clip(peak * 127.0, 1.0, 127.0)
                        
                        idx_inst = inst_indices[inst]
                        onset_target[frame, idx_inst] = 1.0
                        velocity_target[frame, idx_inst] = velocity / 127.0
                        
            # 3. Load random accompaniment chunk for mixing
            accomp_idx = np.random.randint(0, len(accomp_pool))
            y_accomp = accomp_pool[accomp_idx]
            if len(y_accomp) > TARGET_SAMPLES:
                start_a = np.random.randint(0, len(y_accomp) - TARGET_SAMPLES)
            else:
                start_a = 0
            accomp_slice = y_accomp[start_a:start_a+TARGET_SAMPLES]
            if len(accomp_slice) < TARGET_SAMPLES:
                accomp_slice = np.pad(accomp_slice, (0, TARGET_SAMPLES - len(accomp_slice)), mode='constant')
                
            # LUFS/RMS Peak Normalization: scale accompaniment to be -12dB to -6dB relative to drum peak
            drum_peak = np.max(np.abs(drum_resampled)) + 1e-6
            accomp_peak = np.max(np.abs(accomp_slice)) + 1e-6
            
            # Normalize accompaniment relative to drum peak
            accomp_gain = np.random.uniform(0.1, 0.3) # -20dB to -10dB gain
            accomp_scaled = (accomp_slice / accomp_peak) * drum_peak * accomp_gain
            
            # Mix
            y_mix = drum_resampled + accomp_scaled
            # Prevent clipping
            mix_peak = np.max(np.abs(y_mix)) + 1e-6
            if mix_peak > 1.0:
                y_mix = y_mix / mix_peak
                
        else:
            # Deterministic Validation
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
                        
            # Fixed accompaniment segment for validation
            accomp_idx = song_idx % len(accomp_pool)
            y_accomp = accomp_pool[accomp_idx]
            start_a = min(part * TARGET_SAMPLES, max(0, len(y_accomp) - TARGET_SAMPLES))
            accomp_slice = y_accomp[start_a:start_a+TARGET_SAMPLES]
            if len(accomp_slice) < TARGET_SAMPLES:
                accomp_slice = np.pad(accomp_slice, (0, TARGET_SAMPLES - len(accomp_slice)), mode='constant')
                
            # Mix with fixed -15dB gain
            drum_peak = np.max(np.abs(drum_resampled)) + 1e-6
            accomp_peak = np.max(np.abs(accomp_slice)) + 1e-6
            accomp_scaled = (accomp_slice / accomp_peak) * drum_peak * 0.17 # ~ -15dB
            
            y_mix = drum_resampled + accomp_scaled
            mix_peak = np.max(np.abs(y_mix)) + 1e-6
            if mix_peak > 1.0:
                y_mix = y_mix / mix_peak
                
        # 4. Feature Extraction (Log-Mel and Linear-domain Superflux)
        mel = librosa.feature.melspectrogram(y=y_mix, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        
        diff_mel = np.diff(mel, axis=1)
        diff_mel = np.maximum(0, diff_mel)
        diff_mel = np.pad(diff_mel, ((0, 0), (1, 0)), mode='constant')
        log_diff_mel = np.log1p(diff_mel * 1000.0)
        
        # Z-score Normalization
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
        log_diff_mel = (log_diff_mel - log_diff_mel.mean()) / (log_diff_mel.std() + 1e-6)
        
        features = np.stack([log_mel, log_diff_mel], axis=0)
        
        # Align time frames
        n_feat_frames = features.shape[2]
        if n_feat_frames != CHUNK_FRAMES:
            min_frames = min(CHUNK_FRAMES, n_feat_frames)
            features = features[:, :, :min_frames]
            if n_feat_frames < CHUNK_FRAMES:
                pad_w = CHUNK_FRAMES - n_feat_frames
                features = np.pad(features, ((0, 0), (0, 0), (0, pad_w)), mode='constant')
                
        return torch.from_numpy(features).float(), torch.from_numpy(onset_target).float(), torch.from_numpy(velocity_target).float()

# We import the exact model classes from train_phase2
from train_phase2 import SymmetricDrumTCN, gaussian_smooth_targets, propagate_velocity_targets, calculate_metrics

def main():
    print(f"Using device: {DEVICE}")
    
    # Load raw file prefixes
    mix_files = glob.glob(os.path.join(audio_dir, '*#MIX.wav'))
    song_prefixes = [os.path.basename(f).split('#')[0] for f in mix_files]
    
    # Deterministic Song-level Split matching Phase 2
    np.random.seed(42)
    np.random.shuffle(song_prefixes)
    
    train_prefixes = song_prefixes[:80]
    val_prefixes = song_prefixes[80:]
    
    print(f"Dataset split: {len(train_prefixes)} songs for Training, {len(val_prefixes)} songs for Validation.")
    
    # Build dynamic Dataloaders
    train_dataset = DrumSeqMixDataset(train_prefixes, is_training=True)
    val_dataset = DrumSeqMixDataset(val_prefixes, is_training=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize Model and load Phase 2 pre-trained weights
    model = SymmetricDrumTCN().to(DEVICE)
    if os.path.exists('best_drum_model.pth'):
        print("Loading Phase 2 pre-trained weights from best_drum_model.pth...")
        model.load_state_dict(torch.load('best_drum_model.pth', map_location=DEVICE, weights_only=False))
    else:
        print("Warning: best_drum_model.pth not found! Starting training from scratch.")
        
    optimizer = optim.Adam(model.parameters(), lr=0.0005) # Lower learning rate for fine-tuning
    
    start_epoch = 0
    best_val_f1 = -1.0
    best_val_rmse = 999.0
    
    # Checkpoint support
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Found active checkpoint at {CHECKPOINT_PATH}. Resuming training...")
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_f1 = checkpoint['best_val_f1']
            best_val_rmse = checkpoint['best_val_rmse']
            print(f"Successfully resumed from Epoch {start_epoch}. Best Val F1: {best_val_f1:.3f}, Best RMSE: {best_val_rmse:.2f}")
        except Exception as e:
            print(f"Failed to load checkpoint ({e}). Starting Phase 3 from scratch.")
            
    print("\nStarting Phase 3 full mixture training (fine-tuning)...")
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss = 0.0
        train_onset_loss = 0.0
        train_vel_loss = 0.0
        
        beta = 20.0 if epoch < 10 else 10.0 # Shorter epochs in Phase 3
        
        for batch_idx, (feats, onset_hard, vel_hard) in enumerate(train_loader):
            feats = feats.float().to(DEVICE)
            onset_hard = onset_hard.float().to(DEVICE)
            vel_hard = vel_hard.float().to(DEVICE)
            
            # Label smoothing and velocity propagation
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
        
        # Monitor weak onset recall (under 40 velocity) to track the PRD metric: "鬼音漏检率 < 8%"
        # Ghost note leakage rate = FN_weak / (TP_weak + FN_weak)
        # We can estimate it during validation:
        tps_weak = 0
        fns_weak = 0
        threshold = 0.5
        for b in range(val_onset_preds.shape[0]):
            for inst in range(3):
                true_frames = np.where(val_onset_hard[b, :, inst] > 0.5)[0]
                pred_prob = val_onset_preds[b, :, inst]
                pred_peaks = []
                for t in range(1, len(pred_prob) - 1):
                    if pred_prob[t] > threshold and pred_prob[t] > pred_prob[t-1] and pred_prob[t] > pred_prob[t+1]:
                        pred_peaks.append(t)
                
                # Check for weak hits
                for tr in true_frames:
                    tr_vel = val_vel_hard[b, tr, inst] * 127.0
                    if 1.0 <= tr_vel < 40.0:
                        # This is a ghost note. Was it matched?
                        matched = False
                        for pk in pred_peaks:
                            if abs(pk - tr) <= 5:
                                matched = True
                                break
                        if matched:
                            tps_weak += 1
                        else:
                            fns_weak += 1
                            
        ghost_leakage = fns_weak / (tps_weak + fns_weak + 1e-6)
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f} (Onset: {train_onset_loss/len(train_loader):.4f}, Vel: {train_vel_loss/len(train_loader):.4f}) | Val F1: [KD: {f1s[0]:.3f}, SD: {f1s[1]:.3f}, HH: {f1s[2]:.3f}] Mean F1: {mean_f1:.3f} | Val Vel RMSE: {rmse:.2f} | Ghost Leakage: {ghost_leakage*100.0:.1f}% (Beta: {beta})")
        
        # Save best model
        if mean_f1 > best_val_f1 or (abs(mean_f1 - best_val_f1) < 0.01 and rmse < best_val_rmse):
            best_val_f1 = mean_f1
            best_val_rmse = rmse
            torch.save(model.state_dict(), 'best_drum_model.pth')
            print(f"  --> Saved new best model (Mean F1: {mean_f1:.3f}, RMSE: {rmse:.2f}, Ghost Leakage: {ghost_leakage*100.0:.1f}%)")
            
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
            
    print(f"\nPhase 3 training finished. Best Validation Mean F1: {best_val_f1:.3f}, Best Velocity RMSE: {best_val_rmse:.2f}")

if __name__ == '__main__':
    main()
