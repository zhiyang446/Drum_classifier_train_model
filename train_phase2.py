# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Settings
SR = 44100
HOP_LENGTH = 256
N_MELS = 256
CHUNK_FRAMES = 688 # Reduced to ~4 seconds for 2x CPU speedup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
EPOCHS = 45
CHECKPOINT_PATH = 'checkpoint.pth'

class DrumSeqDataset(Dataset):
    def __init__(self, features, onsets, velocities, chunk_frames=688, is_training=True):
        self.features = features
        self.onsets = onsets
        self.velocities = velocities
        self.chunk_frames = chunk_frames
        self.is_training = is_training
        
    def __len__(self):
        if self.is_training:
            return 500 # Reduced from 1000 to 500 for another 2x CPU speedup
        else:
            return len(self.features) * 2
            
    def __getitem__(self, idx):
        if self.is_training:
            song_idx = np.random.randint(0, len(self.features))
        else:
            song_idx = (idx // 2) % len(self.features)
            
        feat = self.features[song_idx]
        onset = self.onsets[song_idx]
        vel = self.velocities[song_idx]
        
        n_frames = feat.shape[2]
        
        if self.is_training:
            if n_frames > self.chunk_frames:
                start_f = np.random.randint(0, n_frames - self.chunk_frames)
            else:
                start_f = 0
        else:
            part = idx % 2
            if part == 0:
                start_f = 0
            else:
                start_f = max(0, n_frames - self.chunk_frames)
                
        end_f = start_f + self.chunk_frames
        
        feat_chunk = feat[:, :, start_f:end_f]
        onset_chunk = onset[start_f:end_f, :]
        vel_chunk = vel[start_f:end_f, :]
        
        # Padding if sequence is too short
        if feat_chunk.shape[2] < self.chunk_frames:
            pad_len = self.chunk_frames - feat_chunk.shape[2]
            feat_chunk = np.pad(feat_chunk, ((0, 0), (0, 0), (0, pad_len)), mode='constant')
            onset_chunk = np.pad(onset_chunk, ((0, pad_len), (0, 0)), mode='constant')
            vel_chunk = np.pad(vel_chunk, ((0, pad_len), (0, 0)), mode='constant')
            
        # Normalize velocity targets to [0, 1]
        vel_chunk = vel_chunk / 127.0
        
        return torch.from_numpy(feat_chunk), torch.from_numpy(onset_chunk), torch.from_numpy(vel_chunk)

# Model Architecture components
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)
        
    def forward(self, x):
        x = self.conv(x)
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, dilation=1):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.relu2(out + residual)

class SharedCNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # Reduced channel capacities for 4x CPU computation speedup
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)) # Freq: 256 -> 128
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)) # Freq: 128 -> 64
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)) # Freq: 64 -> 32
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)) # Freq: 32 -> 16
        )
        
        # Multi-Slot Attention Projection
        # 16 bins split into:
        # Low: 0-4 (5 bins) -> 64 * 5 = 320 -> 32 channels
        self.low_proj = nn.Conv2d(64 * 5, 32, kernel_size=1)
        # Mid: 5-9 (5 bins) -> 64 * 5 = 320 -> 32 channels
        self.mid_proj = nn.Conv2d(64 * 5, 32, kernel_size=1)
        # High: 10-15 (6 bins) -> 64 * 6 = 384 -> 32 channels
        self.high_proj = nn.Conv2d(64 * 6, 32, kernel_size=1)
        
        self.slot_proj = nn.Conv1d(96, 64, kernel_size=1)
        
        # Legacy projection compatibility for older checkpoints
        self.legacy_slot_proj = nn.Conv2d(1024, 64, kernel_size=1)
        self.use_legacy_proj = False
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        B, C, F, T = x.shape
        
        if getattr(self, 'use_legacy_proj', False):
            x_flat = x.contiguous().view(B, C * F, 1, T)
            out = self.legacy_slot_proj(x_flat).squeeze(2)
            return out
            
        # Split frequency bins into 3 bands: Low, Mid, High
        x_low = x[:, :, :5, :].contiguous().view(B, C * 5, 1, T)
        x_mid = x[:, :, 5:10, :].contiguous().view(B, C * 5, 1, T)
        x_high = x[:, :, 10:, :].contiguous().view(B, C * 6, 1, T)
        
        # Project each band separately
        p_low = self.low_proj(x_low)   # [B, 32, 1, T]
        p_mid = self.mid_proj(x_mid)   # [B, 32, 1, T]
        p_high = self.high_proj(x_high) # [B, 32, 1, T]
        
        # Concatenate and squeeze
        p_concat = torch.cat([p_low, p_mid, p_high], dim=1).squeeze(2) # [B, 96, T]
        
        # Project back to 64 channels
        out = self.slot_proj(p_concat) # [B, 64, T]
        return out

class SymmetricDrumTCN(nn.Module):
    def __init__(self, num_classes=3):
        """中文註解：建立可指定輸出鼓件數的模型；預設三類以相容既有 checkpoint。"""
        super().__init__()
        self.num_classes = num_classes
        self.backbone = SharedCNNBackbone()
        
        self.onset_tcn = nn.Sequential(
            TCNBlock(64, 64, kernel_size=5, dilation=1),
            TCNBlock(64, 64, kernel_size=5, dilation=2),
            TCNBlock(64, 64, kernel_size=5, dilation=4),
            TCNBlock(64, 64, kernel_size=5, dilation=8),
            TCNBlock(64, 64, kernel_size=5, dilation=16)
        )
        self.onset_head = nn.Conv1d(64, num_classes, kernel_size=1)
        
        self.velocity_tcn = nn.Sequential(
            TCNBlock(64, 64, kernel_size=5, dilation=1),
            TCNBlock(64, 64, kernel_size=5, dilation=2),
            TCNBlock(64, 64, kernel_size=5, dilation=4),
            TCNBlock(64, 64, kernel_size=5, dilation=8),
            TCNBlock(64, 64, kernel_size=5, dilation=16)
        )
        self.velocity_head = nn.Conv1d(64, num_classes, kernel_size=1)
        
    def forward(self, x):
        """中文註解：輸出每個時間框的 onset 與 velocity logits。"""
        feat = self.backbone(x)
        
        onset_feat = self.onset_tcn(feat)
        onset_logits = self.onset_head(onset_feat)
        
        velocity_feat = self.velocity_tcn(feat)
        velocity_logits = self.velocity_head(velocity_feat)
        
        return onset_logits.transpose(1, 2), velocity_logits.transpose(1, 2)

# Helpers
def gaussian_smooth_targets(targets, device):
    x = targets.transpose(1, 2)
    kernel = torch.tensor([0.05, 0.25, 1.0, 0.25, 0.05], dtype=torch.float32, device=device)
    kernel = kernel.view(1, 1, 5)
    x_padded = nn.functional.pad(x, (2, 2), mode='constant', value=0.0)
    
    smoothed = []
    for i in range(3):
        ch = x_padded[:, i:i+1, :]
        sm = nn.functional.conv1d(ch, kernel, padding=0)
        smoothed.append(sm)
    smoothed = torch.cat(smoothed, dim=1)
    # Clamp to [0.0, 1.0] to prevent overlapping windows from summing above 1.0
    return torch.clamp(smoothed.transpose(1, 2), 0.0, 1.0)


def propagate_velocity_targets(vel_targets):
    x = vel_targets.transpose(1, 2)
    x_propagated = nn.functional.max_pool1d(x, kernel_size=5, stride=1, padding=2)
    return x_propagated.transpose(1, 2)

def calculate_metrics(onset_preds, onset_targets, vel_preds, vel_targets, threshold=0.5):
    tps = np.zeros(3)
    fps = np.zeros(3)
    fns = np.zeros(3)
    vel_errors = []
    
    for b in range(onset_preds.shape[0]):
        for inst in range(3):
            true_frames = np.where(onset_targets[b, :, inst] > 0.5)[0]
            pred_prob = onset_preds[b, :, inst]
            pred_peaks = []
            for t in range(1, len(pred_prob) - 1):
                if pred_prob[t] > threshold and pred_prob[t] > pred_prob[t-1] and pred_prob[t] > pred_prob[t+1]:
                    pred_peaks.append(t)
            pred_peaks = np.array(pred_peaks)
            
            matched_true = set()
            for pk in pred_peaks:
                best_match = -1
                min_dist = 6
                for tr in true_frames:
                    dist = abs(tr - pk)
                    if dist <= 5 and dist < min_dist:
                        min_dist = dist
                        best_match = tr
                if best_match != -1:
                    tps[inst] += 1
                    matched_true.add(best_match)
                    start_look = max(0, pk - 2)
                    end_look = min(onset_preds.shape[1], pk + 3)
                    pred_vel_scaled = np.max(vel_preds[b, start_look:end_look, inst]) * 127.0
                    true_vel_scaled = vel_targets[b, best_match, inst] * 127.0
                    vel_errors.append((pred_vel_scaled - true_vel_scaled) ** 2)
                else:
                    fps[inst] += 1
            fns[inst] += len(true_frames) - len(matched_true)
            
    f1s = []
    for inst in range(3):
        prec = tps[inst] / (tps[inst] + fps[inst] + 1e-6)
        rec = tps[inst] / (tps[inst] + fns[inst] + 1e-6)
        f1 = 2 * prec * rec / (prec + rec + 1e-6)
        f1s.append(f1)
        
    rmse = np.sqrt(np.mean(vel_errors)) if len(vel_errors) > 0 else 0.0
    return f1s, rmse

def main():
    print(f"Using device: {DEVICE}")
    print("Loading preprocessed dataset phase1_dataset.npz...")
    data = np.load('processed_data/phase1_dataset.npz', allow_pickle=True)
    features = data['features']
    onsets = data['onsets']
    velocities = data['velocities']
    
    np.random.seed(42)
    indices = np.arange(len(features))
    np.random.shuffle(indices)
    
    train_idx = indices[:80]
    val_idx = indices[80:]
    
    print(f"Dataset split: {len(train_idx)} songs for Training, {len(val_idx)} songs for Validation.")
    
    train_dataset = DrumSeqDataset(features[train_idx], onsets[train_idx], velocities[train_idx], chunk_frames=CHUNK_FRAMES, is_training=True)
    val_dataset = DrumSeqDataset(features[val_idx], onsets[val_idx], velocities[val_idx], chunk_frames=CHUNK_FRAMES, is_training=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = SymmetricDrumTCN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    start_epoch = 0
    best_val_f1 = -1.0
    best_val_rmse = 999.0
    
    # Auto-resume from checkpoint if it exists
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Found active checkpoint at {CHECKPOINT_PATH}. Resuming training...")
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_f1 = checkpoint['best_val_f1']
            best_val_rmse = checkpoint['best_val_rmse']
            print(f"Successfully resumed from Epoch {start_epoch}. Best Val F1: {best_val_f1:.3f}, Best RMSE: {best_val_rmse:.2f}")
        except Exception as e:
            print(f"Failed to load checkpoint ({e}). Starting training from scratch.")
            
    print("\nStarting Phase 2 model training...")
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss = 0.0
        train_onset_loss = 0.0
        train_vel_loss = 0.0
        
        beta = 20.0 if epoch < 20 else 10.0
        
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
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f} (Onset: {train_onset_loss/len(train_loader):.4f}, Vel: {train_vel_loss/len(train_loader):.4f}) | Val F1: [KD: {f1s[0]:.3f}, SD: {f1s[1]:.3f}, HH: {f1s[2]:.3f}] Mean F1: {mean_f1:.3f} | Val Vel RMSE: {rmse:.2f} (Beta: {beta})")
        
        # Save best model
        if mean_f1 > best_val_f1 or (abs(mean_f1 - best_val_f1) < 0.01 and rmse < best_val_rmse):
            best_val_f1 = mean_f1
            best_val_rmse = rmse
            torch.save(model.state_dict(), 'best_drum_model.pth')
            print(f"  --> Saved new best model (Mean F1: {mean_f1:.3f}, RMSE: {rmse:.2f})")
            
        # Save checkpoint for fault tolerance
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
            
    # Remove checkpoint on success
    if os.path.exists(CHECKPOINT_PATH):
        try:
            os.remove(CHECKPOINT_PATH)
        except:
            pass
            
    print(f"\nPhase 2 training finished. Best Validation Mean F1: {best_val_f1:.3f}, Best Velocity RMSE: {best_val_rmse:.2f}")

if __name__ == '__main__':
    main()
