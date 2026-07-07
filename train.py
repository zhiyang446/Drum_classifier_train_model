# -*- coding: utf-8 -*-
"""
自动打鼓转谱 (ADT) 系统 - 通用场景多标签训练脚本 (train.py)
功能：
1. 同时载入单体轨（*#train.wav）和混合轨（*#MIX.wav）。
2. 在时频域引入 SpecAugment (频率遮蔽与时间遮蔽) 模拟 EQ 裁剪与时值微抖动。
3. 引入随机高斯噪声与 Spectral Roll 频移，模拟不同鼓皮尺寸、调音物理变化及伴奏漏音残留。
4. 增大神经网络容量 (通道数拓宽为 32->64->128)，配合 BCEWithLogitsLoss 与梯度剪裁，完成通用化收敛。
5. 采用数据加载一次，分离实例化训练 Dataset (启用增强) 与验证 Dataset (关闭增强) 的架构设计。
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import glob
import xml.etree.ElementTree as ET
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix

def apply_spec_augment(mel_db, num_freq_masks=2, num_time_masks=2, freq_mask_width=12, time_mask_width=6):
    """
    对标准化后的梅尔频谱图应用 SpecAugment 遮蔽增强。
    以均值 0.0 进行遮挡，模拟 EQ 裁剪与时值扰动。
    """
    augmented = mel_db.copy()
    n_mels, n_frames = augmented.shape
    
    # 频率遮蔽 (Frequency Masking)
    for _ in range(num_freq_masks):
        w = np.random.randint(1, freq_mask_width + 1)
        f0 = np.random.randint(0, n_mels - w)
        augmented[f0:f0+w, :] = 0.0
        
    # 时间遮蔽 (Time Masking)
    for _ in range(num_time_masks):
        w = np.random.randint(1, time_mask_width + 1)
        t0 = np.random.randint(0, n_frames - w)
        augmented[:, t0:t0+w] = 0.0
        
    return augmented

def apply_spectral_roll_and_noise(mel_db, max_roll=2, noise_std=0.08):
    """
    对梅尔频谱图应用高斯白噪声和沿频率轴滚动（模拟变调与漏音）。
    """
    augmented = mel_db.copy()
    
    # 高斯白噪声
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, augmented.shape)
        augmented += noise
        
    # 频谱频移 (Spectral Roll)
    if max_roll > 0:
        roll = np.random.randint(-max_roll, max_roll + 1)
        if roll != 0:
            augmented = np.roll(augmented, roll, axis=0)
            # 滚出边界的部分清零
            if roll > 0:
                augmented[:roll, :] = 0.0
            else:
                augmented[roll:, :] = 0.0
                
    return augmented

def apply_time_shift(mel_db, max_shift=4):
    """
    在线时间轴抖动增强（Time-Shifting Augmentation）
    通过在时间轴（列轴）上随机平移频谱，让模型具备时间平移不变性，克服 Onset 识别微小对齐抖动问题。
    """
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift == 0:
        return mel_db
    augmented = np.full_like(mel_db, mel_db.min())
    if shift > 0:
        augmented[:, shift:] = mel_db[:, :-shift]
    else:
        augmented[:, :shift] = mel_db[:, -shift:]
    return augmented


def extract_all_features(audio_dir, xml_dir, sr=44100, hop_length=512, n_mels=128, target_frames=64):
    """
    预先在内存中加载并提取所有音频切片的梅尔频谱特征。
    """
    samples = []
    labels = []
    
    inst_indices = {'KD': 0, 'SD': 1, 'HH': 2}
    
    # 1. 扫描混合轨 (*#MIX.wav)
    mix_files = glob.glob(os.path.join(audio_dir, '*#MIX.wav'))
    print(f"找到 {len(mix_files)} 个混合音频轨。开始预加载...")
    
    for idx, audio_path in enumerate(mix_files):
        filename = os.path.basename(audio_path)
        prefix = filename.split('#')[0]
        xml_path = os.path.join(xml_dir, f"{prefix}#MIX.xml")
        
        print(f"[MIX {idx+1}/{len(mix_files)}] 正在预载: {filename}")
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
        
        # 30ms 临近 Onset 合并
        onsets_merged = []
        if os.path.exists(xml_path):
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                raw_events = []
                for event in root.findall('.//event'):
                    inst = event.find('instrument').text
                    if inst in inst_indices:
                        onset_sec = float(event.find('onsetSec').text)
                        raw_events.append((onset_sec, inst))
                
                raw_events.sort(key=lambda x: x[0])
                for t, inst in raw_events:
                    merged = False
                    for mo in onsets_merged:
                        if abs(mo['time'] - t) < 0.03:
                            mo['labels'][inst_indices[inst]] = 1.0
                            mo['time'] = (mo['time'] + t) / 2.0
                            merged = True
                            break
                    if not merged:
                        lbl = np.zeros(3, dtype=np.float32)
                        lbl[inst_indices[inst]] = 1.0
                        onsets_merged.append({'time': t, 'labels': lbl})
            except Exception as e:
                print(f"  -> 解析 XML 失败 ({e})，跳过该 MIX 文件。")
                continue
        else:
            continue
            
        # 物理切片提取
        _slice_audio(y, onsets_merged, samples, labels, sr, n_mels, hop_length, target_frames)

    # 2. 扫描单体轨 (*#train.wav)
    train_files = glob.glob(os.path.join(audio_dir, '*#train.wav'))
    print(f"\n找到 {len(train_files)} 个单体连奏轨。开始预加载...")
    
    for idx, audio_path in enumerate(train_files):
        filename = os.path.basename(audio_path)
        parts = filename.split('#')
        if len(parts) < 3:
            continue
        prefix = parts[0]
        inst_code = parts[1]
        
        if inst_code not in inst_indices:
            continue
            
        label_idx = inst_indices[inst_code]
        inst_label = np.zeros(3, dtype=np.float32)
        inst_label[label_idx] = 1.0
        
        xml_name = filename.replace('.wav', '.xml')
        xml_path = os.path.join(xml_dir, xml_name)
        
        print(f"[TRAIN {idx+1}/{len(train_files)}] 正在预载: {filename}")
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
        
        onsets = []
        if os.path.exists(xml_path):
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for event in root.findall('.//event'):
                    onset_sec = event.find('onsetSec')
                    if onset_sec is not None:
                        onsets.append(float(onset_sec.text))
                if not onsets:
                    for point in root.findall('.//point'):
                        frame = point.get('frame')
                        if frame is not None:
                            onsets.append(float(frame) / sr)
            except Exception as e:
                pass
                
        # Onset Fallback
        if not onsets:
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length, backtrack=True)
            onsets = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
            
        onsets_data = [{'time': t, 'labels': inst_label} for t in onsets]
        _slice_audio(y, onsets_data, samples, labels, sr, n_mels, hop_length, target_frames)
        
    return np.array(samples, dtype=np.float32), np.array(labels, dtype=np.float32)

def _slice_audio(y, onsets_data, samples, labels, sr, n_mels, hop_length, target_frames):
    chunk_len = int(0.74 * sr)
    for data in onsets_data:
        t = data['time']
        lbl = data['labels']
        
        start_sample = int((t - 0.05) * sr)
        end_sample = start_sample + chunk_len
        
        # 边界防爆
        if start_sample < 0:
            chunk = y[0:max(0, end_sample)]
            pad_len = chunk_len - len(chunk)
            chunk = np.pad(chunk, (pad_len, 0), mode='constant')
        elif end_sample > len(y):
            chunk = y[max(0, start_sample):]
            pad_len = chunk_len - len(chunk)
            chunk = np.pad(chunk, (0, pad_len), mode='constant')
        else:
            chunk = y[start_sample:end_sample]
            
        # 静音段过滤
        if np.max(np.abs(chunk)) < 1e-4:
            continue
            
        # 计算梅尔谱图并对齐
        mel = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=n_mels, hop_length=hop_length)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        if mel_db.shape[1] < target_frames:
            pad_w = target_frames - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0, 0), (0, pad_w)), mode='constant', constant_values=-80.0)
        else:
            mel_db = mel_db[:, :target_frames]
            
        # 标准化且避免除零
        mean = mel_db.mean()
        std = mel_db.std()
        mel_db = (mel_db - mean) / (std + 1e-6)
        
        samples.append(mel_db)
        labels.append(lbl)


class DrumDataset(Dataset):
    """
    多标签时频增强型 Dataset
    """
    def __init__(self, samples, labels, augment=False):
        self.samples = samples
        self.labels = labels
        self.augment = augment
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        x = self.samples[idx]
        y = self.labels[idx].copy()
        
        # 在线增强与鼓组共时敲击合成机制
        if self.augment:
            # 50% 概率进行随机鼓组共时叠加合成
            if np.random.rand() < 0.5:
                other_idx = np.random.randint(len(self.samples))
                x_other = self.samples[other_idx]
                y_other = self.labels[other_idx]
                
                # 时频叠加（取最大值以模拟两个音色在频谱中的叠加）
                x = np.maximum(x, x_other)
                # 标签进行并集操作，代表共时发生
                y = np.maximum(y, y_other)
                
            x = apply_time_shift(x, max_shift=4) # 加入时间轴随机抖动，最大偏置 4 帧（约 46.4ms）
            x = apply_spec_augment(x)
            x = apply_spectral_roll_and_noise(x)
            
        # 通道扩维 ➔ [1, 128, 64]
        x = np.expand_dims(x, axis=0)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


class DrumCNN(nn.Module):
    """
    大容量多标签卷积神经网络 (32 -> 64 -> 128)
    """
    def __init__(self):
        super(DrumCNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # [B, 32, 64, 32]
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # [B, 64, 32, 16]
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # [B, 128, 16, 8]
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128 * 16 * 8, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 3)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def main():
    audio_dir = 'audio'
    xml_dir = 'annotation_xml'
    
    # 1. 预载入全部样本特征
    print("正在进行音频预载入与时频域特征提取...")
    samples, labels = extract_all_features(audio_dir, xml_dir)
    print(f"\n特征提取完毕，共载入 {len(samples)} 个样本切片。")
    
    if len(samples) == 0:
        return
        
    # 2. 划分训练/验证集并独立封装数据集（分离增强策略）
    train_samples, val_samples, train_labels, val_labels = train_test_split(
        samples, labels, test_size=0.2, random_state=42
    )
    
    # 融入合成波形共时打击数据（仅合并到训练集，保持验证集为纯真实样本评估）
    if os.path.exists('synthetic_data.npz'):
        print("\n检测到合成波形数据集 synthetic_data.npz，正在载入并融合到训练集...")
        synth = np.load('synthetic_data.npz')
        synth_samples = synth['samples']
        synth_labels = synth['labels']
        
        train_samples = np.concatenate([train_samples, synth_samples], axis=0)
        train_labels = np.concatenate([train_labels, synth_labels], axis=0)
        print(f"  -> 融合成功！已将 {synth_samples.shape[0]} 个时域物理合成样本并入训练集。")
        print(f"  -> 最终训练样本数: {train_samples.shape[0]}，验证样本数: {val_samples.shape[0]}")
    
    # 训练集开启动态增强，验证集关闭增强
    train_dataset = DrumDataset(train_samples, train_labels, augment=True)
    val_dataset = DrumDataset(val_samples, val_labels, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 3. 设定计算设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前使用的训练设备为: {device}")
    
    # 4. 初始化模型、损失函数与优化器
    model = DrumCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 5. 模型训练循环
    epochs = 45
    best_loss = float('inf')
    
    print("\n====== 开始通用泛化型多标签训练 ======")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # 梯度截断防爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            
        epoch_loss = train_loss / len(train_samples)
        
        # 验证循环
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
        val_epoch_loss = val_loss / len(val_samples)
        
        print(f"Epoch [{epoch+1:02d}/{epochs}] "
              f"Train Loss: {epoch_loss:.5f} | Val Loss: {val_epoch_loss:.5f}")
              
        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            torch.save(model.state_dict(), 'drum_classifier.pth')
            print("  -> 已保存当前最佳泛化权重模型。")
            
    print(f"\n训练结束！最佳验证 Loss 为: {best_loss:.5f}")
    
    # 6. 载入最优权重输出通用分类评估报告
    model.load_state_dict(torch.load('drum_classifier.pth'))
    model.eval()
    
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int()
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    mcm = multilabel_confusion_matrix(all_targets, all_preds)
    classes = ['Kick (底鼓 / 标签 0)', 'Snare (军鼓 / 标签 1)', 'Hi-Hat (踩镲 / 标签 2)']
    
    print("\n" + "="*20 + " 通用多标签分类评估报告 " + "="*20)
    for i, class_name in enumerate(classes):
        cm = mcm[i]
        tn, fp, fn, tp = cm.ravel()
        
        acc = (tp + tn) / (tn + fp + fn + tp) * 100
        precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        print(f"=== {class_name} ===")
        print(f"  混淆矩阵 (Confusion Matrix):\n  [[TN: {tn:<4} FP: {fp:<4}]\n   [FN: {fn:<4} TP: {tp:<4}]]")
        print(f"  准确率 (Accuracy) : {acc:.2f}%")
        print(f"  精确率 (Precision): {precision:.2f}%")
        print(f"  召回率 (Recall)   : {recall:.2f}%")
        print(f"  F1-Score          : {f1:.2f}%")
        print("-" * 50)
    print("="*64 + "\n")

if __name__ == '__main__':
    main()
