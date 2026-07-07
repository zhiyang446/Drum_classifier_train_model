# -*- coding: utf-8 -*-
"""
自动打鼓转谱 (ADT) 系统 - 时域物理数据合成脚本 (synthesize_data.py)
功能：
1. 从 audio/ 目录中提取 KD、SD、HH 单体轨的纯净打击波形切片。
2. 在时域（波形级）混合不同的单体鼓，并应用随机增益和微小时间偏移（Jitter）以模拟人类真实的共时打击。
3. 提取混合波形的梅尔频谱，进行与 train.py 一致的归一化处理。
4. 将合成特征输出到 synthetic_data.npz 供训练载入。
"""

import os
import glob
import xml.etree.ElementTree as ET
import numpy as np
import librosa
import torch

def load_onsets_from_xml_or_fallback(audio_path, xml_path, sr=44100, hop_length=512):
    """
    加载 XML 中的 Onset，或者 fallback 到 librosa 的 onset 检测器
    """
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
            
    if not onsets:
        # Fallback to onset detector
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length, backtrack=True)
        onsets = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
        
    return onsets

def slice_single_instrument(audio_path, onsets, sr=44100, chunk_len_sec=0.74):
    """
    提取单个音符切片波形
    """
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    chunk_len = int(chunk_len_sec * sr)
    slices = []
    
    for t in onsets:
        start_sample = int((t - 0.05) * sr)
        end_sample = start_sample + chunk_len
        
        # 边界垫零
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
            
        # 过滤静音
        if np.max(np.abs(chunk)) < 1e-4:
            continue
            
        # 归一化振幅，使各音色混音时增益可控
        chunk = chunk / (np.max(np.abs(chunk)) + 1e-6)
        slices.append(chunk)
        
    return slices

def apply_time_shift_waveform(y, shift_samples):
    """
    在时域波形上应用平移
    """
    if shift_samples == 0:
        return y
    if shift_samples > 0:
        return np.pad(y, (shift_samples, 0), mode='constant')[:-shift_samples]
    else:
        return np.pad(y, (0, -shift_samples), mode='constant')[-shift_samples:]

def get_decay_waveform(w, delay_sec, sr=44100):
    """
    提取前一个打击乐器在 delay_sec 秒后的余音衰减波形。
    当前切片起于 onset - 0.05 秒，所以前一打击点对应的偏移量为 delay_sec。
    """
    chunk_len = len(w)
    start_sample = int(delay_sec * sr)
    if start_sample >= chunk_len:
        return np.zeros(chunk_len)
    decay_w = w[start_sample:]
    if len(decay_w) < chunk_len:
        decay_w = np.pad(decay_w, (0, chunk_len - len(decay_w)), mode='constant')
    else:
        decay_w = decay_w[:chunk_len]
    return decay_w

def main():
    sr = 44100
    hop_length = 512
    n_mels = 128
    target_frames = 64
    chunk_len = int(0.74 * sr)
    
    audio_dir = 'audio'
    xml_dir = 'annotation_xml'
    
    print("正在扫描 audio 文件夹以提取单体鼓波形切片...")
    
    kd_files = glob.glob(os.path.join(audio_dir, '*#KD#train.wav'))
    sd_files = glob.glob(os.path.join(audio_dir, '*#SD#train.wav'))
    hh_files = glob.glob(os.path.join(audio_dir, '*#HH#train.wav'))
    
    print(f"找到底鼓(KD)音轨 {len(kd_files)} 个，军鼓(SD)音轨 {len(sd_files)} 个，踩镲(HH)音轨 {len(hh_files)} 个。")
    
    kd_pool = []
    sd_pool = []
    hh_pool = []
    
    # 提取底鼓
    for path in kd_files:
        filename = os.path.basename(path)
        xml_name = filename.replace('.wav', '.xml')
        xml_path = os.path.join(xml_dir, xml_name)
        onsets = load_onsets_from_xml_or_fallback(path, xml_path, sr, hop_length)
        slices = slice_single_instrument(path, onsets, sr)
        kd_pool.extend(slices)
        
    # 提取军鼓
    for path in sd_files:
        filename = os.path.basename(path)
        xml_name = filename.replace('.wav', '.xml')
        xml_path = os.path.join(xml_dir, xml_name)
        onsets = load_onsets_from_xml_or_fallback(path, xml_path, sr, hop_length)
        slices = slice_single_instrument(path, onsets, sr)
        sd_pool.extend(slices)
        
    # 提取踩镲
    for path in hh_files:
        filename = os.path.basename(path)
        xml_name = filename.replace('.wav', '.xml')
        xml_path = os.path.join(xml_dir, xml_name)
        onsets = load_onsets_from_xml_or_fallback(path, xml_path, sr, hop_length)
        slices = slice_single_instrument(path, onsets, sr)
        hh_pool.extend(slices)
        
    print(f"提取完成！\n底鼓切片数: {len(kd_pool)}\n军鼓切片数: {len(sd_pool)}\n踩镲切片数: {len(hh_pool)}")
    
    if len(kd_pool) == 0 or len(sd_pool) == 0 or len(hh_pool) == 0:
        print("错误：单体切片池为空，无法进行合成数据！")
        return
        
    # 合成参数
    num_samples_per_class = 1500
    synthetic_samples = []
    synthetic_labels = []
    
    # 最大时移 35ms 对应的样本数
    max_shift_samples = int(0.035 * sr)
    
    print(f"\n开始在时域自动物理合成共时敲击数据（每类 {num_samples_per_class} 个样本）...")
    
    # 1. KD + SD (底鼓+军鼓) [1.0, 1.0, 0.0]
    print("正在合成 [Kick + Snare]...")
    for _ in range(num_samples_per_class):
        kd_w = kd_pool[np.random.randint(len(kd_pool))]
        sd_w = sd_pool[np.random.randint(len(sd_pool))]
        
        # 随机增益
        g_kd = np.random.uniform(0.7, 1.0)
        g_sd = np.random.uniform(0.6, 0.9)
        
        # 随机时间平移(模拟非对齐)
        shift = np.random.randint(-max_shift_samples, max_shift_samples + 1)
        sd_w_shifted = apply_time_shift_waveform(sd_w, shift)
        
        # 叠加混合与防溢出归一化
        mixed = g_kd * kd_w + g_sd * sd_w_shifted
        mixed = mixed / (np.max(np.abs(mixed)) + 1e-6)
        
        synthetic_samples.append(mixed)
        synthetic_labels.append([1.0, 1.0, 0.0])
        
    # 2. KD + HH (底鼓+踩镲) [1.0, 0.0, 1.0]
    print("正在合成 [Kick + Hi-Hat]...")
    for _ in range(num_samples_per_class):
        kd_w = kd_pool[np.random.randint(len(kd_pool))]
        hh_w = hh_pool[np.random.randint(len(hh_pool))]
        
        g_kd = np.random.uniform(0.7, 1.0)
        g_hh = np.random.uniform(0.15, 0.8)
        
        shift = np.random.randint(-max_shift_samples, max_shift_samples + 1)
        hh_w_shifted = apply_time_shift_waveform(hh_w, shift)
        
        mixed = g_kd * kd_w + g_hh * hh_w_shifted
        mixed = mixed / (np.max(np.abs(mixed)) + 1e-6)
        
        synthetic_samples.append(mixed)
        synthetic_labels.append([1.0, 0.0, 1.0])
        
    # 3. SD + HH (军鼓+踩镲) [0.0, 1.0, 1.0]
    print("正在合成 [Snare + Hi-Hat]...")
    for _ in range(num_samples_per_class):
        sd_w = sd_pool[np.random.randint(len(sd_pool))]
        hh_w = hh_pool[np.random.randint(len(hh_pool))]
        
        g_sd = np.random.uniform(0.7, 1.0)
        g_hh = np.random.uniform(0.4, 0.8)
        
        shift = np.random.randint(-max_shift_samples, max_shift_samples + 1)
        hh_w_shifted = apply_time_shift_waveform(hh_w, shift)
        
        mixed = g_sd * sd_w + g_hh * hh_w_shifted
        mixed = mixed / (np.max(np.abs(mixed)) + 1e-6)
        
        synthetic_samples.append(mixed)
        synthetic_labels.append([0.0, 1.0, 1.0])
        
    # 4. KD + SD + HH (底鼓+军鼓+踩镲) [1.0, 1.0, 1.0]
    print("正在合成 [Kick + Snare + Hi-Hat]...")
    for _ in range(num_samples_per_class):
        kd_w = kd_pool[np.random.randint(len(kd_pool))]
        sd_w = sd_pool[np.random.randint(len(sd_pool))]
        hh_w = hh_pool[np.random.randint(len(hh_pool))]
        
        g_kd = np.random.uniform(0.7, 1.0)
        g_sd = np.random.uniform(0.6, 0.9)
        g_hh = np.random.uniform(0.15, 0.8)
        
        shift_sd = np.random.randint(-max_shift_samples, max_shift_samples + 1)
        shift_hh = np.random.randint(-max_shift_samples, max_shift_samples + 1)
        
        sd_w_shifted = apply_time_shift_waveform(sd_w, shift_sd)
        hh_w_shifted = apply_time_shift_waveform(hh_w, shift_hh)
        
        mixed = g_kd * kd_w + g_sd * sd_w_shifted + g_hh * hh_w_shifted
        mixed = mixed / (np.max(np.abs(mixed)) + 1e-6)
        
        synthetic_samples.append(mixed)
        synthetic_labels.append([1.0, 1.0, 1.0])
        
    # 5. HH + KD decay (踩镲+底鼓余音，训练模型排除对底鼓余音的误判) [0.0, 0.0, 1.0]
    print("正在合成 [Hi-Hat + Kick Decay]...")
    for _ in range(num_samples_per_class):
        hh_w = hh_pool[np.random.randint(len(hh_pool))]
        kd_w = kd_pool[np.random.randint(len(kd_pool))]
        delay = np.random.uniform(0.20, 0.60)
        kd_decay = get_decay_waveform(kd_w, delay, sr)
        
        g_hh = np.random.uniform(0.6, 1.0)
        g_kd = np.random.uniform(0.2, 0.5)
        
        mixed = g_hh * hh_w + g_kd * kd_decay
        mixed = mixed / (np.max(np.abs(mixed)) + 1e-6)
        
        synthetic_samples.append(mixed)
        synthetic_labels.append([0.0, 0.0, 1.0])

    # 6. HH + SD decay (踩镲+军鼓余音，防止对军鼓余音误判) [0.0, 0.0, 1.0]
    print("正在合成 [Hi-Hat + Snare Decay]...")
    for _ in range(num_samples_per_class):
        hh_w = hh_pool[np.random.randint(len(hh_pool))]
        sd_w = sd_pool[np.random.randint(len(sd_pool))]
        delay = np.random.uniform(0.20, 0.60)
        sd_decay = get_decay_waveform(sd_w, delay, sr)
        
        g_hh = np.random.uniform(0.6, 1.0)
        g_sd = np.random.uniform(0.2, 0.5)
        
        mixed = g_hh * hh_w + g_sd * sd_decay
        mixed = mixed / (np.max(np.abs(mixed)) + 1e-6)
        
        synthetic_samples.append(mixed)
        synthetic_labels.append([0.0, 0.0, 1.0])

    # 7. HH + KD decay + SD decay (踩镲+底鼓军鼓双余音，完美匹配雷鬼等节奏的第4拍) [0.0, 0.0, 1.0]
    print("正在合成 [Hi-Hat + Kick/Snare Double Decay]...")
    for _ in range(num_samples_per_class):
        hh_w = hh_pool[np.random.randint(len(hh_pool))]
        kd_w = kd_pool[np.random.randint(len(kd_pool))]
        sd_w = sd_pool[np.random.randint(len(sd_pool))]
        delay = np.random.uniform(0.20, 0.60)
        kd_decay = get_decay_waveform(kd_w, delay, sr)
        sd_decay = get_decay_waveform(sd_w, delay, sr)
        
        g_hh = np.random.uniform(0.6, 1.0)
        g_kd = np.random.uniform(0.15, 0.4)
        g_sd = np.random.uniform(0.15, 0.4)
        
        mixed = g_hh * hh_w + g_kd * kd_decay + g_sd * sd_decay
        mixed = mixed / (np.max(np.abs(mixed)) + 1e-6)
        
        synthetic_samples.append(mixed)
        synthetic_labels.append([0.0, 0.0, 1.0])

    # 8. SD + KD decay (军鼓+底鼓余音，防止将军鼓强击后的余音误触发) [0.0, 1.0, 0.0]
    print("正在合成 [Snare + Kick Decay]...")
    for _ in range(num_samples_per_class):
        sd_w = sd_pool[np.random.randint(len(sd_pool))]
        kd_w = kd_pool[np.random.randint(len(kd_pool))]
        delay = np.random.uniform(0.20, 0.60)
        kd_decay = get_decay_waveform(kd_w, delay, sr)
        
        g_sd = np.random.uniform(0.7, 1.0)
        g_kd = np.random.uniform(0.2, 0.5)
        
        mixed = g_sd * sd_w + g_kd * kd_decay
        mixed = mixed / (np.max(np.abs(mixed)) + 1e-6)
        
        synthetic_samples.append(mixed)
        synthetic_labels.append([0.0, 1.0, 0.0])

    # 9. KD + SD decay (底鼓+军鼓余音) [1.0, 0.0, 0.0]
    print("正在合成 [Kick + Snare Decay]...")
    for _ in range(num_samples_per_class):
        kd_w = kd_pool[np.random.randint(len(kd_pool))]
        sd_w = sd_pool[np.random.randint(len(sd_pool))]
        delay = np.random.uniform(0.20, 0.60)
        sd_decay = get_decay_waveform(sd_w, delay, sr)
        
        g_kd = np.random.uniform(0.7, 1.0)
        g_sd = np.random.uniform(0.2, 0.5)
        
        mixed = g_kd * kd_w + g_sd * sd_decay
        mixed = mixed / (np.max(np.abs(mixed)) + 1e-6)
        
        synthetic_samples.append(mixed)
        synthetic_labels.append([1.0, 0.0, 0.0])
        
    # 10. KD alone (纯底鼓) [1.0, 0.0, 0.0]
    print("正在合成 [Kick Alone]...")
    for _ in range(num_samples_per_class):
        kd_w = kd_pool[np.random.randint(len(kd_pool))]
        g_kd = np.random.uniform(0.5, 1.0)
        mixed = g_kd * kd_w
        synthetic_samples.append(mixed)
        synthetic_labels.append([1.0, 0.0, 0.0])
        
    # 11. SD alone (纯军鼓) [0.0, 1.0, 0.0]
    print("正在合成 [Snare Alone]...")
    for _ in range(num_samples_per_class):
        sd_w = sd_pool[np.random.randint(len(sd_pool))]
        g_sd = np.random.uniform(0.5, 1.0)
        mixed = g_sd * sd_w
        synthetic_samples.append(mixed)
        synthetic_labels.append([0.0, 1.0, 0.0])
        
    # 12. HH alone (纯踩镲) [0.0, 0.0, 1.0]
    print("正在合成 [Hi-Hat Alone]...")
    for _ in range(num_samples_per_class):
        hh_w = hh_pool[np.random.randint(len(hh_pool))]
        g_hh = np.random.uniform(0.4, 0.9)
        mixed = g_hh * hh_w
        synthetic_samples.append(mixed)
        synthetic_labels.append([0.0, 0.0, 1.0])
        
    print(f"\n物理波形合成完毕，共生成 {len(synthetic_samples)} 个样本。")
    print("正在计算梅尔频谱图与特征归一化...")
    
    mel_samples = []
    for idx, w in enumerate(synthetic_samples):
        if (idx + 1) % 1000 == 0 or idx == 0:
            print(f"  进度: {idx+1}/{len(synthetic_samples)}")
            
        mel = librosa.feature.melspectrogram(y=w, sr=sr, n_mels=n_mels, hop_length=hop_length)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # 裁剪/填充到 target_frames
        if mel_db.shape[1] < target_frames:
            pad_w = target_frames - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0, 0), (0, pad_w)), mode='constant', constant_values=-80.0)
        else:
            mel_db = mel_db[:, :target_frames]
            
        # 均值-方差归一化
        mean = mel_db.mean()
        std = mel_db.std()
        mel_db = (mel_db - mean) / (std + 1e-6)
        
        mel_samples.append(mel_db)
        
    mel_samples = np.array(mel_samples, dtype=np.float32)
    labels = np.array(synthetic_labels, dtype=np.float32)
    
    # 保存为 npz 压缩文件
    out_path = 'synthetic_data.npz'
    np.savez_compressed(out_path, samples=mel_samples, labels=labels)
    print(f"\n[成功] 已经将合成数据集导出到: {out_path} ({mel_samples.shape[0]} 样本, 维度: {mel_samples.shape[1:]})")

if __name__ == '__main__':
    main()
