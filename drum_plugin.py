# -*- coding: utf-8 -*-
"""
自动打鼓转谱 (ADT) 系统 - 特征分辨插件模块 (drum_plugin.py)
"""

import os
import numpy as np
import librosa
import torch
import torch.nn as nn


# Import modern Symmetric TCN components and DSP utilities
from train_phase2 import SymmetricDrumTCN
from dsp_utils import extract_features


class DrumClassifierPlugin:
    """
    自动转谱系统的“特征分辨插件”。
    在主系统启动时初始化一次（常驻内存），随后只需调用 predict_slice 即可快速获得概率字典。
    """
    def __init__(self, model_path='best_drum_model.pth', device=None):
        """
        初始化并载入模型。
        :param model_path: 权重文件路径。
        :param device: 指定运行设备 ('cuda' / 'cpu')。若为 None 则自动检测。
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"[ADT 插件] 正在初始化推理引擎，当前设备: {self.device}")
        
        # 载入模型架构
        self.model = SymmetricDrumTCN().to(self.device)
        
        # 检查权重物理路径
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[ADT 插件] 未在指定路径找到权重文件: {model_path}")
            
        # 载入权重并设置为评估状态 (eval)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'backbone.legacy_slot_proj.weight' in checkpoint:
            self.model.backbone.use_legacy_proj = True
        elif 'backbone.slot_proj.weight' in checkpoint and checkpoint['backbone.slot_proj.weight'].shape == torch.Size([64, 1024, 1, 1]):
            self.model.backbone.use_legacy_proj = True
            checkpoint['backbone.legacy_slot_proj.weight'] = checkpoint.pop('backbone.slot_proj.weight')
            checkpoint['backbone.legacy_slot_proj.bias'] = checkpoint.pop('backbone.slot_proj.bias')
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.eval()
        print(f"[ADT 插件] 成功加载模型参数: {model_path}")

    def predict_slice(self, y_slice, sr=44100, n_mels=256, hop_length=256, target_frames=None):
        """
        纯净、健壮的单切片推理接口，支持 Symmetric TCN 双分支预测。
        
        :param y_slice: 一维原始波形数据 (numpy.ndarray)，采样率 44100Hz，长度约 0.74s。
        :param sr: 采样率，默认 44100Hz。
        :param n_mels: 梅尔滤波器阶数，默认 256。
        :param hop_length: 帧移，默认 256。
        :param target_frames: 未使用，保留参数兼容性。
        :return: 包含 3 个鼓组件概率与力度的字典，例如 {"Kick": 0.91, "Snare": 0.02, "Hi-Hat": 0.88, "Kick_vel": 0.75, "Snare_vel": 0.0, "Hi-Hat_vel": 0.65}
        """
        # --- 1. 防御性边界对齐 ---
        # 0.74 秒对应的精确采样点数为 32634。若传入的波形长度不符，进行截断或垫零
        expected_len = int(0.74 * sr)
        if len(y_slice) < expected_len:
            # 长度不足，右侧补零
            y_slice = np.pad(y_slice, (0, expected_len - len(y_slice)), mode='constant')
        elif len(y_slice) > expected_len:
            # 长度过长，直接截断
            y_slice = y_slice[:expected_len]
            
        # --- 2. 防御性静音防爆 ---
        # 若切片为空白噪声或接近静音，直接返回全零概率，避免对静音信号取 log 产生 NaN 特征
        if np.max(np.abs(y_slice)) < 1e-4:
            return {
                "Kick": 0.0, "Snare": 0.0, "Hi-Hat": 0.0,
                "Kick_vel": 0.0, "Snare_vel": 0.0, "Hi-Hat_vel": 0.0
            }

        # --- 3. 特征工程（使用 custom 2-channel hybrid features） ---
        features = extract_features(y_slice, sr=sr, hop_length=hop_length, n_mels=n_mels)
        
        # --- 4. 维度对齐与张量打包 [1, 2, 256, Time] ---
        x_tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)

        # --- 5. 前向计算与概率转换 ---
        with torch.no_grad():
            onset_logits, vel_logits = self.model(x_tensor)
            # 使用 Sigmoid 转换输出概率
            onset_preds = torch.sigmoid(onset_logits).squeeze(0).cpu().numpy() # [Time, 3]
            vel_preds = torch.sigmoid(vel_logits).squeeze(0).cpu().numpy() # [Time, 3]

        # Find maximum probability in the sequence for each channel to represent the slice:
        probs = np.max(onset_preds, axis=0) # shape [3]
        
        # Find the frame index of maximum probability for each channel, and take the velocity at that frame:
        vels = []
        for c in range(3):
            max_frame = np.argmax(onset_preds[:, c])
            vels.append(float(vel_preds[max_frame, c]))

        # --- 6. 结果打包输出 ---
        return {
            "Kick": float(probs[0]),
            "Snare": float(probs[1]),
            "Hi-Hat": float(probs[2]),
            "Kick_vel": float(vels[0]),
            "Snare_vel": float(vels[1]),
            "Hi-Hat_vel": float(vels[2])
        }
