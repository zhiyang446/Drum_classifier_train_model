# -*- coding: utf-8 -*-
"""Log-Mel／True SuperFlux 雙分支 DCNN 與既有 TCN 的最小組合。"""

import torch
import torch.nn as nn

from train_phase2 import SharedCNNBackbone, SymmetricDrumTCN


class DCNNBackbone(nn.Module):
    """以兩個獨立 CNN 分別處理音色與瞬態，再融合為 64 個時間特徵。"""

    def __init__(self):
        """建立單通道雙分支，並把 fusion 初始化為逐通道平均。"""
        super().__init__()
        self.timbre = SharedCNNBackbone(input_channels=1)
        self.transient = SharedCNNBackbone(input_channels=1)
        self.fusion = nn.Conv1d(128, 64, kernel_size=1)
        with torch.no_grad():
            self.fusion.weight.zero_()
            self.fusion.bias.zero_()
            indices = torch.arange(64)
            self.fusion.weight[indices, indices, 0] = 0.5
            self.fusion.weight[indices, indices + 64, 0] = 0.5

    def forward(self, features):
        """將 channel 0/1 分別送入音色／瞬態 CNN，並回傳 late-fusion 特徵。"""
        if features.ndim != 4 or features.shape[1] != 2:
            raise ValueError('DCNN input must have shape [batch, 2, frequency, time]')
        timbre = self.timbre(features[:, 0:1])
        transient = self.transient(features[:, 1:2])
        return self.fusion(torch.cat((timbre, transient), dim=1))


class DCNNDrumTCN(SymmetricDrumTCN):
    """以 DCNN 替換共享頻譜 backbone，保留既有 TCN 與輸出 heads。"""

    def __init__(self, num_classes=6):
        """建立六類預設的 DCNN+TCN 候選模型。"""
        super().__init__(num_classes=num_classes)
        self.backbone = DCNNBackbone()


def transfer_symmetric_state(model, source_state):
    """把 Symmetric 模型的 channel 語意分流至 DCNN，並移植相容 TCN/head tensor。"""
    target = model.state_dict()
    copied = 0
    for branch, source_channel in (('timbre', 0), ('transient', 1)):
        for source_name, value in source_state.items():
            if not source_name.startswith('backbone.'):
                continue
            suffix = source_name[len('backbone.'):]
            target_name = f'backbone.{branch}.{suffix}'
            if target_name not in target:
                continue
            candidate = value[:, source_channel:source_channel + 1] if suffix == 'conv1.0.weight' else value
            if candidate.shape == target[target_name].shape:
                target[target_name].copy_(candidate)
                copied += 1

    for source_name, value in source_state.items():
        if source_name.startswith('backbone.') or source_name not in target:
            continue
        if value.shape == target[source_name].shape:
            target[source_name].copy_(value)
            copied += 1

    model.load_state_dict(target)
    if 'backbone.legacy_slot_proj.weight' in source_state:
        model.backbone.timbre.use_legacy_proj = True
        model.backbone.transient.use_legacy_proj = True
    return copied


def load_dcnn_checkpoint(model, checkpoint_path, device):
    """載入 DCNN 候選並依 state keys 還原兩分支 legacy projection。"""
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'backbone.timbre.legacy_slot_proj.weight' in state:
        model.backbone.timbre.use_legacy_proj = True
        model.backbone.transient.use_legacy_proj = True
    model.load_state_dict(state)
