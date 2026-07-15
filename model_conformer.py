# -*- coding: utf-8 -*-
"""Residual DCNN 與小型 Conformer temporal encoder。"""

import torch
import torch.nn as nn

from model_dcnn import ResidualDCNNBackbone
from train_phase2 import SymmetricDrumTCN


class FeedForwardModule(nn.Module):
    """Conformer 的 pre-norm 兩層前饋模組。"""

    def __init__(self, d_model=64, expansion=2, dropout=0.1):
        """建立維度固定、可使用 half-step residual 的 FFN。"""
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """回傳 FFN 修正量，不在模組內加入 residual。"""
        return self.net(x)


class ConformerConvModule(nn.Module):
    """以 GLU、depthwise convolution 保留局部鼓擊瞬態。"""

    def __init__(self, d_model=64, kernel_size=15, dropout=0.1):
        """建立不改變時間長度的 Conformer convolution module。"""
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError('Conformer kernel_size must be odd')
        self.norm = nn.LayerNorm(d_model)
        self.pointwise_in = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=d_model,
        )
        self.channel_norm = nn.GroupNorm(1, d_model)
        self.pointwise_out = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """處理 `[B,T,D]` 並保持完全相同的 frame 數。"""
        y = self.norm(x).transpose(1, 2)
        y = nn.functional.glu(self.pointwise_in(y), dim=1)
        y = nn.functional.silu(self.channel_norm(self.depthwise(y)))
        return self.dropout(self.pointwise_out(y).transpose(1, 2))


class ConformerBlock(nn.Module):
    """最小標準 Macaron Conformer block。"""

    def __init__(self, d_model=64, num_heads=4, expansion=2, kernel_size=15, dropout=0.1):
        """建立 FFN、attention、convolution 與 final norm。"""
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, expansion, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.conv = ConformerConvModule(d_model, kernel_size, dropout)
        self.ffn2 = FeedForwardModule(d_model, expansion, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """依 Macaron 順序套用四個 residual 子模組。"""
        x = x + 0.5 * self.ffn1(x)
        normalized = self.attn_norm(x)
        attended, _ = self.attn(normalized, normalized, normalized, need_weights=False)
        x = x + self.attn_dropout(attended)
        x = x + self.conv(x)
        return self.final_norm(x + 0.5 * self.ffn2(x))


class SmallConformerEncoder(nn.Module):
    """保持 `[B,C,T]` 介面的兩層 Conformer encoder。"""

    def __init__(self, d_model=64, num_layers=2, num_heads=4, kernel_size=15):
        """建立固定小型 encoder，避免不必要的架構搜尋。"""
        super().__init__()
        self.layers = nn.ModuleList([
            ConformerBlock(d_model, num_heads, 2, kernel_size)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        """轉為 `[B,T,C]` 運算後恢復既有 head 介面。"""
        x = x.transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        return x.transpose(1, 2)


class ResidualDCNNDrumConformer(SymmetricDrumTCN):
    """以 residual DCNN frontend 搭配兩套小型 Conformer。"""

    def __init__(self, num_classes=6):
        """建立六類 frame-level onset/velocity Conformer 候選。"""
        super().__init__(num_classes=num_classes)
        self.backbone = ResidualDCNNBackbone()
        self.onset_tcn = SmallConformerEncoder()
        self.velocity_tcn = SmallConformerEncoder()


def transfer_d3r_state(model, source_state):
    """從 D3R 移植 backbone/head，明確跳過不可相容的 TCN。"""
    target = model.state_dict()
    copied = 0
    for name, value in source_state.items():
        if name.startswith(('onset_tcn.', 'velocity_tcn.')):
            continue
        if name in target and value.shape == target[name].shape:
            target[name].copy_(value)
            copied += 1
    model.load_state_dict(target)
    if 'backbone.shared.legacy_slot_proj.weight' in source_state:
        model.backbone.shared.use_legacy_proj = True
        model.backbone.correction.timbre.use_legacy_proj = True
        model.backbone.correction.transient.use_legacy_proj = True
    return copied


def load_conformer_checkpoint(model, checkpoint_path, device):
    """載入 D4 candidate 並還原 residual DCNN projection 狀態。"""
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'backbone.shared.legacy_slot_proj.weight' in state:
        model.backbone.shared.use_legacy_proj = True
        model.backbone.correction.timbre.use_legacy_proj = True
        model.backbone.correction.transient.use_legacy_proj = True
    model.load_state_dict(state)
