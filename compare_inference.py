# -*- coding: utf-8 -*-
import torch
import numpy as np
from train_phase2 import SymmetricDrumTCN

def main():
    device = torch.device('cpu')

    # 建立隨機輸入特徵 [batch, channels, frequency, time]
    # 在 TCN 中輸入是 [1, 2, 256, N_FRAMES]，通常 CHUNK_FRAMES = 688
    x = torch.randn(1, 2, 256, 688)

    # 載入模型與權重
    model_orig = SymmetricDrumTCN(num_classes=3).to(device)
    state_orig = torch.load('mixed_formal_kick375_snare18_hh12_candidate.pth', map_location=device, weights_only=False)

    # 相容載入
    if 'backbone.legacy_slot_proj.weight' in state_orig:
        model_orig.backbone.use_legacy_proj = True
    model_orig.load_state_dict(state_orig, strict=True)
    model_orig.eval()

    model_v13 = SymmetricDrumTCN(num_classes=6).to(device)
    state_v13 = torch.load('validation_runs/six_class_candidate_v13/six_class_candidate_v13.pth', map_location=device, weights_only=False)
    if 'backbone.legacy_slot_proj.weight' in state_v13:
        model_v13.backbone.use_legacy_proj = True
    model_v13.load_state_dict(state_v13, strict=True)
    model_v13.eval()

    # 前向傳播
    with torch.no_grad():
        logits_orig, _ = model_orig(x)
        logits_v13, _ = model_v13(x)

    logits_orig_np = torch.sigmoid(logits_orig).squeeze(0).numpy()
    logits_v13_np = torch.sigmoid(logits_v13).squeeze(0).numpy()[:, :3]

    diff = np.abs(logits_orig_np - logits_v13_np).max()
    print("Maximum difference of onset probabilities:", diff)

    # 比對前 5 幀的預測機率值以看細節
    print("Orig probe (first 3 frames):\n", logits_orig_np[:3])
    print("V13 probe (first 3 frames):\n", logits_v13_np[:3])

if __name__ == '__main__':
    main()
