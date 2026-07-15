# -*- coding: utf-8 -*-
"""DCNN + TCN 架構與 Symmetric checkpoint 移植的最小檢查。"""

import torch

from model_dcnn import DCNNDrumTCN, transfer_symmetric_state
from train_phase2 import SymmetricDrumTCN


def run_self_check():
    """驗證舊模型相容、雙分支獨立、權重語意切分與輸出 shape。"""
    source = SymmetricDrumTCN(num_classes=6)
    assert source.backbone.conv1[0].weight.shape[1] == 2
    with torch.no_grad():
        source.backbone.conv1[0].weight[:, 0].fill_(1.0)
        source.backbone.conv1[0].weight[:, 1].fill_(2.0)
        source.onset_head.bias.fill_(3.0)

    model = DCNNDrumTCN(num_classes=6)
    copied = transfer_symmetric_state(model, source.state_dict())
    assert copied > 0
    assert torch.all(model.backbone.timbre.conv1[0].weight == 1.0)
    assert torch.all(model.backbone.transient.conv1[0].weight == 2.0)
    assert torch.equal(model.onset_head.bias, source.onset_head.bias)
    assert model.backbone.timbre.conv1[0].weight.data_ptr() != model.backbone.transient.conv1[0].weight.data_ptr()

    model.eval()
    features = torch.randn(1, 2, 256, 32)
    with torch.no_grad():
        onset, velocity = model(features)
    assert onset.shape == velocity.shape == (1, 32, 6)

    try:
        model(torch.randn(1, 1, 256, 32))
    except ValueError:
        pass
    else:
        raise AssertionError('DCNN must reject non-two-channel input')
    print('DCNN model self-check passed.')


if __name__ == '__main__':
    run_self_check()
