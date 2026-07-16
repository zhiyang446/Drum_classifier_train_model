# -*- coding: utf-8 -*-
"""DCNN + TCN 架構與 Symmetric checkpoint 移植的最小檢查。"""

import os
import tempfile

import torch

from model_dcnn import (
    DCNNDrumTCN,
    ResidualDCNNDrumTCN,
    load_dcnn_checkpoint,
    load_residual_dcnn_checkpoint,
    transfer_residual_state,
    transfer_symmetric_state,
)
from train_six_class_candidate import build_full_model_optimizer
from train_phase2 import SymmetricDrumTCN


def run_self_check():
    """驗證舊模型相容、雙分支獨立、權重語意切分與輸出 shape。"""
    source = SymmetricDrumTCN(num_classes=6)
    source.backbone.use_legacy_proj = True
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

    residual = ResidualDCNNDrumTCN(num_classes=6)
    residual_copied = transfer_residual_state(residual, source.state_dict())
    assert residual_copied > copied
    assert residual.backbone.gate.item() == 0.0
    source.eval()
    residual.eval()
    exact_features = torch.randn(1, 2, 256, 32)
    with torch.no_grad():
        source_output = source(exact_features)
        residual_output = residual(exact_features)
    assert torch.equal(source_output[0], residual_output[0])
    assert torch.equal(source_output[1], residual_output[1])

    optimizer, counts = build_full_model_optimizer(residual, 'dcnn-residual-tcn', 1e-4, 1e-6, 5e-5)
    assert [group['lr'] for group in optimizer.param_groups] == [1e-4, 1e-6, 5e-5]
    assert counts['heads'] > 0 and counts['inherited'] > 0 and counts['new_modules'] > 0
    optimizer.zero_grad()
    residual(exact_features)[0].sum().backward()
    assert residual.backbone.gate.grad.abs().item() > 0.0
    optimizer.step()
    optimizer.zero_grad()
    residual(exact_features)[0].sum().backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum().item() > 0.0
        for parameter in residual.backbone.correction.parameters()
    )

    model.eval()
    features = torch.randn(1, 2, 256, 32)
    with torch.no_grad():
        onset, velocity = model(features)
    assert onset.shape == velocity.shape == (1, 32, 6)

    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint = os.path.join(temp_dir, 'dcnn.pth')
        torch.save(model.state_dict(), checkpoint)
        reloaded = DCNNDrumTCN(num_classes=6)
        load_dcnn_checkpoint(reloaded, checkpoint, torch.device('cpu'))
        assert reloaded.backbone.timbre.use_legacy_proj
        assert torch.equal(reloaded.onset_head.bias, model.onset_head.bias)

        residual_checkpoint = os.path.join(temp_dir, 'residual_dcnn.pth')
        torch.save(residual.state_dict(), residual_checkpoint)
        residual_reloaded = ResidualDCNNDrumTCN(num_classes=6)
        load_residual_dcnn_checkpoint(residual_reloaded, residual_checkpoint, torch.device('cpu'))
        assert residual_reloaded.backbone.shared.use_legacy_proj
        assert torch.equal(residual_reloaded.backbone.gate, residual.backbone.gate)

    try:
        model(torch.randn(1, 1, 256, 32))
    except ValueError:
        pass
    else:
        raise AssertionError('DCNN must reject non-two-channel input')
    print('DCNN model self-check passed.')


if __name__ == '__main__':
    run_self_check()
