# -*- coding: utf-8 -*-
"""小型 Conformer shape、移植、reload、backward 與 optimizer 檢查。"""

import os
import tempfile

import torch

from model_conformer import (
    ResidualDCNNDrumConformer,
    ResidualDCNNDrumHybridConformer,
    load_conformer_checkpoint,
    load_hybrid_conformer_checkpoint,
    transfer_d3r_hybrid_state,
    transfer_d3r_state,
)
from model_dcnn import ResidualDCNNDrumTCN, transfer_residual_state
from train_phase2 import SymmetricDrumTCN
from train_six_class_candidate import build_full_model_optimizer


def run_self_check():
    """驗證 D4 保持 frame 對齊、正確移植且新 encoder 可學習。"""
    source = SymmetricDrumTCN(num_classes=6)
    source.backbone.use_legacy_proj = True
    d3r = ResidualDCNNDrumTCN(num_classes=6)
    transfer_residual_state(d3r, source.state_dict())
    with torch.no_grad():
        d3r.backbone.gate.fill_(0.2)
        d3r.onset_head.bias.fill_(0.3)

    model = ResidualDCNNDrumConformer(num_classes=6)
    copied = transfer_d3r_state(model, d3r.state_dict())
    assert copied > 0
    assert torch.equal(model.backbone.gate, d3r.backbone.gate)
    assert torch.equal(model.onset_head.bias, d3r.onset_head.bias)

    features = torch.randn(2, 2, 256, 32)
    onset, velocity = model(features)
    assert onset.shape == velocity.shape == (2, 32, 6)
    assert torch.isfinite(onset).all() and torch.isfinite(velocity).all()

    hybrid = ResidualDCNNDrumHybridConformer(num_classes=6)
    hybrid_copied = transfer_d3r_hybrid_state(hybrid, d3r.state_dict())
    assert hybrid_copied > copied
    assert hybrid.onset_tcn.gate.item() == hybrid.velocity_tcn.gate.item() == 0.0
    d3r.eval()
    hybrid.eval()
    with torch.no_grad():
        d3r_output = d3r(features)
        hybrid_output = hybrid(features)
    assert torch.equal(d3r_output[0], hybrid_output[0])
    assert torch.equal(d3r_output[1], hybrid_output[1])

    hybrid_optimizer, hybrid_counts = build_full_model_optimizer(hybrid, 'dcnn-tcn-conformer', 1e-4, 1e-6, 5e-5)
    assert [group['lr'] for group in hybrid_optimizer.param_groups] == [1e-4, 1e-6, 5e-5]
    assert hybrid_counts['new_modules'] > 0
    hybrid_optimizer.zero_grad()
    hybrid(features)[0].sum().backward()
    assert hybrid.onset_tcn.gate.grad.abs().item() > 0.0
    hybrid_optimizer.step()
    hybrid_optimizer.zero_grad()
    hybrid(features)[0].sum().backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum().item() > 0.0
        for parameter in hybrid.onset_tcn.conformer.parameters()
    )

    resumed_hybrid = ResidualDCNNDrumHybridConformer(num_classes=6)
    resumed_copied = transfer_d3r_hybrid_state(resumed_hybrid, hybrid.state_dict())
    assert resumed_copied == len(hybrid.state_dict())
    assert torch.equal(resumed_hybrid.onset_tcn.gate, hybrid.onset_tcn.gate)
    assert torch.equal(next(resumed_hybrid.onset_tcn.base.parameters()), next(hybrid.onset_tcn.base.parameters()))

    optimizer, counts = build_full_model_optimizer(model, 'dcnn-conformer', 1e-4, 1e-6, 5e-5)
    assert [group['lr'] for group in optimizer.param_groups] == [1e-4, 1e-6, 5e-5]
    assert counts['new_modules'] > 0
    optimizer.zero_grad()
    (onset.square().mean() + velocity.square().mean()).backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum().item() > 0.0
        for parameter in model.onset_tcn.parameters()
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint = os.path.join(temp_dir, 'conformer.pth')
        torch.save(model.state_dict(), checkpoint)
        reloaded = ResidualDCNNDrumConformer(num_classes=6)
        load_conformer_checkpoint(reloaded, checkpoint, torch.device('cpu'))
        assert reloaded.backbone.shared.use_legacy_proj
        assert torch.equal(reloaded.onset_head.bias, model.onset_head.bias)

        hybrid_checkpoint = os.path.join(temp_dir, 'hybrid.pth')
        torch.save(hybrid.state_dict(), hybrid_checkpoint)
        hybrid_reloaded = ResidualDCNNDrumHybridConformer(num_classes=6)
        load_hybrid_conformer_checkpoint(hybrid_reloaded, hybrid_checkpoint, torch.device('cpu'))
        assert hybrid_reloaded.backbone.shared.use_legacy_proj
        assert torch.equal(hybrid_reloaded.onset_tcn.gate, hybrid.onset_tcn.gate)
    print('Conformer model self-check passed.')


if __name__ == '__main__':
    run_self_check()
