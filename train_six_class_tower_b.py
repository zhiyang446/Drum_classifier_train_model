# -*- coding: utf-8 -*-
"""
Model B (Rare Classes Specialization Tower) training script.
Pre-weights TOM/CRASH/RIDE positive losses to strongly optimize Recall.
"""
import argparse
import json
import os

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from run_six_class_smoke import CHUNK_FRAMES, LABELS, SR, TARGET_SAMPLES, build_window, load_accompaniment, load_compatible_weights
from train_star_smoke import freeze_batchnorm_stats
from train_phase2 import SymmetricDrumTCN, propagate_velocity_targets
from train_six_class_candidate import build_schedule, batch_from_schedule, gaussian_smooth_targets


def rare_competition_loss(onset_logits, onset_target):
    """只在單一罕見類別真值 frame 計算 TOM/CRASH/RIDE 競爭損失。"""
    rare_target = onset_target[:, :, 3:6]
    single_rare_mask = rare_target.sum(dim=-1) == 1.0
    if not torch.any(single_rare_mask):
        return onset_logits.sum() * 0.0
    rare_class = rare_target.argmax(dim=-1)
    return F.cross_entropy(onset_logits[:, :, 3:6][single_rare_mask], rare_class[single_rare_mask])


def rare_focal_loss(onset_logits, onset_target, positive_weights, gamma, adversarial_weight):
    """只對 TOM/CRASH/RIDE 計算 focal BCE，並加重 core-hit frame 的罕見類別負樣本。"""
    rare_logits = onset_logits[:, :, 3:6]
    rare_target = onset_target[:, :, 3:6]
    probabilities = torch.sigmoid(rare_logits)
    bce = F.binary_cross_entropy_with_logits(rare_logits, rare_target, reduction='none')
    pt = rare_target * probabilities + (1.0 - rare_target) * (1.0 - probabilities)
    weights = torch.where(rare_target > 0.0, positive_weights.expand_as(rare_target), torch.ones_like(rare_target))
    has_core_hit = onset_target[:, :, 0:3].sum(dim=-1, keepdim=True) > 0.0
    adversarial_mask = has_core_hit & (rare_target == 0.0)
    weights = torch.where(adversarial_mask, torch.full_like(weights, adversarial_weight), weights)
    return (((1.0 - pt) ** gamma) * bce * weights).mean()


def run_self_check():
    """確認競爭損失只處理單一 rare 真值，且正確類別分數較低。"""
    target = torch.zeros(1, 2, 6)
    target[0, 0, 3] = 1.0
    target[0, 1, 3:5] = 1.0
    correct = torch.zeros(1, 2, 6)
    wrong = torch.zeros(1, 2, 6)
    correct[0, 0, 3] = 5.0
    wrong[0, 0, 4] = 5.0
    assert rare_competition_loss(correct, target) < rare_competition_loss(wrong, target)
    assert rare_competition_loss(torch.zeros(1, 1, 6), torch.zeros(1, 1, 6)).item() == 0.0
    focal_target = torch.zeros(1, 1, 6)
    focal_target[0, 0, 3] = 1.0
    focal_weights = torch.ones(1, 1, 3)
    focal_correct = torch.zeros(1, 1, 6)
    focal_wrong = torch.zeros(1, 1, 6)
    focal_correct[0, 0, 3] = 5.0
    focal_wrong[0, 0, 3] = -5.0
    assert rare_focal_loss(focal_correct, focal_target, focal_weights, 2.0, 1.0) < rare_focal_loss(focal_wrong, focal_target, focal_weights, 2.0, 1.0)
    print('Rare competition self-check passed.')


def main():
    parser = argparse.ArgumentParser(description='Train Model B specialized for TOM/CRASH/RIDE.')
    parser.add_argument('--meta', help='Path to metadata json')
    parser.add_argument('--checkpoint', default=r'validation_runs/six_class_candidate_v14/six_class_candidate_v14.pth')
    parser.add_argument('--output-dir', default='validation_runs/six_class_tower_b')
    parser.add_argument('--per-class', type=int, default=24)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-5, help='Heads learning rate')
    parser.add_argument('--backbone-lr', type=float, default=1e-6, help='Backbone micro learning rate')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--gaussian-targets', action='store_true', default=True)
    parser.add_argument('--positive-weight', type=float, default=20.0, help='Base positive weight for KD/SD/HH')
    parser.add_argument('--rare-positive-weight', type=float, default=50.0, help='Enhanced positive weight for TOM/CRASH/RIDE')
    parser.add_argument('--adversarial-weight', type=float, default=40.0, help='Adversarial negative loss multiplier for rare classes')
    parser.add_argument('--rare-competition-weight', type=float, default=0.5, help='Single-rare class competition loss multiplier')
    parser.add_argument('--rare-head-only', action='store_true', help='Freeze backbone and optimize only rare head outputs')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Rare focal loss gamma')
    parser.add_argument('--rare-positive-weights', default='4,8,8', help='TOM,CRASH,RIDE focal positive weights')
    parser.add_argument('--accompaniment', help='Optional non-gate no-drums WAV for online domain mixing')
    parser.add_argument('--accompaniment-gain-min', type=float, default=0.10)
    parser.add_argument('--accompaniment-gain-max', type=float, default=0.30)
    parser.add_argument('--freeze-bn', action='store_true', default=True)
    parser.add_argument('--log-every', type=int, default=1)
    parser.add_argument('--candidate-name', default='six_class_tower_b_candidate.pth')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()

    if args.self_check:
        run_self_check()
        return
    if not args.meta:
        parser.error('--meta is required unless --self-check is used')
    rare_positive_weights = [float(value.strip()) for value in args.rare_positive_weights.split(',')]
    if len(rare_positive_weights) != 3 or min(rare_positive_weights) <= 0.0:
        parser.error('--rare-positive-weights must contain three positive values')
    if args.focal_gamma < 0.0:
        parser.error('--focal-gamma must be non-negative')
    if not 0.0 <= args.accompaniment_gain_min <= args.accompaniment_gain_max:
        parser.error('accompaniment gain range is invalid')

    torch.manual_seed(1337)
    np.random.seed(1337)
    with open(args.meta, encoding='utf-8') as handle:
        metadata = json.load(handle)
        
    schedule = build_schedule(metadata, args.per_class)
    accompaniment_pool = [load_accompaniment(args.accompaniment)] if args.accompaniment else None
    if len(schedule) % args.batch_size:
        raise ValueError('Schedule length must divide evenly by batch size.')
        
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'train_schedule.json'), 'w', encoding='utf-8') as handle:
        json.dump(schedule, handle, indent=2)

    # 設置加權的 pos_weight
    class_weights = {}
    for label in LABELS:
        if label in ('TOM', 'CRASH', 'RIDE'):
            class_weights[label] = args.rare_positive_weight
        else:
            class_weights[label] = args.positive_weight

    print(f"[Specialization Setup] Pos weights assigned: {class_weights}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # 初始化並加載 V14 權重
    model = SymmetricDrumTCN(num_classes=len(LABELS)).to(device)
    transferred = load_compatible_weights(model, args.checkpoint, device)
    
    # 進行 Full Model Backbone 解凍微調
    head_params = list(model.onset_head.parameters()) + list(model.velocity_head.parameters())
    head_ids = {id(parameter) for parameter in head_params}
    backbone_params = [parameter for parameter in model.parameters() if id(parameter) not in head_ids]
    if args.rare_head_only:
        for parameter in backbone_params:
            parameter.requires_grad_(False)
        optimizer = torch.optim.Adam(head_params, lr=args.lr)
    else:
        optimizer = torch.optim.Adam([
            {'params': head_params, 'lr': args.lr},
            {'params': backbone_params, 'lr': args.backbone_lr},
        ])

    losses = []
    positive_weight = torch.tensor([class_weights[label] for label in LABELS], dtype=torch.float32, device=device).view(1, 1, -1)
    rare_focal_weights = torch.tensor(rare_positive_weights, dtype=torch.float32, device=device).view(1, 1, -1)
    batches_per_epoch = len(schedule) // args.batch_size
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_bn:
            freeze_batchnorm_stats(model)
            
        for start in range(0, len(schedule), args.batch_size):
            feature, onset, velocity = batch_from_schedule(
                schedule, metadata, start, args.batch_size,
                accompaniment_pool=accompaniment_pool,
                gain_range=(args.accompaniment_gain_min, args.accompaniment_gain_max),
            )
            x = torch.from_numpy(feature).float().to(device)
            onset_target = torch.from_numpy(onset).float().to(device)
            velocity_target = torch.from_numpy(velocity).float().to(device)
            
            onset_logits, velocity_logits = model(x)
            
            onset_for_loss = gaussian_smooth_targets(onset_target) if args.gaussian_targets else onset_target
            velocity_for_loss = propagate_velocity_targets(velocity_target) if args.gaussian_targets else velocity_target
            if args.rare_head_only:
                onset_loss = rare_focal_loss(
                    onset_logits, onset_for_loss, rare_focal_weights, args.focal_gamma, args.adversarial_weight,
                )
                rare_active = onset_target[:, :, 3:6].sum().clamp_min(1.0)
                velocity_loss = (
                    (torch.sigmoid(velocity_logits[:, :, 3:6]) - velocity_for_loss[:, :, 3:6]).pow(2)
                    * onset_for_loss[:, :, 3:6]
                ).sum() / rare_active
            else:
                bce = F.binary_cross_entropy(torch.sigmoid(onset_logits), onset_for_loss, reduction='none')
                onset_weight = torch.where(onset_for_loss > 0.0, positive_weight.expand_as(onset_for_loss), torch.ones_like(onset_for_loss))

                # 罕見類別在 core 擊打 frame 為負樣本時提高權重，抑制跨類誤報。
                has_main_hit = (onset_target[:, :, 0] > 0.0) | (onset_target[:, :, 1] > 0.0) | (onset_target[:, :, 2] > 0.0)
                for c_rare in (3, 4, 5):
                    is_neg_rare = has_main_hit & (onset_target[:, :, c_rare] == 0.0)
                    onset_weight[:, :, c_rare] = torch.where(is_neg_rare, torch.tensor(args.adversarial_weight, device=device), onset_weight[:, :, c_rare])

                onset_loss = (bce * onset_weight).mean()
                active = onset_target.sum().clamp_min(1.0)
                velocity_loss = ((torch.sigmoid(velocity_logits) - velocity_for_loss).pow(2) * onset_for_loss).sum() / active
            
            competition_loss = rare_competition_loss(onset_logits, onset_target)
            loss = onset_loss + velocity_loss + args.rare_competition_weight * competition_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            # 注意：為 Model B 微調，我們不進行 3-class 物理梯度鎖定，使其全面優化 rare channels
            optimizer.step()
            losses.append(float(loss.item()))
            
            batch_number = start // args.batch_size + 1
            if batch_number % args.log_every == 0 or batch_number == batches_per_epoch:
                print(f'epoch={epoch}/{args.epochs} batch={batch_number}/{batches_per_epoch} loss={losses[-1]:.4f}', flush=True)
                
        # 儲存每個 Epoch 的 checkpoints 方便後續評估篩選
        epoch_cand_name = f"six_class_tower_b_epoch{epoch}.pth"
        torch.save(model.state_dict(), os.path.join(args.output_dir, epoch_cand_name))
        
    candidate_path = os.path.join(args.output_dir, args.candidate_name)
    torch.save(model.state_dict(), candidate_path)
    
    report = {
        'status': 'pass', 'labels': LABELS, 'schedule_windows': len(schedule),
        'per_class': args.per_class, 'batch_size': args.batch_size, 'epochs': args.epochs, 'batches': len(losses),
        'head_learning_rate': args.lr, 'backbone_learning_rate': args.backbone_lr,
        'gaussian_targets': args.gaussian_targets, 'class_positive_weights': class_weights,
        'checkpoint': os.path.abspath(args.checkpoint), 'adversarial_weight': args.adversarial_weight,
        'rare_competition_weight': args.rare_competition_weight,
        'rare_head_only': args.rare_head_only, 'focal_gamma': args.focal_gamma,
        'rare_focal_positive_weights': rare_positive_weights,
        'accompaniment': os.path.abspath(args.accompaniment) if args.accompaniment else None,
        'accompaniment_gain_range': [args.accompaniment_gain_min, args.accompaniment_gain_max],
        'first_loss': losses[0], 'last_loss': losses[-1],
        'transferred_compatible_tensors': transferred, 'candidate': os.path.abspath(candidate_path),
    }
    with open(os.path.join(args.output_dir, 'train_report.json'), 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
