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

from run_six_class_smoke import CHUNK_FRAMES, LABELS, SR, TARGET_SAMPLES, build_window, load_compatible_weights
from train_star_smoke import freeze_batchnorm_stats
from train_phase2 import SymmetricDrumTCN, propagate_velocity_targets
from train_six_class_candidate import build_schedule, batch_from_schedule, gaussian_smooth_targets


def main():
    parser = argparse.ArgumentParser(description='Train Model B specialized for TOM/CRASH/RIDE.')
    parser.add_argument('--meta', required=True, help='Path to metadata json')
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
    parser.add_argument('--freeze-bn', action='store_true', default=True)
    parser.add_argument('--log-every', type=int, default=1)
    parser.add_argument('--candidate-name', default='six_class_tower_b_candidate.pth')
    args = parser.parse_args()

    torch.manual_seed(1337)
    with open(args.meta, encoding='utf-8') as handle:
        metadata = json.load(handle)
        
    schedule = build_schedule(metadata, args.per_class)
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
    optimizer = torch.optim.Adam([
        {'params': head_params, 'lr': args.lr},
        {'params': backbone_params, 'lr': args.backbone_lr},
    ])

    losses = []
    positive_weight = torch.tensor([class_weights[label] for label in LABELS], dtype=torch.float32, device=device).view(1, 1, -1)
    batches_per_epoch = len(schedule) // args.batch_size
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_bn:
            freeze_batchnorm_stats(model)
            
        for start in range(0, len(schedule), args.batch_size):
            feature, onset, velocity = batch_from_schedule(schedule, metadata, start, args.batch_size)
            x = torch.from_numpy(feature).float().to(device)
            onset_target = torch.from_numpy(onset).float().to(device)
            velocity_target = torch.from_numpy(velocity).float().to(device)
            
            onset_logits, velocity_logits = model(x)
            
            onset_for_loss = gaussian_smooth_targets(onset_target) if args.gaussian_targets else onset_target
            bce = F.binary_cross_entropy(torch.sigmoid(onset_logits), onset_for_loss, reduction='none')
            onset_weight = torch.where(onset_for_loss > 0.0, positive_weight.expand_as(onset_for_loss), torch.ones_like(onset_for_loss))
            
            # --- Adversarial Negative Sampling (V22) ---
            # Gate rare class (TOM/CRASH/RIDE) false positives at backbeat (KD/SD/HH) hit frames
            has_main_hit = (onset_target[:, :, 0] > 0.0) | (onset_target[:, :, 1] > 0.0) | (onset_target[:, :, 2] > 0.0)
            for c_rare in (3, 4, 5):
                is_neg_rare = has_main_hit & (onset_target[:, :, c_rare] == 0.0)
                onset_weight[:, :, c_rare] = torch.where(is_neg_rare, torch.tensor(args.adversarial_weight, device=device), onset_weight[:, :, c_rare])
                
            onset_loss = (bce * onset_weight).mean()
            
            active = onset_target.sum().clamp_min(1.0)
            velocity_for_loss = propagate_velocity_targets(velocity_target) if args.gaussian_targets else velocity_target
            velocity_loss = ((torch.sigmoid(velocity_logits) - velocity_for_loss).pow(2) * onset_for_loss).sum() / active
            
            loss = onset_loss + velocity_loss
            
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
        'first_loss': losses[0], 'last_loss': losses[-1],
        'transferred_compatible_tensors': transferred, 'candidate': os.path.abspath(candidate_path),
    }
    with open(os.path.join(args.output_dir, 'train_report.json'), 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
