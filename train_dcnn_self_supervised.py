# -*- coding: utf-8 -*-
"""D22：以既有訓練切分對 SharedCNNBackbone 進行遮罩自監督預訓練。"""

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

from run_six_class_smoke import TARGET_SAMPLES, build_window
from train_phase2 import SharedCNNBackbone


SOURCE_ORDER = ('star', 'egmd', 'idmt')


class ReconstructionHead(nn.Module):
    """中文註解：預訓練期間把 64 維 backbone 特徵暫時還原成雙通道頻譜；此 head 不會儲存至候選權重。"""

    def __init__(self):
        """中文註解：建立最小 1x1 重建投影，輸出維度固定對應兩個 256-bin 特徵通道。"""
        super().__init__()
        self.projection = nn.Conv1d(64, 2 * 256, kernel_size=1)

    def forward(self, latent):
        """中文註解：把 [batch, 64, time] latent 還原為 [batch, 2, 256, time]。"""
        batch_size, _, frames = latent.shape
        return self.projection(latent).view(batch_size, 2, 256, frames)


def file_sha256(path):
    """中文註解：計算 metadata 檔案雜湊，讓候選權重可追溯到固定資料清單。"""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_for_item(key, item, manifest_kind):
    """中文註解：把既有 metadata 的缺省來源欄位轉為 D22 明確允許的三個資料來源。"""
    if manifest_kind == 'idmt':
        return 'idmt'
    source = item.get('source')
    if source == 'egmd_pitch_weighted':
        return 'egmd'
    if source in (None, '') and key.startswith('star_'):
        return 'star'
    raise AssertionError(f'Unsupported D22 source for key={key!r}: {source!r}')


def choose_anchor(item, audio_path):
    """中文註解：只用已註冊 train 音檔內事件的中位時間定位窗口；事件標籤不參與自監督 loss。"""
    event_times = sorted(float(event['time']) for event in item.get('events', []))
    if not event_times:
        return None
    info = sf.info(audio_path)
    half_window = TARGET_SAMPLES / float(info.samplerate) / 2.0
    valid = [time for time in event_times if half_window <= time <= (info.frames / float(info.samplerate)) - half_window]
    if not valid:
        return None
    return valid[len(valid) // 2]


def assert_no_train_heldout_overlap(rows):
    """中文註解：在讀取音訊特徵前，強制拒絕 train 音檔與任何 held-out split 的路徑重疊。"""
    train_paths = {row['audio_path'] for row in rows if row['split'] == 'train'}
    heldout_paths = {row['audio_path'] for row in rows if row['split'] != 'train'}
    overlap = train_paths & heldout_paths
    if overlap:
        raise AssertionError(f'Train/held-out audio overlap detected: {sorted(overlap)[:3]}')


def audit_metadata(meta_path, idmt_meta_path):
    """中文註解：載入兩份既有 metadata，僅回傳可用 train rows，並輸出完整隔離稽核資訊。"""
    manifests = ((Path(meta_path), 'star_egmd'), (Path(idmt_meta_path), 'idmt'))
    rows = []
    manifest_hashes = {}
    for path, manifest_kind in manifests:
        if not path.is_file():
            raise FileNotFoundError(f'Metadata not found: {path}')
        manifest_hashes[str(path.resolve())] = file_sha256(path)
        with path.open(encoding='utf-8') as handle:
            metadata = json.load(handle)
        for key, item in metadata.items():
            audio_path = str(Path(item['audio_path']).resolve())
            split = item.get('split')
            if split not in ('train', 'validation', 'test'):
                raise AssertionError(f'Unexpected split for key={key!r}: {split!r}')
            rows.append({
                'key': key,
                'item': item,
                'audio_path': audio_path,
                'split': split,
                'source': source_for_item(key, item, manifest_kind),
            })

    assert_no_train_heldout_overlap(rows)
    missing = [row['audio_path'] for row in rows if not Path(row['audio_path']).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing audio files: {missing[:3]}')

    train_rows = [row for row in rows if row['split'] == 'train']
    source_counts = Counter(row['source'] for row in train_rows)
    if set(source_counts) != set(SOURCE_ORDER):
        raise AssertionError(f'Expected all D22 sources, got {dict(source_counts)}')
    return train_rows, {
        'train_items_by_source': dict(sorted(source_counts.items())),
        'heldout_items': sum(row['split'] != 'train' for row in rows),
        'train_heldout_audio_overlap': 0,
        'missing_audio': 0,
        'manifest_sha256': manifest_hashes,
    }


def build_train_schedule(train_rows, max_windows):
    """中文註解：以三來源 round-robin 建立固定 train-only 排程，讓小型 IDMT 不被 STAR 數量淹沒。"""
    grouped = defaultdict(list)
    for row in train_rows:
        anchor = choose_anchor(row['item'], row['audio_path'])
        if anchor is not None:
            grouped[row['source']].append({
                'key': row['key'],
                'item': row['item'],
                'audio_path': row['audio_path'],
                'source': row['source'],
                'anchor': anchor,
            })
    for source in SOURCE_ORDER:
        grouped[source].sort(key=lambda row: (row['key'], row['anchor']))
        if not grouped[source]:
            raise AssertionError(f'No valid four-second train windows for source={source}')

    schedule = []
    source_index = {source: 0 for source in SOURCE_ORDER}
    while len(schedule) < max_windows:
        for source in SOURCE_ORDER:
            if len(schedule) >= max_windows:
                break
            candidates = grouped[source]
            schedule.append(candidates[source_index[source] % len(candidates)])
            source_index[source] += 1
    return schedule, dict(sorted((source, len(rows)) for source, rows in grouped.items()))


def mask_time_frames(features, mask_ratio, generator):
    """中文註解：遮罩每筆特徵的完整時間欄，並回傳只在遮罩位置計算重建 loss 的布林 mask。"""
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError('--mask-ratio must be between 0 and 1.')
    batch_size, _, _, frame_count = features.shape
    mask = torch.rand((batch_size, frame_count), generator=generator, device=features.device) < mask_ratio
    for index in range(batch_size):
        if not bool(mask[index].any()):
            mask[index, index % frame_count] = True
    masked = features.masked_fill(mask[:, None, None, :], 0.0)
    return masked, mask


def masked_reconstruction_loss(prediction, target, frame_mask):
    """中文註解：只計算被遮罩時間欄的平均 MSE，避免模型靠未遮罩欄位取得虛假低 loss。"""
    weighted_error = (prediction - target).pow(2) * frame_mask[:, None, None, :]
    denominator = frame_mask.sum().to(target.dtype) * target.shape[1] * target.shape[2]
    return weighted_error.sum() / denominator.clamp_min(1.0)


def batch_features(schedule, start, batch_size):
    """中文註解：沿用既有 build_window 特徵管線讀取固定訓練排程，忽略其 onset/velocity target。"""
    features = []
    for row in schedule[start:start + batch_size]:
        feature, _, _, _ = build_window(row['item'], row['anchor'], use_true_superflux=True)
        features.append(feature)
    return np.stack(features)


def run_self_check():
    """中文註解：以合成特徵測試遮罩、backward 與候選 backbone 權重重新載入，不讀取任何資料集音檔。"""
    torch.manual_seed(1337)
    features = torch.ones(1, 2, 256, 16)
    generator = torch.Generator().manual_seed(1337)
    masked, frame_mask = mask_time_frames(features, 0.25, generator)
    assert masked.shape == features.shape and frame_mask.shape == (1, 16)
    assert bool(frame_mask.any()) and torch.all(masked[:, :, :, frame_mask[0]] == 0.0)
    backbone = SharedCNNBackbone()
    head = ReconstructionHead()
    prediction = head(backbone(features))
    loss = masked_reconstruction_loss(prediction, features, frame_mask)
    loss.backward()
    assert torch.isfinite(loss)
    restored = SharedCNNBackbone()
    restored.load_state_dict(backbone.state_dict(), strict=True)
    assert tuple(prediction.shape) == (1, 2, 256, 16)
    print('Self-check passed.')


def train(args, train_rows, audit_report):
    """中文註解：依固定 D22 配方訓練 backbone，儲存獨立候選與可追溯報告。"""
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'Output directory already contains files: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = torch.Generator(device=device).manual_seed(args.seed)
    schedule, valid_windows_by_source = build_train_schedule(train_rows, args.max_windows)

    backbone = SharedCNNBackbone().to(device)
    head = ReconstructionHead().to(device)
    optimizer = torch.optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr=args.lr)
    epoch_losses = []
    for epoch in range(1, args.epochs + 1):
        backbone.train()
        head.train()
        losses = []
        for start in range(0, len(schedule), args.batch_size):
            features = torch.from_numpy(batch_features(schedule, start, args.batch_size)).float().to(device)
            masked, frame_mask = mask_time_frames(features, args.mask_ratio, generator)
            prediction = head(backbone(masked))
            loss = masked_reconstruction_loss(prediction, features, frame_mask)
            if not torch.isfinite(loss):
                raise AssertionError(f'Non-finite loss at epoch={epoch}, start={start}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        epoch_loss = float(np.mean(losses))
        epoch_losses.append(epoch_loss)
        print(f'Epoch {epoch}/{args.epochs}: masked_mse={epoch_loss:.8f}', flush=True)

    candidate_path = output_dir / 'shared_backbone_pretrain.pth'
    torch.save(backbone.state_dict(), candidate_path)
    reloaded = SharedCNNBackbone().to(device)
    reloaded.load_state_dict(torch.load(candidate_path, map_location=device, weights_only=True), strict=True)
    reloaded.eval()
    with torch.no_grad():
        reload_shape = list(reloaded(torch.zeros(1, 2, 256, 688, device=device)).shape)
    if reload_shape != [1, 64, 688]:
        raise AssertionError(f'Unexpected reloaded output shape: {reload_shape}')

    report = {
        'status': 'pass',
        'phase': 'D22',
        'research_only': True,
        'objective': 'masked_feature_reconstruction',
        'updated_modules': ['SharedCNNBackbone', 'temporary ReconstructionHead'],
        'not_updated_or_loaded': ['TCN', 'Conformer', 'onset_head', 'velocity_head', 'decoder', 'product_checkpoint'],
        'feature_shape': [2, 256, 688],
        'candidate': str(candidate_path.resolve()),
        'candidate_sha256': file_sha256(candidate_path),
        'reload_output_shape': reload_shape,
        'device': str(device),
        'recipe': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'max_windows': args.max_windows,
            'mask_ratio': args.mask_ratio,
            'lr': args.lr,
            'seed': args.seed,
        },
        'schedule_items_by_source': dict(sorted(Counter(row['source'] for row in schedule).items())),
        'valid_windows_by_source': valid_windows_by_source,
        'epoch_masked_mse': epoch_losses,
        'data_audit': audit_report,
    }
    report_path = output_dir / 'pretrain_report.json'
    with report_path.open('w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    """中文註解：提供 D22 資料稽核、合成 self-check 與固定預算 backbone 預訓練 CLI。"""
    parser = argparse.ArgumentParser(description='D22 train-only masked self-supervised DCNN pretraining.')
    parser.add_argument('--meta', default='processed_data/star_egmd_six_class_d4d.json')
    parser.add_argument('--idmt-meta', default='processed_data/local_xml_meta.json')
    parser.add_argument('--output-dir', default='validation_runs/d22_dcnn_ssl')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-windows', type=int, default=2048)
    parser.add_argument('--mask-ratio', type=float, default=0.15)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()

    if args.self_check:
        run_self_check()
        return
    train_rows, audit_report = audit_metadata(args.meta, args.idmt_meta)
    if args.audit_only:
        print(json.dumps(audit_report, ensure_ascii=False, indent=2))
        return
    if args.epochs <= 0 or args.batch_size <= 0 or args.max_windows <= 0:
        parser.error('--epochs, --batch-size and --max-windows must be positive.')
    train(args, train_rows, audit_report)


if __name__ == '__main__':
    main()
