"""Evaluation script for D3PM pseudo-label denoiser.

Computes mIoU, per-class IoU, and also reports the baseline mIoU
of the raw pseudo-labels (before denoising) for comparison.

Usage:
    python tools/test.py \
        configs/denoiser/d3pm_concat_uniform_512x512_100k.py \
        work_dirs/d3pm_concat_uniform/latest.pth \
        --num-steps 50
"""

import argparse
import os
import os.path as osp
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))

from mmcv.utils import Config
from diffusion_denoiser.models.diffusion_denoiser import DiffusionDenoiserModel
from diffusion_denoiser.datasets.pseudo_label_dataset import PseudoLabelDiffusionDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate D3PM denoiser')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--num-steps', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def compute_miou(pred, gt, num_classes, ignore_index=255):
    """Compute per-class IoU."""
    intersection = np.zeros(num_classes)
    union = np.zeros(num_classes)
    for c in range(num_classes):
        valid = gt != ignore_index
        pred_c = (pred == c) & valid
        gt_c = (gt == c) & valid
        intersection[c] = (pred_c & gt_c).sum()
        union[c] = (pred_c | gt_c).sum()
    iou = intersection / (union + 1e-10)
    return iou


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    device = torch.device(args.device)

    # Build model
    model_cfg = cfg.model.copy()
    model_cfg.pop('type', None)
    model = DiffusionDenoiserModel(**model_cfg)

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    if 'ema' in ckpt:
        state_dict = model.state_dict()
        for k, v in ckpt['ema'].items():
            if k in state_dict:
                state_dict[k] = v
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(ckpt['model'])

    model = model.to(device)
    model.eval()

    num_classes = cfg.model.num_classes

    # Build test dataset
    test_cfg = cfg.data.test.copy()
    test_cfg.pop('type', None)
    test_dataset = PseudoLabelDiffusionDataset(**test_cfg)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=2)

    # Evaluate
    all_iou_pred = []
    all_iou_pseudo = []

    for batch in tqdm(test_loader, desc='Evaluating'):
        satellite = batch['satellite_img'].to(device)
        pseudo = batch['pseudo_label'].to(device)
        clean = batch['clean_label'].numpy()[0]

        # Denoise
        with torch.no_grad():
            pred = model.denoise(
                satellite, pseudo,
                num_steps=args.num_steps,
                temperature=args.temperature)
            pred_np = pred.squeeze(0).cpu().numpy()

        pseudo_np = pseudo.squeeze(0).cpu().numpy()

        # Compute IoU for denoised prediction
        iou_pred = compute_miou(pred_np, clean, num_classes)
        all_iou_pred.append(iou_pred)

        # Compute IoU for raw pseudo-label (baseline)
        iou_pseudo = compute_miou(pseudo_np, clean, num_classes)
        all_iou_pseudo.append(iou_pseudo)

    # Aggregate
    mean_iou_pred = np.mean(all_iou_pred, axis=0)
    mean_iou_pseudo = np.mean(all_iou_pseudo, axis=0)

    print('\n' + '=' * 60)
    print('Evaluation Results')
    print('=' * 60)
    print(f'\n{"Class":<10} {"Pseudo (baseline)":<20} {"Denoised (ours)":<20} {"Δ":<10}')
    print('-' * 60)
    for c in range(num_classes):
        delta = mean_iou_pred[c] - mean_iou_pseudo[c]
        sign = '+' if delta >= 0 else ''
        print(f'{c:<10} {mean_iou_pseudo[c]:<20.4f} {mean_iou_pred[c]:<20.4f} '
              f'{sign}{delta:<10.4f}')
    print('-' * 60)
    miou_pseudo = mean_iou_pseudo.mean()
    miou_pred = mean_iou_pred.mean()
    delta = miou_pred - miou_pseudo
    sign = '+' if delta >= 0 else ''
    print(f'{"mIoU":<10} {miou_pseudo:<20.4f} {miou_pred:<20.4f} '
          f'{sign}{delta:<10.4f}')
    print('=' * 60)


if __name__ == '__main__':
    main()
