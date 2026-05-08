"""Inference script: denoise pseudo-labels via D3PM reverse diffusion.

Usage:
    python tools/inference.py \
        configs/denoiser/d3pm_concat_uniform_512x512_100k.py \
        work_dirs/d3pm_concat_uniform/latest.pth \
        --img-dir data/test/images \
        --pseudo-dir data/test/pseudo_labels \
        --out-dir data/test/refined_labels \
        --num-classes 7 \
        --num-steps 50
"""

import argparse
import os
import os.path as osp
import sys
from contextlib import redirect_stdout

import mmcv
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))

from mmcv.utils import Config
from diffusion_denoiser.models.diffusion_denoiser import DiffusionDenoiserModel


def parse_args():
    parser = argparse.ArgumentParser(description='D3PM inference')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--img-dir', required=True)
    parser.add_argument('--pseudo-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--num-classes', type=int, default=7)
    parser.add_argument('--num-steps', type=int, default=None,
                        help='Override denoising steps (default: use T from config)')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--img-suffix', default='.tif')
    parser.add_argument('--pseudo-suffix', default='.png')
    parser.add_argument('--split-file', default=None,
                        help='Optional file listing images/stems to process')
    parser.add_argument('--tile-size', type=int, default=None,
                        help='Run tiled inference with this square tile size')
    parser.add_argument('--tile-stride', type=int, default=None,
                        help='Stride for tiled inference (default: tile-size)')
    parser.add_argument('--gt-dir', default=None,
                        help='Optional ground-truth label dir for metrics')
    parser.add_argument('--gt-suffix', default=None,
                        help='GT suffix (default: use --pseudo-suffix)')
    parser.add_argument('--metrics-out', default=None,
                        help='Optional path to save the final metrics text')
    parser.add_argument('--class-names', nargs='*', default=None,
                        help='Optional class names, e.g. background building')
    parser.add_argument('--ignore-index', type=int, default=255)
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def _load_image_list(img_dir, img_suffix, split_file=None):
    if split_file is None:
        return sorted([
            f for f in os.listdir(img_dir)
            if f.endswith(img_suffix)])

    with open(split_file, 'r', encoding='utf-8') as f:
        names = [line.strip() for line in f if line.strip()]

    img_files = [
        name if name.endswith(img_suffix) else f'{name}{img_suffix}'
        for name in names
    ]
    return sorted(img_files)


def _tile_starts(size, tile_size, stride):
    if size <= tile_size:
        return [0]

    starts = list(range(0, size - tile_size + 1, stride))
    last = size - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _pad_tile(img, pseudo, tile_size):
    h, w = pseudo.shape
    pad_h = max(tile_size - h, 0)
    pad_w = max(tile_size - w, 0)
    if pad_h == 0 and pad_w == 0:
        return img, pseudo

    img = np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
    pseudo = np.pad(pseudo, ((0, pad_h), (0, pad_w)), mode='edge')
    return img, pseudo


def _denoise_array(model, img, pseudo, device, num_steps, temperature):
    img_t = torch.from_numpy(img).unsqueeze(0).float().to(device)
    pseudo_t = torch.from_numpy(
        pseudo.astype(np.int64)).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model.denoise(
            img_t, pseudo_t,
            num_steps=num_steps,
            temperature=temperature)
    return pred.squeeze(0).cpu().numpy().astype(np.uint8)


def _denoise_tiled(model, img, pseudo, device, num_steps, temperature,
                   num_classes, tile_size, tile_stride):
    h, w = pseudo.shape
    stride = tile_stride or tile_size
    y_starts = _tile_starts(h, tile_size, stride)
    x_starts = _tile_starts(w, tile_size, stride)
    votes = np.zeros((num_classes, h, w), dtype=np.uint16)

    for top in y_starts:
        for left in x_starts:
            bottom = min(top + tile_size, h)
            right = min(left + tile_size, w)
            img_tile = img[:, top:bottom, left:right]
            pseudo_tile = pseudo[top:bottom, left:right]
            img_tile, pseudo_tile = _pad_tile(img_tile, pseudo_tile, tile_size)

            pred_tile = _denoise_array(
                model, img_tile, pseudo_tile, device, num_steps, temperature)
            pred_tile = pred_tile[:bottom - top, :right - left]

            for c in range(num_classes):
                votes[c, top:bottom, left:right] += (pred_tile == c)

    return votes.argmax(axis=0).astype(np.uint8)


def _load_label(path):
    label = np.array(Image.open(path))
    if label.ndim == 3:
        label = label[:, :, 0]
    return label


def _compute_metrics(file_names, pred_dir, pseudo_dir, gt_dir, num_classes,
                     pseudo_suffix, gt_suffix, ignore_index):
    intersection_pred_total = np.zeros(num_classes, dtype=np.float64)
    union_pred_total = np.zeros(num_classes, dtype=np.float64)
    intersection_pseudo_total = np.zeros(num_classes, dtype=np.float64)
    union_pseudo_total = np.zeros(num_classes, dtype=np.float64)
    per_image_pred = []
    per_image_pseudo = []

    for file_name in file_names:
        pred_path = osp.join(pred_dir, file_name)
        pseudo_path = osp.join(pseudo_dir, file_name)
        gt_name = file_name
        if gt_suffix != pseudo_suffix and file_name.endswith(pseudo_suffix):
            gt_name = file_name[:-len(pseudo_suffix)] + gt_suffix
        gt_path = osp.join(gt_dir, gt_name)

        missing = [
            path for path in (pred_path, pseudo_path, gt_path)
            if not osp.exists(path)
        ]
        if missing:
            raise FileNotFoundError(
                f'Missing files for metrics: {", ".join(missing)}')

        pred = _load_label(pred_path)
        pseudo = _load_label(pseudo_path)
        gt = _load_label(gt_path)
        if pred.shape != gt.shape:
            raise ValueError(
                f'Prediction/GT shape mismatch for {file_name}: '
                f'{pred.shape} vs {gt.shape}')
        if pseudo.shape != gt.shape:
            raise ValueError(
                f'Pseudo/GT shape mismatch for {file_name}: '
                f'{pseudo.shape} vs {gt.shape}')

        valid = gt != ignore_index
        pred_iou = np.zeros(num_classes, dtype=np.float64)
        pseudo_iou = np.zeros(num_classes, dtype=np.float64)

        for c in range(num_classes):
            pred_c = (pred == c) & valid
            pseudo_c = (pseudo == c) & valid
            gt_c = (gt == c) & valid

            inter_pred = (pred_c & gt_c).sum()
            pred_union = (pred_c | gt_c).sum()
            inter_pseudo = (pseudo_c & gt_c).sum()
            pseudo_union = (pseudo_c | gt_c).sum()

            intersection_pred_total[c] += inter_pred
            union_pred_total[c] += pred_union
            intersection_pseudo_total[c] += inter_pseudo
            union_pseudo_total[c] += pseudo_union
            pred_iou[c] = inter_pred / (pred_union + 1e-10)
            pseudo_iou[c] = inter_pseudo / (pseudo_union + 1e-10)

        per_image_pred.append(pred_iou)
        per_image_pseudo.append(pseudo_iou)

    global_pred = intersection_pred_total / (union_pred_total + 1e-10)
    global_pseudo = (
        intersection_pseudo_total / (union_pseudo_total + 1e-10))
    mean_image_pred = np.mean(per_image_pred, axis=0)
    mean_image_pseudo = np.mean(per_image_pseudo, axis=0)
    return dict(
        global_pseudo=global_pseudo,
        global_pred=global_pred,
        mean_image_pseudo=mean_image_pseudo,
        mean_image_pred=mean_image_pred)


def _print_metrics(metrics, num_images, class_names):
    print('\n' + '=' * 72)
    print('Final Metrics')
    print('=' * 72)
    print(f'Evaluated images: {num_images}')

    sections = [
        ('Global pixel-aggregated IoU',
         metrics['global_pseudo'], metrics['global_pred']),
        ('Mean of per-image IoU',
         metrics['mean_image_pseudo'], metrics['mean_image_pred']),
    ]
    for title, pseudo, pred in sections:
        print('\n' + title)
        print('-' * 72)
        print(f'{"Class":<18} {"Pseudo":>12} {"Denoised":>12} {"Delta":>12}')
        print('-' * 72)
        for idx, name in enumerate(class_names):
            delta = pred[idx] - pseudo[idx]
            print(f'{name:<18} {pseudo[idx] * 100:>11.4f}% '
                  f'{pred[idx] * 100:>11.4f}% '
                  f'{delta * 100:>+11.4f}%')

        pseudo_miou = pseudo.mean()
        pred_miou = pred.mean()
        delta_miou = pred_miou - pseudo_miou
        print('-' * 72)
        print(f'{"mIoU":<18} {pseudo_miou * 100:>11.4f}% '
              f'{pred_miou * 100:>11.4f}% '
              f'{delta_miou * 100:>+11.4f}%')
    print('=' * 72)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    num_classes = cfg.model.get('num_classes', args.num_classes)
    class_names = args.class_names
    if class_names is None:
        class_names = [f'class_{idx}' for idx in range(num_classes)]
    if len(class_names) != num_classes:
        raise ValueError(
            f'Expected {num_classes} class names, got {len(class_names)}.')
    gt_suffix = args.gt_suffix or args.pseudo_suffix

    # Build model
    model_cfg = cfg.model.copy()
    model_cfg.pop('type', None)
    model = DiffusionDenoiserModel(**model_cfg)

    # Load checkpoint (handle EMA)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    if 'ema' in ckpt:
        # Use EMA weights for inference
        state_dict = model.state_dict()
        for k, v in ckpt['ema'].items():
            if k in state_dict:
                state_dict[k] = v
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(ckpt['model'])

    model = model.to(args.device)
    model.eval()

    # Normalization
    img_norm_cfg = cfg.data.get('img_norm_cfg', cfg.data.train.get(
        'img_norm_cfg', dict(mean=[123.675, 116.28, 103.53],
                             std=[58.395, 57.12, 57.375])))
    mean = np.array(img_norm_cfg['mean'], dtype=np.float32)
    std = np.array(img_norm_cfg['std'], dtype=np.float32)

    os.makedirs(args.out_dir, exist_ok=True)

    img_files = _load_image_list(args.img_dir, args.img_suffix,
                                 args.split_file)

    print(f'Found {len(img_files)} images. Denoising with '
          f'{args.num_steps or cfg.model.num_timesteps} steps...')

    processed_files = []
    for img_file in tqdm(img_files, desc='Denoising'):
        # Load satellite image
        img_path = osp.join(args.img_dir, img_file)
        if not osp.exists(img_path):
            print(f'Warning: {img_file} not found, skipping.')
            continue
        img = mmcv.imread(img_path).astype(np.float32)
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)  # (3, H, W)

        # Load pseudo-label
        pseudo_file = img_file.replace(args.img_suffix, args.pseudo_suffix)
        pseudo_path = osp.join(args.pseudo_dir, pseudo_file)
        if not osp.exists(pseudo_path):
            print(f'Warning: {pseudo_file} not found, skipping.')
            continue
        pseudo = mmcv.imread(pseudo_path, flag='unchanged')
        if pseudo.ndim == 3:
            pseudo = pseudo[:, :, 0]

        # Denoise
        if args.tile_size is None:
            pred = _denoise_array(
                model, img, pseudo, args.device,
                args.num_steps, args.temperature)
        else:
            pred = _denoise_tiled(
                model, img, pseudo, args.device,
                args.num_steps, args.temperature, num_classes,
                args.tile_size, args.tile_stride)

        # Save
        out_file = img_file.replace(args.img_suffix, args.pseudo_suffix)
        Image.fromarray(pred).save(osp.join(args.out_dir, out_file))
        processed_files.append(out_file)

    print(f'Done. Refined labels saved to {args.out_dir}')

    if args.gt_dir is not None:
        metrics = _compute_metrics(
            processed_files, args.out_dir, args.pseudo_dir, args.gt_dir,
            num_classes, args.pseudo_suffix, gt_suffix, args.ignore_index)
        _print_metrics(metrics, len(processed_files), class_names)

        if args.metrics_out is not None:
            os.makedirs(osp.dirname(args.metrics_out) or '.', exist_ok=True)
            with open(args.metrics_out, 'w', encoding='utf-8') as f:
                with redirect_stdout(f):
                    _print_metrics(metrics, len(processed_files), class_names)
            print(f'Metrics saved to {args.metrics_out}')


if __name__ == '__main__':
    main()
