"""Training script for D3PM discrete diffusion pseudo-label denoiser.

Usage:
    # Single GPU
    python tools/train.py configs/denoiser/d3pm_concat_uniform_512x512_100k.py

    # Multi-GPU
    torchrun --nproc_per_node=4 tools/train.py \
        configs/denoiser/d3pm_hybrid_uniform_512x512_100k.py --launcher pytorch

    # Resume
    python tools/train.py configs/denoiser/d3pm_concat_uniform_512x512_100k.py \
        --resume-from work_dirs/d3pm_concat_uniform/latest.pth
"""

import argparse
import copy
import logging
import os
import os.path as osp
import platform
import shutil
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))

from mmcv.utils import Config
from diffusion_denoiser.models.diffusion_denoiser import DiffusionDenoiserModel
from diffusion_denoiser.datasets.pseudo_label_dataset import PseudoLabelDiffusionDataset


# ---------------------------------------------------------------------------
# mmseg-style logger setup
# ---------------------------------------------------------------------------
def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get the root logger with mmseg-style formatting.

    The logger prints to both console and a log file (if provided),
    mirroring the mmseg/mmcv convention of ``<timestamp> - <name> - <level> - <msg>``.
    """
    logger = logging.getLogger('diffusion_denoiser')
    # Avoid adding duplicate handlers when called multiple times
    if logger.hasHandlers():
        return logger
    logger.setLevel(log_level)

    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    if log_file is not None:
        fh = logging.FileHandler(log_file, mode='w')
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def log_env_info(logger, cfg, args):
    """Log environment, config, and system info (mmseg style)."""
    dash_line = '-' * 60
    logger.info('Environment info:')
    logger.info(dash_line)
    logger.info(f'sys.platform: {sys.platform}')
    logger.info(f'Python: {sys.version}')
    logger.info(f'PyTorch: {torch.__version__}')
    logger.info(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        logger.info(f'CUDA version: {torch.version.cuda}')
        logger.info(f'GPU: {torch.cuda.get_device_name(0)}')
        logger.info(f'GPU count: {torch.cuda.device_count()}')
    logger.info(f'Hostname: {platform.node()}')
    logger.info(dash_line)
    logger.info(f'Config file: {args.config}')
    logger.info(f'Config:\n{cfg.pretty_text}')
    logger.info(dash_line)


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {k: v.clone().detach()
                       for k, v in model.named_parameters() if v.requires_grad}

    def update(self, model: nn.Module):
        with torch.no_grad():
            for k, v in model.named_parameters():
                if v.requires_grad and k in self.shadow:
                    self.shadow[k].mul_(self.decay).add_(
                        v.data, alpha=1 - self.decay)

    def apply(self, model: nn.Module):
        """Replace model params with EMA params."""
        for k, v in model.named_parameters():
            if v.requires_grad and k in self.shadow:
                v.data.copy_(self.shadow[k])

    def restore(self, model: nn.Module, backup: dict):
        """Restore model params from backup."""
        for k, v in model.named_parameters():
            if k in backup:
                v.data.copy_(backup[k])


def parse_args():
    parser = argparse.ArgumentParser(description='Train D3PM denoiser')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('--work-dir', help='Working directory')
    parser.add_argument('--resume-from', help='Checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--launcher', choices=['none', 'pytorch'],
                        default='none')
    parser.add_argument('--local_rank', type=int, default=0)
    return parser.parse_args()


def build_model(cfg) -> DiffusionDenoiserModel:
    """Build model from config dict."""
    model_cfg = cfg.model.copy()
    model_cfg.pop('type', None)
    return DiffusionDenoiserModel(**model_cfg)


def build_dataset(data_cfg, is_train=True):
    """Build dataset from config dict."""
    dataset_cfg = data_cfg.copy()
    dataset_cfg.pop('type', None)
    return PseudoLabelDiffusionDataset(**dataset_cfg)


def evaluate(model, val_loader, device, num_steps=10):
    """Evaluate by denoising validation pseudo-labels and computing mIoU."""
    model.eval()
    intersection = torch.zeros(model.num_classes, device=device)
    union = torch.zeros(model.num_classes, device=device)

    for batch in val_loader:
        satellite = batch['satellite_img'].to(device)
        pseudo = batch['pseudo_label'].to(device)
        clean = batch['clean_label'].to(device)

        # Denoise with reduced steps for speed
        pred = model.denoise(satellite, pseudo, num_steps=num_steps)

        # Compute per-class IoU
        for c in range(model.num_classes):
            pred_c = (pred == c)
            gt_c = (clean == c)
            intersection[c] += (pred_c & gt_c).sum()
            union[c] += (pred_c | gt_c).sum()

    iou = intersection / (union + 1e-10)
    miou = iou.mean().item()
    model.train()
    return miou, iou.cpu().numpy()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # Setup distributed
    distributed = args.launcher != 'none'
    if distributed:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
    device = torch.device(f'cuda:{args.local_rank}')
    rank = args.local_rank if distributed else 0

    # Work dir
    if args.work_dir:
        work_dir = args.work_dir
    else:
        work_dir = osp.join(
            cfg.get('log_dir', 'work_dirs'),
            osp.splitext(osp.basename(args.config))[0])
    if rank == 0:
        os.makedirs(work_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # mmseg-style logging: <timestamp>.log in work_dir
    # -----------------------------------------------------------------------
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(work_dir, f'{timestamp}.log') if rank == 0 else None
    logger = get_root_logger(log_file=log_file)

    if rank == 0:
        log_env_info(logger, cfg, args)
        # Dump a copy of config to work_dir (mmseg convention)
        cfg.dump(osp.join(work_dir, osp.basename(args.config)))
        logger.info(f'Work dir: {work_dir}')
        logger.info(f'Log file: {log_file}')

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if rank == 0:
        logger.info(f'Set random seed to {args.seed}')

    # Build model
    model = build_model(cfg).to(device)
    if distributed:
        model = DDP(model, device_ids=[args.local_rank])
    raw_model = model.module if distributed else model

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f'Model parameters: {n_params:,} total, {n_train:,} trainable')

    # EMA
    use_ema = cfg.get('use_ema', True)
    ema = EMA(raw_model, cfg.get('ema_decay', 0.9999)) if use_ema else None

    # Optimizer
    opt_cfg = cfg.get('optimizer', dict(type='AdamW', lr=1e-4))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg.get('lr', 1e-4),
        betas=opt_cfg.get('betas', (0.9, 0.999)),
        weight_decay=opt_cfg.get('weight_decay', 0.01))

    # LR scheduler
    max_iters = cfg.get('max_iters', 100000)
    warmup_iters = cfg.get('lr_scheduler', {}).get('warmup_iters', 5000)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_iters - warmup_iters, eta_min=1e-6)

    # Dataset
    train_dataset = build_dataset(cfg.data.train, is_train=True)
    val_dataset = build_dataset(cfg.data.val, is_train=False)

    train_sampler = DistributedSampler(train_dataset) if distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.samples_per_gpu,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.data.workers_per_gpu,
        pin_memory=True,
        drop_last=True)

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True)

    if rank == 0:
        logger.info(f'Train dataset: {len(train_dataset)} samples')
        logger.info(f'Val dataset:   {len(val_dataset)} samples')

    # Resume
    start_iter = 0
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device)
        raw_model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_iter = ckpt.get('iter', 0)
        if ema and 'ema' in ckpt:
            ema.shadow = ckpt['ema']
        if rank == 0:
            logger.info(f'Resumed from checkpoint: {args.resume_from} (iter {start_iter})')

    # Training loop
    model.train()
    data_iter = iter(train_loader)
    log_interval = cfg.get('log_interval', 100)
    ckpt_interval = cfg.get('checkpoint_interval', 5000)
    eval_interval = cfg.get('eval_interval', 1000)

    if rank == 0:
        logger.info(f'Starting training for {max_iters} iterations ...')
        logger.info(f'  Model:      {cfg.model.get("type", "DiffusionDenoiserModel")}')
        logger.info(f'  Cond type:  {cfg.model.get("cond_type", "N/A")}')
        logger.info(f'  Noise type: {cfg.model.get("transition_type", "N/A")}')
        logger.info(f'  Log every:  {log_interval} iters')
        logger.info(f'  Ckpt every: {ckpt_interval} iters')
        logger.info(f'  Eval every: {eval_interval} iters')

    best_miou = 0.0
    train_start = time.time()
    iter_time = time.time()

    for iteration in range(start_iter, max_iters):
        # Get batch (with cycling)
        try:
            batch = next(data_iter)
        except StopIteration:
            if train_sampler:
                train_sampler.set_epoch(iteration)
            data_iter = iter(train_loader)
            batch = next(data_iter)

        data_time = time.time() - iter_time

        satellite = batch['satellite_img'].to(device)
        clean_label = batch['clean_label'].to(device)

        # Forward
        losses = model(clean_label, satellite)
        loss = losses['loss_total']

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # LR warmup
        if iteration < warmup_iters:
            lr_scale = min(1.0, (iteration + 1) / warmup_iters)
            for pg in optimizer.param_groups:
                pg['lr'] = cfg.optimizer.lr * lr_scale
        else:
            scheduler.step()

        # EMA update
        if ema:
            ema.update(raw_model)

        batch_time = time.time() - iter_time

        # ---------------------------------------------------------------
        # Logging (mmseg style: Iter [x/y], lr, loss, time, memory)
        # ---------------------------------------------------------------
        if rank == 0 and (iteration + 1) % log_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            loss_str = ', '.join(
                f'{k}: {v.item():.4f}' for k, v in losses.items())

            eta_seconds = batch_time * (max_iters - iteration - 1)
            eta_str = str(int(eta_seconds // 3600)).zfill(2) + ':' + \
                      str(int(eta_seconds % 3600 // 60)).zfill(2) + ':' + \
                      str(int(eta_seconds % 60)).zfill(2)

            mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

            logger.info(
                f'Iter [{iteration + 1}/{max_iters}]  '
                f'lr: {lr:.2e}, '
                f'{loss_str}, '
                f'time: {batch_time:.3f}, data_time: {data_time:.3f}, '
                f'memory: {mem_mb:.0f}, '
                f'eta: {eta_str}')

        # ---------------------------------------------------------------
        # Checkpoint (mmseg style: iter_XXXX.pth + latest.pth symlink)
        # ---------------------------------------------------------------
        if rank == 0 and (iteration + 1) % ckpt_interval == 0:
            ckpt_path = osp.join(work_dir, f'iter_{iteration + 1}.pth')
            meta = dict(
                iter=iteration + 1,
                timestamp=timestamp,
                config=args.config,
                CLASSES=cfg.get('CLASSES', None),
                best_miou=best_miou,
            )
            save_dict = dict(
                model=raw_model.state_dict(),
                optimizer=optimizer.state_dict(),
                iter=iteration + 1,
                meta=meta)
            if ema:
                save_dict['ema'] = ema.shadow
            torch.save(save_dict, ckpt_path)

            # Symlink latest.pth -> iter_XXXX.pth
            latest = osp.join(work_dir, 'latest.pth')
            if osp.islink(latest) or osp.exists(latest):
                os.remove(latest)
            os.symlink(osp.basename(ckpt_path), latest)
            logger.info(f'Saving checkpoint at {iteration + 1} iterations: {ckpt_path}')

        # ---------------------------------------------------------------
        # Evaluation
        # ---------------------------------------------------------------
        if rank == 0 and (iteration + 1) % eval_interval == 0:
            logger.info(f'--- Validation @ iter {iteration + 1} ---')
            eval_start = time.time()

            # Apply EMA for evaluation
            if ema:
                backup = {k: v.data.clone() for k, v in raw_model.named_parameters()}
                ema.apply(raw_model)

            miou, per_class_iou = evaluate(raw_model, val_loader, device)
            eval_elapsed = time.time() - eval_start

            is_best = miou > best_miou
            if is_best:
                best_miou = miou
                # Save best checkpoint
                best_path = osp.join(work_dir, 'best_mIoU.pth')
                save_dict = dict(
                    model=raw_model.state_dict(),
                    iter=iteration + 1,
                    meta=dict(
                        iter=iteration + 1,
                        miou=miou,
                        per_class_iou=per_class_iou.tolist(),
                    ))
                if ema:
                    save_dict['ema'] = ema.shadow
                torch.save(save_dict, best_path)

            logger.info(
                f'mIoU: {miou:.4f} | '
                f'best: {best_miou:.4f} | '
                f'eval_time: {eval_elapsed:.1f}s'
                f'{" [NEW BEST]" if is_best else ""}')
            logger.info(
                f'Per-class IoU: {np.array2string(per_class_iou, precision=4, separator=", ")}')

            if ema:
                ema.restore(raw_model, backup)

            model.train()

        iter_time = time.time()

    # End of training
    total_time = time.time() - train_start
    total_time_str = str(int(total_time // 3600)).zfill(2) + ':' + \
                     str(int(total_time % 3600 // 60)).zfill(2) + ':' + \
                     str(int(total_time % 60)).zfill(2)
    if rank == 0:
        logger.info(f'Training complete. Total time: {total_time_str}')
        logger.info(f'Best mIoU: {best_miou:.4f}')
        logger.info(f'Checkpoints and logs saved to: {work_dir}')


if __name__ == '__main__':
    main()
