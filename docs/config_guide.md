# DifusionDenoiser — Config Guide & Experiment Design

## Hệ Thống Config

Codebase sử dụng **mmcv Config** — hệ thống config phân tầng kiểu Python files, cho phép kế thừa và ghi đè (override).

### Cấu Trúc 4 Tầng

```
configs/denoiser/<experiment>.py          ← EXPERIMENT (ghi đè cuối cùng)
    └── inherits from:
        configs/_base_/models/<model>.py      ← MODEL architecture
        configs/_base_/datasets/<data>.py     ← DATASET layout
        configs/_base_/schedules/<sched>.py   ← TRAINING schedule
        configs/_base_/default_runtime.py     ← RUNTIME defaults
```

### Ví Dụ: Đọc Config Đầy Đủ

```python
# configs/denoiser/d3pm_crossattn_uniform_segformer_512x512_100k.py

_base_ = [                                    # Kế thừa 4 base configs
    '../_base_/models/d3pm_crossattn_uniform_segformer.py',
    '../_base_/datasets/pseudo_label_diffusion.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_100k.py',
]

data = dict(samples_per_gpu=4, workers_per_gpu=4)  # Override batch size
optimizer = dict(lr=5e-5)                           # Override LR (pretrained)
```

Kết quả merged config sẽ chứa tất cả fields từ 4 base files, với `data.samples_per_gpu` và `optimizer.lr` được ghi đè.

---

## Base Configs Chi Tiết

### 1. Model Configs (`_base_/models/`)

#### Tham Số Chung (tất cả variants)

| Tham số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| `num_classes` | 7 | 7 classes (hoặc 8 nếu dùng OEM đầy đủ) |
| `num_timesteps` | 100 | Số bước diffusion |
| `base_channels` | 128 | Kênh cơ bản UNet |
| `channel_mult` | (1, 2, 4, 8) | Multiplier → [128, 256, 512, 1024] channels |
| `num_res_blocks` | 2 | ResBlocks per level |
| `attn_resolutions` | (2, 4) | Self/Cross attention ở ds-factor 2× và 4× |
| `dropout` | 0.1 | Dropout rate |
| `cond_channels` | 3 | RGB input |
| `cond_base_channels` | 64 | Lightweight encoder base channels |
| `beta_schedule` | 'cosine' | Cosine noise schedule |
| `loss_type` | 'hybrid' | KL + 0.01×CE |
| `hybrid_lambda` | 0.01 | CE weight |

#### Ma Trận Biến Thể

| Model Config | `cond_type` | `transition_type` | `cond_encoder_type` |
|-------------|-------------|-------------------|-------------------|
| `d3pm_concat_uniform` | concat | uniform | — |
| `d3pm_concat_absorbing` | concat | absorbing | — |
| `d3pm_crossattn_uniform` | crossattn | uniform | lightweight |
| `d3pm_crossattn_absorbing` | crossattn | absorbing | lightweight |
| `d3pm_crossattn_uniform_segformer` | crossattn | uniform | **pretrained** (SegFormer-B2) |
| `d3pm_crossattn_absorbing_resnet50` | crossattn | absorbing | **pretrained** (ResNet50) |
| `d3pm_hybrid_uniform` | hybrid | uniform | lightweight |
| `d3pm_hybrid_absorbing` | hybrid | absorbing | lightweight |
| `d3pm_hybrid_uniform_resnet101` | hybrid | uniform | **pretrained** (ResNet101) |

#### Pretrained Encoder Configs

**SegFormer-B2:**
```python
pretrained_cond_cfg=dict(
    backbone_type='segformer_b2',
    pretrained='pretrain/mit_b2.pth',    # Local checkpoint
    freeze_stages=2,                      # Freeze stages 0-1
    out_channels=[64, 128, 256, 512],    # 1×1 projection targets
)
```

**ResNet50:**
```python
pretrained_cond_cfg=dict(
    backbone_type='resnet50',
    pretrained='open-mmlab://resnet50_v1c',  # MMSeg model zoo
    freeze_stages=1,                          # Freeze stem + layer1
    out_channels=[64, 128, 256, 512],
)
```

### 2. Dataset Config (`_base_/datasets/pseudo_label_diffusion.py`)

```python
dataset_type = 'PseudoLabelDiffusionDataset'
data_root = 'data/my_dataset'       # ← ĐỔI THÀNH PATH THỰC TẾ
num_classes = 7

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],  # ImageNet BGR mean
    std=[58.395, 57.12, 57.375])     # ImageNet BGR std

crop_size = (512, 512)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(..., img_dir='images/train', ann_dir='clean_labels/train',
               pseudo_label_dir='pseudo_labels/train', is_train=True),
    val=dict(..., img_dir='images/val', is_train=False),
    test=dict(..., img_dir='images/val', is_train=False),
)
```

### 3. Schedule Config (`_base_/schedules/schedule_100k.py`)

```python
optimizer = dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01)
lr_scheduler = dict(type='cosine', warmup_iters=5000, warmup_ratio=1e-6, min_lr=1e-6)

max_iters = 100000
checkpoint_interval = 10000   # Save every 10k iters
eval_interval = 10000         # Evaluate every 10k iters
log_interval = 100            # Log losses every 100 iters

use_ema = True
ema_decay = 0.9999
```

### 4. Runtime Config (`_base_/default_runtime.py`)

```python
log_interval = 50
checkpoint_interval = 5000
seed = 42
cudnn_benchmark = True
log_dir = 'work_dirs'
```

> **Lưu ý:** `schedule_100k.py` ghi đè `log_interval` và `checkpoint_interval` từ `default_runtime.py`.

---

## Cách Tạo Config Mới

### Ví dụ 1: Thay đổi số class (cho OEM Building)

```python
# configs/denoiser/d3pm_crossattn_uniform_building_512x512_100k.py
_base_ = [
    '../_base_/models/d3pm_crossattn_uniform.py',
    '../_base_/datasets/pseudo_label_diffusion.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_100k.py',
]

# Override model
model = dict(num_classes=2)  # Binary: background + building

# Override dataset
data_root = '/path/to/OEM_v2_Building'
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(data_root=data_root, num_classes=2),
    val=dict(data_root=data_root, num_classes=2),
    test=dict(data_root=data_root, num_classes=2),
)
```

### Ví dụ 2: Training ngắn hơn (50k iters)

```python
_base_ = [...]
max_iters = 50000
eval_interval = 5000
checkpoint_interval = 5000
lr_scheduler = dict(warmup_iters=2500)
```

### Ví dụ 3: Sử dụng data root khác

```bash
# Thay vì sửa config, dùng --cfg-options (nếu train.py hỗ trợ)
# Hoặc đơn giản tạo symlink:
ln -s /path/to/your/data data/my_dataset
```

---

## Experiment Naming Convention

Format: `d3pm_{cond}_{noise}[_{encoder}]_{res}_{iters}.py`

| Part | Options |
|------|---------|
| `cond` | `concat`, `crossattn`, `hybrid` |
| `noise` | `uniform`, `absorbing` |
| `encoder` | (none) = lightweight, `segformer`, `resnet50`, `resnet101` |
| `res` | `512x512` |
| `iters` | `100k` |

---

## Ablation Variables

Khi thiết kế ablation study, 3 trục chính:

### 1. Conditioning Strategy
| Strategy | Ưu điểm | Nhược điểm |
|----------|---------|-----------|
| **Concat** | Đơn giản, nhanh, không cần encoder riêng | Thông tin condition bị pha loãng qua nhiều layers |
| **CrossAttn** | Injection mạnh tại bottleneck & decoder | Tốn memory (attention HW × HW), cần encoder riêng |
| **Hybrid** | Mạnh nhất: condition ở cả input lẫn attention | Tốn nhất, có thể redundant |

### 2. Noise Type
| Type | Ưu điểm | Nhược điểm |
|------|---------|-----------|
| **Uniform** | Natural entropy progression | Có thể tạo confusion patterns không realistic |
| **Absorbing** | Mask state → clear denoising signal | Cần class K-1 làm mask, giảm 1 class thực |

### 3. Condition Encoder
| Encoder | Params | Benefit |
|---------|--------|---------|
| **Lightweight** | ~5M | Nhanh, ít overfitting |
| **SegFormer-B2** | ~25M | Pretrained visual features, nhưng domain gap? |
| **ResNet50/101** | ~25M/44M | Cổ điển, robust, Pretrained ImageNet |
