# DifusionDenoiser — Codebase Documentation

## Tổng Quan

**DifusionDenoiser** là một hệ thống dựa trên **D3PM (Discrete Denoising Diffusion Probabilistic Model)** nhằm tinh chỉnh (refine) pseudo-label cho bài toán semantic segmentation ảnh vệ tinh. Hệ thống lấy pseudo-label (bản đồ nhãn noisy từ model CISC-R) kết hợp với ảnh vệ tinh RGB gốc làm condition, thực hiện quá trình khuếch tán rời rạc ngược (reverse discrete diffusion) để tạo ra bản đồ nhãn gần ground truth hơn.

### Tại sao dùng Discrete Diffusion?

Khác với diffusion truyền thống (Gaussian noise trên continuous domain), bài toán này hoạt động trên **discrete state space** (integer class IDs). D3PM sử dụng **transition matrices** thay vì Gaussian noise, phù hợp tự nhiên cho label maps. Mỗi pixel chuyển đổi giữa K classes theo xác suất được định nghĩa bởi ma trận chuyển tiếp Q_t.

---

## Cấu Trúc Thư Mục

```
DifusionDenoiser/
├── setup.py                          # Package installation config
├── configs/                          # MMConfig-style configs
│   ├── _base_/
│   │   ├── datasets/
│   │   │   └── pseudo_label_diffusion.py    # Dataset defaults
│   │   ├── models/
│   │   │   ├── d3pm_concat_uniform.py       # Concat + uniform noise
│   │   │   ├── d3pm_concat_absorbing.py     # Concat + absorbing noise
│   │   │   ├── d3pm_crossattn_uniform.py    # CrossAttn + uniform
│   │   │   ├── d3pm_crossattn_absorbing.py  # CrossAttn + absorbing
│   │   │   ├── d3pm_crossattn_absorbing_resnet50.py   # Pretrained ResNet50
│   │   │   ├── d3pm_crossattn_uniform_segformer.py    # Pretrained SegFormer-B2
│   │   │   ├── d3pm_hybrid_uniform.py       # Hybrid + uniform
│   │   │   ├── d3pm_hybrid_absorbing.py     # Hybrid + absorbing
│   │   │   └── d3pm_hybrid_uniform_resnet101.py       # Pretrained ResNet101
│   │   ├── schedules/
│   │   │   └── schedule_100k.py             # Training: 100k iters, cosine LR
│   │   └── default_runtime.py               # Logging, seed, checkpoint
│   └── denoiser/                     # 9 experiment configs (composed from _base_)
│       ├── d3pm_concat_uniform_512x512_100k.py
│       ├── d3pm_concat_absorbing_512x512_100k.py
│       ├── d3pm_crossattn_uniform_512x512_100k.py
│       ├── d3pm_crossattn_absorbing_512x512_100k.py
│       ├── d3pm_crossattn_uniform_segformer_512x512_100k.py
│       ├── d3pm_crossattn_absorbing_resnet50_512x512_100k.py
│       ├── d3pm_hybrid_uniform_512x512_100k.py
│       ├── d3pm_hybrid_absorbing_512x512_100k.py
│       └── d3pm_hybrid_uniform_resnet101_512x512_100k.py
├── diffusion_denoiser/               # Core Python package
│   ├── __init__.py                   # Re-exports all submodules
│   ├── models/
│   │   ├── __init__.py
│   │   ├── conditional_unet.py       # ★ UNet chính (~854 lines)
│   │   └── diffusion_denoiser.py     # ★ Top-level model wrapper
│   ├── diffusion/
│   │   ├── __init__.py
│   │   ├── d3pm.py                   # ★ D3PM training + sampling logic
│   │   └── noise_schedule.py         # ★ Transition matrices + schedule
│   ├── datasets/
│   │   ├── __init__.py
│   │   └── pseudo_label_dataset.py   # Dataset loader (img + pseudo + GT)
│   └── losses/
│       └── __init__.py               # Placeholder (losses in D3PM)
├── tools/
│   ├── train.py                      # Training script (single/multi-GPU)
│   ├── test.py                       # Evaluation script (mIoU)
│   ├── inference.py                  # Inference: denoise pseudo-labels
│   └── run_ablation.sh               # Full ablation: train + eval + aggregate
├── pretrain/
│   └── mit_b2.pth                    # Pretrained SegFormer-B2 weights (~105MB)
├── data/
│   └── OEM_v2_aDanh -> ...           # Symlink to dataset
└── docs/
    ├── research_briefing.md           # Detailed noise characterization
    └── codebase_overview.md           # ← Bạn đang đọc file này
```

---

## Kiến Trúc Hệ Thống

### Sơ Đồ Tổng Quát

```
                                    ┌──────────────────────────┐
                                    │   DiffusionDenoiserModel  │
                                    │   (top-level wrapper)     │
                                    └──────┬───────────────────┘
                                           │
                              ┌────────────┼────────────────┐
                              │            │                │
                    ┌─────────▼─────┐  ┌───▼─────────┐  ┌──▼────────────────┐
                    │  ConditionalUNet│  │    D3PM     │  │DiscreteNoiseSchedule│
                    │  (denoising    │  │ (train +    │  │ (transition       │
                    │   network)     │  │  sample)    │  │  matrices)        │
                    └────────────────┘  └─────────────┘  └───────────────────┘
```

### Pipeline Chi Tiết

```
TRAINING:
  x_0 (clean label)  ──┐
                        ├──▶ q_sample(x_0, t) ──▶ x_t (noisy label)
  t (random timestep)──┘                              │
                                                       ▼
  satellite_img ──▶ CondEncoder ──▶ cond_feats ──▶ ConditionalUNet(x_t, t, cond)
                                                       │
                                                       ▼
                                                  x_0_logits (predicted clean)
                                                       │
                                              ┌────────┼────────┐
                                              ▼                 ▼
                                         CE loss           KL loss
                                              └────────┬────────┘
                                                       ▼
                                             hybrid loss = KL + λ·CE


INFERENCE (reverse diffusion):
  x_T (noisy/random) ──▶ for t = T-1, ..., 0:
                              x_0_pred = UNet(x_t, t, satellite)
                              posterior = q(x_{t-1} | x_t, x_0_pred)
                              x_{t-1} ~ posterior
                         ──▶ x_0 (denoised label)
```

---

## Thành Phần Chính

### 1. `DiffusionDenoiserModel` — Top-level Wrapper

**File:** `diffusion_denoiser/models/diffusion_denoiser.py` (131 lines)

Đây là lớp chính được instantiate từ config. Kết nối 3 thành phần:
- `ConditionalUNet` — mạng denoising
- `DiscreteNoiseSchedule` — bảng chuyển tiếp noise
- `D3PM` — logic training và sampling

**API chính:**
| Phương thức | Input | Output | Mô tả |
|------------|-------|--------|-------|
| `forward(clean_label, satellite_img)` | `(B,H,W)` + `(B,3,H,W)` | `dict[str, Tensor]` | Training forward, trả loss dict |
| `denoise(satellite_img, noisy_label, num_steps, temp)` | `(B,3,H,W)` + optional | `(B,H,W)` | Reverse diffusion sampling |

**Hyperparameters mặc định:**
| Tham số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| `num_classes` | 7 | Số class segmentation |
| `num_timesteps` | 100 | Số bước diffusion T |
| `base_channels` | 128 | Số kênh cơ bản UNet |
| `cond_type` | `'concat'` | Cách inject condition |
| `transition_type` | `'uniform'` | Loại noise D3PM |
| `loss_type` | `'hybrid'` | KL + λ·CE |
| `hybrid_lambda` | 0.01 | Trọng số CE trong hybrid loss |

---

### 2. `ConditionalUNet` — Mạng Denoising

**File:** `diffusion_denoiser/models/conditional_unet.py` (854 lines)

UNet tiêu chuẩn với **timestep modulation** và 3 cơ chế conditioning:

#### Cơ Chế Conditioning

| `cond_type` | Mô tả | Input channels | Tính toán |
|-------------|--------|----------------|-----------|
| `'concat'` | Nối ảnh vệ tinh với x_t tại input | K + 3 | Đơn giản, không cần encoder riêng |
| `'crossattn'` | Cross-attention từ UNet features sang condition features | K | Cần ConditionEncoder riêng |
| `'hybrid'` | Cả concat và cross-attention | K + 3 | Mạnh nhất, tốn tính toán nhất |

#### Các Building Blocks

| Class | Chức năng |
|-------|----------|
| `SinusoidalTimestepEmbedding` | Embedding timestep dạng sin/cos (Vaswani et al.) |
| `TimestepMLP` | Project embedding → model dimension (2-layer MLP) |
| `ResBlock` | Residual block với **scale-shift modulation** từ timestep |
| `SelfAttention` | Multi-head self-attention cho spatial features |
| `CrossAttention` | Cross-attention: UNet features (Q) attend to condition (K,V) |
| `Downsample` | Strided conv 3×3 (stride=2) |
| `Upsample` | Nearest interpolate 2× rồi conv 3×3 |

#### Architecture Flow

```
Input (K+3 or K channels)
  │
  ▼ input_conv (3×3)
  │
  ├── Encoder Level 0: [ResBlock × num_res_blocks] [+ SelfAttn] [+ CrossAttn] → Downsample
  ├── Encoder Level 1: [ResBlock × num_res_blocks] [+ SelfAttn] [+ CrossAttn] → Downsample
  ├── Encoder Level 2: [ResBlock × num_res_blocks] [+ SelfAttn] [+ CrossAttn] → Downsample
  └── Encoder Level 3: [ResBlock × num_res_blocks] [+ SelfAttn] [+ CrossAttn]
                        │
                        ▼
                   Bottleneck: ResBlock → SelfAttn → [CrossAttn] → ResBlock
                        │
                        ▼
  ┌── Decoder Level 3: [ResBlock × (num_res_blocks+1)] [+ SelfAttn] [+ CrossAttn] → Upsample
  ├── Decoder Level 2: [ResBlock × (num_res_blocks+1)] [+ SelfAttn] [+ CrossAttn] → Upsample
  ├── Decoder Level 1: [ResBlock × (num_res_blocks+1)] [+ SelfAttn] [+ CrossAttn] → Upsample
  └── Decoder Level 0: [ResBlock × (num_res_blocks+1)] [+ SelfAttn] [+ CrossAttn]
                        │
                        ▼
                   GroupNorm → SiLU → output_conv (3×3) → logits (B, K, H, W)
```

**Attention chỉ áp dụng tại resolutions specified bởi `attn_resolutions`.**
Mặc định `(2, 4)` = downsample factor 2× và 4× (level 1 và 2).

---

### 3. Condition Encoders

#### `ConditionEncoder` (Lightweight CNN)

**File:** `conditional_unet.py` (L205-257)

Encoder nhẹ cho ảnh vệ tinh, tạo multi-scale features:
- **Stem:** Conv 7×7 stride 2 → GroupNorm → SiLU
- **Stages 1-3:** Mỗi stage gồm 2× Conv 3×3 với stride 2 ở đầu
- **Output channels mặc định:** `[64, 128, 256, 512]`
- **Không cần pretrained weights**

#### `PretrainedConditionEncoder` (SegFormer / ResNet)

**File:** `conditional_unet.py` (L264-601)

Encoder sử dụng backbone pretrained từ `mmseg`:

| Backbone | Class | Feature Channels | Kích thước |
|----------|-------|-----------------|-----------|
| `segformer_b2` | MixVisionTransformer | [64, 128, 320, 512] | ~25M params |
| `resnet50` | ResNetV1c | [256, 512, 1024, 2048] | ~25M params |
| `resnet101` | ResNetV1c | [256, 512, 1024, 2048] | ~44M params |

**Tính năng:**
- **Freeze stages:** Đóng băng N stages đầu (mặc định freeze 2 stages đầu)
- **1×1 projection:** Map backbone channels → target channels `[64, 128, 256, 512]`
- **Load pretrained:** Hỗ trợ local `.pth` files và `open-mmlab://` model zoo URLs
- **BN eval mode:** Tự động giữ frozen BN layers ở eval mode khi training

---

### 4. `D3PM` — Discrete Diffusion Process

**File:** `diffusion_denoiser/diffusion/d3pm.py` (289 lines)

Implement đầy đủ D3PM (Austin et al., NeurIPS 2021) cho pseudo-label denoising.

#### Quá Trình Forward (Training)

```python
# 1. Random timestep
t = randint(0, T)

# 2. Forward diffusion: thêm noise rời rạc
x_t = noise_schedule.q_sample(x_0, t)   # x_0 → x_t qua cumulative Q_bar_t

# 3. Model dự đoán x_0 từ x_t
x_0_logits = UNet(one_hot(x_t), t, satellite)

# 4. Tính loss
loss = KL(q(x_{t-1}|x_t,x_0) || p_θ(x_{t-1}|x_t)) + λ·CE(x_0_logits, x_0)
```

#### Quá Trình Reverse (Sampling/Inference)

```python
x_T = random or noisy_pseudo_label

for t = T-1, T-2, ..., 1:
    x_0_pred = UNet(x_t, t, satellite)           # Predict clean x_0
    x_0_probs = softmax(x_0_pred / temperature)
    posterior = Σ_k p(x_0=k) · q(x_{t-1}|x_t, x_0=k)  # Marginalize
    x_{t-1} ~ Categorical(posterior)               # Sample

# At t=0: directly argmax
x_0 = argmax(UNet(x_1, 0, satellite))
```

#### Loss Functions

| `loss_type` | Công thức | Ý nghĩa |
|-------------|-----------|---------|
| `'ce'` | CE(x_0_logits, x_0) | Direct x_0 prediction accuracy |
| `'kl'` | KL(q_true \|\| p_pred) | Diffusion posterior matching |
| `'hybrid'` | KL + λ·CE | Kết hợp cả hai (mặc định λ=0.01) |

#### KL Loss Chi Tiết

```
KL = Σ_j q_true(x_{t-1}=j | x_t, x_0) · log[q_true(j) / p_pred(j)]
```

Trong đó:
- `q_true`: posterior thật từ GT x_0
- `p_pred`: posterior dự đoán từ model's soft x_0 predictions (marginalized qua tất cả possible x_0)

---

### 5. `DiscreteNoiseSchedule` — Transition Matrices

**File:** `diffusion_denoiser/diffusion/noise_schedule.py` (225 lines)

Precompute tất cả transition matrices cho D3PM.

#### Loại Transition

| `transition_type` | Ma trận Q_t | Ý nghĩa |
|-------------------|-------------|---------|
| `'uniform'` | `Q_t = (1-β_t)·I + (β_t/K)·11^T` | Mỗi pixel có xác suất β_t nhảy tới bất kỳ class nào |
| `'absorbing'` | Non-absorbing → absorbing state (class K-1) với xác suất β_t | Class cuối là mask state, noise "hấp thụ" pixels |

#### Beta Schedule

| `beta_schedule` | Công thức |
|-----------------|-----------|
| `'linear'` | `β_t = linspace(β_start, β_end, T)` |
| `'cosine'` | `β_t = 1 - ᾱ_t/ᾱ_{t-1}` với `ᾱ_t = cos²((t/T + 0.008)/1.008 · π/2)` |

#### Buffers (registered, non-learnable)

| Buffer | Shape | Ý nghĩa |
|--------|-------|---------|
| `betas` | `(T,)` | Noise schedule β_1..β_T |
| `Q_t` | `(T, K, K)` | Single-step transition matrices |
| `Q_bar` | `(T, K, K)` | Cumulative products: `Q_bar_t = Q_1 · Q_2 · ... · Q_t` |

#### Key Methods

| Method | Input | Output | Mô tả |
|--------|-------|--------|-------|
| `q_sample(x_0, t)` | `(B,H,W)` + `(B,)` | `(B,H,W)` | Sample x_t from q(x_t\|x_0) |
| `q_posterior(x_0, x_t, t)` | 3× `(B,H,W)` | `(B,H,W,K)` | True posterior q(x_{t-1}\|x_t,x_0) |

---

### 6. `PseudoLabelDiffusionDataset` — Data Loading

**File:** `diffusion_denoiser/datasets/pseudo_label_dataset.py` (166 lines)

#### Cấu Trúc Dữ Liệu Yêu Cầu

```
data_root/
├── images/train/           # Satellite images (.tif)
├── images/val/
├── pseudo_labels/train/    # Pseudo-labels (.png)
├── pseudo_labels/val/
├── clean_labels/train/     # Ground truth (.png)
└── clean_labels/val/
```

#### Output Sample

```python
{
    'satellite_img':  Tensor(3, 512, 512),   # float32, normalized
    'pseudo_label':   Tensor(512, 512),       # int64, class indices
    'clean_label':    Tensor(512, 512),       # int64, class indices
    'filename':       str,
}
```

#### Augmentations

| Mode | Augmentations |
|------|--------------|
| Train | Random crop (512×512), random horizontal flip, random vertical flip |
| Val/Test | Center crop (hoặc pad nếu ảnh nhỏ hơn crop_size) |

**Normalization mặc định:** ImageNet mean/std (BGR order từ mmcv.imread).

---

## Training Pipeline

### Script: `tools/train.py`

**Feature highlights:**
- **EMA (Exponential Moving Average):** Decay 0.9999, dùng cho cả eval và checkpoint
- **LR Warmup:** Linear warmup trong 5000 iter đầu, sau đó cosine annealing
- **Gradient Clipping:** `clip_grad_norm_(params, 1.0)`
- **Multi-GPU:** Hỗ trợ `torchrun` với DistributedDataParallel (DDP)
- **Checkpoint:** Save model + optimizer + EMA + iter count, symlink `latest.pth`
- **Periodic Eval:** Denoise validation pseudo-labels, tính mIoU

### Lệnh Training

```bash
# Single GPU
python tools/train.py configs/denoiser/d3pm_concat_uniform_512x512_100k.py

# Multi-GPU (4 GPUs)
torchrun --nproc_per_node=4 tools/train.py \
    configs/denoiser/d3pm_hybrid_uniform_512x512_100k.py --launcher pytorch

# Resume
python tools/train.py configs/denoiser/d3pm_concat_uniform_512x512_100k.py \
    --resume-from work_dirs/d3pm_concat_uniform/latest.pth
```

---

## Evaluation & Inference

### Script: `tools/test.py`

Đánh giá model bằng mIoU, so sánh denoised vs raw pseudo-label:

```bash
python tools/test.py \
    configs/denoiser/d3pm_concat_uniform_512x512_100k.py \
    work_dirs/d3pm_concat_uniform/latest.pth \
    --num-steps 50
```

Output:
```
==========================================================
Class      Pseudo (baseline)    Denoised (ours)      Δ
----------------------------------------------------------
0          0.0820               0.XXXX               +0.XXXX
...
mIoU       0.4907               0.YYYY               +0.ZZZZ
==========================================================
```

### Script: `tools/inference.py`

Denoise batch of pseudo-labels và save kết quả:

```bash
python tools/inference.py \
    configs/denoiser/d3pm_concat_uniform_512x512_100k.py \
    work_dirs/d3pm_concat_uniform/latest.pth \
    --img-dir data/test/images \
    --pseudo-dir data/test/pseudo_labels \
    --out-dir data/test/refined_labels \
    --num-steps 50
```

---

## Ablation Study Framework

### Script: `tools/run_ablation.sh`

Tự động chạy **6 thí nghiệm** (3 conditioning × 2 noise types):

| Conditioning | Uniform Noise | Absorbing Noise |
|-------------|--------------|----------------|
| Concat | ✅ | ✅ |
| CrossAttn | ✅ | ✅ |
| Hybrid | ✅ | ✅ |

```bash
# Full ablation
bash tools/run_ablation.sh

# Multi-GPU
bash tools/run_ablation.sh --gpus 4

# Evaluate only
bash tools/run_ablation.sh --eval-only

# Custom settings
bash tools/run_ablation.sh --data-root /path/to/dataset --eval-steps 50
```

**Output:** Bảng aggregated results + CSV file trong `work_dirs/ablation_results.{txt,csv}`

---

## Ma Trận Config: 9 Biến Thể

Codebase hỗ trợ **9 config combinations** (tăng so với 6 cơ bản bằng pretrained encoders):

| Config | Cond Type | Noise | Encoder | LR |
|--------|-----------|-------|---------|-----|
| concat_uniform | concat | uniform | — | 1e-4 |
| concat_absorbing | concat | absorbing | — | 1e-4 |
| crossattn_uniform | crossattn | uniform | lightweight | 1e-4 |
| crossattn_absorbing | crossattn | absorbing | lightweight | 1e-4 |
| crossattn_uniform_segformer | crossattn | uniform | **SegFormer-B2** | **5e-5** |
| crossattn_absorbing_resnet50 | crossattn | absorbing | **ResNet50** | **5e-5** |
| hybrid_uniform | hybrid | uniform | lightweight | 1e-4 |
| hybrid_absorbing | hybrid | absorbing | lightweight | 1e-4 |
| hybrid_uniform_resnet101 | hybrid | uniform | **ResNet101** | LR phụ thuộc |

> **Lưu ý:** Configs pretrained dùng LR thấp hơn (5e-5) để tránh catastrophic forgetting.

---

## Dependencies

| Package | Version | Vai trò |
|---------|---------|---------|
| `torch` | ≥1.10 | Core deep learning |
| `mmcv-full` | ≥1.3.0 | Config system, image I/O |
| `mmsegmentation` | (optional) | Backbones cho PretrainedConditionEncoder |
| `numpy` | — | Numeric computation |
| `tqdm` | — | Progress bars |
| `PIL` | — | Image saving (inference) |

---

## Lưu Ý Kỹ Thuật Quan Trọng

### 1. Discrete vs Continuous Diffusion
- **Input/Output:** Integer class indices `(B, H, W)` ∈ [0, K-1]
- **Internal:** One-hot encoded `(B, K, H, W)` float cho UNet
- **Noise:** Categorical transition (Q matrices) thay vì Gaussian
- **Sampling:** `torch.multinomial` thay vì reparameterization

### 2. Precision Considerations
- Transition matrices `Q_t`, `Q_bar` được tính ở `float64` rồi cast sang `float32`
- Posterior clamp `min=1e-10` để tránh log(0)
- KL divergence tính trên `(B, H, W, K)` — rất tốn memory cho resolution lớn

### 3. Missing Features (so với full D3PM paper)
- **Không có auxiliary denoising loss** (L_T term)
- **Không có importance sampling** cho timestep t
- **Dataset không return pseudo_label cho training** — training dùng clean_label + forward diffusion noise (model tự tạo noise x_t từ x_0, không dùng pseudo_label có sẵn)
- **Chưa có mixed precision training (AMP)**
- **Losses module trống** — tất cả loss logic trong D3PM class

### 4. Luồng Dữ Liệu Chú Ý
- **Training:** `forward()` nhận `clean_label` (x_0), model tự tạo noise x_t → dự đoán x_0
- **Inference:** `denoise()` nhận noisy `pseudo_label` hoặc random → reverse diffusion → x_0
- **pseudo_label trong dataset chỉ dùng khi inference/evaluation**, không dùng khi training forward
