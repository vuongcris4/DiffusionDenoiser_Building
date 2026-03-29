# DifusionDenoiser — API Reference & Function Map

## Module: `diffusion_denoiser.models.diffusion_denoiser`

### `DiffusionDenoiserModel(nn.Module)`
Top-level model class. Instantiated từ config file.

| Phương thức | Signature | Mô tả |
|------------|-----------|-------|
| `__init__` | `(num_classes=7, num_timesteps=100, base_channels=128, channel_mult=(1,2,4,8), num_res_blocks=2, attn_resolutions=(2,4), dropout=0.1, cond_type='concat', cond_channels=3, cond_base_channels=64, cond_encoder_type='lightweight', pretrained_cond_cfg=None, transition_type='uniform', beta_schedule='cosine', loss_type='hybrid', hybrid_lambda=0.01)` | Khởi tạo UNet + NoiseSchedule + D3PM |
| `forward` | `(clean_label: Tensor(B,H,W), satellite_img: Tensor(B,3,H,W)) → dict` | Training forward. Trả `{'loss_total', 'loss_ce', 'loss_kl'}` |
| `denoise` | `(satellite_img: Tensor(B,3,H,W), noisy_label: Tensor(B,H,W)=None, num_steps: int=None, temperature: float=1.0) → Tensor(B,H,W)` | Inference reverse diffusion. `@torch.no_grad()` |

---

## Module: `diffusion_denoiser.models.conditional_unet`

### `SinusoidalTimestepEmbedding(nn.Module)`
| Method | `forward(t: Tensor(B,)) → Tensor(B, dim)` |
|--------|-----|
| Logic | Sinusoidal positional encoding cho timestep |

### `TimestepMLP(nn.Module)`
| Method | `forward(t_emb: Tensor(B, t_dim)) → Tensor(B, out_dim)` |
|--------|-----|
| Logic | 2-layer MLP: Linear → SiLU → Linear |

### `ResBlock(nn.Module)`
| Method | `forward(x: Tensor(B,C,H,W), t_emb: Tensor(B,t_dim)) → Tensor(B,C_out,H,W)` |
|--------|-----|
| Logic | ConvBlock + timestep scale-shift modulation + skip connection |
| Key | `h = norm(h) * (1 + scale) + shift` where scale,shift from t_emb |

### `SelfAttention(nn.Module)`
| Method | `forward(x: Tensor(B,C,H,W)) → Tensor(B,C,H,W)` |
|--------|-----|
| Logic | Multi-head self-attention with residual |

### `CrossAttention(nn.Module)`
| Method | `forward(x: Tensor(B,C,H,W), cond: Tensor(B,C_cond,H_c,W_c)) → Tensor(B,C,H,W)` |
|--------|-----|
| Logic | Q from x, K/V from cond. Auto-resize cond spatial dims. Residual. |

### `ConditionEncoder(nn.Module)`
| Method | `forward(x: Tensor(B,3,H,W)) → list[Tensor]` |
|--------|-----|
| Logic | Lightweight CNN: stem (7×7 s2) + 3 stages (3×3 s2). Output 4 scales: `[64, 128, 256, 512]` |

### `PretrainedConditionEncoder(nn.Module)`
| Method | `forward(x: Tensor(B,3,H,W)) → list[Tensor]` |
|--------|-----|
| Logic | mmseg backbone (SegFormer/ResNet) + 1×1 projections |
| Key Methods | `_build_backbone`, `_load_pretrained`, `_freeze_stages`, `_freeze_segformer_stages`, `_freeze_resnet_stages` |

### `ConditionalUNet(nn.Module)`
| Method | `forward(x_t: Tensor(B,K,H,W), t: Tensor(B,), condition: Tensor(B,3,H,W)) → Tensor(B,K,H,W)` |
|--------|-----|
| Logic | Full UNet: input_conv → encoder (4 levels) → bottleneck → decoder (4 levels) → output_conv |
| Output | x_0 logits, unnormalized |

---

## Module: `diffusion_denoiser.diffusion.d3pm`

### `D3PM(nn.Module)`

| Phương thức | Signature | Mô tả |
|------------|-----------|-------|
| `forward` | `(x_0: Tensor(B,H,W), condition: Tensor(B,3,H,W)) → dict` | Full training: sample t → q_sample → predict x_0 → compute loss |
| `_predict_x0` | `(x_t, t, condition) → Tensor(B,K,H,W)` | One-hot x_t → UNet → logits |
| `_compute_loss` | `(x_0, x_t, x_0_logits, t) → dict` | CE + KL loss computation |
| `_kl_loss` | `(x_0, x_t, x_0_logits, t) → Tensor` | KL(q_true \|\| p_pred) |
| `_soft_posterior` | `(x_0_probs, x_t, t) → Tensor(B,H,W,K)` | Predicted posterior marginalized over soft x_0 |
| `sample` | `(condition, noisy_label=None, num_steps=None, temperature=1.0) → Tensor(B,H,W)` | Full reverse sampling loop |

---

## Module: `diffusion_denoiser.diffusion.noise_schedule`

### `DiscreteNoiseSchedule(nn.Module)`

| Phương thức | Signature | Mô tả |
|------------|-----------|-------|
| `__init__` | `(num_classes, num_timesteps=100, transition_type='uniform', beta_schedule='cosine', beta_start=1e-4, beta_end=0.02)` | Build và cache Q_t, Q_bar |
| `_get_beta_schedule` | `(schedule, T, beta_start, beta_end) → ndarray` | Linear hoặc cosine schedule |
| `_build_transition_matrices` | `(betas, K, transition_type) → Tensor(T,K,K)` | Uniform hoặc absorbing Q_t |
| `_compute_cumulative_products` | `(Q_t) → Tensor(T,K,K)` | Q_bar_t = Q_1 · ... · Q_t |
| `q_sample` | `(x_0: Tensor(B,H,W), t: Tensor(B,)) → Tensor(B,H,W)` | Forward diffusion sampling |
| `q_posterior` | `(x_0, x_t, t) → Tensor(B,H,W,K)` | True posterior via Bayes |

---

## Module: `diffusion_denoiser.datasets.pseudo_label_dataset`

### `PseudoLabelDiffusionDataset(Dataset)`

| Phương thức | Signature | Mô tả |
|------------|-----------|-------|
| `__init__` | `(data_root, img_dir, pseudo_label_dir, ann_dir, num_classes, crop_size=(512,512), img_suffix='.tif', label_suffix='.png', img_norm_cfg=None, is_train=True, ignore_index=255)` | Load file list |
| `__getitem__` | `(idx) → dict` | Return `{satellite_img, pseudo_label, clean_label, filename}` |
| `_random_crop` | `(img, pseudo, clean) → tuple` | Training augmentation |
| `_center_crop_or_pad` | `(img, pseudo, clean) → tuple` | Validation preprocessing |
| `_pad` | `(img, pseudo, clean, h, w) → tuple` | Reflect pad image, const pad labels |

---

## Module: `tools/train.py`

### `EMA`
| Method | Mô tả |
|--------|-------|
| `__init__(model, decay=0.9999)` | Clone tất cả trainable params |
| `update(model)` | `shadow = decay * shadow + (1-decay) * param` |
| `apply(model)` | Replace params with EMA |
| `restore(model, backup)` | Restore original params |

### Functions
| Function | Mô tả |
|----------|-------|
| `build_model(cfg) → DiffusionDenoiserModel` | From config dict |
| `build_dataset(data_cfg, is_train) → Dataset` | From config dict |
| `evaluate(model, val_loader, device, num_steps=10) → (miou, per_class_iou)` | Denoise + mIoU |
| `main()` | Full training loop with EMA, warmup, eval, checkpointing |

---

## Module: `tools/test.py`

| Function | Mô tả |
|----------|-------|
| `compute_miou(pred, gt, num_classes, ignore_index=255) → ndarray` | Per-class IoU |
| `main()` | Load checkpoint (prefer EMA) → evaluate → print comparison table |

---

## Module: `tools/inference.py`

| Function | Mô tả |
|----------|-------|
| `main()` | Load images + pseudo-labels → denoise → save as PNG |

---

## Call Graph (Training)

```
main() [tools/train.py]
 ├── Config.fromfile(args.config)
 ├── build_model(cfg) → DiffusionDenoiserModel.__init__()
 │    ├── ConditionalUNet.__init__()
 │    │    ├── SinusoidalTimestepEmbedding()
 │    │    ├── TimestepMLP()
 │    │    ├── ConditionEncoder() or PretrainedConditionEncoder()
 │    │    ├── ResBlock() × many
 │    │    ├── SelfAttention() × many
 │    │    └── CrossAttention() × many
 │    ├── DiscreteNoiseSchedule.__init__()
 │    │    ├── _get_beta_schedule()
 │    │    ├── _build_transition_matrices()
 │    │    └── _compute_cumulative_products()
 │    └── D3PM.__init__()
 ├── build_dataset(cfg.data.train) → PseudoLabelDiffusionDataset()
 ├── EMA(model)
 └── Training loop:
      ├── model(clean_label, satellite) → DiffusionDenoiserModel.forward()
      │    └── D3PM.forward(x_0, condition)
      │         ├── noise_schedule.q_sample(x_0, t) → x_t
      │         ├── _predict_x0(x_t, t, condition) → logits
      │         │    └── ConditionalUNet.forward(x_t_onehot, t, condition)
      │         │         ├── time_embed(t) → time_mlp(emb)
      │         │         ├── cond_encoder(condition) → multi-scale feats
      │         │         ├── Encoder: ResBlock + SelfAttn + CrossAttn
      │         │         ├── Bottleneck: ResBlock + SelfAttn + CrossAttn
      │         │         └── Decoder: ResBlock + SelfAttn + CrossAttn
      │         └── _compute_loss(x_0, x_t, logits, t)
      │              ├── F.cross_entropy(logits, x_0) → CE loss
      │              └── _kl_loss()
      │                   ├── noise_schedule.q_posterior(x_0, x_t, t)
      │                   └── _soft_posterior(softmax(logits), x_t, t)
      ├── loss.backward()
      ├── clip_grad_norm_(params, 1.0)
      ├── optimizer.step()
      ├── ema.update(model)
      └── [periodic] evaluate(model, val_loader)
           └── model.denoise(satellite, pseudo, num_steps)
                └── D3PM.sample(condition, noisy_label, steps, temp)
                     └── Loop T → 0:
                          ├── _predict_x0(x_t, t, cond)
                          ├── softmax(logits/temp) → x_0_probs
                          ├── _soft_posterior(x_0_probs, x_t, t)
                          └── multinomial(posterior) → x_{t-1}
```
