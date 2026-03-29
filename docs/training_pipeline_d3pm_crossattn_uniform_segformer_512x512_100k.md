# Training Pipeline: `d3pm_crossattn_uniform_segformer_512x512_100k`

Tài liệu này mô tả training pipeline thực tế của config:

`/home/ubuntu/vuong_denoiser/BUILDING/DifusionDenoiser/DifusionDenoiser/configs/denoiser/d3pm_crossattn_uniform_segformer_512x512_100k.py`

Nội dung dưới đây bám theo implementation hiện tại trong:

- `tools/train.py`
- `diffusion_denoiser/datasets/pseudo_label_dataset.py`
- `diffusion_denoiser/models/diffusion_denoiser.py`
- `diffusion_denoiser/diffusion/d3pm.py`
- `diffusion_denoiser/models/conditional_unet.py`

## 1. Config được merge từ đâu

Experiment config:

```python
_base_ = [
    '../_base_/models/d3pm_crossattn_uniform_segformer.py',
    '../_base_/datasets/pseudo_label_diffusion.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_100k.py',
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(num_classes=2),
    val=dict(num_classes=2),
    test=dict(num_classes=2))

model = dict(num_classes=2)
optimizer = dict(lr=5e-5)
```

## 2. Merged config cuối cùng dùng để train

Sau khi merge 4 base configs và override ở experiment file, pipeline train thực tế dùng các tham số chính sau:

### Model

- `type='DiffusionDenoiserModel'`
- `num_classes=2`
- `num_timesteps=100`
- `base_channels=128`
- `channel_mult=(1, 2, 4, 8)`
- `num_res_blocks=2`
- `attn_resolutions=(2, 4)`
- `dropout=0.1`
- `cond_type='crossattn'`
- `cond_encoder_type='pretrained'`
- `transition_type='uniform'`
- `beta_schedule='cosine'`
- `loss_type='hybrid'`
- `hybrid_lambda=0.01`

### Condition encoder

- backbone: `SegFormer-B2`
- pretrained checkpoint: `pretrain/mit_b2.pth`
- `freeze_stages=2`
- multi-scale projected channels: `[64, 128, 256, 512]`

### Dataset

- `data_root='data/OEM_v2_Building'`
- image dir: `images`
- clean label dir: `labels`
- pseudo-label dir: `pseudolabels`
- splits: `train.txt`, `val.txt`, `test.txt`
- `img_suffix='.tif'`
- `label_suffix='.tif'`
- `crop_size=(512, 512)`
- `samples_per_gpu=4`
- `workers_per_gpu=4`

### Optimization / runtime

- optimizer: `AdamW`
- learning rate: `5e-5`
- betas: `(0.9, 0.999)`
- weight decay: `0.01`
- `max_iters=100000`
- warmup: `5000` iterations
- LR schedule: cosine annealing tới `1e-6`
- gradient clipping: `max_norm=1.0`
- EMA enabled: `decay=0.9999`
- `log_interval=100`
- `checkpoint_interval=10000`
- `eval_interval=10000`

## 3. Dữ liệu đi vào model như thế nào

`PseudoLabelDiffusionDataset` đọc mỗi sample thành một bộ 3:

- `satellite_img`: ảnh vệ tinh, shape `(3, H, W)`, normalized theo mean/std ImageNet
- `pseudo_label`: nhãn giả, shape `(H, W)`, class index map
- `clean_label`: nhãn sạch, shape `(H, W)`, class index map

### Augmentation

Trong train:

- random crop `512x512`
- random horizontal flip
- random vertical flip

Trong val/test:

- center crop hoặc pad về `512x512`

### Ghi chú quan trọng

Trong implementation train hiện tại, `pseudo_label` từ dataset **không được dùng trong forward training loss**. Train loop chỉ lấy:

- `satellite_img`
- `clean_label`

`pseudo_label` chỉ được dùng ở bước evaluation để khởi tạo reverse diffusion.

Điều này có nghĩa pipeline hiện tại đang học:

- từ `clean_label` sinh ra `x_t` bằng forward diffusion nội bộ,
- rồi dùng ảnh vệ tinh để dự đoán lại `x_0`,

chứ **không train trực tiếp theo cặp** `(pseudo_label, clean_label)`.

## 4. Luồng train end-to-end

### Bước 1: tạo batch

`DataLoader` trả về batch:

- `satellite_img`: `(B, 3, 512, 512)`
- `clean_label`: `(B, 512, 512)`
- `pseudo_label`: `(B, 512, 512)` nhưng không dùng trong train

Với config này:

- `B = 4`

### Bước 2: gọi model

Train loop trong `tools/train.py` gọi:

```python
losses = model(clean_label, satellite)
```

`DiffusionDenoiserModel.forward()` chỉ chuyển tiếp sang `self.d3pm(clean_label, satellite_img)`.

### Bước 3: sample timestep ngẫu nhiên

Trong `D3PM.forward()`:

```python
t = torch.randint(0, self.num_timesteps, (B,), device=device)
```

Mỗi sample trong batch nhận một timestep `t` ngẫu nhiên trong `[0, 99]`.

### Bước 4: forward diffusion để tạo `x_t`

Từ `clean_label = x_0`, noise schedule nội bộ tạo ra `x_t ~ q(x_t | x_0)`.

Với config này:

- noise type: `uniform`
- beta schedule: `cosine`

Ý nghĩa:

- nhãn sạch bị corruption dần trong không gian nhãn rời rạc
- corruption không dùng Gaussian như diffusion ảnh liên tục
- state transitions được mô tả bằng ma trận chuyển lớp theo từng timestep

### Bước 5: convert `x_t` sang one-hot

`x_t` là class index map `(B, H, W)`, sau đó được đổi thành:

- `x_t_onehot`: `(B, K, H, W)`

với `K = 2`.

### Bước 6: condition encoder xử lý ảnh vệ tinh

Vì config này dùng:

- `cond_type='crossattn'`
- `cond_encoder_type='pretrained'`

nên ảnh vệ tinh đi qua `PretrainedConditionEncoder` dùng backbone SegFormer-B2.

Pipeline phần condition:

1. Ảnh vệ tinh vào SegFormer-B2.
2. Backbone trích xuất feature maps nhiều scale.
3. Mỗi scale được chiếu qua `1x1 conv` để về các channel mục tiêu:
   - `64`
   - `128`
   - `256`
   - `512`
4. Hai stage đầu của backbone bị freeze.
5. Các stage sau và projection heads được finetune cùng UNet.

## 5. Kiến trúc denoiser

Network denoise cốt lõi là `ConditionalUNet`.

Input của denoiser:

- noisy label one-hot `x_t_onehot`
- timestep embedding `t`
- satellite condition features từ SegFormer

### Các thành phần chính

- sinusoidal timestep embedding
- MLP cho timestep
- UNet encoder-decoder với residual blocks
- self-attention ở các resolution nằm trong `attn_resolutions=(2, 4)`
- cross-attention để UNet feature attend vào condition feature

### Cách inject condition

Vì `cond_type='crossattn'`, feature ảnh vệ tinh được inject bằng cross-attention ở bottleneck và decoder, thay vì concat trực tiếp vào input.

Ý nghĩa thực tế:

- branch diffusion xử lý trạng thái nhãn nhiễu `x_t`
- branch SegFormer cung cấp visual prior giàu ngữ nghĩa từ ảnh vệ tinh
- cross-attention ghép hai nguồn này khi UNet dự đoán `x_0`

## 6. Model học dự đoán cái gì

Pipeline này dùng:

- `parameterization='x0'`

Nghĩa là model không dự đoán noise, cũng không dự đoán trực tiếp `x_{t-1}`.
Model dự đoán:

- logits của clean label `x_0`

Output của denoiser:

- `x_0_logits` có shape `(B, K, H, W)`

## 7. Loss được tính như thế nào

Config dùng:

- `loss_type='hybrid'`
- `hybrid_lambda=0.01`

Nên tổng loss là:

```text
loss_total = loss_kl + 0.01 * loss_ce
```

### `loss_ce`

Cross-entropy trực tiếp giữa:

- `x_0_logits`
- ground truth `clean_label`

### `loss_kl`

KL divergence giữa:

- posterior thật: `q(x_{t-1} | x_t, x_0)`
- posterior dự đoán: `p_theta(x_{t-1} | x_t)`

Posterior dự đoán được suy ra từ:

- softmax của `x_0_logits`
- noise schedule rời rạc

Nói ngắn gọn:

- CE giữ cho dự đoán `x_0` đúng lớp
- KL buộc reverse process khớp với posterior rời rạc của D3PM

## 8. Backprop và tối ưu

Sau khi có `loss_total`, train loop chạy:

1. `optimizer.zero_grad()`
2. `loss.backward()`
3. `clip_grad_norm_(..., 1.0)`
4. `optimizer.step()`

### Learning rate

Warmup:

- 5000 iter đầu scale tuyến tính từ rất nhỏ lên `5e-5`

Sau warmup:

- dùng `CosineAnnealingLR`

### EMA

Nếu bật EMA:

- sau mỗi iteration, shadow weights được cập nhật với decay `0.9999`

EMA chỉ được dùng ở evaluation, không thay thế trọng số train online trong mỗi step.

## 9. Checkpointing

Cứ mỗi `10000` iterations:

- lưu checkpoint `iter_<N>.pth`
- cập nhật symlink `latest.pth`

Checkpoint chứa:

- `model`
- `optimizer`
- `iter`
- `ema` nếu có

Work dir mặc định:

```text
work_dirs/d3pm_crossattn_uniform_segformer_512x512_100k
```

## 10. Validation / evaluation pipeline

Cứ mỗi `10000` iterations, rank 0 chạy evaluation.

### Quy trình eval

1. Nếu có EMA, áp EMA weights tạm thời vào model.
2. Dùng `pseudo_label` của validation set làm `noisy_label` đầu vào.
3. Gọi:

```python
pred = model.denoise(satellite, pseudo, num_steps=10)
```

4. Reverse diffusion chạy từ `t = 9 -> 0`.
5. So sánh `pred` với `clean_label`.
6. Tính IoU từng lớp và mIoU.
7. Nếu có EMA, restore lại online weights.

### Ghi chú quan trọng

Training dùng `100` timesteps, nhưng evaluation hiện mặc định chỉ denoise với:

- `num_steps=10`

Đây là một quyết định để tăng tốc eval, không phải full reverse chain 100 bước.

## 11. Pseudo-code pipeline

```text
for each iteration:
    batch = next(train_loader)
    satellite = batch['satellite_img']
    clean = batch['clean_label']

    sample timestep t
    x_t = q_sample(clean, t)
    x_t_onehot = one_hot(x_t)

    cond_feats = segformer_b2(satellite)
    x0_logits = conditional_unet(x_t_onehot, t, cond_feats)

    loss_ce = CE(x0_logits, clean)
    loss_kl = KL(q(x_{t-1}|x_t,x_0) || p_theta(x_{t-1}|x_t))
    loss_total = loss_kl + 0.01 * loss_ce

    backward(loss_total)
    grad_clip(1.0)
    optimizer.step()
    scheduler.step()
    ema.update()
```

## 12. Ý nghĩa của config này trong study

Config này đang test tổ hợp:

- conditioning mạnh bằng `cross-attention`
- visual prior pretrained bằng `SegFormer-B2`
- discrete transition kiểu `uniform`
- objective `hybrid`
- binary segmentation (`num_classes=2`)

Nó phù hợp khi muốn trả lời câu hỏi:

- pretrained encoder có giúp pseudo-label denoising tốt hơn lightweight encoder không
- cross-attention có khai thác ảnh vệ tinh tốt hơn concat không
- uniform discrete corruption có đủ ổn định cho bài toán building segmentation nhị phân không

## 13. Điểm cần lưu ý khi dùng doc này

### A. Mismatch giữa train và eval input

Train hiện tại:

- tạo `x_t` từ `clean_label`

Eval hiện tại:

- khởi tạo reverse process bằng `pseudo_label`

Vì vậy training objective đang học mô hình denoise theo noise process tổng quát, chứ không trực tiếp học sửa lỗi kiểu pseudo-label distribution thực tế.

### B. `num_classes=2`

Experiment file đã override model và dataset splits sang 2 lớp. Nếu dữ liệu thực tế không đúng mapping nhị phân, cần sửa lại ở config hoặc dataset chuẩn bị trước.

### C. SegFormer checkpoint là local dependency

Config này cần:

- `pretrain/mit_b2.pth`

Nếu checkpoint sai hoặc thiếu, model pretrained branch sẽ fail khi build.

## 14. Lệnh train tương ứng

Single GPU:

```bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate denoiser
python tools/train.py \
  configs/denoiser/d3pm_crossattn_uniform_segformer_512x512_100k.py
```

Resume:

```bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate denoiser
python tools/train.py \
  configs/denoiser/d3pm_crossattn_uniform_segformer_512x512_100k.py \
  --resume-from work_dirs/d3pm_crossattn_uniform_segformer_512x512_100k/latest.pth
```

## 15. Tóm tắt ngắn

Pipeline này là một D3PM discrete diffusion denoiser cho segmentation mask, trong đó:

- `clean_label` được corruption thành `x_t`
- model dùng `satellite_img` + `x_t` để dự đoán lại `x_0`
- condition branch là `SegFormer-B2` pretrained
- condition được inject bằng `cross-attention`
- loss là `KL + 0.01 * CE`
- eval khởi tạo từ `pseudo_label` và đo `mIoU` với `clean_label`

Nếu cần, bước tiếp theo hợp lý là viết thêm một tài liệu thứ hai so sánh:

- training pipeline hiện tại
- training pipeline lý tưởng nếu muốn dùng `pseudo_label` trực tiếp trong train objective

