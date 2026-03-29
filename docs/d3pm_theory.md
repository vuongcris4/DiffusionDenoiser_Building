# D3PM Theory — Discrete Denoising Diffusion cho Pseudo-label Refinement

## Tổng Quan: Tại Sao D3PM?

### Bài Toán
- **Input:** Pseudo-label map `x` ∈ {0, 1, ..., K-1}^{H×W} — discrete, integer
- **Output:** Refined label map gần ground truth
- **Condition:** Ảnh vệ tinh RGB

### Tại Sao Không Dùng Gaussian Diffusion?
Label maps là **discrete** (integer class IDs). Gaussian noise thêm vào integer thì vô nghĩa. D3PM (Austin et al., NeurIPS 2021) thiết kế cho **discrete state spaces** — mỗi pixel chuyển đổi giữa các class theo xác suất rời rạc.

### D3PM vs DDPM

| | DDPM (continuous) | D3PM (discrete) |
|---|---|---|
| State space | ℝ^d (continuous pixel values) | {0, 1, ..., K-1} (class indices) |
| Forward noise | `x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε` | `x_t ~ Categorical(x_0 · Q̄_t)` |
| Noise type | Gaussian ε ~ N(0,I) | Transition matrix Q_t ∈ ℝ^{K×K} |
| Sampling | `x_{t-1} = μ_θ(x_t, t) + σ · z` | `x_{t-1} ~ Categorical(posterior)` |

---

## Forward Process (Thêm Noise)

### Transition Matrix Q_t

Tại mỗi timestep t, mỗi pixel chuyển từ class i sang class j với xác suất `Q_t[i, j]`:

```
x_t ~ Categorical(one_hot(x_{t-1}) · Q_t)
```

### Loại 1: Uniform Transition

```
Q_t = (1 - β_t) · I + (β_t / K) · 1·1^T
```

- Với xác suất `(1 - β_t)`: pixel **giữ nguyên** class
- Với xác suất `β_t`: pixel **nhảy đều** tới bất kỳ class nào (bao gồm cả chính nó)
- Ở t lớn (β_t → 1), phân bố → uniform `1/K` cho mọi class

### Loại 2: Absorbing Transition

```
Q_t[i, K-1] = β_t      (for i < K-1, transition to mask state)
Q_t[K-1, K-1] = 1.0    (mask stays mask)
Q_t[i, i] = 1 - β_t    (stay with prob 1-β_t, for i < K-1)
```

- Class cuối cùng (K-1) là **absorbing state** (trạng thái hấp thụ = mask)
- Pixels dần bị "hấp thụ" vào mask state
- Ở t lớn, gần như tất cả pixels đều là mask

### Cumulative Transition Q̄_t

```
Q̄_t = Q_1 · Q_2 · ... · Q_t
```

Cho phép nhảy thẳng từ x_0 đến x_t mà không cần sequential:

```
p(x_t | x_0) = one_hot(x_0) · Q̄_t
x_t ~ Categorical(p(x_t | x_0))
```

### Beta Schedule

#### Linear
```
β_t = linspace(β_start, β_end, T)
```
Đơn giản nhưng thường quá agressive ở bước cuối.

#### Cosine (Nichol & Dhariwal, 2021)
```
ᾱ_t = cos²((t/T + 0.008) / 1.008 · π/2)
β_t = 1 - ᾱ_t / ᾱ_{t-1}
β_t = clip(β_t, 0, 0.999)
```
Smooth hơn, noise tăng chậm ở đầu và cuối.

---

## Reverse Process (Khử Noise)

### Posterior Thật: q(x_{t-1} | x_t, x_0)

Bằng Bayes' rule:

```
q(x_{t-1} = j | x_t, x_0) ∝ q(x_t | x_{t-1} = j) · q(x_{t-1} = j | x_0)
                            = Q_t[j, x_t] · Q̄_{t-1}[x_0, j]
```

Đây là posterior **analytical** — tính được chính xác khi biết cả x_0 và x_t.

### Model Prediction: p_θ(x_{t-1} | x_t, cond)

Model **không trực tiếp dự đoán** posterior p_θ(x_{t-1} | x_t). Thay vào đó, nó dự đoán x_0:

```
x̂_0 = UNet(one_hot(x_t), t, satellite_image)   → logits (B, K, H, W)
p_θ(x_0 = k) = softmax(x̂_0)[k]
```

Rồi tính predicted posterior bằng cách **marginalize** over x_0:

```
p_θ(x_{t-1} = j | x_t) = Σ_k p_θ(x_0 = k) · q(x_{t-1} = j | x_t, x_0 = k)
```

Công thức chi tiết:
```
p_θ(x_{t-1} = j | x_t) ∝ q(x_t | x_{t-1} = j) · Σ_k [p_θ(x_0 = k) · Q̄_{t-1}[k, j]]
```

### Sampling Loop

```
Khởi tạo: x_T ~ Uniform(0, K-1) hoặc x_T = noisy_pseudo_label

For t = T-1, T-2, ..., 1:
    1. x̂_0 = UNet(x_t, t, satellite)
    2. p_θ(x_0) = softmax(x̂_0 / temperature)
    3. posterior(j) = Σ_k p_θ(x_0=k) · q(x_{t-1}=j | x_t, x_0=k)
    4. Normalize posterior
    5. x_{t-1} ~ Categorical(posterior)

At t = 0:
    x_0 = argmax(UNet(x_1, 0, satellite))   ← deterministic final step
```

---

## Training Loss

### Loss 1: Cross-Entropy (CE)

```
L_CE = CrossEntropy(x̂_0_logits, x_0_true)
     = -Σ_{b,h,w} log[softmax(x̂_0)[x_0_true[b,h,w]]]
```

Trực tiếp đo chất lượng dự đoán x_0.

### Loss 2: KL Divergence

```
L_KL = KL(q(x_{t-1} | x_t, x_0) || p_θ(x_{t-1} | x_t))
     = Σ_j q_true(j) · log[q_true(j) / p_pred(j)]
```

Đo khoảng cách giữa posterior thật và posterior dự đoán. Đây là loss "chuẩn ELBO" của D3PM.

### Loss 3: Hybrid (Mặc Định)

```
L_total = L_KL + λ · L_CE        (λ = 0.01 mặc định)
```

CE giúp ổn định training ở giai đoạn đầu khi KL chưa hội tụ.

---

## Kết Nối Với Bài Toán Pseudo-label

### Training
Model **không dùng pseudo-label** để train. Thay vào đó:
1. Lấy `clean_label` (ground truth) làm x_0
2. Tự thêm noise qua forward diffusion → x_t
3. Model học: từ x_t + satellite → dự đoán lại x_0

Điều này có nghĩa model học "denoising" nói chung, không specific cho pattern noise của CISC-R.

### Inference
1. Lấy `pseudo_label` (noisy, từ CISC-R) làm x_T
2. Reverse diffusion: x_T → x_{T-1} → ... → x_0
3. Output x_0 là refined label

> **Mismatch tiềm ẩn:** Forward noise (uniform/absorbing) có thể khác pattern noise thật của CISC-R (structured, class-dependent, boundary-concentrated). Đây là design choice có thể optimize.

---

## Toán Học Triển Khai Trong Code

### `q_sample(x_0, t)` — Forward Diffusion

```python
# Code: noise_schedule.py L140-171
Q_bar_t = self.Q_bar[t]                          # (B, K, K)
x_0_onehot = F.one_hot(x_0, K).float()           # (B, H, W, K)
x_0_flat = x_0_onehot.view(B, H*W, K)            # (B, H*W, K)
probs = torch.bmm(x_0_flat, Q_bar_t.T)           # (B, H*W, K)
x_t = torch.multinomial(probs.view(-1, K), 1)     # Sample
```

### `q_posterior(x_0, x_t, t)` — True Posterior

```python
# Code: noise_schedule.py L173-224
# q(x_{t-1}=j | x_0) = x_0_oh · Q̄_{t-1}^T
prob_xtm1_given_x0 = bmm(x_0_flat, Q_bar_prev.T)

# q(x_t | x_{t-1}=j) = Q_t[j, :] · x_t_oh
prob_xt_given_xtm1 = bmm(Q_t, x_t_flat.T).T

# Bayes
posterior ∝ prob_xt_given_xtm1 * prob_xtm1_given_x0
posterior = posterior / posterior.sum(dim=-1, keepdim=True)
```

### `_soft_posterior(x_0_probs, x_t, t)` — Predicted Posterior

```python
# Code: d3pm.py L192-239
# q(x_t | x_{t-1}=j) for each j
prob_xt_given_xtm1 = bmm(Q_t, x_t_oh.T).T       # (B, H*W, K)

# Σ_k p(x_0=k) · Q̄_{t-1}[k, j]
prob_xtm1 = bmm(x_0_probs_flat, Q_bar_prev.T)    # (B, H*W, K)

# Combined
posterior ∝ prob_xt_given_xtm1 * prob_xtm1
posterior = posterior / posterior.sum(...)
```

---

## Memory & Computation Considerations

### Transition Matrices
- `Q_t`: `(T, K, K)` = `(100, 7, 7)` = 19.6K floats ≈ 78KB
- `Q_bar`: same size
- **Negligible memory** cho K nhỏ

### Forward/Backward Per Step
- **x_t one-hot:** `(B, K, H, W)` = `(4, 7, 512, 512)` = 28MB @ float32
- **UNet forward:** Phụ thuộc architecture, ~1-2GB cho 512×512
- **KL loss:** cần `(B, H, W, K)` posteriors × 2 ≈ 28MB × 2

### Scale Concerns
- **K lớn (nhiều classes):** Q_t ∝ K², `q_posterior` ∝ K² — vẫn ok cho K ≤ 50
- **Resolution lớn:** Attention layers ∝ (H×W)² — bottleneck chính
- **T lớn (nhiều steps):** Training không ảnh hưởng (chỉ sample 1 t), inference ∝ T

---

## Tham Khảo

1. Austin et al., "Structured Denoising Diffusion Models in Discrete State-Spaces", NeurIPS 2021
2. Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models", ICML 2021 (cosine schedule)
3. Vaswani et al., "Attention Is All You Need", NeurIPS 2017 (sinusoidal embeddings)
