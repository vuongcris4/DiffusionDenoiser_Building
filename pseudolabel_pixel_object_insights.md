# Insight pixel-level và object-level: pseudolabel vs Ground Truth

## Dữ liệu

- Pseudolabel: `/home/ubuntu/vuong_denoiser/BUILDING/DifusionDenoiser/DifusionDenoiser/data/test_oem_raw/test_oem_raw/pseudolabels_binary`
- Ground truth: `/home/ubuntu/vuong_denoiser/BUILDING/DifusionDenoiser/DifusionDenoiser/data/test_oem_raw/test_oem_raw/labels`
- Số file pseudolabel: `100`
- Số file ground truth: `220`
- Số file được so sánh cùng tên: `100`
- Quy ước nhị phân: `pixel > 0` là building/foreground, `pixel == 0` là background.
- Object-level dùng connected components 8-neighbor trên mask nhị phân.

## Executive summary

Pseudolabel khá tốt nếu nhìn ở pixel-level, với `IoU = 0.7411` và `F1/Dice = 0.8513`. Tuy nhiên object-level yếu hơn nhiều: object recall ở ngưỡng IoU `0.5` chỉ `54.94%`, object precision chỉ `41.94%`.

Lỗi chính không phải là pseudo phủ quá nhiều diện tích, mà là pseudo vừa có nhiều object nhỏ/noise, vừa bỏ sót hoặc cắt hụt building thật. Tổng diện tích building của pseudo nhỏ hơn GT khoảng `0.838%` tổng pixel, nhưng số object pseudo lại nhiều hơn GT `1.31x`.

## Pixel-level insight

| Metric | Giá trị |
|---|---:|
| Tổng pixel được so sánh | 100,369,604 |
| Pixel khác nhau `(FP + FN)` | 4,694,922 |
| Tỷ lệ khác nhau | 4.678% |
| Accuracy | 0.9532 |
| IoU building | 0.7411 |
| F1/Dice building | 0.8513 |
| Precision building | 0.8746 |
| Recall building | 0.8292 |
| Pseudolabel building area | 15.309% |
| Ground truth building area | 16.147% |
| Pseudo - GT area | -0.838% |
| Pseudo / GT area | 0.9481 |

## Pixel confusion matrix

| Pseudolabel \ Ground truth | Building | Background | Tổng |
|---|---:|---:|---:|
| Building | 13,438,404 | 1,926,710 | 15,365,114 |
| Background | 2,768,212 | 82,236,278 | 85,004,490 |
| Tổng | 16,206,616 | 84,162,988 | 100,369,604 |

## Pixel-level interpretation

- `FP = 1,926,710` pixel, tương đương `1.920%` tổng pixel: pseudo dự đoán building nhưng GT là background.
- `FN = 2,768,212` pixel, tương đương `2.758%` tổng pixel: GT là building nhưng pseudo bỏ sót.
- `FN > FP`, nên lỗi chính ở pixel-level là under-segmentation, tức pseudo thiếu building nhiều hơn là vẽ dư building.
- Pseudo building area thấp hơn GT: `15.309%` so với `16.147%`.

## Object-level insight

| Metric | Giá trị |
|---|---:|
| GT objects | 7,421 |
| Pseudo objects | 9,721 |
| Pseudo - GT object count | +2,300 |
| Pseudo / GT object count | 1.31x |
| GT object có overlap với pseudo | 88.76% |
| Pseudo object có overlap với GT | 73.83% |
| Object recall @ IoU 0.10 | 71.15% |
| Object precision @ IoU 0.10 | 54.32% |
| Object recall @ IoU 0.25 | 67.50% |
| Object precision @ IoU 0.25 | 51.53% |
| Object recall @ IoU 0.50 | 54.94% |
| Object precision @ IoU 0.50 | 41.94% |
| Object recall @ IoU 0.75 | 34.77% |
| Object precision @ IoU 0.75 | 26.54% |
| GT objects không overlap pseudo | 834 |
| Pseudo objects không overlap GT | 2,544 |
| GT no-overlap pixels | 150,140 |
| Pseudo no-overlap pixels | 330,534 |
| GT object bị split đáng kể | 544 |
| Pseudo object merge nhiều GT đáng kể | 586 |

## Object-level interpretation

- Pixel-level nhìn tốt vì phần lớn pixel building nằm trong các object vừa/lớn.
- Object-level thấp vì nhiều object nhỏ bị bỏ sót hoặc chỉ overlap rất ít.
- Có `2,544` pseudo objects không overlap GT, nhưng median diện tích false pseudo object chỉ `31 px`, nghĩa là nhiều object giả rất nhỏ.
- Có `834` GT objects không overlap pseudo, median diện tích missed GT object là `95 px`, nghĩa là phần lớn object bị miss cũng nhỏ.
- Pseudo có nhiều object hơn GT nhưng diện tích lại nhỏ hơn GT. Điều này cho thấy pseudo bị fragment/noise ở object nhỏ, đồng thời vẫn thiếu diện tích building thật.

## Recall theo kích thước GT object

| GT object size | Số GT object | % GT building pixels | Any-overlap recall | Recall @ IoU 0.10 | Recall @ IoU 0.25 | Recall @ IoU 0.50 |
|---|---:|---:|---:|---:|---:|---:|
| 1-50 px | 336 | 0.057% | 26.79% | 11.01% | 7.74% | 3.57% |
| 51-200 px | 1,049 | 0.781% | 64.35% | 42.04% | 36.99% | 26.22% |
| 201-1k px | 2,827 | 9.337% | 92.89% | 69.37% | 65.72% | 53.48% |
| 1k-5k px | 2,585 | 35.500% | 99.50% | 87.04% | 84.14% | 71.10% |
| >5k px | 624 | 54.325% | 100.00% | 94.71% | 90.06% | 70.51% |

## Insight theo kích thước object

- Object nhỏ `1-50 px` gần như không được khớp tốt: recall @ IoU `0.5` chỉ `3.57%`.
- Object `51-200 px` cũng yếu: recall @ IoU `0.5` chỉ `26.22%`.
- Từ `201 px` trở lên, pseudo bắt đầu ổn hơn.
- Building lớn gần như luôn có overlap, nhưng IoU chưa chắc cao do boundary bị hụt, split hoặc merge.

## Component size statistics

| Loại object | Số object | Mean area | Median | Q75 | Q90 | Q95 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| GT object | 7,421 | 2,183.89 | 784 | 1,989 | 4,304 | 7,730 | 112,924 |
| Pseudo object | 9,721 | 1,580.61 | 367 | 1,412 | 3,465 | 6,067 | 109,800 |
| Missed GT no-overlap | 834 | 180.02 | 95 | 206 | 476 | 635 | 3,671 |
| False pseudo no-overlap | 2,544 | 129.93 | 31 | 105 | 268 | 507 | 7,636 |

## Split và merge

| Hiện tượng | Số lượng | Tỷ lệ |
|---|---:|---:|
| GT object bị split đáng kể | 544 | 7.33% GT objects |
| Pseudo object merge nhiều GT đáng kể | 586 | 6.03% pseudo objects |

Ảnh split nhiều:

| File | GT obj | Pseudo obj | Split GT | Merge pseudo | TP @ IoU 0.5 | Pixel diff |
|---|---:|---:|---:|---:|---:|---:|
| `accra_68.tif` | 219 | 452 | 46 | 13 | 76 | 22.246% |
| `coxsbazar_62.tif` | 344 | 341 | 34 | 29 | 189 | 7.707% |
| `maputo_11.tif` | 175 | 208 | 29 | 19 | 58 | 9.188% |
| `zanzibar_98.tif` | 432 | 314 | 27 | 67 | 134 | 11.799% |
| `koeln_28.tif` | 233 | 316 | 26 | 8 | 181 | 6.071% |

Ảnh merge nhiều:

| File | GT obj | Pseudo obj | Split GT | Merge pseudo | TP @ IoU 0.5 | Pixel diff |
|---|---:|---:|---:|---:|---:|---:|
| `zanzibar_98.tif` | 432 | 314 | 27 | 67 | 134 | 11.799% |
| `kyoto_14.tif` | 240 | 249 | 26 | 37 | 99 | 11.694% |
| `kyoto_7.tif` | 244 | 223 | 11 | 33 | 114 | 9.322% |
| `coxsbazar_62.tif` | 344 | 341 | 34 | 29 | 189 | 7.707% |
| `maputo_11.tif` | 175 | 208 | 29 | 19 | 58 | 9.188% |

## Ảnh đáng chú ý

Ảnh khác nhiều nhất ở pixel-level:

| File | Pixel diff | IoU | FP | FN | Pseudo area | GT area |
|---|---:|---:|---:|---:|---:|---:|
| `accra_68.tif` | 22.246% | 0.5423 | 1.795% | 20.451% | 28.155% | 46.811% |
| `ica_37.tif` | 20.840% | 0.6243 | 5.743% | 15.097% | 40.368% | 49.722% |
| `paris_46.tif` | 19.895% | 0.3932 | 9.443% | 10.451% | 22.334% | 23.342% |
| `rio_34.tif` | 19.504% | 0.5409 | 2.379% | 17.126% | 25.362% | 40.109% |
| `chincha_45.tif` | 17.557% | 0.6867 | 5.353% | 12.204% | 43.841% | 50.693% |

Ảnh miss nhiều GT object nhất:

| File | GT obj | Pseudo obj | GT no-overlap | Pseudo no-overlap | TP @ IoU 0.5 | Pixel diff |
|---|---:|---:|---:|---:|---:|---:|
| `coxsbazar_62.tif` | 344 | 341 | 62 | 15 | 189 | 7.707% |
| `zanzibar_98.tif` | 432 | 314 | 53 | 22 | 134 | 11.799% |
| `aachen_46.tif` | 208 | 251 | 36 | 55 | 126 | 7.876% |
| `ica_46.tif` | 101 | 136 | 33 | 21 | 31 | 9.247% |
| `kyoto_4.tif` | 216 | 158 | 32 | 104 | 31 | 8.576% |

Ảnh có nhiều pseudo false object nhất:

| File | GT obj | Pseudo obj | GT no-overlap | Pseudo no-overlap | TP @ IoU 0.5 | Pixel diff |
|---|---:|---:|---:|---:|---:|---:|
| `kyoto_4.tif` | 216 | 158 | 32 | 104 | 31 | 8.576% |
| `bielefeld_33.tif` | 208 | 295 | 10 | 100 | 151 | 7.744% |
| `lohur_9.tif` | 91 | 169 | 1 | 95 | 50 | 5.307% |
| `kyoto_27.tif` | 161 | 204 | 19 | 90 | 74 | 5.271% |
| `kyoto_5.tif` | 137 | 153 | 17 | 71 | 50 | 8.640% |

## Effect of filtering small pseudo components

| Min pseudo object area | Pseudo obj | Obj precision @0.5 | Obj recall @0.5 | Any precision | Any recall | Pixel IoU | Pixel F1 | Pixel precision | Pixel recall | Diff | FP | FN | Pseudo area |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 px | 9,721 | 41.94% | 54.94% | 73.83% | 88.76% | 0.7411 | 0.8513 | 0.8746 | 0.8292 | 4.678% | 1.920% | 2.758% | 15.309% |
| 5 px | 8,960 | 45.50% | 54.94% | 77.22% | 88.30% | 0.7411 | 0.8513 | 0.8747 | 0.8292 | 4.677% | 1.919% | 2.759% | 15.307% |
| 10 px | 8,534 | 47.77% | 54.94% | 79.20% | 87.89% | 0.7411 | 0.8513 | 0.8748 | 0.8291 | 4.676% | 1.917% | 2.760% | 15.304% |
| 20 px | 8,041 | 50.70% | 54.94% | 81.57% | 87.19% | 0.7411 | 0.8513 | 0.8750 | 0.8289 | 4.675% | 1.912% | 2.762% | 15.297% |
| 50 px | 7,283 | 55.75% | 54.71% | 85.73% | 85.58% | 0.7412 | 0.8514 | 0.8758 | 0.8283 | 4.669% | 1.897% | 2.772% | 15.272% |
| 100 px | 6,542 | 60.65% | 53.47% | 89.64% | 82.94% | 0.7411 | 0.8513 | 0.8772 | 0.8268 | 4.665% | 1.869% | 2.797% | 15.219% |
| 200 px | 5,672 | 66.41% | 50.76% | 94.04% | 78.37% | 0.7395 | 0.8503 | 0.8799 | 0.8226 | 4.678% | 1.813% | 2.865% | 15.095% |
| 500 px | 4,377 | 71.72% | 42.30% | 97.03% | 67.08% | 0.7258 | 0.8411 | 0.8838 | 0.8025 | 4.894% | 1.704% | 3.190% | 14.661% |
| 1000 px | 3,039 | 74.07% | 30.33% | 98.32% | 52.43% | 0.6878 | 0.8150 | 0.8880 | 0.7531 | 5.521% | 1.534% | 3.987% | 13.693% |

## Recommendation

Nếu dùng pseudolabel cho semantic segmentation, nên cân nhắc post-process bằng cách bỏ pseudo components nhỏ hơn khoảng `50 px`.

Lý do:

- Object precision @ IoU `0.5` tăng từ `41.94%` lên `55.75%`.
- Object recall @ IoU `0.5` gần như giữ nguyên: `54.94%` xuống `54.71%`.
- Pixel IoU gần như không đổi: `0.7411` lên `0.7412`.
- Pixel F1 gần như không đổi: `0.8513` lên `0.8514`.

Không nên lọc quá mạnh như `>= 500 px` hoặc `>= 1000 px`, vì lúc đó recall giảm rõ và pixel IoU tụt.

Kết luận ngắn:

- Pixel-level: pseudo đủ tốt về diện tích tổng thể, nhưng hơi thiếu building.
- Object-level: pseudo còn yếu với building nhỏ, có nhiều object noise nhỏ, và có cả split/merge.
- Ngưỡng lọc component nhỏ `50 px` là lựa chọn hợp lý để làm sạch pseudolabel mà ít ảnh hưởng đến chất lượng pixel-level.
