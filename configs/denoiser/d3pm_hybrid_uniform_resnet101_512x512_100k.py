_base_ = [
    '../_base_/models/d3pm_hybrid_uniform_resnet101.py',
    '../_base_/datasets/pseudo_label_diffusion.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_100k.py',
]

# Lower LR for pretrained condition encoder to prevent catastrophic forgetting
optimizer = dict(lr=5e-5)

# Override for binary building segmentation (0=other, 1=building).
# Lighter UNet to fit on a single 22 GB GPU:
#   base_channels  128 -> 64
#   num_res_blocks 2   -> 1
#   attn_resolutions only at coarsest 64x64 stage
model = dict(
    num_classes=2,
    base_channels=64,
    num_res_blocks=1,
    attn_resolutions=(8,))

# Hybrid conditioning (concat + cross-attn) is heavier than pure cross-attn,
# so we keep batch size 1 and crop to 384x384 — same as the SegFormer run.
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(num_classes=2, crop_size=(384, 384)),
    val=dict(num_classes=2, crop_size=(384, 384)),
    test=dict(num_classes=2, crop_size=(384, 384)))
