_base_ = [
    '../_base_/models/d3pm_crossattn_uniform_segformer.py',
    '../_base_/datasets/pseudo_label_diffusion.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_100k.py',
]

# Pretrained backbone uses BN; smaller batch may require adjustment
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(num_classes=2),
    val=dict(num_classes=2),
    test=dict(num_classes=2))

# Lower LR for pretrained condition encoder to prevent catastrophic forgetting
model = dict(num_classes=2)
optimizer = dict(lr=5e-5)

# Avoid full spatial attention at 256x256 / 128x128, which is not
# tractable on 22 GB GPUs for 512x512 inputs. Restrict attention to the
# coarsest 64x64 stage.
model = dict(
    num_classes=2,
    base_channels=64,
    num_res_blocks=1,
    attn_resolutions=(8,))

# This diffusion model is memory-heavy; batch size 1 is the safer default
# on a single 22 GB GPU.
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(num_classes=2, crop_size=(384, 384)),
    val=dict(num_classes=2, crop_size=(384, 384)),
    test=dict(num_classes=2, crop_size=(384, 384)))
