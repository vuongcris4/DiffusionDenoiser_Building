_base_ = [
    '../_base_/models/d3pm_crossattn_absorbing_resnet50.py',
    '../_base_/datasets/pseudo_label_diffusion.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_100k.py',
]

model = dict(num_classes=3)

data = dict(samples_per_gpu=4, workers_per_gpu=4)

# Lower LR for pretrained condition encoder to prevent catastrophic forgetting
optimizer = dict(lr=5e-5)
