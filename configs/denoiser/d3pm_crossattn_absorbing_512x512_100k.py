_base_ = [
    '../_base_/models/d3pm_crossattn_absorbing.py',
    '../_base_/datasets/pseudo_label_diffusion.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_100k.py',
]

data = dict(samples_per_gpu=4, workers_per_gpu=4)
