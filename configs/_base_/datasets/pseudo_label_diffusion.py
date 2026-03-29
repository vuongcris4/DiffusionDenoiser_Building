# Dataset config for D3PM pseudo-label denoising on OEM_v2_Building.

dataset_type = 'PseudoLabelDiffusionDataset'
data_root = 'data/OEM_v2_Building'
dataset_num_classes = 3

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375])

crop_size = (512, 512)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='labels',
        pseudo_label_dir='pseudolabels',
        split_file='train.txt',
        num_classes=dataset_num_classes,
        crop_size=crop_size,
        img_suffix='.tif',
        label_suffix='.tif',
        img_norm_cfg=img_norm_cfg,
        is_train=True),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='labels',
        pseudo_label_dir='pseudolabels',
        split_file='val.txt',
        num_classes=dataset_num_classes,
        crop_size=crop_size,
        img_suffix='.tif',
        label_suffix='.tif',
        img_norm_cfg=img_norm_cfg,
        is_train=False),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='labels',
        pseudo_label_dir='pseudolabels',
        split_file='test.txt',
        num_classes=dataset_num_classes,
        crop_size=crop_size,
        img_suffix='.tif',
        label_suffix='.tif',
        img_norm_cfg=img_norm_cfg,
        is_train=False))
