# Dataset config for D3PM pseudo-label denoising.
# Same directory structure as MMsegDenoiser.

dataset_type = 'PseudoLabelDiffusionDataset'
data_root = 'data/my_dataset'
num_classes = 7

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
        img_dir='images/train',
        ann_dir='clean_labels/train',
        pseudo_label_dir='pseudo_labels/train',
        num_classes=num_classes,
        crop_size=crop_size,
        img_suffix='.tif',
        label_suffix='.png',
        img_norm_cfg=img_norm_cfg,
        is_train=True),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='clean_labels/val',
        pseudo_label_dir='pseudo_labels/val',
        num_classes=num_classes,
        crop_size=crop_size,
        img_suffix='.tif',
        label_suffix='.png',
        img_norm_cfg=img_norm_cfg,
        is_train=False),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='clean_labels/val',
        pseudo_label_dir='pseudo_labels/val',
        num_classes=num_classes,
        crop_size=crop_size,
        img_suffix='.tif',
        label_suffix='.png',
        img_norm_cfg=img_norm_cfg,
        is_train=False))
