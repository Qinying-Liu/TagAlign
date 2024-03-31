_base_ = ["../custom_import.py"]
# dataset settings
dataset_type = "ImageNetSDataset"
subset = 300
data_root = "/data/liuqy/ImagenetS_dataset/ImageNetS" + str(subset)
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageNetSImageFromFile', downsample_large_image=True),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(2048, 448),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="validation",
        ann_dir="validation-segmentation",
        pipeline=test_pipeline,
        subset=subset,
    )
)

test_cfg = dict(bg_thresh=0.1,
                scale=30.0, 
                clip_w=0.5,
                mode="slide", 
                stride=(56, 56), 
                crop_size=(448, 448))
