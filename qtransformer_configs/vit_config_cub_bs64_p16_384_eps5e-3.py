_base_ = [
    './vit_config_cub.py'
]

sampling_error=0.005
r_size=384
# dataset settings

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformerSampling',
        arch='b',
        img_size=r_size,
        patch_size=16,
        drop_rate=0.1,
        sampling_error=sampling_error,
        init_cfg=[
            dict(
                type='Kaiming',
                layer='Conv2d',
                mode='fan_in',
                nonlinearity='linear')
        ]),
    neck=None,
    head=dict(
        type='VisionTransformerClsHeadSampling',
        num_classes=200,
        in_channels=768,
        sampling_error=sampling_error,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
