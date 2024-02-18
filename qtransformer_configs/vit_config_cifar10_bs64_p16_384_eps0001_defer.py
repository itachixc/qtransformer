_base_ = [
    './vit_config_cifar10.py'
]

sampling_error=0.001
r_size=384
# dataset settings

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformerSamplingDefer',
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
        num_classes=10,
        in_channels=768,
        sampling_error=sampling_error,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

