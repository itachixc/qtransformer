_base_ = [
    './qtransformer_configs/vit_config_cifar10.py'
]

sampling_error=0.001
sampling_mode=1
sampling_order=200
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
        sampling_mode=sampling_mode,
        sampling_order=sampling_order,
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
        # type='VisionTransformerClsHead',
        num_classes=10,
        in_channels=768,
        sampling_error=sampling_error,
        sampling_mode=sampling_mode,
        sampling_order=10,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))


train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    sampler=dict(type='DefaultSampler', shuffle=False),
)


load_from='epoch_1.pth'

