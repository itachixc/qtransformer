_base_ = [
    './vit_config_cifar10.py'
]

sampling_error=0.003
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
        num_classes=10,
        in_channels=768,
        sampling_error=sampling_error,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.00001, weight_decay=0.05),
    clip_grad=dict(max_norm=1.0),
    # specific to vit pretrain
    paramwise_cfg=dict(custom_keys={
        '.cls_token': dict(decay_mult=0.0),
        '.pos_embed': dict(decay_mult=0.0),
        'backbone': dict(lr_mult=0.1,decay_mult=1.0),
    }),
)

