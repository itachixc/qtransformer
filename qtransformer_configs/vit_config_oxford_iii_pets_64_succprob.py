_base_ = [
    '../configs/_base_/default_runtime.py'
]

sampling_error=0.001
r_size=64
# dataset settings
dataset_type = 'OxfordIIITPet'


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
        num_classes=37,
        in_channels=768,
        sampling_error=sampling_error,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))



# data_preprocessor = dict(
#     num_classes=200,
#     # RGB format normalization parameters
#     mean=[123.675, 116.28, 103.53],
#     std=[58.395, 57.12, 57.375],
#     # convert image from BGR to RGB
#     to_rgb=False,
# )

data_preprocessor = dict(
    num_classes=37,
    # RGB format normalization parameters
    mean=[122.676, 114.538, 100.903],
    std=[58.258, 57.524, 58.017],
    # convert image from BGR to RGB
    to_rgb=False,
)

# Mean: [0.48108368 0.44916949 0.39569984]
# Standard Deviation: [0.22846428 0.22558406 0.22751722]



train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=r_size),
    dict(type='RandomCrop', crop_size=r_size),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=r_size),
    dict(type='CenterCrop', crop_size=r_size),
    dict(type='PackInputs'),
]




train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root='data/Oxford_IIII_Pets',
        split='trainval',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root='data/Oxford_IIII_Pets',
        split='test',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1,5 ))

test_dataloader = val_dataloader
test_evaluator = val_evaluator




# optimizer
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.05),
    clip_grad=dict(max_norm=1.0),
    # specific to vit pretrain
    paramwise_cfg=dict(custom_keys={
        '.cls_token': dict(decay_mult=0.0),
        '.pos_embed': dict(decay_mult=0.0),
        'backbone': dict(lr_mult=0.1,decay_mult=1.0),
    }),
)

# learning policy
# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-2,
        by_epoch=False,
        begin=0,
        end=1000,
        # update by iter
        # convert_to_iter_based=True
        ),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        # T_max=10,
        by_epoch=False,
        begin=2000,
        end=3000,
    )
]


# load_from='vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth'
load_from=None
# load_from='vit-base-p32_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-9cea8599.pth'

# train, val, test setting
# train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=1)
train_cfg = dict(by_epoch=False, max_iters=500, val_interval=3000)
val_cfg = dict()
test_cfg = dict()

# configure default hooks
default_hooks = dict(
    # print log every 100 iterations.
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    checkpoint=dict(type='CheckpointHook', interval=1)
)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=128)
