NUM_CLASS=3
TimeStep=3
_base_ = [
    f'./_base_/GAMMA.py',
    './_base_/default_runtime.py',
    './_base_/schedule.py'
]
custom_imports = dict(imports='mmcls.models', allow_failed_imports=False)
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='IOMSG',
    timesteps=TimeStep,
    bit_scale=0.01,
    pretrained=None,
    backbone=dict(),
    neck=[],
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=0,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=NUM_CLASS,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4)
        ),
    decode_head=dict(
        type='DeformableHeadWithTime',
        in_channels=[256],
        channels=256,
        in_index=[0],
        dropout_ratio=0.,
        num_classes=NUM_CLASS,
        dataset_name="GAMMA",
        norm_cfg=norm_cfg,
        align_corners=False,
        num_feature_levels=1,
        encoder=dict(
            type='DetrTransformerEncoder',
            num_layers=3,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                use_time_mlp=True,
                attn_cfgs=dict(
                    type='MultiScaleDeformableAttention',
                    embed_dims=256,
                    num_levels=1,
                    num_heads=8,
                    dropout=0.),
                ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=256,
                    feedforward_channels=1024,
                    ffn_drop=0.,
                    act_cfg=dict(type='GELU')),
                operation_order=('self_attn', 'norm', 'ffn', 'norm'))
        ),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0)),
        # loss_decode=dict(
        #     type='DiceLoss',
        #     loss_name='loss_dice',
        #     loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
)
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=1.)
        }))
lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
find_unused_parameters = True
evaluation = dict(interval=5000, metric='mIoU', save_best='mIoU')
