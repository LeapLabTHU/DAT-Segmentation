_base_ = [
    '../_base_/models/fpn_dat.py', 
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py'
]

pretrained = '<path-to-pretrained-model>'

model = dict(
    backbone=dict(
        type='DAT',
        dim_stem=96,
        dims=[96, 192, 384, 768],
        depths=[2, 4, 18, 2],
        stage_spec=[
            ["N", "D"], 
            ["N", "D", "N", "D"], 
            ["N", "D", "N", "D", "N", "D", "N", "D", "N", "D", "N", "D", "N", "D", "N", "D", "N", "D"], 
            ["D", "D"]],
        heads=[3, 6, 12, 24],
        groups=[1, 2, 3, 6],
        use_pes=[True, True, True, True],
        strides=[8, 4, 2, 1],
        offset_range_factor=[-1, -1, -1, -1],
        use_dwc_mlps=[True, True, True, True],
        use_lpus=[True, True, True, True],
        use_conv_patches=True,
        ksizes=[9, 7, 5, 3],
        nat_ksizes=[7, 7, 7, 7],
        drop_path_rate=0.4,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)
    ),
    neck=dict(in_channels=[96, 192, 384, 768]),
    decode_head=dict(num_classes=150)
)

gpu_multiples = 2  # we use 8 gpu instead of 4 in mmsegmentation, so lr*2 and max_iters/2
# optimizer
optimizer = dict(type='AdamW', lr=0.0001*gpu_multiples, weight_decay=0.0001, 
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.), 
                                  'relative_position_bias_table': dict(decay_mult=0.),
                                  'rpe_table': dict(decay_mult=0.),
                                  'norm': dict(decay_mult=0.)
                                 }))
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=0.0, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=80000//gpu_multiples)
checkpoint_config = dict(by_epoch=False, interval=8000//gpu_multiples)
evaluation = dict(interval=8000//gpu_multiples, metric='mIoU')

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])
auto_resume = True