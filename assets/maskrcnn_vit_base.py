default_scope = 'mmdet'

_base_ = [
    '../../../configs/_base_/models/mask-rcnn_r50_fpn.py',
]

#####################################################

work_dir = '/path/to/save_dir'

# pretrained = None
pretrained = '/path/to/your/pretrained'


dataset_type = 'CocoDataset'
data_root = '/path/to/data/root'
image_size = (256, 256)

classes = ('class1', 'class2', 'class3',) # dataset class' names
num_classes = len(classes)

#####################################################

custom_imports = dict(imports=['projects.ViTDet.vitdet'])
port = int(29500)	# port for tensorboard
env_cfg = dict(
cudnn_benchmark=True,
mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
dist_cfg=dict(backend='nccl', port=port))

backbone_norm_cfg = dict(type='LN', requires_grad=True)
norm_cfg = dict(type='LN2d', requires_grad=True)
batch_augments = [
    dict(type='BatchFixedSizePad', size=image_size, pad_mask=True)
]

# model settings
model = dict(
    data_preprocessor=dict(pad_size_divisor=32, batch_augments=batch_augments,
                           mean=None,
                           std=None,
                           bgr_to_rgb=False,
                           ),
    backbone=dict(
        _delete_=True,
        type='ViT',
        img_size=256,
        in_chans=2,         # edit here for image channels
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        drop_path_rate=0.1,
        window_size=14,
        mlp_ratio=4,
        qkv_bias=True,
        norm_cfg=backbone_norm_cfg,
        window_block_indexes=[
            0,
            1,
            3,
            4,
            6,
            7,
            9,
            10,
        ],
        use_rel_pos=True,
        # init_cfg=dict(type=None), # if training from scratch
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        _delete_=True,
        type='SimpleFPN',
        backbone_channel=768,
        in_channels=[192, 384, 768, 768],
        out_channels=256,
        num_outs=5,
        norm_cfg=norm_cfg),
    rpn_head=dict(
            type='RPNHead',
            in_channels=256,
            feat_channels=256,
            num_convs=2,            # edit here
            anchor_generator=dict(
                type='AnchorGenerator',
                scales=[8],
                ratios=[0.5, 1.0, 2.0],
                strides=[4, 8, 16, 32, 64]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            conv_out_channels=256,
            norm_cfg=norm_cfg,
            num_classes=1),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            norm_cfg=norm_cfg,
            num_classes=1,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)))
)

custom_hooks = [dict(type='Fp16CompresssionHook')]

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromNDArray', backend_args=backend_args, to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomResize',
        scale=image_size,
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=image_size,
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='Pad', size=image_size),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromNDArray', backend_args=backend_args, to_float32=True),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    dict(type='Pad', size_divisor=32),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]


train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='path/to/train_json_file',
        data_prefix=dict(img='path/to/train_img_file'), # from data_root
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        metainfo=dict(classes=classes))
        )

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='path/to/val_json_file',
        data_prefix=dict(img='path/to/val_img_file'), # from data_root
        test_mode=True,
        pipeline=test_pipeline,
        metainfo=dict(classes=classes)))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'path/to/val_json_file',
    metric=['segm'],
    format_only=False)

test_evaluator = val_evaluator

optim_wrapper = dict(
    type='AmpOptimWrapper',
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg={
        'decay_rate': 0.7,
        'decay_type': 'layer_wise',
        'num_layers': 12,
    },
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.1,
    ))

# calculate your max_iter based on the sample of data, epochs, and (total) batch size
# For example, fine-tuning COCO 100 epoch, batch size 64, COCO has ~118000 imgs
# so the max_iter = 118000 / 64 * 100
# COCO
# 100 ep = 184375 iters * 64 images/iter / 118000 images/ep
# max_iters = 184375

max_iters = None # calculate your max_iter

interval = 1000
dynamic_intervals = [(max_iters // interval * interval + 1, max_iters)]
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=250),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_iters,
        by_epoch=False,
        # COCO
        # 88 ep = [163889 iters * 64 images/iter / 118000 images/ep
        # 96 ep = [177546 iters * 64 images/iter / 118000 images/ep
        # milestones=[163889, 177546],

        milestones=[None, None],
        gamma=0.1)
]

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=interval,
    dynamic_intervals=dynamic_intervals)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=100),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        save_last=True,
        save_best='coco/segm_mAP',
        rule="greater",
        interval=interval,
        max_keep_ckpts=5))
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='mmdet.DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)

auto_scale_lr = dict(base_batch_size=64)    # batch size 64

# load_from = None  # Load model checkpoint as a pre-trained model from a given path. This will not resume training.
# resume = False    # whether resume from checkpoint
