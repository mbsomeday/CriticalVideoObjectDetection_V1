"""Common MMDetection 3.x inference config for R-50 nuImages checkpoints.

The 10 foreground categories and detector preprocessing match nuImages.
This standalone base does not require the nuImages images/dataset itself.
"""

_default_classes = (
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
)
default_scope = 'mmdet'

model = dict(
    type='CascadeRCNN',
    train_cfg=None,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True, pad_mask=True, pad_size_divisor=32,
    ),
    backbone=dict(
        type='ResNet', depth=50, num_stages=4,
        out_indices=(0, 1, 2, 3), frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True, style='pytorch',
    ),
    neck=dict(type='FPN', in_channels=[256, 512, 1024, 2048],
              out_channels=256, num_outs=5),
    rpn_head=dict(
        type='RPNHead', in_channels=256, feat_channels=256,
        anchor_generator=dict(type='AnchorGenerator', scales=[8],
                              ratios=[0.5, 1.0, 2.0],
                              strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(type='DeltaXYWHBBoxCoder',
                        target_means=[0., 0., 0., 0.],
                        target_stds=[1., 1., 1., 1.]),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.),
        loss_bbox=dict(type='SmoothL1Loss', beta=1. / 9., loss_weight=1.),
    ),
    test_cfg=dict(
        rpn=dict(nms_pre=1000, max_per_img=1000,
                 nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0),
        rcnn=dict(score_thr=0.001, nms=dict(type='nms', iou_threshold=0.5),
                  max_per_img=100, mask_thr_binary=0.5),
    ),
)

# Image inference uses original nuScenes CAM_FRONT size (1600 x 900).
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1600, 900), keep_ratio=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]
test_dataloader = dict(dataset=dict(type='CocoDataset',
                                    metainfo=dict(classes=_default_classes),
                                    pipeline=test_pipeline))
