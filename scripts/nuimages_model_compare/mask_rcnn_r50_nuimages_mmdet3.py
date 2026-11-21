"""Standalone MMDetection 3.x inference config for the official nuImages R50 Mask R-CNN.

Compatible model topology with the legacy nuImages model zoo checkpoint.
Does NOT require mmdetection3d or the nuImages dataset at inference time.
"""

_default_classes = (
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier',
)
default_scope = 'mmdet'

model = dict(
    type='MaskRCNN',
    train_cfg=None,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32,
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
                        target_means=[0.0, 0.0, 0.0, 0.0],
                        target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
    ),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256, featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead', in_channels=256, fc_out_channels=1024,
            roi_feat_size=7, num_classes=10,
            bbox_coder=dict(type='DeltaXYWHBBoxCoder',
                            target_means=[0.0, 0.0, 0.0, 0.0],
                            target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256, featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead', num_convs=4, in_channels=256,
            conv_out_channels=256, num_classes=10,
            loss_mask=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
    ),
    test_cfg=dict(
        rpn=dict(nms_pre=1000, max_per_img=1000,
                 nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0),
        rcnn=dict(score_thr=0.05, nms=dict(type='nms', iou_threshold=0.5),
                  max_per_img=100, mask_thr_binary=0.5),
    ),
)

# The official nuImages test scale is 1600x900, which matches nuScenes CAM_FRONT.
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1600, 900), keep_ratio=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

# Required by MMDetection 3.x's inference_detector(). This is never iterated.
test_dataloader = dict(
    dataset=dict(
        type='CocoDataset',
        metainfo=dict(classes=_default_classes),
        pipeline=test_pipeline,
    )
)
