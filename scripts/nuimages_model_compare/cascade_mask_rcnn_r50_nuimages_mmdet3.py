"""Cascade Mask R-CNN R-50, nuImages, 10 foreground classes, MMDet 3.x."""
_base_ = './nuimages_base_mmdet3.py'

model = dict(roi_head=dict(
    type='CascadeRoIHead', num_stages=3, stage_loss_weights=[1., 0.5, 0.25],
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
        out_channels=256, featmap_strides=[4, 8, 16, 32],
    ),
    bbox_head=[dict(
        type='Shared2FCBBoxHead', in_channels=256, fc_out_channels=1024,
        roi_feat_size=7, num_classes=10,
        bbox_coder=dict(type='DeltaXYWHBBoxCoder',
                        target_means=[0., 0., 0., 0.], target_stds=stds),
        reg_class_agnostic=True,
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.),
        loss_bbox=dict(type='SmoothL1Loss', beta=1., loss_weight=1.),
    ) for stds in (
        [0.1, 0.1, 0.2, 0.2],
        [0.05, 0.05, 0.1, 0.1],
        [0.033, 0.033, 0.067, 0.067],
    )],
    mask_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
        out_channels=256, featmap_strides=[4, 8, 16, 32],
    ),
    mask_head=dict(
        type='FCNMaskHead', num_convs=4, in_channels=256,
        conv_out_channels=256, num_classes=10,
        loss_mask=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.),
    ),
))
