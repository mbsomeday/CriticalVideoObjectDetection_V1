"""HTC R-50 nuImages (IN+COCO-20e, nuImages 1x) for MMDetection 3.x inference.

This config targets the nuImages HTC R-50 checkpoint whose original config is:
  configs/nuimages/htc_r50_fpn_coco-20e_1x_nuim.py (MMDetection3D v0.7.0)

It requires nuimages_base_mmdet3.py in the same folder and does not require
nuImages annotation files for inference on nuScenes CAM_FRONT frames.

Use the *nuImages-finetuned* checkpoint; the COCO-20e checkpoint was only
used to initialize training, and is not the final nuImages detector.
"""

_base_ = './nuimages_base_mmdet3.py'

# Inherited from the base:
#   R-50 backbone, FPN, RPN, input normalization, and CAM_FRONT test pipeline.
# Override detector type and all HTC-specific heads.
model = dict(
    type='HybridTaskCascade',
    train_cfg=None,
    data_preprocessor=dict(pad_seg=True),
    roi_head=dict(
        type='HybridTaskCascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1.0, 0.5, 0.25],
        interleaved=True,
        mask_info_flow=True,
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=10,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=list(stds),
                ),
                reg_class_agnostic=True,
                loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
            )
            for stds in (
                (0.1, 0.1, 0.2, 0.2),
                (0.05, 0.05, 0.1, 0.1),
                (0.033, 0.033, 0.067, 0.067),
            )
        ],
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),
        mask_head=[
            dict(
                type='HTCMaskHead',
                with_conv_res=False,
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=10,
                loss_mask=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
            ),
            dict(
                type='HTCMaskHead',
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=10,
                loss_mask=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
            ),
            dict(
                type='HTCMaskHead',
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=10,
                loss_mask=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
            ),
        ],
        semantic_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[8],
        ),
        semantic_head=dict(
            type='FusedSemanticHead',
            num_ins=5,
            fusion_level=1,
            seg_scale_factor=1 / 8,
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=32,
            loss_seg=dict(type='CrossEntropyLoss', ignore_index=0, loss_weight=0.2),
        ),
    ),
)
