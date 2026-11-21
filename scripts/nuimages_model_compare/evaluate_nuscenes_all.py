"""Run nuImages-pretrained MMDetection pedestrian models on nuScenes CAM_FRONT.

Saves ONLY per-frame predicted bounding boxes and confidence scores. Ground truth
and all metrics are handled separately by formate_res_updated.py.

Requires the same local MMDetection 3.x model config files as the prior script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

NUIM_CLASSES = (
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier',
)

MODEL_CONFIGS = {
    'maskrcnn': 'mask_rcnn_r50_nuimages_mmdet3.py',
    'cascade': 'cascade_mask_rcnn_r50_nuimages_mmdet3.py',
    'htc': 'htc_r50_nuimages_mmdet3.py',
}


def scene_samples(nusc, name):
    scene = next((s for s in nusc.scene if s['name'] == name), None)
    if scene is None:
        raise ValueError(f'Scene {name!r} not found')
    token = scene['first_sample_token']
    index = 0
    while token:
        sample = nusc.get('sample', token)
        yield index, sample
        token = sample['next']
        index += 1


def run(args):
    import torch
    from nuscenes.nuscenes import NuScenes
    from mmdet.apis import init_detector, inference_detector

    config = args.config or Path(__file__).with_name(MODEL_CONFIGS[args.model])
    if not config.is_file():
        raise FileNotFoundError(f'Model config not found: {config}')
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint}')

    device = ('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use --device cpu')

    nusc = NuScenes(version=args.version, dataroot=str(args.data_root), verbose=False)
    scene_names = ([s['name'] for s in nusc.scene]
                   if args.scene.lower() == 'all' else [args.scene])
    if not scene_names:
        raise ValueError('No scenes found in the requested dataset')
    model = init_detector(str(config), str(args.checkpoint), device=device, palette='random')
    model.dataset_meta['classes'] = NUIM_CLASSES
    bbox_heads = model.roi_head.bbox_head
    if not isinstance(bbox_heads, (list, tuple, torch.nn.ModuleList)):
        bbox_heads = [bbox_heads]
    for i, head in enumerate(bbox_heads):
        if head.fc_cls.out_features - 1 != len(NUIM_CLASSES):
            raise RuntimeError(f'BBox head {i} is not a 10-class nuImages head')
    if args.model != 'maskrcnn' and len(bbox_heads) != 3:
        raise RuntimeError(f'{args.model}: expected 3 cascade stages')

    # Ensure the model does not discard candidates above the requested export
    # threshold internally. This still respects model NMS and max_per_img.
    test_cfg = getattr(model.roi_head, 'test_cfg', None)
    if test_cfg is not None:
        test_cfg['score_thr'] = args.score_thr

    pedestrian_id = NUIM_CLASSES.index('pedestrian')
    saved_files = []

    for scene_name in scene_names:
        samples = list(scene_samples(nusc, scene_name))
        if args.scene.lower() == 'all':
            # In all mode, --output denotes an output DIRECTORY.
            out_dir = args.output or Path('results')
            if out_dir.suffix.lower() == '.json':
                raise ValueError('When --scene all is used, --output must be a directory, not a .json file')
            out_path = out_dir / f'{args.model}_{scene_name}.json'
        else:
            out_path = args.output or Path(f'{args.model}_{scene_name}.json')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frames = []

        for frame_idx, sample in samples:
            sd = nusc.get('sample_data', sample['data']['CAM_FRONT'])
            img_path = Path(nusc.dataroot) / sd['filename']
            if not img_path.is_file():
                raise FileNotFoundError(f'CAM_FRONT image missing: {img_path}')

            result = inference_detector(model, str(img_path))
            instances = result.pred_instances.cpu().numpy()
            detections = []
            for label, score, bbox in zip(instances.labels, instances.scores, instances.bboxes):
                if int(label) != pedestrian_id or float(score) < args.score_thr:
                    continue
                detections.append({
                    'bbox': [float(v) for v in bbox.tolist()],
                    'score': float(score),
                    'category': 'pedestrian',
                })
            frames.append({'frame_idx': frame_idx, 'detections': detections})
            print(f'{scene_name}: {frame_idx + 1}/{len(samples)} - '
                  f'{len(detections)} pedestrian predictions')

            if args.save_vis:
                from PIL import Image, ImageDraw
                vis_dir = out_path.parent / f'{scene_name}_{args.model}_vis'
                vis_dir.mkdir(parents=True, exist_ok=True)
                with Image.open(img_path) as f:
                    canvas = f.convert('RGB')
                draw = ImageDraw.Draw(canvas)
                for det in detections:
                    draw.rectangle(det['bbox'], outline='red', width=2)
                    draw.text(tuple(det['bbox'][:2]), f"{det['score']:.2f}", fill='red')
                canvas.save(vis_dir / f'{frame_idx:03d}.jpg')

        payload = {
            'model': args.model,
            'checkpoint': args.checkpoint.name,
            'score_threshold': args.score_thr,
            'scene': scene_name,
            'frames': frames,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        saved_files.append(out_path)
        print(f'Saved: {out_path}')

    print(f'Completed {len(saved_files)} scene(s)')
    return saved_files


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path,
                        default=Path(r'E:\Linux-ubuntu\datasets\nuscenes_mini'))
    parser.add_argument('--version', default='v1.0-mini')
    parser.add_argument('--scene', default='scene-0061',
                        help="Scene name or 'all' for every scene")
    parser.add_argument('--model', choices=list(MODEL_CONFIGS), default='htc')
    parser.add_argument('--config', type=Path, default=r'htc_r50_nuimages_mmdet3.py',
                        help='Optional MMDetection 3.x config; defaults by --model')
    parser.add_argument('--checkpoint', type=Path, default=r'D:\my_phd\on_git\a_Work2\data_and_results\pretrained_weights/htc_r50_fpn_coco-20e_1x_nuim_20201010_070203-0b53a65e.pth')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--score-thr', type=float, default=0.001,
                        help='Export threshold; use a low threshold for AP')
    parser.add_argument('--output', type=Path, default=Path(r'D:\my_phd\on_git\a_Work2\CriticalVideoObjectDetection_V1\scripts\nuimages_model_compare\htc'),
                        help='JSON filepath for one scene; output directory when --scene all')
    parser.add_argument('--save-vis', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.score_thr <= 1:
        parser.error('--score-thr must be between 0 and 1')
    return args


if __name__ == '__main__':
    run(parse_args())
