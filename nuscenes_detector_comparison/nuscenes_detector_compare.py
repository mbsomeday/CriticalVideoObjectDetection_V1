#!/usr/bin/env python3
"""Compare 2D person detectors on the SAME annotated nuScenes samples.

Primary scopes: TXT states 1+2 (overall), and TXT state 2 (critical).
No intermediate camera frames and no recomputation of criticality.
Python >=3.10. See README_zh.md for installation and commands.
"""
from __future__ import annotations
import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np

CAMERAS = (
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
)


@dataclass
class GTBox:
    pid: str
    category: str
    xyxy: np.ndarray


@dataclass
class Prediction:
    xyxy: np.ndarray
    score: float


def load_importance_annotations(annotation_path: str | Path,
                                expected_frames: int) -> dict[str, dict]:
    """Parse the existing scene TXT without recomputing criticality or visibility.

    Verify that each listed track has exactly N state values, drawn from 0/1/2,
    and num_frames equals the count of nonzero states.
    """
    path = Path(annotation_path)
    if not path.is_file():
        raise FileNotFoundError(f"Precomputed importance annotation not found: {path}")
    tracks: dict[str, dict] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        fields = [s.strip() for s in raw.split(",", 3)]
        if len(fields) != 4:
            raise ValueError(f"{path}:{line_number}: expected 'pid, category, num_frames, states'")
        pid, category, count_text, states_text = fields
        if not pid or not category.startswith("human.pedestrian."):
            raise ValueError(f"{path}:{line_number}: invalid PID/category")
        if pid in tracks:
            raise ValueError(f"{path}:{line_number}: duplicate PID {pid}")
        try:
            count = int(count_text)
            states = [int(v) for v in states_text.split()]
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid integer") from exc
        if len(states) != expected_frames:
            raise ValueError(
                f"{path}:{line_number}: {pid}: got {len(states)} states, "
                f"but scene contains {expected_frames} frames"
            )
        if any(value not in (0, 1, 2) for value in states):
            raise ValueError(f"{path}:{line_number}: states must be 0, 1, or 2")
        if count != sum(value != 0 for value in states):
            raise ValueError(f"{path}:{line_number}: num_frames does not equal nonzero states")
        tracks[pid] = {
            "category": category,
            "num_frames": count,
            "states": states,
            "pred": [0] * expected_frames,
            "matched_score": [None] * expected_frames,
            "matched_iou": [None] * expected_frames,
        }
    if not tracks:
        raise ValueError(f"No pedestrian tracks in {path}")
    return tracks


def metadata_from_zip_if_needed(dataroot: Path, zip_path: str | None) -> None:
    """Only extract missing nuScenes metadata files (does not recreate labels)."""
    if not zip_path:
        return
    destination = dataroot / "v1.0-mini"
    destination.mkdir(parents=True, exist_ok=True)
    filenames = (
        "attribute.json", "calibrated_sensor.json", "category.json",
        "ego_pose.json", "instance.json", "log.json", "map.json",
        "sample.json", "sample_annotation.json", "sample_data.json",
        "scene.json", "sensor.json", "visibility.json",
    )
    with zipfile.ZipFile(zip_path) as archive:
        for name in filenames:
            target = destination / name
            if target.exists():
                continue
            with archive.open(f"v1.0-mini/{name}") as source, target.open("wb") as dest:
                shutil.copyfileobj(source, dest)


def scene_samples(nusc: Any, scene_name: str) -> tuple[dict, list[dict]]:
    scene = next((s for s in nusc.scene if scene_name in (s["name"], s["token"])), None)
    if scene is None:
        raise ValueError(f"Scene not found: {scene_name}")
    samples = []
    token = scene["first_sample_token"]
    while token:
        sample = nusc.get("sample", token)
        samples.append(sample)
        if len(samples) > scene["nbr_samples"]:
            raise RuntimeError("Scene sample chain exceeds nbr_samples")
        token = sample["next"]
    if len(samples) != scene["nbr_samples"]:
        raise RuntimeError("Scene sample chain length differs from nbr_samples")
    return scene, samples


def project_box_to_image(box: Any, intrinsic: np.ndarray,
                         width: int, height: int, near_plane: float = 0.1) -> np.ndarray | None:
    """Get a clipped 2D box from nuScenes camera-frame 3D box corners.

    A proxy 2D ground-truth box: does not account for occlusion or tight masks.
    """
    corners = np.asarray(box.corners(), dtype=np.float64).T
    if corners.shape != (8, 3):
        raise ValueError("nuScenes Box.corners() must return (3,8)")
    edges = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    points = [p for p in corners if p[2] >= near_plane]
    for i, j in edges:
        p, q = corners[i], corners[j]
        if (p[2] >= near_plane) != (q[2] >= near_plane):
            fraction = (near_plane - p[2]) / (q[2] - p[2])
            points.append(p + fraction * (q - p))
    if not points:
        return None
    xyz = np.asarray(points, dtype=np.float64).T
    projected = np.asarray(intrinsic, dtype=np.float64) @ xyz
    xy = projected[:2, :] / projected[2:3, :]
    x1, y1 = xy.min(axis=1)
    x2, y2 = xy.max(axis=1)
    x1, x2 = np.clip([x1, x2], 0, width)
    y1, y2 = np.clip([y1, y2], 0, height)
    if not np.isfinite([x1, y1, x2, y2]).all() or x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return np.array([x1, y1, x2, y2], dtype=np.float64)


def load_frame_ground_truth(nusc: Any, sample: dict, camera: str) -> tuple[Path, list[GTBox]]:
    """Read GT coordinates for IoU only. Does not assign criticality or select tracks."""
    from nuscenes.utils.geometry_utils import BoxVisibility

    camera_token = sample["data"][camera]
    img_path, boxes, intrinsic = nusc.get_sample_data(
        camera_token, box_vis_level=BoxVisibility.ANY
    )
    camera_data = nusc.get("sample_data", camera_token)
    width, height = int(camera_data["width"]), int(camera_data["height"])
    gt = []
    for box in boxes:
        if not box.name or not box.name.startswith("human.pedestrian."):
            continue
        xyxy = project_box_to_image(box, intrinsic, width, height)
        if xyxy is None:
            continue
        ann = nusc.get("sample_annotation", box.token)
        gt.append(GTBox(
            pid=ann["instance_token"], category=ann["category_name"], xyxy=xyxy
        ))
    return Path(img_path), gt


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    wh = np.maximum(0.0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    intersection = float(np.prod(wh))
    area_a = float(np.prod(np.maximum(a[2:] - a[:2], 0.0)))
    area_b = float(np.prod(np.maximum(b[2:] - b[:2], 0.0)))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def match_one_frame(predictions: list[Prediction], gt: list[GTBox],
                    iou_threshold: float, score_threshold: float = 0.0) -> dict:
    """Confidence-sorted one-to-one matching of person boxes to all pedestrian GT."""
    pred_indices = sorted(
        (i for i, p in enumerate(predictions) if p.score >= score_threshold),
        key=lambda i: (-predictions[i].score, i),
    )
    used_gt = set()
    matched = {}
    tp_pred = {}
    for i in pred_indices:
        best_j, best_iou = None, -1.0
        for j, g in enumerate(gt):
            if j in used_gt:
                continue
            iou = bbox_iou(predictions[i].xyxy, g.xyxy)
            if iou > best_iou:
                best_j, best_iou = j, iou
        valid = best_j is not None and best_iou >= iou_threshold
        tp_pred[i] = valid
        if valid:
            used_gt.add(best_j)
            matched[best_j] = (i, predictions[i].score, best_iou)
    tp = len(matched)
    return {"matched": matched, "tp": tp, "fp": len(pred_indices) - tp,
            "fn": len(gt) - tp, "tp_pred": tp_pred}



def safe_ratio(a, b):
    return a / b if b else None


def summarize_pairs(rows):
    """Rows with state 0 are not detection opportunities (not true negatives)."""
    result = {}
    for name, states in [('overall', {1, 2}), ('critical', {2}), ('noncritical', {1})]:
        chosen = [r for r in rows if r['state'] in states]
        detected = sum(r['detected'] for r in chosen)
        result[name] = dict(present=len(chosen), detected=detected,
                            missed=len(chosen)-detected,
                            recall=safe_ratio(detected, len(chosen)),
                            miss_rate=safe_ratio(len(chosen)-detected, len(chosen)))
    return result


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_frames(nusc, scene_name, camera, annotation_path):
    """Fail before model loading if inputs cannot support the agreed denominator."""
    scene, samples = scene_samples(nusc, scene_name)
    tracks = load_importance_annotations(annotation_path, len(samples))
    frames, errors, zero_with_proxy = [], [], []
    for idx, sample in enumerate(samples):
        image, gt = load_frame_ground_truth(nusc, sample, camera)
        if not image.is_file():
            errors.append(f'frame {idx+1}: missing image {image}')
        ids = [g.pid for g in gt]
        if len(set(ids)) != len(ids):
            errors.append(f'frame {idx+1}: duplicate GT instance IDs')
        by_id = {g.pid: g for g in gt}
        for pid, track in tracks.items():
            state = track['states'][idx]
            if state > 0:
                if pid not in by_id:
                    errors.append(f'frame {idx+1}, {pid}: TXT state={state}, but no projected GT')
                elif track['category'] != by_id[pid].category:
                    errors.append(f'frame {idx+1}, {pid}: category mismatch')
            elif pid in by_id:
                # TXT remains authoritative for primary scopes. A geometric
                # projected box can still exist when the person is not visible.
                zero_with_proxy.append(dict(frame_number=idx+1, pid=pid))
        sd = nusc.get('sample_data', sample['data'][camera])
        frames.append(dict(frame_index=idx, frame_number=idx+1,
                           sample_token=sample['token'], sample_data_token=sd['token'],
                           timestamp_us=sd['timestamp'], image_path=image, gt=gt))
    if errors:
        raise ValueError('Preflight failed (no metrics generated):\n' + '\n'.join(errors[:30])
                         + f'\nTotal errors: {len(errors)}')
    return scene, tracks, frames, zero_with_proxy


class Detector:
    """Normalize torchvision/HF predictions to original-image pixel xyxy."""
    def __init__(self, name, device, dfine_checkpoint, dfine_revision=None):
        import torch
        self.name, self.device = name, torch.device(device)
        if name == 'dfine_l':
            try:
                from transformers import AutoImageProcessor, DFineForObjectDetection
            except ImportError as exc:
                raise RuntimeError('D-FINE requires transformers>=4.52,<5; install requirements.txt') from exc
            kwargs = {'revision': dfine_revision} if dfine_revision else {}
            self.preprocess = AutoImageProcessor.from_pretrained(dfine_checkpoint, **kwargs)
            self.model = DFineForObjectDetection.from_pretrained(dfine_checkpoint, **kwargs).to(self.device).eval()
            person = [int(k) for k, v in self.model.config.id2label.items() if v.lower() == 'person']
            if len(person) != 1:
                raise ValueError('Checkpoint must contain exactly one class named person')
            self.person_class = person[0]
            self.metadata = dict(backend='transformers', checkpoint=dfine_checkpoint,
                                 revision=dfine_revision,
                                 resolved_commit=getattr(self.model.config, '_commit_hash', None),
                                 preprocessing=self.preprocess.to_dict())
        else:
            from torchvision.models.detection import (
                ssd300_vgg16, SSD300_VGG16_Weights,
                fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights)
            if name == 'ssd':
                weights = SSD300_VGG16_Weights.DEFAULT
                self.model = ssd300_vgg16(weights=weights).to(self.device).eval()
            elif name == 'fasterrcnn':
                weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
                self.model = fasterrcnn_resnet50_fpn_v2(weights=weights).to(self.device).eval()
            else:
                raise ValueError(f'Unknown detector: {name}')
            self.preprocess = weights.transforms()
            self.person_class = weights.meta['categories'].index('person')
            self.metadata = dict(backend='torchvision', checkpoint=str(weights), weights_url=weights.url,
                                 preprocessing=repr(self.model.transform))
        self.metadata.update(model=name, person_class=self.person_class, device=str(self.device),
                             dtype='float32', native_postprocessing=True)

    def predict(self, image_path):
        import torch
        from PIL import Image
        with Image.open(image_path) as im:
            image = im.convert('RGB')
        width, height = image.size
        with torch.inference_mode():
            if self.name == 'dfine_l':
                inputs = self.preprocess(images=image, return_tensors='pt').to(self.device)
                output = self.model(**inputs)
                output = self.preprocess.post_process_object_detection(
                    output, target_sizes=[(height, width)], threshold=0.0)[0]
            else:
                output = self.model([self.preprocess(image).to(self.device)])[0]
        keep = output['labels'] == self.person_class
        boxes = output['boxes'][keep].detach().cpu().numpy()
        scores = output['scores'][keep].detach().cpu().numpy()
        predictions = []
        for box, score in zip(boxes, scores):
            box = np.asarray(box, dtype=np.float64).copy()
            if not np.isfinite(box).all() or not np.isfinite(score):
                raise ValueError(f'Nonfinite prediction from {self.name}: {image_path}')
            box[[0, 2]] = box[[0, 2]].clip(0, width)
            box[[1, 3]] = box[[1, 3]].clip(0, height)
            if box[2] > box[0] and box[3] > box[1]:
                predictions.append(Prediction(box, float(score)))
        return predictions


def evaluate_predictions(frames, tracks, predictions, score_threshold, iou_threshold):
    """Match ONCE against all geometric pedestrian GT; then subset by TXT.

    This preserves the original evaluator's matching convention. Unlisted
    people and state-0 projected boxes may consume a detection, but cannot
    inflate the TXT recall. Auxiliary precision uses all projected GT.
    """
    if len(frames) != len(predictions):
        raise ValueError('Prediction/frame count mismatch')
    pair_rows, frame_rows = [], []
    for frame, pred in zip(frames, predictions):
        idx, gt = frame['frame_index'], frame['gt']
        by_id = {g.pid: j for j, g in enumerate(gt)}
        matched = match_one_frame(pred, gt, iou_threshold, score_threshold)
        rows = []
        for pid, track in tracks.items():
            state = track['states'][idx]
            if state > 0 and pid not in by_id:
                raise ValueError(f'Missing GT: {pid}, frame {idx+1}')
            hit = matched['matched'].get(by_id.get(pid)) if state else None
            rows.append(dict(pid=pid, category=track['category'], frame_index=idx,
                             frame_number=idx+1, sample_token=frame['sample_token'],
                             timestamp_us=frame['timestamp_us'], state=state,
                             detected=int(hit is not None),
                             matched_score=hit[1] if hit else None,
                             matched_iou=hit[2] if hit else None))
        stats = summarize_pairs(rows)
        fr = dict(frame_index=idx, frame_number=idx+1, sample_token=frame['sample_token'],
                  timestamp_us=frame['timestamp_us'], all_camera_gt=len(gt),
                  all_camera_tp=matched['tp'], all_camera_fp=matched['fp'], all_camera_fn=matched['fn'])
        for scope, values in stats.items():
            fr.update({scope+'_'+key: value for key, value in values.items()})
        frame_rows.append(fr)
        pair_rows.extend(rows)
    metrics = summarize_pairs(pair_rows)
    tp = sum(r['all_camera_tp'] for r in frame_rows)
    fp = sum(r['all_camera_fp'] for r in frame_rows)
    fn = sum(r['all_camera_fn'] for r in frame_rows)
    metrics['all_camera_auxiliary'] = dict(TP=tp, FP=fp, FN=fn,
                                          precision=safe_ratio(tp, tp+fp), recall=safe_ratio(tp, tp+fn))
    return metrics, pair_rows, frame_rows


def save_overlay(frame, predictions, tracks, pair_rows, threshold, output):
    from PIL import Image, ImageDraw
    with Image.open(frame['image_path']) as im:
        image = im.convert('RGB')
    draw = ImageDraw.Draw(image)
    idx = frame['frame_index']
    hits = {r['pid']: r['detected'] for r in pair_rows if r['frame_index'] == idx}
    for g in frame['gt']:
        state = tracks.get(g.pid, {}).get('states', [0]*(idx+1))[idx]
        if not state:
            continue
        color = 'red' if state == 2 else 'orange'
        draw.rectangle(g.xyxy.tolist(), outline=color, width=3)
        draw.text((float(g.xyxy[0]), float(g.xyxy[1])),
                  f'{g.pid[:6]} state={state} hit={hits[g.pid]}', fill=color)
    for p in predictions:
        if p.score >= threshold:
            draw.rectangle(p.xyxy.tolist(), outline='lime', width=2)
            draw.text((float(p.xyxy[0]), max(0, float(p.xyxy[1])-12)), f'{p.score:.3f}', fill='lime')
    image.save(output)


def package_versions():
    versions = {}
    for name in ['torch', 'torchvision', 'transformers', 'nuscenes-devkit', 'numpy', 'Pillow']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataroot', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--scene', default='scene-0061')
    parser.add_argument('--version', default='v1.0-mini')
    parser.add_argument('--camera', choices=CAMERAS, default='CAM_FRONT')
    parser.add_argument('--models', nargs='+', choices=['ssd', 'fasterrcnn', 'dfine_l'],
                        default=['ssd', 'fasterrcnn', 'dfine_l'])
    parser.add_argument('--dfine-checkpoint', default='ustc-community/dfine-large-coco')
    parser.add_argument('--dfine-revision', default=None)
    parser.add_argument('--score-threshold', type=float, default=0.3)
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda or cuda:0')
    parser.add_argument('--outdir', type=Path, default=Path('detector_comparison_results'))
    parser.add_argument('--metadata-zip', default=None, help='Optional v1.0-mini metadata ZIP')
    parser.add_argument('--validate-only', action='store_true', help='Check metadata, TXT and images without loading detectors')
    parser.add_argument('--save-overlays', action='store_true')
    args = parser.parse_args()
    if not (0 <= args.score_threshold <= 1 and 0 < args.iou_threshold <= 1):
        parser.error('score threshold must be in [0,1], IoU threshold in (0,1]')
    if args.metadata_zip and args.version != 'v1.0-mini':
        parser.error('--metadata-zip supports v1.0-mini only')
    metadata_from_zip_if_needed(args.dataroot, args.metadata_zip)
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=args.version, dataroot=str(args.dataroot), verbose=False)
    scene, tracks, frames, zero_with_proxy = prepare_frames(nusc, args.scene, args.camera, args.annotations)
    args.outdir.mkdir(parents=True, exist_ok=True)
    annotation_hash = hashlib.sha256(args.annotations.read_bytes()).hexdigest()
    write_json(args.outdir/'input_manifest.json', dict(
        scene=scene['name'], version=args.version, camera=args.camera,
        annotation_sha256=annotation_hash, num_frames=len(frames),
        num_tracks=len(tracks), overall_present=sum(t['num_frames'] for t in tracks.values()),
        critical_present=sum(t['states'].count(2) for t in tracks.values()),
        state_zero_with_projected_gt=zero_with_proxy,
        frames=[{k: str(v) if isinstance(v, Path) else v for k,v in f.items() if k != 'gt'} for f in frames]))
    print(f'Validated {len(frames)} frames, {len(tracks)} tracks; TXT states preserved.')
    if args.validate_only:
        return
    import torch
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA is not available. Use --device cpu or install a compatible CUDA PyTorch build.')
    comparison = []
    for name in dict.fromkeys(args.models):
        print(f'\nLoading {name} on {device} ...', flush=True)
        detector = Detector(name, device, args.dfine_checkpoint, args.dfine_revision)
        model_dir = args.outdir/name
        model_dir.mkdir(parents=True, exist_ok=True)
        predictions = []
        for f in frames:
            predictions.append(detector.predict(f['image_path']))
            print(f"{name}: frame {f['frame_number']}/{len(frames)}", flush=True)
        metrics, pairs, frame_rows = evaluate_predictions(
            frames, tracks, predictions, args.score_threshold, args.iou_threshold)
        write_json(model_dir/'metrics.json', dict(scene=scene['name'], camera=args.camera,
            num_frames=len(frames), annotation_sha256=annotation_hash,
            settings=dict(**detector.metadata, score_threshold=args.score_threshold,
                          iou_threshold=args.iou_threshold,
                          matching='confidence-sorted 1:1 matching against all projected pedestrian GT, then subset by TXT',
                          gt_source='projected 3D box; proxy 2D GT, not occlusion-aware',
                          scope='overall=TXT 1+2; critical=TXT 2; absent=TXT 0 excluded'),
            package_versions=package_versions(), metrics=metrics))
        write_csv(model_dir/'pedestrian_frames.csv', pairs)
        write_csv(model_dir/'frame_metrics.csv', frame_rows)
        write_json(model_dir/'predictions.json', [dict(
            sample_token=f['sample_token'], sample_data_token=f['sample_data_token'],
            predictions=[dict(xyxy=p.xyxy.tolist(), score=p.score) for p in pred])
            for f, pred in zip(frames, predictions)])
        # Same row order and 39 binary flags as the original TXT output convention.
        with (model_dir/'detection_flags.txt').open('w', encoding='utf-8') as stream:
            for pid, track in tracks.items():
                flags = [r['detected'] for r in pairs if r['pid'] == pid]
                stream.write(f"{pid}, {track['category']}, {track['num_frames']}, "
                             + ' '.join(map(str, flags)) + '\n')
        if args.save_overlays:
            overlay_dir = model_dir/'overlays'
            overlay_dir.mkdir(exist_ok=True)
            for f, pred in zip(frames, predictions):
                save_overlay(f, pred, tracks, pairs, args.score_threshold,
                             overlay_dir/f"frame_{f['frame_number']:03}.jpg")
        row = dict(model=name, checkpoint=detector.metadata['checkpoint'],
                   score_threshold=args.score_threshold, iou_threshold=args.iou_threshold,
                   num_frames=len(frames))
        for scope in ['overall', 'critical', 'noncritical']:
            row.update({scope+'_'+k: v for k,v in metrics[scope].items()})
        row['all_camera_precision_auxiliary'] = metrics['all_camera_auxiliary']['precision']
        comparison.append(row)
        write_csv(args.outdir/'comparison.csv', comparison)
        write_json(args.outdir/'comparison.json', comparison)
        print(json.dumps(metrics, indent=2))
        del detector
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f'\nComparison saved to {(args.outdir/"comparison.csv").resolve()}')


if __name__ == '__main__':
    main()
