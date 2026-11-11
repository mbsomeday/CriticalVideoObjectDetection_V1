"""Export all CAM_FRONT-projectable pedestrians per nuScenes scene and keyframe.

The 3D -> 2D projection follows the nuScenes devkit's
scripts/export_2d_annotations_as_json.py (global -> ego -> camera,
project box corners, and clip their convex hull to the image canvas).

Reachset / TTC functions are preserved from the supplied nuscenes_scene_anno.py:
causal backward velocity and constant-velocity circular reachable sets.
TTC is the earliest *possible* overlap of conservative reachable sets, not
an actual collision time or a guarantee of a future collision.

Default behavior:
- All available human.pedestrian.* annotations projected into CAM_FRONT.
- No distance or visibility_token filtering.
- All scene keyframes are retained, including frames with zero pedestrians.
- TTC starts at frame_idx 2 (third frame); earlier TTC values are null.
- Output: one JSON per scene, with frame-centric pedestrian records.

Usage:
    python nuscenes_scene_frames.py --scene scene-0061
    python nuscenes_scene_frames.py --scene all
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# Same method as the uploaded official script (nuscene_2dT3d.py).
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.scripts.export_2d_annotations_as_json import post_process_coords
from nuscenes.utils.geometry_utils import view_points


# ---------- TTC and reachsets (from uploaded nuscenes_scene_anno.py) ----------

def causal_velocity(xy, timestamp_s, history, lookback_s=1.0):
    """Backward difference using up to 1 second of past positions."""
    valid = [(t, p) for t, p in history
             if 1e-6 < timestamp_s - t <= lookback_s + 0.05]
    if not valid:
        return np.zeros(2, dtype=float), False
    past_t, past_xy = min(valid, key=lambda item: item[0])
    return (xy - past_xy) / (timestamp_s - past_t), True


def reachset_cv(state, *, horizon_s, dt_s, velocity_error_mps,
                accel_bound_mps2, unknown_speed_bound_mps):
    """Return circular reachable sets at discrete times in [0, horizon_s]."""
    if horizon_s <= 0 or dt_s <= 0:
        raise ValueError("horizon_s and dt_s must be positive")
    if min(velocity_error_mps, accel_bound_mps2, unknown_speed_bound_mps) < 0:
        raise ValueError("uncertainty bounds must be nonnegative")
    n_steps = int(np.ceil(horizon_s / dt_s))
    times = np.linspace(0.0, horizon_s, n_steps + 1)
    xy = np.asarray(state["xy"], dtype=float)
    velocity = np.asarray(state["velocity"], dtype=float)
    centers = xy[None, :] + times[:, None] * velocity[None, :]
    radii = (float(state["footprint_radius_m"])
             + velocity_error_mps * times
             + 0.5 * accel_bound_mps2 * times ** 2)
    if not state["velocity_known"]:
        radii += unknown_speed_bound_mps * times
    return {
        "times_s": times,
        "centers_xy_m": centers,
        "radii_m": radii,
        "velocity_known": bool(state["velocity_known"]),
    }


def predict_ego_reachset(state, *, horizon_s=3.0, dt_s=0.1,
                         velocity_error_mps=0.3, accel_bound_mps2=1.0,
                         unknown_speed_bound_mps=15.0):
    return reachset_cv(
        state, horizon_s=horizon_s, dt_s=dt_s,
        velocity_error_mps=velocity_error_mps,
        accel_bound_mps2=accel_bound_mps2,
        unknown_speed_bound_mps=unknown_speed_bound_mps,
    )


def predict_pedestrian_reachset(state, *, horizon_s=3.0, dt_s=0.1,
                                velocity_error_mps=0.2, accel_bound_mps2=0.7,
                                unknown_speed_bound_mps=3.0):
    return reachset_cv(
        state, horizon_s=horizon_s, dt_s=dt_s,
        velocity_error_mps=velocity_error_mps,
        accel_bound_mps2=accel_bound_mps2,
        unknown_speed_bound_mps=unknown_speed_bound_mps,
    )


def earliest_possible_ttc(ego_reach, ped_reach):
    """First sampled time when circular ego/ped reachable sets intersect."""
    if not np.allclose(ego_reach["times_s"], ped_reach["times_s"]):
        raise ValueError("Reachset time grids must be identical")
    distances = np.linalg.norm(
        ego_reach["centers_xy_m"] - ped_reach["centers_xy_m"], axis=1
    )
    overlaps = distances <= ego_reach["radii_m"] + ped_reach["radii_m"]
    indices = np.flatnonzero(overlaps)
    return float(ego_reach["times_s"][indices[0]]) if indices.size else None


# ---------- 3D -> 2D (same transforms/projection as official export) ----------

def project_pedestrian_boxes(nusc, sample, cam_token):
    """Return (annotation, 2D bbox) for all pedestrian GT boxes in CAM_FRONT.

    Uses the official projection algorithm, rather than a pre-exported
    image_annotations.json file. Bounding boxes are xyxy, clipped to image.
    A pedestrian with no valid in-image projection cannot be exported.
    """
    sd_rec = nusc.get("sample_data", cam_token)
    if sd_rec["sensor_modality"] != "camera" or not sd_rec["is_key_frame"]:
        raise ValueError("Expected a camera keyframe sample_data token")
    cs_rec = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_rec = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    camera_intrinsic = np.asarray(cs_rec["camera_intrinsic"], dtype=float)
    image_size = (int(sd_rec["width"]), int(sd_rec["height"]))

    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        if not ann["category_name"].startswith("human.pedestrian."):
            continue

        # The following transformations match the supplied official script.
        box = nusc.get_box(ann_token)  # fresh 3D box in global coordinates
        box.translate(-np.asarray(pose_rec["translation"], dtype=float))
        box.rotate(Quaternion(pose_rec["rotation"]).inverse)
        box.translate(-np.asarray(cs_rec["translation"], dtype=float))
        box.rotate(Quaternion(cs_rec["rotation"]).inverse)

        corners_3d = box.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]
        if corners_3d.shape[1] < 3:
            # Degenerate polygon: insufficient front-facing corners.
            continue

        corner_coords = view_points(
            corners_3d, camera_intrinsic, normalize=True
        ).T[:, :2].tolist()
        # Same convex-hull/intersection method as official post_process_coords.
        coords = post_process_coords(corner_coords, imsize=image_size)
        if coords is None:
            continue
        x1, y1, x2, y2 = [float(v) for v in coords]
        if x2 <= x1 or y2 <= y1:
            continue
        yield ann, [x1, y1, x2, y2]


# ---------- Per-frame export ----------

def analyze_scene(nusc, my_scene, *, horizon_s=3.0, dt_s=0.1,
                  history_s=1.0, ttc_start_frame=2,
                  ego_length_m=4.7, ego_width_m=2.0,
                  ego_reachset_fn=predict_ego_reachset,
                  ped_reachset_fn=predict_pedestrian_reachset):
    """Produce one scene -> frames -> pedestrians JSON-serializable dictionary."""
    if history_s <= 0 or ttc_start_frame < 0:
        raise ValueError("history_s must be positive; ttc_start_frame >= 0")

    output = {"scene": my_scene["name"], "frames": []}
    ego_history = []
    ped_history = defaultdict(list)
    ego_radius = 0.5 * float(np.hypot(ego_length_m, ego_width_m))

    sample_token = my_scene["first_sample_token"]
    frame_idx = 0
    while sample_token:
        sample = nusc.get("sample", sample_token)
        sample_t = float(sample["timestamp"]) * 1e-6
        cam_token = sample["data"]["CAM_FRONT"]
        cam_sd = nusc.get("sample_data", cam_token)

        # Use LIDAR_TOP-aligned ego pose as in the uploaded TTC code.
        lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
        ego_xy = np.asarray(ego_pose["translation"][:2], dtype=float)
        ego_t = float(ego_pose["timestamp"]) * 1e-6
        ego_v, ego_known = causal_velocity(ego_xy, ego_t, ego_history, history_s)
        ego_state = {
            "xy": ego_xy,
            "velocity": ego_v,
            "velocity_known": ego_known,
            "footprint_radius_m": ego_radius,
        }
        ego_reach = (ego_reachset_fn(ego_state, horizon_s=horizon_s, dt_s=dt_s)
                     if frame_idx >= ttc_start_frame else None)

        ego_history.append((ego_t, ego_xy))
        ego_history = [(t, xy) for t, xy in ego_history
                       if ego_t - t <= history_s + 0.05]

        frame_output = {
            "frame_idx": frame_idx,
            "front_cam": cam_sd["filename"],
            "pedestrians": [],
        }
        for ann, bbox in project_pedestrian_boxes(nusc, sample, cam_token):
            pid = ann["instance_token"]
            ped_xy = np.asarray(ann["translation"][:2], dtype=float)
            distance = float(np.linalg.norm(ped_xy - ego_xy))

            ped_v, ped_known = causal_velocity(
                ped_xy, sample_t, ped_history[pid], history_s
            )
            width_m, length_m = map(float, ann["size"][:2])
            ped_state = {
                "xy": ped_xy,
                "velocity": ped_v,
                "velocity_known": ped_known,
                "footprint_radius_m": 0.5 * float(np.hypot(width_m, length_m)),
            }
            ttc = None
            if ego_reach is not None:
                ped_reach = ped_reachset_fn(
                    ped_state, horizon_s=horizon_s, dt_s=dt_s
                )
                ttc = earliest_possible_ttc(ego_reach, ped_reach)

            x1, y1, x2, y2 = bbox
            frame_output["pedestrians"].append({
                "Instance": pid,
                "bbox": [round(v, 2) for v in bbox],
                "bbox_area": round((x2 - x1) * (y2 - y1), 2),
                "visibility_token": int(ann["visibility_token"]),
                "distance/m": round(distance, 2),
                "ttc/s": None if ttc is None else round(ttc, 2),
            })

            # Match existing history policy: only FOV-projectable GT observations.
            ped_history[pid].append((sample_t, ped_xy))
            ped_history[pid] = [(t, xy) for t, xy in ped_history[pid]
                                if sample_t - t <= history_s + 0.05]

        output["frames"].append(frame_output)
        sample_token = sample["next"]
        frame_idx += 1

    if frame_idx != int(my_scene["nbr_samples"]):
        raise ValueError(
            f"Scene {my_scene['name']}: traversed {frame_idx} samples, "
            f"expected {my_scene['nbr_samples']}"
        )
    return output


def save_scene_json(result, output_dir="./results_frames"):
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{result['scene']}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    return path


def main():
    parser = argparse.ArgumentParser(description="nuScenes CAM_FRONT per-frame pedestrian annotations")
    parser.add_argument("--dataroot", default=r"E:\Linux-ubuntu\datasets\nuscenes_mini")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--scene", default="all", help="scene name (e.g. scene-0061) or 'all'")
    parser.add_argument("--output-dir", default="./results_frames")
    parser.add_argument("--horizon-s", type=float, default=3.0)
    parser.add_argument("--dt-s", type=float, default=0.1)
    parser.add_argument("--ttc-start-frame", type=int, default=2)
    args = parser.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    scenes = (nusc.scene if args.scene == "all"
              else [scene for scene in nusc.scene if scene["name"] == args.scene])
    if not scenes:
        raise ValueError(f"Scene not found: {args.scene}")

    for scene in scenes:
        result = analyze_scene(
            nusc, scene,
            horizon_s=args.horizon_s,
            dt_s=args.dt_s,
            ttc_start_frame=args.ttc_start_frame,
        )
        path = save_scene_json(result, args.output_dir)
        count = sum(len(frame["pedestrians"]) for frame in result["frames"])
        print(f"Saved {path} | frames={len(result['frames'])} "
              f"| pedestrian frame-observations={count}")


if __name__ == "__main__":
    main()
