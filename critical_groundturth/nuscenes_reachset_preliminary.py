"""nuScenes mini: conservative 2D reach-set TTC for pedestrians (preliminary experiment).

Requirements: numpy, nuscenes-devkit. Python >=3.10.

The circles produced by functions 3 and 4 are conservative planar over-approximations
of oriented object footprints under constant-velocity nominal motion and bounded
velocity/acceleration error. Thus the reported TTC is the earliest *possible*
intersection at the chosen time resolution, NOT a guaranteed crash time.

All velocities are estimated causally (no access to future annotations).
Only camera-FOV pedestrian frames are classified; non-FOV frames remain 0.
Each qualifying pedestrian can be exported as one line in a per-scene TXT file.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from nuscenes.utils.geometry_utils import BoxVisibility

# Keep the uploaded script's default: front camera only. Pass all six to use all cameras.
DEFAULT_CAMERAS = ("CAM_FRONT",)


def _scene_samples(nusc: Any, scene: dict) -> list[dict]:
    """Enumerate annotation samples in chronological order."""
    samples = []
    token = scene["first_sample_token"]
    while token:
        sample = nusc.get("sample", token)
        samples.append(sample)
        if len(samples) > scene["nbr_samples"]:
            raise ValueError("Scene sample chain exceeds scene['nbr_samples']")
        token = sample["next"]
    if len(samples) != scene["nbr_samples"]:
        raise ValueError("Scene sample chain length does not equal nbr_samples")
    return samples


def _get_scene(nusc: Any, scene_name_or_token: str) -> dict:
    for scene in nusc.scene:
        if scene_name_or_token in (scene["name"], scene["token"]):
            return scene
    raise ValueError(f"Scene not found: {scene_name_or_token}")


def function2_select_pedestrians(
    nusc: Any,
    samples: list[dict],
    min_frames_exclusive: int = 3,
    cameras: tuple[str, ...] = DEFAULT_CAMERAS,
) -> list[dict]:
    """Select pedestrian instances inside a selected camera FOV in >3 keyframes.

    Each instance is counted at most once per sample, even if multiple cameras
    contain its 3D bounding box. `visible_frame_tokens` is used by function 1
    so frames outside the camera FOV are labeled 0.

    BoxVisibility.ANY checks geometry in the camera image, not occlusion.
    """
    if not cameras:
        raise ValueError("At least one camera must be specified")
    if min_frames_exclusive < 0:
        raise ValueError("min_frames_exclusive must be nonnegative")

    pedestrians: dict[str, dict] = {}
    for sample in samples:
        visible_ann_tokens: set[str] = set()
        for camera in cameras:
            cam_token = sample["data"].get(camera)
            if cam_token is None:
                continue
            _, boxes, _ = nusc.get_sample_data(
                cam_token,
                box_vis_level=BoxVisibility.ANY,
            )
            for box in boxes:
                if box.name is not None and box.name.startswith("human.pedestrian."):
                    visible_ann_tokens.add(box.token)

        # Intersect with this sample's annotations so each instance is counted
        # once per keyframe even when it appears in multiple camera images.
        for ann_token in sample["anns"]:
            if ann_token not in visible_ann_tokens:
                continue
            ann = nusc.get("sample_annotation", ann_token)
            category = ann["category_name"]
            if not category.startswith("human.pedestrian."):
                continue
            instance = ann["instance_token"]
            if instance not in pedestrians:
                pedestrians[instance] = {
                    "instance_token": instance,
                    "first_frame_token": sample["token"],
                    "first_annotation_token": ann_token,
                    "category": category,
                    "num_frames": 0,
                    "visible_frame_tokens": [],
                }
            rec = pedestrians[instance]
            rec["num_frames"] += 1
            rec["visible_frame_tokens"].append(sample["token"])

    return [
        rec for rec in pedestrians.values()
        if rec["num_frames"] > min_frames_exclusive
    ]


def _causal_velocity(xy: np.ndarray, timestamp_s: float,
                     history: list[tuple[float, np.ndarray]],
                     lookback_s: float = 1.0) -> tuple[np.ndarray, bool]:
    """Backward finite difference across up to one second of past observations.

    This is valid even when a track disappears for a short period. Larger gaps
    are not bridged; in that case the velocity is treated as unknown.
    """
    valid = [(t, p) for (t, p) in history
             if 1e-6 < timestamp_s - t <= lookback_s + 0.05]
    if not valid:
        return np.zeros(2, dtype=float), False
    past_t, past_xy = min(valid, key=lambda pair: pair[0])
    return (xy - past_xy) / (timestamp_s - past_t), True


def _reachset_cv(state: dict, *, horizon_s: float, dt_s: float,
                 velocity_error_mps: float, accel_bound_mps2: float,
                 unknown_speed_bound_mps: float) -> dict:
    """Over-approximating circular reach sets, evaluated at discrete future times.

    centers(t) = current_xy + nominal_velocity * t
    radius(t)  = footprint_radius + velocity_error*t + 0.5*accel_bound*t^2
    If velocity is unknown, an additional unknown_speed_bound*t is included.
    All lengths in meters, times in seconds.
    """
    if horizon_s <= 0 or dt_s <= 0 or velocity_error_mps < 0 or accel_bound_mps2 < 0:
        raise ValueError("Expected positive horizon/dt and nonnegative bounds")
    if unknown_speed_bound_mps < 0:
        raise ValueError("unknown_speed_bound_mps must be nonnegative")
    n_steps = int(np.ceil(horizon_s / dt_s))
    times = np.linspace(0.0, horizon_s, n_steps + 1)
    xy = np.asarray(state["xy"], dtype=float)
    v = np.asarray(state["velocity"], dtype=float)
    centers = xy[None, :] + times[:, None] * v[None, :]
    radius = (float(state["footprint_radius_m"])
              + velocity_error_mps * times
              + 0.5 * accel_bound_mps2 * times ** 2)
    if not state["velocity_known"]:
        radius = radius + unknown_speed_bound_mps * times
    return {
        "times_s": times,
        "centers_xy_m": centers,
        "radii_m": radius,
        "velocity_known": bool(state["velocity_known"]),
    }


def function3_predict_ego_reachset(state: dict, *, horizon_s: float = 3.0,
                                   dt_s: float = 0.1,
                                   velocity_error_mps: float = 0.3,
                                   accel_bound_mps2: float = 1.0,
                                   unknown_speed_bound_mps: float = 15.0) -> dict:
    """Predict ego vehicle's 2D reachable sets for the next 3 seconds.

    The footprint radius is derived from the assumed ego dimensions in state.
    Defaults are EXAMPLE assumptions, not learned/calibrated safety bounds.
    """
    return _reachset_cv(state, horizon_s=horizon_s, dt_s=dt_s,
                        velocity_error_mps=velocity_error_mps,
                        accel_bound_mps2=accel_bound_mps2,
                        unknown_speed_bound_mps=unknown_speed_bound_mps)


def function4_predict_pedestrian_reachset(state: dict, *, horizon_s: float = 3.0,
                                          dt_s: float = 0.1,
                                          velocity_error_mps: float = 0.2,
                                          accel_bound_mps2: float = 0.7,
                                          unknown_speed_bound_mps: float = 3.0) -> dict:
    """Predict pedestrian's 2D reachable sets for the next 3 seconds."""
    return _reachset_cv(state, horizon_s=horizon_s, dt_s=dt_s,
                        velocity_error_mps=velocity_error_mps,
                        accel_bound_mps2=accel_bound_mps2,
                        unknown_speed_bound_mps=unknown_speed_bound_mps)


def _earliest_possible_ttc_s(ego_reachset: dict, ped_reachset: dict) -> float | None:
    """First future time with intersection of 2D circular reachable sets."""
    if not np.allclose(ego_reachset["times_s"], ped_reachset["times_s"]):
        raise ValueError("Reach-set time grids must be identical")
    distances = np.linalg.norm(
        ego_reachset["centers_xy_m"] - ped_reachset["centers_xy_m"], axis=1)
    intersects = distances <= (ego_reachset["radii_m"] + ped_reachset["radii_m"])
    indices = np.flatnonzero(intersects)
    if indices.size == 0:
        return None
    return float(ego_reachset["times_s"][indices[0]])


def function1_classify_scene(nusc: Any, scene_name_or_token: str, *,
                             ttc_threshold_s: float = 3.0,
                             horizon_s: float = 3.0,
                             pred_dt_s: float = 0.1,
                             history_s: float = 1.0,
                             ego_length_m: float = 4.7,
                             ego_width_m: float = 2.0,
                             min_frames_exclusive: int = 3,
                             cameras: tuple[str, ...] = DEFAULT_CAMERAS) -> dict:
    """Compute camera-FOV pedestrian safety-critical labels for each scene frame.

    0 = no camera-FOV pedestrian annotation at this sample
        (the object may exist in nuScenes GT outside the selected camera FOV)
    1 = visible in camera FOV, no predicted intersection within TTC threshold
    2 = visible in camera FOV, reach sets may intersect before TTC threshold

    `num_frames` is the number of camera-FOV-visible annotated frames, not the
    number of frames where a pedestrian exists anywhere in the scene.
    """
    if ttc_threshold_s <= 0 or horizon_s <= 0 or history_s <= 0:
        raise ValueError("TTC threshold, horizon and history must be positive")
    if ttc_threshold_s > horizon_s:
        raise ValueError("horizon_s must cover ttc_threshold_s")
    if ego_length_m <= 0 or ego_width_m <= 0:
        raise ValueError("Ego dimensions must be positive")

    scene = _get_scene(nusc, scene_name_or_token)
    samples = _scene_samples(nusc, scene)
    pedestrians = function2_select_pedestrians(
        nusc,
        samples,
        min_frames_exclusive=min_frames_exclusive,
        cameras=cameras,
    )
    n_frames = len(samples)
    result = {
        "scene_name": scene["name"],
        "scene_token": scene["token"],
        "num_frames": n_frames,
        "frame_tokens": [sample["token"] for sample in samples],
        "pedestrians": {},
    }

    # Pre-index visible instance IDs by sample token. A pedestrian outside the
    # selected camera FOV must stay at state 0, even if present in GT annotations.
    visible_by_sample: dict[str, set[str]] = defaultdict(set)
    for p in pedestrians:
        pid = p["instance_token"]
        result["pedestrians"][pid] = {
            **p,
            "states": [0] * n_frames,
            "ttc_seconds": [None] * n_frames,
            "velocity_known": [None] * n_frames,
        }
        for frame_token in p["visible_frame_tokens"]:
            visible_by_sample[frame_token].add(pid)

    ego_history: list[tuple[float, np.ndarray]] = []
    ped_history: dict[str, list[tuple[float, np.ndarray]]] = defaultdict(list)
    ego_radius = 0.5 * float(np.hypot(ego_length_m, ego_width_m))

    for frame_idx, sample in enumerate(samples):
        # Ego GT pose from LIDAR_TOP keyframe, in global coordinates.
        lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
        ego_xy = np.asarray(pose["translation"][:2], dtype=float)
        ego_t = float(pose["timestamp"]) * 1e-6
        ego_v, ego_known = _causal_velocity(ego_xy, ego_t, ego_history, history_s)
        ego_state = {
            "xy": ego_xy,
            "velocity": ego_v,
            "velocity_known": ego_known,
            "footprint_radius_m": ego_radius,
        }
        ego_reach = function3_predict_ego_reachset(
            ego_state, horizon_s=horizon_s, dt_s=pred_dt_s)
        ego_history.append((ego_t, ego_xy))
        ego_history = [
            (t, p) for t, p in ego_history if ego_t - t <= history_s + 0.05
        ]

        visible_pids = visible_by_sample.get(sample["token"], set())
        if not visible_pids:
            continue

        sample_t = float(sample["timestamp"]) * 1e-6
        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            instance = ann["instance_token"]
            if instance not in visible_pids:
                continue  # Not visible to the selected camera(s) at this frame.

            ped_xy = np.asarray(ann["translation"][:2], dtype=float)
            ped_v, ped_known = _causal_velocity(
                ped_xy, sample_t, ped_history[instance], history_s)
            # nuScenes annotation size = [width, length, height].
            width_m, length_m = map(float, ann["size"][:2])
            ped_radius = 0.5 * float(np.hypot(width_m, length_m))
            ped_state = {
                "xy": ped_xy,
                "velocity": ped_v,
                "velocity_known": ped_known,
                "footprint_radius_m": ped_radius,
            }
            ped_reach = function4_predict_pedestrian_reachset(
                ped_state, horizon_s=horizon_s, dt_s=pred_dt_s)
            ttc = _earliest_possible_ttc_s(ego_reach, ped_reach)

            ped_output = result["pedestrians"][instance]
            ped_output["states"][frame_idx] = (
                2 if ttc is not None and ttc < ttc_threshold_s else 1
            )
            ped_output["ttc_seconds"][frame_idx] = (
                None if ttc is None else round(ttc, 3)
            )
            ped_output["velocity_known"][frame_idx] = ped_known

            # Use camera-FOV-observable annotation history only. Do not use
            # hidden-frame GT positions to estimate pedestrian velocity.
            ped_history[instance].append((sample_t, ped_xy))
            ped_history[instance] = [
                (t, p) for t, p in ped_history[instance]
                if sample_t - t <= history_s + 0.05
            ]

    for pid, ped in result["pedestrians"].items():
        if sum(state != 0 for state in ped["states"]) != ped["num_frames"]:
            raise RuntimeError(f"Visible frame count mismatch for {pid}")

    return result


def save_scene_pedestrians(result: dict, output_dir: str | Path = "./results") -> Path:
    """Write one row per qualified pedestrian: pid, category, num_frames, states.

    Each output file is named <scene_name>.txt, and the states list contains one
    value per annotation sample in chronological order (e.g. 40 for a mini scene).
    Empty scenes still produce an empty TXT file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{result['scene_name']}.txt"

    with output_path.open("w", encoding="utf-8") as f:
        for pid, ped in result["pedestrians"].items():
            states = ped["states"]
            if len(states) != result["num_frames"]:
                raise ValueError(f"Incorrect states length for {pid}")
            states_text = " ".join(map(str, states))
            f.write(
                f"{pid}, {ped['category']}, {ped['num_frames']}, {states_text}\n"
            )
    return output_path


if __name__ == "__main__":
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(
        version="v1.0-mini",
        dataroot=r"E:\Linux-ubuntu\datasets\nuscenes_mini",
        verbose=False,
    )

    # Default: CAM_FRONT only (as in the uploaded code).
    # For all six cameras, set cameras=("CAM_FRONT", "CAM_FRONT_LEFT", ...).
    cameras = DEFAULT_CAMERAS
    output_dir = r"../ww_results"

    for scene in nusc.scene:
        result = function1_classify_scene(
            nusc,
            scene_name_or_token=scene["name"],
            ttc_threshold_s=3.0,
            horizon_s=3.0,
            cameras=cameras,
            min_frames_exclusive=3,
        )
        output_path = save_scene_pedestrians(result, output_dir=output_dir)
        print(
            f"Saved {output_path} | "
            f"frames={result['num_frames']} | "
            f"pedestrians={len(result['pedestrians'])}"
        )
