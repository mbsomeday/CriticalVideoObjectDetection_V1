"""CAM_FRONT pedestrians

2D boxes are read from nuScenes' official image_annotations.json export.
TTC is evaluated from the third keyframe (frame_idx >= 2).
Reachsets use the supplied constant-velocity circular approximation.
"""

from collections import defaultdict
import json
from pathlib import Path
import numpy as np


# ---------- Reachset functions: replace/inject these to change the method ----------

def causal_velocity(xy, timestamp_s, history, lookback_s=1.0):
    """Use up to 1 second of *past* positions only (as in the supplied file)."""
    valid = [(t, p) for t, p in history
             if 1e-6 < timestamp_s - t <= lookback_s + 0.05]
    if not valid:
        return np.zeros(2, dtype=float), False
    past_t, past_xy = min(valid, key=lambda item: item[0])
    return (xy - past_xy) / (timestamp_s - past_t), True


def reachset_cv(state, *, horizon_s, dt_s, velocity_error_mps,
                accel_bound_mps2, unknown_speed_bound_mps):
    """2D circular reachset at times t in [0, horizon_s]."""
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


def predict_ego_reachset(state, *, horizon_s=5.0, dt_s=0.1,
                         velocity_error_mps=0.3, accel_bound_mps2=1.0,
                         unknown_speed_bound_mps=15.0):
    return reachset_cv(
        state, horizon_s=horizon_s, dt_s=dt_s,
        velocity_error_mps=velocity_error_mps,
        accel_bound_mps2=accel_bound_mps2,
        unknown_speed_bound_mps=unknown_speed_bound_mps,
    )


def predict_pedestrian_reachset(state, *, horizon_s=5.0, dt_s=0.1,
                                velocity_error_mps=0.2, accel_bound_mps2=0.7,
                                unknown_speed_bound_mps=3.0):
    return reachset_cv(
        state, horizon_s=horizon_s, dt_s=dt_s,
        velocity_error_mps=velocity_error_mps,
        accel_bound_mps2=accel_bound_mps2,
        unknown_speed_bound_mps=unknown_speed_bound_mps,
    )


def earliest_possible_ttc(ego_reach, ped_reach):
    """First possible circle intersection; None if no intersection in horizon."""
    if not np.allclose(ego_reach["times_s"], ped_reach["times_s"]):
        raise ValueError("Ego and pedestrian reachsets need identical time grids")
    distances = np.linalg.norm(
        ego_reach["centers_xy_m"] - ped_reach["centers_xy_m"], axis=1
    )
    overlaps = distances <= ego_reach["radii_m"] + ped_reach["radii_m"]
    indices = np.flatnonzero(overlaps)
    return float(ego_reach["times_s"][indices[0]]) if indices.size else None


# ---------- Official 2D annotation data ----------

def load_image_annotations(json_path):
    """Index official nuScenes 2D annotations by camera sample_data_token.

    Each record contains the original annotation's sample_annotation_token
    and bbox_corners = [xmin, ymin, xmax, ymax]. Load once per dataset.
    """
    with Path(json_path).open("r", encoding="utf-8") as f:
        annotations = json.load(f)

    by_camera = defaultdict(list)
    for record in annotations:
        if record.get("category_name", "").startswith("human.pedestrian."):
            by_camera[record["sample_data_token"]].append(record)
    return by_camera


# ---------- Main analysis ----------

def analyze_scene(nusc, my_scene, *, horizon_s=5.0, dt_s=0.1,
                  history_s=1.0, ego_length_m=4.7, ego_width_m=2.0,
                  ego_reachset_fn=predict_ego_reachset,
                  ped_reachset_fn=predict_pedestrian_reachset,
                  egovehicle_velocity=None, image_ann_by_sd=None,
                  ttc_start_frame=2,
                  max_visibility_token=3):
    """Return qualified pedestrian records and optional per-frame ego velocities.

    ttc_start_frame=2 means the third keyframe when indexing from zero.
    No distance cutoff is applied; visibility_token is compared numerically.
    Frames before ttc_start_frame are saved with ttc_s=None (not evaluated).
    TTC can now be detected up to 5 s ahead with the default reachset horizon.
    """
    if image_ann_by_sd is None:
        raise ValueError("Pass image_ann_by_sd=load_image_annotations(json_path)")
    if ttc_start_frame < 0:
        raise ValueError("ttc_start_frame must be nonnegative")
    if max_visibility_token not in (1, 2, 3, 4):
        raise ValueError("max_visibility_token must be 1, 2, 3, or 4")
    pedestrians = defaultdict(list)
    ego_history = []
    ped_history = defaultdict(list)

    sample_token = my_scene["first_sample_token"]
    frame_idx = 0
    ego_radius = 0.5 * float(np.hypot(ego_length_m, ego_width_m))

    while sample_token:
        sample = nusc.get("sample", sample_token)
        sample_t = float(sample["timestamp"]) * 1e-6

        # Ego position: LIDAR_TOP pose in the global coordinate system,
        # as in the uploaded reachset script.
        lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
        ego_xy = np.asarray(ego_pose["translation"][:2], dtype=float)
        ego_t = float(ego_pose["timestamp"]) * 1e-6
        ego_v, ego_known = causal_velocity(ego_xy, ego_t, ego_history, history_s)

        # Export exactly the velocity used by the ego reachset calculation.
        # Unknown velocity (typically frame 0) is represented as null in JSON.
        if egovehicle_velocity is not None:
            egovehicle_velocity.append({
                "frame": frame_idx,
                "velocity_xy": ([round(float(v), 8) for v in ego_v]
                                if ego_known else None),
                "speed_mps": (round(float(np.linalg.norm(ego_v)), 5)
                              if ego_known else None),
            })

        ego_state = {
            "xy": ego_xy,
            "velocity": ego_v,
            "velocity_known": ego_known,
            "footprint_radius_m": ego_radius,
        }
        # Do not compute any reachset/TTC before the third keyframe.
        ego_reach = (
            ego_reachset_fn(ego_state, horizon_s=horizon_s, dt_s=dt_s)
            if frame_idx >= ttc_start_frame else None
        )
        ego_history.append((ego_t, ego_xy))
        ego_history = [(t, xy) for t, xy in ego_history
                       if ego_t - t <= history_s + 0.05]

        # Use official 2D annotations for this exact CAM_FRONT keyframe.
        cam_token = sample["data"]["CAM_FRONT"]
        for ann_2d in image_ann_by_sd.get(cam_token, []):
            # Official 2D export contains sample_annotation_token and bbox_corners.
            visibility_token = int(ann_2d["visibility_token"])
            if visibility_token > max_visibility_token:  # <= 3
                continue

            ann = nusc.get("sample_annotation", ann_2d["sample_annotation_token"])
            ped_xy = np.asarray(ann["translation"][:2], dtype=float)
            distance = float(np.linalg.norm(ped_xy - ego_xy))

            pid = ann["instance_token"]
            bbox_2d = [float(x) for x in ann_2d["bbox_corners"]]
            if len(bbox_2d) != 4:
                raise ValueError(f"Invalid bbox_corners for annotation {ann['token']}")
            bbox_area = (bbox_2d[2] - bbox_2d[0]) * (bbox_2d[3] - bbox_2d[1])

            ped_v, ped_known = causal_velocity(
                ped_xy, sample_t, ped_history[pid], history_s
            )
            # nuScenes annotation size: [width, length, height].
            width_m, length_m = map(float, ann["size"][:2])
            ped_state = {
                "xy": ped_xy,
                "velocity": ped_v,
                "velocity_known": ped_known,
                "footprint_radius_m": 0.5 * float(np.hypot(width_m, length_m)),
            }
            # Requirement 3: first TTC evaluation at frame_idx == 2.
            if frame_idx >= ttc_start_frame:
                ped_reach = ped_reachset_fn(
                    ped_state, horizon_s=horizon_s, dt_s=dt_s
                )
                ttc = earliest_possible_ttc(ego_reach, ped_reach)
            else:
                ped_reach = None
                ttc = None

            pedestrians[pid].append({
                "frame": frame_idx,
                "front_cam": ann_2d["filename"],
                "bbox_area": bbox_area,       # pixels^2, official 2D bbox area
                "bbox_2d": bbox_2d,          # [xmin, ymin, xmax, ymax]
                "visibility_token": visibility_token,
                "distance": distance,         # meters, horizontal center-to-center
                "ego_reachset": ego_reach,
                "ped_reachset": ped_reach,
                "ttc_s": ttc,                 # earliest possible intersection
            })

            # Only selected CAM_FRONT frames enter pedestrian history;
            # this includes qualifying frames 0/1 for later velocity estimates.
            ped_history[pid].append((sample_t, ped_xy))
            ped_history[pid] = [
                (t, xy) for t, xy in ped_history[pid]
                if sample_t - t <= history_s + 0.05
            ]

        sample_token = sample["next"]
        frame_idx += 1

    return dict(pedestrians)


# ---------- Export one scene's pedestrian annotations as JSON ----------

def build_scene_annotations(my_scene, pedestrians, *, critical_ttc_s=3.0,
                            egovehicle_velocity=None):
    """Produce 0/1/2 labels for all keyframes, plus per-visible-frame data.

    0 = does not meet CAM_FRONT visibility<=3 or image-projection criteria;
    1 = meets criteria, TTC absent or >= critical_ttc_s;
    2 = meets criteria, evaluated TTC < critical_ttc_s.

    Note: the first two frames can be assigned 1 although TTC was not evaluated.
    They should not be interpreted as *verified* non-critical frames.
    """
    if critical_ttc_s <= 0:
        raise ValueError("critical_ttc_s must be positive")

    num_video_frames = int(my_scene["nbr_samples"])
    output = {
        "scene": my_scene["name"],
        "num_video_frames": num_video_frames,
        "pedestrians": [],
    }
    if egovehicle_velocity is not None:
        if len(egovehicle_velocity) != num_video_frames:
            raise ValueError("egovehicle_velocity must contain one entry per frame")
        output["egovehicle_velocity_mps"] = [
            record["speed_mps"] for record in egovehicle_velocity
        ]

    for pid, records in pedestrians.items():
        states = [0] * num_video_frames
        frame_annotations = []

        for record in records:
            frame = int(record["frame"])
            if not 0 <= frame < num_video_frames:
                raise ValueError(f"Invalid frame index {frame} for {pid}")
            if states[frame] != 0:
                raise ValueError(f"Duplicate frame index {frame} for {pid}")

            ttc = record["ttc_s"]
            states[frame] = 2 if ttc is not None and ttc < critical_ttc_s else 1
            area = record["bbox_area"]
            frame_annotations.append({
                "frame": frame,
                "front_cam": record["front_cam"],
                "bbox_2d": (
                    [round(float(v), 2) for v in record["bbox_2d"]]
                    if record["bbox_2d"] is not None else None
                ),
                "bbox_area": None if area is None else round(float(area), 2),
                "visibility_token": record["visibility_token"],
                "distance/m": round(float(record["distance"]), 2),
                "ttc/s": None if ttc is None else round(float(ttc), 2),
            })

        frame_annotations.sort(key=lambda row: row["frame"])
        output["pedestrians"].append({
            "Instance": pid,
            "Number of frames": len(frame_annotations),
            "anno_in_video": " ".join(map(str, states)),
            "each_frame_anno": frame_annotations,
        })

    return output


def save_scene_json(my_scene, pedestrians, *, output_dir="./results",
                    critical_ttc_s=3.0, egovehicle_velocity=None):
    """Save JSON at <output_dir>/<scene name>.json; return saved path."""
    output = build_scene_annotations(
        my_scene, pedestrians, critical_ttc_s=critical_ttc_s,
        egovehicle_velocity=egovehicle_velocity,
    )
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / f"{my_scene['name']}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, allow_nan=False)
    return output_path


if __name__ == "__main__":
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(
        version="v1.0-mini",
        dataroot=r"E:\Linux-ubuntu\datasets\nuscenes_mini",
        verbose=False,
    )
    # Official 2D exports must be located in the nuScenes version directory.
    image_annotations_path = Path(nusc.dataroot) / nusc.version / "image_annotations.json"
    image_ann_by_sd = load_image_annotations(image_annotations_path)

    for my_scene in nusc.scene:
        egovehicle_velocity = []
        pedestrians = analyze_scene(
            nusc, my_scene, horizon_s=5.0, dt_s=0.1,
            egovehicle_velocity=egovehicle_velocity,
            image_ann_by_sd=image_ann_by_sd,
            ttc_start_frame=2,       # 第3帧：frame_idx 2
            max_visibility_token=3, # <= 3
        )
        output_path = save_scene_json(
            my_scene, pedestrians,
            output_dir="./nu_scene_anno",
            critical_ttc_s=3.0,  # Retain original critical-frame threshold
            egovehicle_velocity=egovehicle_velocity,
        )
        print(f"Saved: {output_path}")
