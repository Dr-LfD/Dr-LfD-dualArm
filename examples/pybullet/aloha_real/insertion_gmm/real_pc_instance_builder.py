import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

ROOT_PATH = next(
    (
        "/" + os.path.join(*os.path.dirname(os.path.abspath(__file__)).split(os.sep)[: i + 1]) + os.sep
        for i in range(len(os.path.dirname(os.path.abspath(__file__)).split(os.sep)))
        if os.path.dirname(os.path.abspath(__file__)).split(os.sep)[i] == "pddlstream_aloha"
    ),
    None,
)
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)

DEFAULT_ENV_ARGS = {
    "type": "aloha_real_pc_instance",
}


def parse_text_prompt_objects(text_prompt):
    if isinstance(text_prompt, str):
        return [obj.strip() for obj in text_prompt.split(".") if obj.strip()]
    return [str(obj).strip() for obj in text_prompt if str(obj).strip()]


def parse_json_maybe_bytes(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.ndarray):
        value = value.item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
    if isinstance(value, str):
        return json.loads(value)
    return value


def ensure_sam3_cuda_available(torch_module=None):
    if torch_module is None:
        import torch as torch_module

    if not torch_module.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for SAM3 real-data preprocessing. "
            "Run this pipeline on a CUDA-enabled machine with the SAM3 runtime dependencies installed."
        )


def resolve_sam3_checkpoint_path(checkpoint_path=None, model_dir=None):
    if checkpoint_path:
        return checkpoint_path
    if not model_dir:
        return None

    candidate_names = [
        "sam3.1_multiplex.pt",
        "sam3.pt",
        "model.pt",
        "pytorch_model.bin",
        "model.safetensors",
    ]
    for candidate_name in candidate_names:
        candidate_path = os.path.join(model_dir, candidate_name)
        if os.path.isfile(candidate_path):
            return candidate_path
    return None


def resolve_sam3_version(checkpoint_path, requested_version="auto"):
    inferred_version = "sam3.1"
    if checkpoint_path is not None:
        checkpoint_name = os.path.basename(checkpoint_path).lower()
        if checkpoint_name == "sam3.pt":
            inferred_version = "sam3"
        elif ("sam3.1" in checkpoint_name) or ("multiplex" in checkpoint_name):
            inferred_version = "sam3.1"

    if requested_version == "auto":
        return inferred_version
    if checkpoint_path is not None and requested_version != inferred_version:
        raise ValueError(
            f"Requested SAM3 version {requested_version!r} is incompatible with checkpoint {checkpoint_path!r}"
        )
    return requested_version


def summarize_builder_output(raw_output):
    interesting_lines = []
    for line in raw_output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "dynamic_multimask_via_stability is reset to False" in stripped:
            interesting_lines.append("dynamic_multimask_via_stability reset to False")
        elif stripped.startswith("Missing keys"):
            interesting_lines.append(stripped)
        elif stripped.startswith("Unexpected keys"):
            interesting_lines.append(stripped)
    if not interesting_lines:
        return None
    return " | ".join(interesting_lines)


def build_sam3_predictor(
    *,
    sam3_path,
    sam3_model_dir=None,
    sam3_checkpoint=None,
    sam3_version="auto",
    output_prob_thresh=0.5,
):
    ensure_sam3_cuda_available()
    if not sam3_path or not os.path.isdir(sam3_path):
        raise FileNotFoundError(
            f"SAM3 repo path does not exist: {sam3_path}. "
            "Set sam3_path to the SAM3 repo root."
        )

    checkpoint_path = resolve_sam3_checkpoint_path(
        checkpoint_path=sam3_checkpoint,
        model_dir=sam3_model_dir,
    )
    version = resolve_sam3_version(checkpoint_path, requested_version=sam3_version)

    sys.path.insert(0, sam3_path)
    try:
        from sam3 import build_sam3_predictor
    except ImportError as exc:
        raise RuntimeError(
            f"Failed to import SAM3 runtime from {sam3_path}. "
            "Install the SAM3 dependencies in that environment first."
        ) from exc
    finally:
        sys.path.pop(0)

    build_stdout = io.StringIO()
    with contextlib.redirect_stdout(build_stdout):
        predictor = build_sam3_predictor(
            version=version,
            checkpoint_path=checkpoint_path,
            compile=False,
            warm_up=False,
            async_loading_frames=False,
            default_output_prob_thresh=output_prob_thresh,
        )
    build_summary = summarize_builder_output(build_stdout.getvalue())
    if build_summary:
        print(f"[BUILD] {build_summary}", flush=True)
    return predictor


def write_frames_to_tempdir(color_frames):
    frame_dir = tempfile.mkdtemp(prefix="real_pc_instance_frames_")
    for frame_idx, frame in enumerate(color_frames):
        frame_path = os.path.join(frame_dir, f"{frame_idx:05d}.jpg")
        if not cv2.imwrite(frame_path, frame):
            raise RuntimeError(f"Failed to write temporary frame {frame_path}")
    return frame_dir


def select_best_mask(outputs):
    obj_ids = np.asarray(outputs.get("out_obj_ids", []))
    probs = np.asarray(outputs.get("out_probs", []), dtype=np.float32)
    boxes = np.asarray(outputs.get("out_boxes_xywh", []), dtype=np.float32)
    masks = np.asarray(outputs.get("out_binary_masks", []), dtype=bool)
    if len(probs) == 0:
        return None
    best_idx = int(np.argmax(probs))
    best_mask = masks[best_idx]
    if best_mask.ndim == 3:
        best_mask = best_mask[0]
    return {
        "obj_id": int(obj_ids[best_idx]) if len(obj_ids) else 0,
        "score": float(probs[best_idx]),
        "box_xywh": boxes[best_idx].tolist() if len(boxes) else [0.0, 0.0, 0.0, 0.0],
        "mask": np.asarray(best_mask, dtype=bool),
    }


def run_prompt_pass(predictor, frame_dir, object_name, output_prob_thresh=0.5):
    response = predictor.handle_request({"type": "start_session", "resource_path": frame_dir})
    session_id = response["session_id"]
    try:
        predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": object_name,
                "output_prob_thresh": output_prob_thresh,
            }
        )
        frame_results = {}
        for response in predictor.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": "forward",
                "output_prob_thresh": output_prob_thresh,
            }
        ):
            frame_results[response["frame_index"]] = select_best_mask(response["outputs"])
        return frame_results
    finally:
        predictor.handle_request({"type": "close_session", "session_id": session_id})


def segment_episode_with_sam3(
    predictor,
    color_frames_by_cam,
    object_names,
    output_prob_thresh=0.5,
):
    results = {}
    for cam_name, color_frames in color_frames_by_cam.items():
        frame_dir = write_frames_to_tempdir(color_frames)
        try:
            results[cam_name] = {}
            for object_name in object_names:
                results[cam_name][object_name] = run_prompt_pass(
                    predictor,
                    frame_dir,
                    object_name,
                    output_prob_thresh=output_prob_thresh,
                )
        finally:
            shutil.rmtree(frame_dir, ignore_errors=True)
    return results


def build_grounded_sam_predictors(
    *,
    grounded_sam_path,
    sam2_checkpoint=None,
    sam2_model_config="sam2_hiera_l.yaml",
    gdino_config=None,
    gdino_checkpoint=None,
):
    import torch

    ensure_sam3_cuda_available(torch_module=torch)

    if not grounded_sam_path:
        raise FileNotFoundError(
            "grounded_sam_path is required. Set it to the Grounded-SAM-2 repo root."
        )
    grounded_sam_root = os.path.abspath(os.path.expanduser(grounded_sam_path))
    if not os.path.isdir(grounded_sam_root):
        raise FileNotFoundError(
            f"Grounded-SAM-2 repo path does not exist: {grounded_sam_path}. "
            "Set grounded_sam_path to the Grounded-SAM-2 repo root."
        )

    def _resolve(arg, rel_default):
        p = arg or rel_default
        if not os.path.isabs(p):
            p = os.path.join(grounded_sam_root, p)
        return os.path.normpath(p)

    sam2_checkpoint = _resolve(sam2_checkpoint, "checkpoints/sam2_hiera_large.pt")
    gdino_config = _resolve(
        gdino_config, "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    )
    gdino_checkpoint = _resolve(gdino_checkpoint, "gdino_checkpoints/groundingdino_swint_ogc.pth")

    for label, path in [
        ("SAM2 checkpoint", sam2_checkpoint),
        ("Grounding DINO config", gdino_config),
        ("Grounding DINO checkpoint", gdino_checkpoint),
    ]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Grounded-SAM-2 {label} not found: {path}")

    if grounded_sam_root not in sys.path:
        sys.path.insert(0, grounded_sam_root)

    try:
        from sam2.build_sam import build_sam2_video_predictor
        from grounding_dino.groundingdino.util.inference import (
            load_image,
            load_model,
            predict as gdino_predict,
        )
    except ImportError as exc:
        raise RuntimeError(
            f"Failed to import Grounded-SAM-2 runtime from {grounded_sam_root}. "
            "Install the Grounded-SAM-2 dependencies in that environment first."
        ) from exc

    sam2_video_predictor = build_sam2_video_predictor(sam2_model_config, sam2_checkpoint, device="cuda")
    gdino_model = load_model(gdino_config, gdino_checkpoint, device="cuda")
    return gdino_model, sam2_video_predictor, load_image, gdino_predict


def run_grounded_sam_for_object(
    gdino_model,
    sam2_video_predictor,
    frame_dir,
    frames_bgr,
    object_name,
    gdino_load_image_fn,
    gdino_predict_fn,
    box_threshold=0.35,
    text_threshold=0.25,
    lost_mask_min_pixels=50,
):
    import torch

    num_frames = len(frames_bgr)
    if num_frames == 0:
        return {}

    text_prompt = str(object_name).strip().lower()
    if not text_prompt.endswith("."):
        text_prompt = f"{text_prompt}."

    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def _gdino_detect(frame_idx):
        frame_path = os.path.join(frame_dir, f"{frame_idx:05d}.jpg")
        image_np_rgb, image_tensor = gdino_load_image_fn(frame_path)

        boxes_cxcywh, confidences, _ = gdino_predict_fn(
            gdino_model, image_tensor, text_prompt, box_threshold, text_threshold
        )
        if boxes_cxcywh is None or len(boxes_cxcywh) == 0:
            return None

        boxes_np = (
            boxes_cxcywh.detach().cpu().numpy()
            if hasattr(boxes_cxcywh, "detach")
            else np.asarray(boxes_cxcywh, dtype=np.float32)
        )
        confs_np = (
            confidences.detach().cpu().numpy()
            if hasattr(confidences, "detach")
            else np.asarray(confidences, dtype=np.float32)
        )
        if len(confs_np) == 0:
            return None

        best = int(np.argmax(confs_np))
        image_h, image_w = image_np_rgb.shape[:2]
        cx, cy, w, h = boxes_np[best] * np.array(
            [image_w, image_h, image_w, image_h], dtype=np.float32
        )
        x1 = float(max(0.0, cx - w / 2.0))
        y1 = float(max(0.0, cy - h / 2.0))
        x2 = float(min(image_w - 1, cx + w / 2.0))
        y2 = float(min(image_h - 1, cy + h / 2.0))
        if x2 <= x1 or y2 <= y1:
            return None
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    inference_state = sam2_video_predictor.init_state(video_path=frame_dir)
    frame_results = {}
    current_start = 0
    object_id = 1

    while current_start < num_frames:
        box_xyxy = _gdino_detect(current_start)
        if box_xyxy is None:
            frame_results[current_start] = None
            current_start += 1
            continue

        sam2_video_predictor.reset_state(inference_state)
        sam2_video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=current_start,
            obj_id=object_id,
            box=box_xyxy,
        )

        lost_at = None
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            for frame_idx, obj_ids, video_res_masks in sam2_video_predictor.propagate_in_video(
                inference_state, start_frame_idx=current_start
            ):
                obj_ids_list = list(obj_ids)
                if not obj_ids_list:
                    frame_results[frame_idx] = None
                    lost_at = frame_idx
                    break

                try:
                    mask_slot = obj_ids_list.index(object_id)
                except ValueError:
                    mask_slot = 0

                mask_logits = video_res_masks[mask_slot]  # shape (1, H, W)
                mask = (mask_logits > 0).detach().cpu().numpy().astype(bool).squeeze()
                if mask.ndim != 2:
                    raise RuntimeError(
                        f"Expected 2-D mask for {object_name!r} at frame {frame_idx}, "
                        f"got shape {mask.shape}"
                    )

                if mask.sum() < lost_mask_min_pixels:
                    frame_results[frame_idx] = None
                    lost_at = frame_idx
                    break

                rows, cols = np.where(mask)
                score = float(torch.sigmoid(mask_logits).amax().detach().cpu())
                frame_results[frame_idx] = {
                    "obj_id": object_id,
                    "score": score,
                    "box_xywh": [
                        int(cols.min()),
                        int(rows.min()),
                        int(cols.max() - cols.min()),
                        int(rows.max() - rows.min()),
                    ],
                    "mask": mask,
                }

        if current_start not in frame_results:
            frame_results[current_start] = None
            current_start += 1
            continue
        if lost_at is None:
            break
        if lost_at <= current_start:
            current_start += 1
        else:
            current_start = lost_at

    for frame_idx in range(num_frames):
        frame_results.setdefault(frame_idx, None)
    return frame_results


def segment_episode_with_grounded_sam(
    gdino_model,
    sam2_video_predictor,
    color_frames_by_cam,
    object_names,
    gdino_load_image_fn,
    gdino_predict_fn,
    box_threshold=0.35,
    text_threshold=0.25,
    lost_mask_min_pixels=50,
):
    results = {}
    for cam_name, color_frames in color_frames_by_cam.items():
        frame_dir = write_frames_to_tempdir(color_frames)
        try:
            results[cam_name] = {}
            for object_name in object_names:
                results[cam_name][object_name] = run_grounded_sam_for_object(
                    gdino_model,
                    sam2_video_predictor,
                    frame_dir,
                    color_frames,
                    object_name,
                    gdino_load_image_fn,
                    gdino_predict_fn,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    lost_mask_min_pixels=lost_mask_min_pixels,
                )
        finally:
            shutil.rmtree(frame_dir, ignore_errors=True)
    return results


def project_mask_to_world_points(depth_img, mask, camera_info, cam_pose_json_file, camlink2optical):
    from examples.pybullet.aloha_real.openworld_aloha.estimation.geometry import cloud_from_depth
    from examples.pybullet.aloha_real.openworld_aloha.policy_simp import get_compatible_campose
    from examples.pybullet.utils.pybullet_tools.utils import tform_points

    with open(cam_pose_json_file, "r") as f:
        cam_pose_dict = json.load(f)

    camera_pose = get_compatible_campose(cam_pose_dict, camlink2optical=camlink2optical)
    depth_camera_matrix = np.array(camera_info["K"]).reshape(3, 3)
    point_cloud = cloud_from_depth(depth_camera_matrix, depth_img, top_left_origin=True)
    valid_mask = np.asarray(mask, dtype=bool) & np.isfinite(depth_img) & (depth_img > 0)
    points_cam = point_cloud[valid_mask]
    if len(points_cam) > 0:
        points_cam = points_cam[np.any(points_cam != 0, axis=1)]
    if len(points_cam) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    points_world = tform_points(camera_pose, points_cam)
    return np.asarray(points_world, dtype=np.float32)


def pad_or_sample_points(points, max_points):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(-1, 3)
    if len(points) >= max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
        return points[indices]
    padded = np.zeros((max_points, 3), dtype=np.float32)
    if len(points) > 0:
        padded[: len(points)] = points
    return padded


def build_episode_point_clouds(
    segmentation_results,
    depth_imgs_by_cam,
    camera_infos_by_cam,
    cam_extparam_mapping,
    calibrate_mapping,
    object_names,
    max_points,
    point_projector_fn=project_mask_to_world_points,
):
    demo_len = len(next(iter(depth_imgs_by_cam.values())))
    obj_point_clouds = {}
    obj_visible = {}

    for object_name in object_names:
        frame_clouds = []
        visible_flags = []
        for frame_idx in range(demo_len):
            world_points = []
            for cam_name, frame_results in segmentation_results.items():
                frame_result = frame_results[object_name].get(frame_idx)
                if frame_result is None:
                    continue
                mask = frame_result["mask"]
                if mask is None or not np.any(mask):
                    continue
                points_world = point_projector_fn(
                    depth_imgs_by_cam[cam_name][frame_idx],
                    mask,
                    camera_infos_by_cam[cam_name],
                    cam_extparam_mapping[cam_name],
                    calibrate_mapping[cam_name],
                )
                if len(points_world) > 0:
                    world_points.append(points_world)

            if world_points:
                points = np.concatenate(world_points, axis=0)
                frame_clouds.append(pad_or_sample_points(points, max_points))
                visible_flags.append(True)
            else:
                frame_clouds.append(np.zeros((max_points, 3), dtype=np.float32))
                visible_flags.append(False)

        obj_point_clouds[object_name] = np.asarray(frame_clouds, dtype=np.float32)
        obj_visible[object_name] = np.asarray(visible_flags, dtype=bool)

    return obj_point_clouds, obj_visible


def build_aloha_robot_streams(obs_qpos, robot_config, dt=0.02):
    from examples.pybullet.aloha_real.scripts.aloha_tamp_constants import (
        PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN,
        qpos_to_eetrans,
    )

    obs_qpos = np.asarray(obs_qpos, dtype=np.float32)
    if robot_config.get("robot_name") != "aloha":
        raise NotImplementedError(
            f"Unsupported robot_name {robot_config.get('robot_name')!r} for real pc_instance preprocessing"
        )

    if obs_qpos.shape[1] < 14:
        raise ValueError(f"Expected ALOHA qpos width >= 14, got {obs_qpos.shape}")

    left_joint_slice = slice(*robot_config.get("left_joint_slice", (0, 6)))
    right_joint_slice = slice(*robot_config.get("right_joint_slice", (7, 13)))
    left_gripper_index = robot_config.get("left_gripper_index", 6)
    right_gripper_index = robot_config.get("right_gripper_index", 13)
    robots = robot_config.get("robots", ["robot0", "robot1"])
    if len(robots) != 2:
        raise ValueError(f"Expected exactly two robots for ALOHA preprocessing, got {robots}")

    left_joint = obs_qpos[:, left_joint_slice]
    right_joint = obs_qpos[:, right_joint_slice]
    left_gripper_qpos = PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(obs_qpos[:, left_gripper_index]).astype(np.float32)
    right_gripper_qpos = PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(obs_qpos[:, right_gripper_index]).astype(np.float32)

    def build_eef_stream(joint_vals, hand_id):
        positions = []
        quats = []
        for joint_val in joint_vals:
            transform = qpos_to_eetrans(joint_val, hand_id)
            positions.append(transform[:3, 3].astype(np.float32))
            quats.append(Rotation.from_matrix(transform[:3, :3]).as_quat().astype(np.float32))
        return np.asarray(positions, dtype=np.float32), np.asarray(quats, dtype=np.float32)

    left_eef_pos, left_eef_quat = build_eef_stream(left_joint, 0)
    right_eef_pos, right_eef_quat = build_eef_stream(right_joint, 1)

    left_gripper_qvel = np.gradient(left_gripper_qpos, dt).astype(np.float32).reshape(-1, 1)
    right_gripper_qvel = np.gradient(right_gripper_qpos, dt).astype(np.float32).reshape(-1, 1)

    return {
        f"{robots[0]}_joint_pos": left_joint.astype(np.float32),
        f"{robots[0]}_eef_pos": left_eef_pos,
        f"{robots[0]}_eef_quat": left_eef_quat,
        f"{robots[0]}_gripper_qpos": left_gripper_qpos.reshape(-1, 1),
        f"{robots[0]}_gripper_qvel": left_gripper_qvel,
        f"{robots[1]}_joint_pos": right_joint.astype(np.float32),
        f"{robots[1]}_eef_pos": right_eef_pos,
        f"{robots[1]}_eef_quat": right_eef_quat,
        f"{robots[1]}_gripper_qpos": right_gripper_qpos.reshape(-1, 1),
        f"{robots[1]}_gripper_qvel": right_gripper_qvel,
    }


def read_demo_realsense(input_hdf5_path, camera_names=("",)):
    with h5py.File(input_hdf5_path, "r") as f:
        data_dict = {}
        obs_grp = f["observations"]
        data_dict["qpos"] = obs_grp["qpos"][()]
        if "qvel" in obs_grp:
            data_dict["qvel"] = obs_grp["qvel"][()]
        data_dict["action"] = f["action"][()]

        for rs_cam in camera_names:
            data_dict[f"color_img_{rs_cam}"] = f[f"color_img_{rs_cam}"][()]
            data_dict[f"depth_img_{rs_cam}"] = f[f"depth_img_{rs_cam}"][()]
            data_dict[f"camera_info_{rs_cam}"] = f[f"camera_info_{rs_cam}"][()]
        return data_dict


def iter_episode_paths(hdf5_dir, n_playback=None, debug_ep_id=None):
    root = Path(hdf5_dir)
    if debug_ep_id is not None:
        path = root / f"episode_{debug_ep_id}.hdf5"
        return [path] if path.exists() else []

    paths = sorted(
        root.glob("episode_*.hdf5"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    if n_playback is not None:
        return paths[:n_playback]
    return paths


def build_real_pc_instance_hdf5(
    *,
    raw_hdf5_dir,
    output_hdf5_path,
    object_names,
    camera_names,
    cam_extparam_mapping,
    calibrate_mapping,
    segment_episode_fn,
    point_projector_fn=project_mask_to_world_points,
    robot_stream_builder_fn=build_aloha_robot_streams,
    robot_config=None,
    n_playback=None,
    debug_ep_id=None,
    cache_masks=True,
    discard_mask_cache=False,
    max_points=2048,
    env_args=None,
):
    episode_paths = iter_episode_paths(raw_hdf5_dir, n_playback=n_playback, debug_ep_id=debug_ep_id)
    if not episode_paths:
        raise FileNotFoundError(f"No raw episode_*.hdf5 files found under {raw_hdf5_dir}")

    output_parent = os.path.dirname(output_hdf5_path)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    with h5py.File(output_hdf5_path, "w") as f_out:
        data_group = f_out.create_group("data")
        env_args_dict = dict(DEFAULT_ENV_ARGS)
        if env_args is not None:
            env_args_dict.update(env_args)
        data_group.attrs["env_args"] = json.dumps(env_args_dict)

        for demo_idx, episode_path in enumerate(episode_paths):
            raw_data = read_demo_realsense(str(episode_path), camera_names=camera_names)
            color_imgs_by_cam = {}
            depth_imgs_by_cam = {}
            camera_infos_by_cam = {}
            for cam_name in camera_names:
                color_imgs_by_cam[cam_name] = raw_data[f"color_img_{cam_name}"]
                depth_imgs_by_cam[cam_name] = raw_data[f"depth_img_{cam_name}"].astype(np.float32) / 1000.0
                camera_infos_by_cam[cam_name] = parse_json_maybe_bytes(raw_data[f"camera_info_{cam_name}"])

            segmentation_results = segment_episode_fn(
                color_imgs_by_cam=color_imgs_by_cam,
                object_names=object_names,
            )
            obj_point_clouds, obj_visible = build_episode_point_clouds(
                segmentation_results,
                depth_imgs_by_cam,
                camera_infos_by_cam,
                cam_extparam_mapping,
                calibrate_mapping,
                object_names,
                max_points,
                point_projector_fn=point_projector_fn,
            )
            if robot_config is None:
                raise ValueError("robot_config is required for real pc_instance preprocessing")
            robot_streams = robot_stream_builder_fn(raw_data["qpos"], robot_config)

            demo_group = data_group.create_group(f"demo_{demo_idx}")
            demo_group.attrs["source_episode"] = episode_path.name
            obs_group = demo_group.create_group("obs")
            for key, value in robot_streams.items():
                obs_group.create_dataset(key, data=value)
            for object_name in object_names:
                obs_group.create_dataset(f"{object_name}_point_cloud", data=obj_point_clouds[object_name])
                obs_group.create_dataset(f"{object_name}_visible", data=obj_visible[object_name])

            if cache_masks and not discard_mask_cache:
                cache_group = obs_group.create_group("sam3_cache")
                for cam_name in camera_names:
                    cam_group = cache_group.create_group(cam_name)
                    for object_name in object_names:
                        obj_group = cam_group.create_group(object_name)
                        frame_results = segmentation_results[cam_name][object_name]
                        demo_len = len(color_imgs_by_cam[cam_name])
                        masks = np.zeros(color_imgs_by_cam[cam_name].shape[:3], dtype=np.uint8)
                        boxes = np.zeros((demo_len, 4), dtype=np.float32)
                        scores = np.zeros((demo_len,), dtype=np.float32)
                        for frame_idx, result in frame_results.items():
                            if result is None:
                                continue
                            masks[frame_idx] = result["mask"].astype(np.uint8)
                            boxes[frame_idx] = np.asarray(result["box_xywh"], dtype=np.float32)
                            scores[frame_idx] = float(result["score"])
                        obj_group.create_dataset("masks", data=masks)
                        obj_group.create_dataset("boxes_xywh", data=boxes)
                        obj_group.create_dataset("scores", data=scores)

            demo_group.create_dataset("actions", data=raw_data["action"])


def infer_interested_primitives(schema_config_path):
    with open(schema_config_path, "r") as f:
        data = json.load(f)
    seen = set()
    primitives = []
    for item in data.get("ModeChangeDetection", []):
        description = (item.get("description") or "").lower()
        if "grasp" in description and "grasp" not in seen:
            seen.add("grasp")
            primitives.append("grasp")
        if ("place" in description or "release" in description) and "place" not in seen:
            seen.add("place")
            primitives.append("place")
        if "bimanual" in description and "bimanual" not in seen:
            seen.add("bimanual")
            primitives.append("bimanual")
    return primitives
