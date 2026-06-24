import argparse
import copy
import json
import os
import sys

root_path = next(
    (
        "/" + os.path.join(*os.path.dirname(os.path.abspath(__file__)).split(os.sep)[: i + 1]) + os.sep
        for i in range(len(os.path.dirname(os.path.abspath(__file__)).split(os.sep)))
        if os.path.dirname(os.path.abspath(__file__)).split(os.sep)[i] == "pddlstream_aloha"
    ),
    None,
)
sys.path.append(root_path) if root_path not in sys.path else None
if root_path in sys.path:
    sys.path.remove(root_path)
sys.path.insert(0, root_path)
os.chdir(root_path)

from examples.pybullet.aloha_real.insertion_gmm.postprocess_template_base import (
    parse_prior_graphs,
    postprocess_for_sgs,
)
from examples.pybullet.aloha_real.insertion_gmm.real_pc_instance_builder import (
    build_grounded_sam_predictors,
    build_real_pc_instance_hdf5,
    build_sam3_predictor,
    infer_interested_primitives,
    parse_text_prompt_objects,
    segment_episode_with_grounded_sam,
    segment_episode_with_sam3,
)
from examples.pybullet.aloha_real.openworld_aloha.open_world_utils import (
    get_camera_mappings,
    load_yaml_params,
)


def build_real_segment_episode_fn(
    *,
    seg_backend="sam3",
    sam3_path=None,
    sam3_model_dir=None,
    sam3_checkpoint=None,
    sam3_version="auto",
    output_prob_thresh=0.5,
    grounded_sam_path=None,
    gsam_sam2_checkpoint=None,
    gsam_sam2_model_config="sam2_hiera_l.yaml",
    gsam_gdino_config=None,
    gsam_gdino_checkpoint=None,
    gsam_box_threshold=0.35,
    gsam_text_threshold=0.25,
    gsam_lost_mask_min_pixels=50,
):
    if seg_backend == "sam3":
        predictor = build_sam3_predictor(
            sam3_path=sam3_path,
            sam3_model_dir=sam3_model_dir,
            sam3_checkpoint=sam3_checkpoint,
            sam3_version=sam3_version,
            output_prob_thresh=output_prob_thresh,
        )

        def _segment_episode_fn(*, color_imgs_by_cam, object_names):
            return segment_episode_with_sam3(
                predictor,
                color_imgs_by_cam,
                object_names,
                output_prob_thresh=output_prob_thresh,
            )

        return _segment_episode_fn

    if seg_backend == "grounded_sam":
        gdino_model, sam2_video_predictor, gdino_load_image_fn, gdino_predict_fn = (
            build_grounded_sam_predictors(
                grounded_sam_path=grounded_sam_path,
                sam2_checkpoint=gsam_sam2_checkpoint,
                sam2_model_config=gsam_sam2_model_config,
                gdino_config=gsam_gdino_config,
                gdino_checkpoint=gsam_gdino_checkpoint,
            )
        )

        def _segment_episode_fn(*, color_imgs_by_cam, object_names):
            return segment_episode_with_grounded_sam(
                gdino_model,
                sam2_video_predictor,
                color_imgs_by_cam,
                object_names,
                gdino_load_image_fn,
                gdino_predict_fn,
                box_threshold=gsam_box_threshold,
                text_threshold=gsam_text_threshold,
                lost_mask_min_pixels=gsam_lost_mask_min_pixels,
            )

        return _segment_episode_fn

    raise ValueError(f"Unknown seg_backend: {seg_backend!r}. Choose 'sam3' or 'grounded_sam'.")


def _get_robot_config(parameters):
    return parameters.get("real_pc_instance_robot_config") or {
        "robot_name": parameters.get("robot_name", "aloha"),
        "robots": ["robot0", "robot1"],
        "left_joint_slice": (0, 6),
        "right_joint_slice": (7, 13),
        "left_gripper_index": 6,
        "right_gripper_index": 13,
    }


def build_real_sg_params(parameters, skill_name, input_hdf5_path, output_hdf5_path):
    skill_params = copy.deepcopy(parameters[skill_name])
    interested_objs = list(dict.fromkeys(skill_params["pre_obj_names"] + skill_params["eff_obj_names"]))

    schema_path = skill_params.get("schema_config_path")
    if schema_path is None:
        schema_cfg = skill_params.get("schema", {})
        schema_path = schema_cfg.get("path")
    if schema_path is None:
        raise ValueError(f"Missing schema path for skill {skill_name}")
    if not os.path.isabs(schema_path):
        schema_path = os.path.join(
            os.path.dirname(parameters["skill_yaml_paths"][0]),
            schema_path,
        )
    schema_path = os.path.normpath(schema_path)

    sg_params = copy.deepcopy(skill_params)
    robot_config = _get_robot_config(parameters)
    sg_params["robots"] = robot_config["robots"]
    sg_params["interested_objs"] = interested_objs
    sg_params["interested_primitives"] = infer_interested_primitives(schema_path)
    sg_params["task_name"] = skill_params.get("skill_name", skill_name)
    sg_params["input_hdf5_path"] = input_hdf5_path
    sg_params["output_hdf5_path"] = output_hdf5_path
    sg_params["obj_obj_dist_threshold"] = skill_params.get(
        "obj_obj_dist_threshold",
        skill_params.get("hand_obj_dist_threshold", 0.1),
    )
    return sg_params, schema_path


def postprocess_real_data(args):
    parameters = load_yaml_params(
        args.base_cfg,
        skill_yaml_paths=[args.skill_yaml],
        task_name=args.task_name,
        mode="real",
        is_testing=False,
    )
    skill_name = parameters["skill_names"][0]
    skill_params = parameters[skill_name]
    task_name = args.task_name or skill_params.get("skill_name", skill_name)
    raw_hdf5_dir = args.hdf5_dir or skill_params["h5_parent_dir"]
    output_dir = args.output_dir or skill_params["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    pc_instance_hdf5 = args.intermediate_hdf5 or os.path.join(output_dir, f"{task_name}_pc_instance.hdf5")
    sg_output_hdf5 = args.output_hdf5 or pc_instance_hdf5.replace(".hdf5", f"_sg_{args.n_playback}.hdf5")

    cam_dir_mapping, cam_extparam_mapping, calibrate_mapping = get_camera_mappings(parameters)
    camera_names = list(cam_extparam_mapping.keys())
    object_names = parse_text_prompt_objects(parameters["text_prompt"])
    robot_config = _get_robot_config(parameters)

    seg_backend = args.seg_backend or parameters.get("seg_backend", "sam3")
    segment_episode_fn = build_real_segment_episode_fn(
        seg_backend=seg_backend,
        sam3_path=args.sam3_path or parameters.get("sam3_path") or parameters.get("sam_path"),
        sam3_model_dir=args.sam3_model_dir or parameters.get("sam3_model_dir"),
        sam3_checkpoint=args.sam3_checkpoint or parameters.get("sam3_checkpoint"),
        sam3_version=args.sam3_version,
        output_prob_thresh=args.output_prob_thresh,
        grounded_sam_path=(
            args.grounded_sam_path
            or parameters.get("grounded_sam_path")
            or parameters.get("sam_path")
        ),
        gsam_sam2_checkpoint=(
            args.gsam_sam2_checkpoint or parameters.get("gsam_sam2_checkpoint")
        ),
        gsam_sam2_model_config=(
            args.gsam_sam2_model_config
            or parameters.get("gsam_sam2_model_config")
            or "sam2_hiera_l.yaml"
        ),
        gsam_gdino_config=args.gsam_gdino_config or parameters.get("gsam_gdino_config"),
        gsam_gdino_checkpoint=(
            args.gsam_gdino_checkpoint or parameters.get("gsam_gdino_checkpoint")
        ),
        gsam_box_threshold=(
            args.gsam_box_threshold
            if args.gsam_box_threshold is not None
            else parameters.get("gsam_box_threshold", 0.35)
        ),
        gsam_text_threshold=(
            args.gsam_text_threshold
            if args.gsam_text_threshold is not None
            else parameters.get("gsam_text_threshold", 0.25)
        ),
        gsam_lost_mask_min_pixels=(
            args.gsam_lost_mask_min_pixels
            if args.gsam_lost_mask_min_pixels is not None
            else parameters.get("gsam_lost_mask_min_pixels", 50)
        ),
    )

    build_real_pc_instance_hdf5(
        raw_hdf5_dir=raw_hdf5_dir,
        output_hdf5_path=pc_instance_hdf5,
        object_names=object_names,
        camera_names=camera_names,
        cam_extparam_mapping=cam_extparam_mapping,
        calibrate_mapping=calibrate_mapping,
        segment_episode_fn=segment_episode_fn,
        robot_config=robot_config,
        n_playback=args.n_playback,
        debug_ep_id=args.debug_ep_id,
        cache_masks=not args.no_cache_masks,
        discard_mask_cache=args.discard_mask_cache,
        max_points=args.max_points,
        env_args={"task_name": task_name, "raw_hdf5_dir": raw_hdf5_dir},
    )

    sg_params, schema_path = build_real_sg_params(
        parameters,
        skill_name,
        pc_instance_hdf5,
        sg_output_hdf5,
    )

    with open(schema_path, "r") as f:
        prior_graphs = parse_prior_graphs(json.load(f))

    postprocess_for_sgs(
        sg_params,
        prior_graphs=prior_graphs,
        n_playback=args.n_playback,
        num_workers=args.num_workers,
        visualize=args.visualize,
        debug_ep_id=args.debug_ep_id,
    )

    print(f"------{task_name} real data postprocessed via pc_instance plugin!------")


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-cfg",
        type=str,
        default="examples/pybullet/aloha_real/openworld_aloha/configs/sgBase.yaml",
    )
    parser.add_argument(
        "--skill-yaml",
        type=str,
        default="examples/pybullet/aloha_real/openworld_aloha/configs/skill_cfg/handoff_cup_per_skill.yaml",
    )
    parser.add_argument("--task-name", type=str, default=None)
    parser.add_argument("--hdf5-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--intermediate-hdf5", type=str, default=None)
    parser.add_argument("--output-hdf5", type=str, default=None)
    parser.add_argument("--sam3-path", type=str, default=None)
    parser.add_argument("--sam3-model-dir", type=str, default=None)
    parser.add_argument("--sam3-checkpoint", type=str, default=None)
    parser.add_argument("--seg-backend", type=str, default=None, choices=["sam3", "grounded_sam"])
    parser.add_argument("--grounded-sam-path", type=str, default=None)
    parser.add_argument("--gsam-sam2-checkpoint", type=str, default=None)
    parser.add_argument("--gsam-sam2-model-config", type=str, default=None)
    parser.add_argument("--gsam-gdino-config", type=str, default=None)
    parser.add_argument("--gsam-gdino-checkpoint", type=str, default=None)
    parser.add_argument("--gsam-box-threshold", type=float, default=None)
    parser.add_argument("--gsam-text-threshold", type=float, default=None)
    parser.add_argument("--gsam-lost-mask-min-pixels", type=int, default=None)
    parser.add_argument("--sam3-version", type=str, default="auto", choices=["auto", "sam3", "sam3.1"])
    parser.add_argument("--output-prob-thresh", type=float, default=0.5)
    parser.add_argument("--max-points", type=int, default=2048)
    parser.add_argument("--n-playback", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--debug-ep-id", type=int, default=None)
    parser.add_argument("--no-cache-masks", action="store_true")
    parser.add_argument("--discard-mask-cache", action="store_true")
    return parser


if __name__ == "__main__":
    postprocess_real_data(build_arg_parser().parse_args())
