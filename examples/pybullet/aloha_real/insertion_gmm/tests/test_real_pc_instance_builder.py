import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import h5py
import numpy as np
import pytest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


BUILDER_PATH = Path(ROOT) / "examples/pybullet/aloha_real/insertion_gmm/real_pc_instance_builder.py"
PLUGIN_PATH = Path(ROOT) / "examples/pybullet/aloha_real/insertion_gmm/postprocess_real_plugin.py"


def _load_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_text_prompt_objects():
    module = _load_module("test_real_pc_instance_builder", BUILDER_PATH)
    assert module.parse_text_prompt_objects("cup.sponge.") == ["cup", "sponge"]
    assert module.parse_text_prompt_objects(["cup", "sponge"]) == ["cup", "sponge"]


def test_pad_or_sample_points_pads_short_sequences():
    module = _load_module("test_real_pc_instance_builder_pad", BUILDER_PATH)
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    padded = module.pad_or_sample_points(points, 4)

    assert padded.shape == (4, 3)
    assert np.allclose(padded[:2], points)
    assert np.allclose(padded[2:], 0.0)


def test_ensure_sam3_cuda_available_raises_clear_error():
    module = _load_module("test_real_pc_instance_builder_cuda", BUILDER_PATH)

    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    class FakeTorch:
        cuda = FakeCuda()

    with pytest.raises(RuntimeError, match="CUDA GPU is required for SAM3 real-data preprocessing"):
        module.ensure_sam3_cuda_available(FakeTorch())


def test_build_real_pc_instance_hdf5_writes_expected_contract(tmp_path):
    module = _load_module("test_real_pc_instance_builder_contract", BUILDER_PATH)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_episode = raw_dir / "episode_0.hdf5"
    camera_name = "camera_2"
    color_imgs = np.zeros((2, 2, 2, 3), dtype=np.uint8)
    depth_imgs = np.ones((2, 2, 2), dtype=np.uint16) * 1000
    qpos = np.zeros((2, 14), dtype=np.float32)
    actions = np.zeros((2, 14), dtype=np.float32)
    with h5py.File(raw_episode, "w") as f:
        obs = f.create_group("observations")
        obs.create_dataset("qpos", data=qpos)
        f.create_dataset("action", data=actions)
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset(f"camera_info_{camera_name}", data=json.dumps({"K": [1, 0, 0, 0, 1, 0, 0, 0, 1]}), dtype=dt)
        f.create_dataset(f"color_img_{camera_name}", data=color_imgs)
        f.create_dataset(f"depth_img_{camera_name}", data=depth_imgs)

    ext_json = tmp_path / "camera_pose.json"
    ext_json.write_text(json.dumps({"xyz": [0, 0, 0], "wxyz": [1, 0, 0, 0]}), encoding="utf-8")

    def fake_segment_episode_fn(*, color_imgs_by_cam, object_names):
        return {
            camera_name: {
                "cup": {
                    0: {
                        "mask": np.array([[True, False], [False, False]], dtype=bool),
                        "box_xywh": [0.1, 0.1, 0.2, 0.2],
                        "score": 0.9,
                    },
                    1: None,
                }
            }
        }

    def fake_point_projector_fn(_depth_img, mask, _camera_info, _pose_path, _calink):
        if np.any(mask):
            return np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        return np.zeros((0, 3), dtype=np.float32)

    def fake_robot_stream_builder_fn(_qpos, _robot_config):
        return {
            "robot0_joint_pos": np.zeros((2, 6), dtype=np.float32),
            "robot0_eef_pos": np.zeros((2, 3), dtype=np.float32),
            "robot0_eef_quat": np.zeros((2, 4), dtype=np.float32),
            "robot0_gripper_qpos": np.zeros((2, 1), dtype=np.float32),
            "robot0_gripper_qvel": np.zeros((2, 1), dtype=np.float32),
            "robot1_joint_pos": np.zeros((2, 6), dtype=np.float32),
            "robot1_eef_pos": np.zeros((2, 3), dtype=np.float32),
            "robot1_eef_quat": np.zeros((2, 4), dtype=np.float32),
            "robot1_gripper_qpos": np.zeros((2, 1), dtype=np.float32),
            "robot1_gripper_qvel": np.zeros((2, 1), dtype=np.float32),
        }

    output_hdf5 = tmp_path / "handoff_cup_pc_instance.hdf5"
    module.build_real_pc_instance_hdf5(
        raw_hdf5_dir=str(raw_dir),
        output_hdf5_path=str(output_hdf5),
        object_names=["cup"],
        camera_names=[camera_name],
        cam_extparam_mapping={camera_name: str(ext_json)},
        calibrate_mapping={camera_name: False},
        segment_episode_fn=fake_segment_episode_fn,
        point_projector_fn=fake_point_projector_fn,
        robot_stream_builder_fn=fake_robot_stream_builder_fn,
        robot_config={"robot_name": "aloha", "robots": ["robot0", "robot1"]},
        n_playback=1,
        cache_masks=True,
        discard_mask_cache=False,
        max_points=4,
        env_args={"task_name": "handoff_cup"},
    )

    with h5py.File(output_hdf5, "r") as f:
        assert "data" in f
        assert json.loads(f["data"].attrs["env_args"])["task_name"] == "handoff_cup"
        obs = f["data/demo_0/obs"]
        assert obs["cup_point_cloud"].shape == (2, 4, 3)
        assert obs["cup_visible"][()].tolist() == [True, False]
        assert "sam3_cache" in obs
        assert obs["sam3_cache"][camera_name]["cup"]["masks"].shape == (2, 2, 2)
        assert f["data/demo_0/actions"].shape == (2, 14)


def test_project_mask_to_world_points_filters_invalid_depth(monkeypatch, tmp_path):
    module = _load_module("test_real_pc_instance_builder_depth", BUILDER_PATH)

    geom = types.ModuleType("examples.pybullet.aloha_real.openworld_aloha.estimation.geometry")
    geom.cloud_from_depth = lambda *_args, **_kwargs: np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]],
        ],
        dtype=np.float32,
    )
    monkeypatch.setitem(sys.modules, "examples.pybullet.aloha_real.openworld_aloha.estimation.geometry", geom)

    policy = types.ModuleType("examples.pybullet.aloha_real.openworld_aloha.policy_simp")
    policy.get_compatible_campose = lambda *_args, **_kwargs: "pose"
    monkeypatch.setitem(sys.modules, "examples.pybullet.aloha_real.openworld_aloha.policy_simp", policy)

    utils = types.ModuleType("examples.pybullet.utils.pybullet_tools.utils")
    utils.tform_points = lambda _pose, pts: pts
    monkeypatch.setitem(sys.modules, "examples.pybullet.utils.pybullet_tools.utils", utils)

    pose_path = tmp_path / "pose.json"
    pose_path.write_text(json.dumps({"xyz": [0, 0, 0], "wxyz": [1, 0, 0, 0]}), encoding="utf-8")

    depth = np.array([[0.0, 1.0], [0.0, 2.0]], dtype=np.float32)
    mask = np.array([[True, True], [True, True]], dtype=bool)
    points = module.project_mask_to_world_points(
        depth,
        mask,
        {"K": [1, 0, 0, 0, 1, 0, 0, 0, 1]},
        str(pose_path),
        False,
    )

    assert points.shape == (2, 3)
    assert np.allclose(points, np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32))


def test_postprocess_real_data_builds_intermediate_then_calls_template(tmp_path, monkeypatch):
    template_module = types.ModuleType("examples.pybullet.aloha_real.insertion_gmm.postprocess_template_base")
    template_calls = {}

    def fake_postprocess_for_sgs(sg_params, **kwargs):
        template_calls["sg_params"] = sg_params
        template_calls["kwargs"] = kwargs

    template_module.parse_prior_graphs = lambda data: {"init": data}
    template_module.postprocess_for_sgs = fake_postprocess_for_sgs
    monkeypatch.setitem(sys.modules, "examples.pybullet.aloha_real.insertion_gmm.postprocess_template_base", template_module)

    builder_module = types.ModuleType("examples.pybullet.aloha_real.insertion_gmm.real_pc_instance_builder")
    builder_calls = {}

    def fake_build_real_pc_instance_hdf5(**kwargs):
        builder_calls.update(kwargs)
        output_path = Path(kwargs["output_hdf5_path"])
        with h5py.File(output_path, "w") as f:
            data = f.create_group("data")
            data.attrs["env_args"] = json.dumps({"task_name": "handoff_cup"})

    builder_module.build_real_pc_instance_hdf5 = fake_build_real_pc_instance_hdf5
    builder_module.build_sam3_predictor = lambda **_kwargs: object()
    builder_module.infer_interested_primitives = lambda _path: ["grasp", "bimanual"]
    builder_module.parse_text_prompt_objects = lambda prompt: [obj for obj in prompt.strip(".").split(".") if obj]
    builder_module.segment_episode_with_sam3 = lambda *_args, **_kwargs: {}
    monkeypatch.setitem(sys.modules, "examples.pybullet.aloha_real.insertion_gmm.real_pc_instance_builder", builder_module)

    ow_utils = types.ModuleType("examples.pybullet.aloha_real.openworld_aloha.open_world_utils")
    schema_json = tmp_path / "schema.json"
    schema_json.write_text(json.dumps({"initial_graph": [], "ModeChangeDetection": [{"description": "grasp"}]}), encoding="utf-8")
    skill_yaml = tmp_path / "handoff_cup_per_skill.yaml"
    skill_yaml.write_text("dummy", encoding="utf-8")
    ow_utils.load_yaml_params = lambda *_args, **_kwargs: {
        "skill_names": ["handoff_cup_per_skill"],
        "skill_yaml_paths": [str(skill_yaml)],
        "text_prompt": "cup.",
        "sam3_path": "/tmp/sam3",
        "handoff_cup_per_skill": {
            "skill_name": "handoff_cup",
            "pre_obj_names": ["cup"],
            "eff_obj_names": ["cup"],
            "output_dir": str(tmp_path / "out"),
            "h5_parent_dir": str(tmp_path / "raw"),
            "hand_obj_dist_threshold": 0.12,
            "holding_truncation": 10,
            "left": {"grasp": {}, "release": {}},
            "right": {"grasp": {}, "release": {}},
            "schema": {"path": str(schema_json)},
        },
    }
    ow_utils.get_camera_mappings = lambda _para: (
        {"camera_2": str(tmp_path / "realrobot")},
        {"camera_2": str(tmp_path / "camera_pose.json")},
        {"camera_2": False},
    )
    monkeypatch.setitem(sys.modules, "examples.pybullet.aloha_real.openworld_aloha.open_world_utils", ow_utils)

    plugin = _load_module("test_postprocess_real_plugin", PLUGIN_PATH)

    args = types.SimpleNamespace(
        base_cfg="base.yaml",
        skill_yaml=str(skill_yaml),
        task_name=None,
        hdf5_dir=str(tmp_path / "raw"),
        output_dir=str(tmp_path / "out"),
        intermediate_hdf5=None,
        output_hdf5=None,
        sam3_path="/tmp/sam3",
        sam3_model_dir=None,
        sam3_checkpoint=None,
        sam3_version="auto",
        output_prob_thresh=0.5,
        max_points=2048,
        n_playback=5,
        num_workers=2,
        visualize=False,
        debug_ep_id=None,
        no_cache_masks=False,
        discard_mask_cache=False,
    )

    plugin.postprocess_real_data(args)

    assert builder_calls["raw_hdf5_dir"] == str(tmp_path / "raw")
    assert builder_calls["object_names"] == ["cup"]
    assert builder_calls["cache_masks"] is True
    assert builder_calls["robot_config"]["robot_name"] == "aloha"
    assert builder_calls["robot_config"]["robots"] == ["robot0", "robot1"]
    assert template_calls["sg_params"]["input_hdf5_path"].endswith("handoff_cup_pc_instance.hdf5")
    assert template_calls["sg_params"]["output_hdf5_path"].endswith("_sg_5.hdf5")
    assert template_calls["sg_params"]["robots"] == ["robot0", "robot1"]
    assert template_calls["sg_params"]["interested_objs"] == ["cup"]


def test_postprocess_real_data_passes_debug_ep_id_to_template(tmp_path, monkeypatch):
    template_module = types.ModuleType("examples.pybullet.aloha_real.insertion_gmm.postprocess_template_base")
    seen = {}
    template_module.parse_prior_graphs = lambda data: {"init": data}
    template_module.postprocess_for_sgs = lambda _sg_params, **kwargs: seen.update(kwargs)
    monkeypatch.setitem(sys.modules, "examples.pybullet.aloha_real.insertion_gmm.postprocess_template_base", template_module)

    builder_module = types.ModuleType("examples.pybullet.aloha_real.insertion_gmm.real_pc_instance_builder")
    builder_module.build_real_pc_instance_hdf5 = lambda **kwargs: h5py.File(kwargs["output_hdf5_path"], "w").create_group("data").file.close()
    builder_module.build_sam3_predictor = lambda **_kwargs: object()
    builder_module.infer_interested_primitives = lambda _path: ["grasp"]
    builder_module.parse_text_prompt_objects = lambda prompt: [obj for obj in prompt.strip(".").split(".") if obj]
    builder_module.segment_episode_with_sam3 = lambda *_args, **_kwargs: {}
    monkeypatch.setitem(sys.modules, "examples.pybullet.aloha_real.insertion_gmm.real_pc_instance_builder", builder_module)

    ow_utils = types.ModuleType("examples.pybullet.aloha_real.openworld_aloha.open_world_utils")
    schema_json = tmp_path / "schema.json"
    schema_json.write_text(json.dumps({"initial_graph": [], "ModeChangeDetection": [{"description": "grasp"}]}), encoding="utf-8")
    skill_yaml = tmp_path / "handoff_cup_per_skill.yaml"
    skill_yaml.write_text("dummy", encoding="utf-8")
    ow_utils.load_yaml_params = lambda *_args, **_kwargs: {
        "skill_names": ["handoff_cup_per_skill"],
        "skill_yaml_paths": [str(skill_yaml)],
        "text_prompt": "cup.",
        "robot_name": "aloha",
        "handoff_cup_per_skill": {
            "skill_name": "handoff_cup",
            "pre_obj_names": ["cup"],
            "eff_obj_names": ["cup"],
            "output_dir": str(tmp_path / "out"),
            "h5_parent_dir": str(tmp_path / "raw"),
            "hand_obj_dist_threshold": 0.12,
            "holding_truncation": 10,
            "left": {"grasp": {}, "release": {}},
            "right": {"grasp": {}, "release": {}},
            "schema": {"path": str(schema_json)},
        },
    }
    ow_utils.get_camera_mappings = lambda _para: (
        {"camera_2": str(tmp_path / "realrobot")},
        {"camera_2": str(tmp_path / "camera_pose.json")},
        {"camera_2": False},
    )
    monkeypatch.setitem(sys.modules, "examples.pybullet.aloha_real.openworld_aloha.open_world_utils", ow_utils)

    plugin = _load_module("test_postprocess_real_plugin_debug", PLUGIN_PATH)
    args = types.SimpleNamespace(
        base_cfg="base.yaml",
        skill_yaml=str(skill_yaml),
        task_name=None,
        hdf5_dir=str(tmp_path / "raw"),
        output_dir=str(tmp_path / "out"),
        intermediate_hdf5=None,
        output_hdf5=None,
        sam3_path="/tmp/sam3",
        sam3_model_dir=None,
        sam3_checkpoint=None,
        sam3_version="auto",
        output_prob_thresh=0.5,
        max_points=2048,
        n_playback=5,
        num_workers=2,
        visualize=False,
        debug_ep_id=7,
        no_cache_masks=False,
        discard_mask_cache=False,
    )

    plugin.postprocess_real_data(args)
    assert seen["debug_ep_id"] == 7
