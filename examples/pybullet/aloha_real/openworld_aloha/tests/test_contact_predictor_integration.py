import importlib
import os
import sys
import tempfile
import types

import numpy as np
import pytest

from conftest import stub_module, import_robosuite_base, ROOT

# ContactPredictorWrapper requires contact_repo_root to be an existing directory (it is added to
# sys.path for the lazy contact_pred model import). For unit tests the model is monkeypatched, so
# any existing directory works; point CONTACT_PREDICTION_ROOT at a real ContactPrediction checkout
# to exercise the model end-to-end, otherwise fall back to a throwaway temp dir.
_CONTACT_PRED_ROOT = os.environ.get("CONTACT_PREDICTION_ROOT") or tempfile.mkdtemp(
    prefix="contact_pred_stub_"
)

THREADING_SINGLE_WRIST_CHECKPOINT = (
    "/home/user/yzchen_ws/imitation_learning/contact_prediction/data/outputs/"
    "2026.05.04/threading/checkpoints/latest.ckpt"
)


def _install_symbolic_utils_stubs(monkeypatch):
    monkeypatch.setitem(sys.modules, "pddlstream", stub_module("pddlstream"))
    monkeypatch.setitem(sys.modules, "pddlstream.language", stub_module("pddlstream.language"))
    monkeypatch.setitem(sys.modules, "pddlstream.language.constants", stub_module(
        "pddlstream.language.constants",
        PDDLProblem=object,
    ))
    monkeypatch.setitem(sys.modules, "pddlstream.algorithms", stub_module("pddlstream.algorithms"))
    monkeypatch.setitem(sys.modules, "pddlstream.algorithms.constraints", stub_module(
        "pddlstream.algorithms.constraints",
        WILD=object(),
    ))
    monkeypatch.setitem(sys.modules, "examples.pybullet.aloha_real.openworld_aloha.primitives", stub_module(
        "examples.pybullet.aloha_real.openworld_aloha.primitives",
        Conf=object,
        GroupConf=object,
        Grasp=object,
        Trajectory=object,
        GroupTrajectory=object,
        Sequence=object,
        Command=object,
    ))


def _import_symbolic_utils(monkeypatch):
    _install_symbolic_utils_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules,
        "examples.pybullet.aloha_real.openworld_aloha.symbolic_utils",
        raising=False,
    )
    return importlib.import_module(
        "examples.pybullet.aloha_real.openworld_aloha.symbolic_utils"
    )


def _import_schema_construction(monkeypatch):
    monkeypatch.delitem(
        sys.modules,
        "examples.pybullet.aloha_real.scripts.standalone_scripts.schema_construction",
        raising=False,
    )
    return importlib.import_module(
        "examples.pybullet.aloha_real.scripts.standalone_scripts.schema_construction"
    )


def test_check_effect_achieved_uses_geometric_attach_when_no_backend(monkeypatch):
    module = _import_symbolic_utils(monkeypatch)

    skill_meta = {
        "skill_name": "robot0_grasp_piece_1",
        "matched_streams": ["LearnedAttach"],
        "grounding_arm": "robot0",
        "effect_detection": {"obj_eef_dist_threshold": 0.2},
    }
    sensor_data = {
        "eef_xyz": {"robot0": [0.0, 0.0, 0.0], "robot1": [1.0, 1.0, 1.0]},
        "obj_pose": [0.05, 0.0, 0.0],
        "obj_visible": True,
    }

    assert module.check_effect_achieved(skill_meta, sensor_data, env_type="sim") is True


def test_check_effect_achieved_contact_predictor_dispatches_to_wrapper(monkeypatch):
    module = _import_symbolic_utils(monkeypatch)
    calls = []

    class FakeContactPredictor:
        def predict_effect(self, skill_meta):
            calls.append(skill_meta["skill_name"])
            return True

    skill_meta = {
        "skill_name": "robot0_grasp_piece_1",
        "matched_streams": ["LearnedAttach"],
        "effect_detection": {
            "backend": "contact_predictor",
            "checkpoint": "/tmp/contact.ckpt",
            "contact_edge": "robot0_piece_1",
            "skip_frame": 3,
        },
    }
    sensor_data = {
        "eef_xyz": {"robot0": [0.0, 0.0, 0.0], "robot1": [1.0, 1.0, 1.0]},
        "obj_pose": None,
        "obj_visible": False,
        "contact_predictor": FakeContactPredictor(),
    }

    assert module.check_effect_achieved(skill_meta, sensor_data, env_type="sim") is True
    assert calls == ["robot0_grasp_piece_1"]


@pytest.mark.parametrize(
    "effect_detection, match",
    [
        (
            {"backend": "contact_predictor", "contact_edge": "robot0_piece_1"},
            "checkpoint",
        ),
        (
            {"backend": "contact_predictor", "checkpoint": "/tmp/contact.ckpt"},
            "contact_edge",
        ),
        (
            {
                "backend": "contact_predictor",
                "checkpoint": "/tmp/contact.ckpt",
                "contact_edge": "robot0_piece_1",
                "skip_frame": 0,
            },
            "skip_frame",
        ),
    ],
)
def test_check_effect_achieved_contact_predictor_validates_metadata(monkeypatch, effect_detection, match):
    module = _import_symbolic_utils(monkeypatch)

    class ValidatingPredictor:
        def predict_effect(self, skill_meta):
            det = dict(skill_meta.get("effect_detection") or {})
            if not det.get("checkpoint"):
                raise ValueError("effect_detection['checkpoint'] is required for contact_predictor.")
            if not det.get("contact_edge"):
                raise ValueError("effect_detection['contact_edge'] is required for contact_predictor.")
            skip_frame = det.get("skip_frame", 1)
            if not isinstance(skip_frame, int) or skip_frame < 1:
                raise ValueError("effect_detection['skip_frame'] must be an integer >= 1.")
            return True

    skill_meta = {
        "skill_name": "robot0_grasp_piece_1",
        "matched_streams": ["LearnedAttach"],
        "effect_detection": effect_detection,
    }
    sensor_data = {
        "eef_xyz": {"robot0": [0.0, 0.0, 0.0], "robot1": [1.0, 1.0, 1.0]},
        "obj_pose": None,
        "obj_visible": True,
        "contact_predictor": ValidatingPredictor(),
    }

    with pytest.raises(ValueError, match=match):
        module.check_effect_achieved(skill_meta, sensor_data, env_type="sim")


def test_biop_completion_monitor_timeout_only(monkeypatch):
    module = _import_symbolic_utils(monkeypatch)
    monitor = module.BiopCompletionMonitor(
        {
            "skill_name": "bimanual_clean",
            "matched_streams": [],
            "effect_detection": {"timeout_t": 3},
        },
        env_type="real",
    )

    assert monitor.update({}) is False
    assert monitor.update({}) is False
    assert monitor.update({}) is True


def test_biop_completion_monitor_timeout_requires_effect_when_configured(monkeypatch):
    module = _import_symbolic_utils(monkeypatch)
    monitor = module.BiopCompletionMonitor(
        {
            "skill_name": "bimanual_packing",
            "matched_streams": ["LearnedBiKeyPose"],
            "grounding_arm1": "left_arm",
            "grounding_arm2": "right_arm",
            "effect_detection": {
                "timeout_t": 3,
                "hand_hand_dist_threshold": 0.2,
                "hand_hand_dist_comparison": "lt",
            },
        },
        env_type="real",
    )
    sensor_data = {
        "eef_xyz": {
            "left_arm": [0.0, 0.0, 0.0],
            "right_arm": [0.1, 0.0, 0.0],
        },
        "gripper_vals": {
            "left_arm": 0.0,
            "right_arm": 0.0,
        },
    }

    assert monitor.update(sensor_data) is False
    assert monitor.update(sensor_data) is False
    assert monitor.update(sensor_data) is True


def test_biop_completion_monitor_timeout_with_unsatisfied_effect_stays_false(monkeypatch):
    module = _import_symbolic_utils(monkeypatch)
    monitor = module.BiopCompletionMonitor(
        {
            "skill_name": "bimanual_packing",
            "matched_streams": ["LearnedBiKeyPose"],
            "grounding_arm1": "left_arm",
            "grounding_arm2": "right_arm",
            "effect_detection": {
                "timeout_t": 2,
                "hand_hand_dist_threshold": 0.2,
                "hand_hand_dist_comparison": "lt",
            },
        },
        env_type="real",
    )
    sensor_data = {
        "eef_xyz": {
            "left_arm": [0.0, 0.0, 0.0],
            "right_arm": [0.3, 0.0, 0.0],
        },
        "gripper_vals": {
            "left_arm": 0.0,
            "right_arm": 0.0,
        },
    }

    assert monitor.update(sensor_data) is False
    assert monitor.update(sensor_data) is False
    assert monitor.update(sensor_data) is False


def test_sim_detector_uses_contact_predictor_without_target_object(monkeypatch):
    module = _import_symbolic_utils(monkeypatch)
    calls = []

    class FakeContactPredictor:
        def predict_effect(self, skill_meta):
            calls.append(skill_meta["skill_name"])
            return False

    detector = module.SimPrimitiveSubgoalDetector(
        {
            "robot0_grasp_piece_1": {
                "skill_name": "robot0_grasp_piece_1",
                "matched_streams": ["LearnedAttach"],
                "effect_detection": {
                    "backend": "contact_predictor",
                    "checkpoint": "/tmp/contact.ckpt",
                    "contact_edge": "robot0_piece_1",
                    "skip_frame": 3,
                },
            }
        },
        lfd=types.SimpleNamespace(
            get_cur_eef_xyz_robosuite=lambda: {
                "left_arm": [0.0, 0.0, 0.0],
                "right_arm": [1.0, 1.0, 1.0],
            }
        ),
        contact_predictor=FakeContactPredictor(),
    )

    lane_results = detector.detect(
        {"left": {"subgoal": [("DoneSkill", "robot0_grasp_piece_1")]}}
    )

    assert lane_results == {"left": False}
    assert calls == ["robot0_grasp_piece_1"]


def test_contact_predictor_wrapper_runs_every_call_when_skip_frame_one(monkeypatch, tmp_path):
    module = import_robosuite_base(monkeypatch)
    checkpoint = tmp_path / "contact.ckpt"
    checkpoint.write_bytes(b"")
    wrapper = module.ContactPredictorWrapper(
        lfd=types.SimpleNamespace(raw_obs={"robot0_eye_in_hand_image": [[[0, 0, 0]]]}),
        contact_repo_root=_CONTACT_PRED_ROOT,
    )
    skill_meta = {
        "skill_name": "robot0_grasp_piece_1",
        "effect_detection": {
            "backend": "contact_predictor",
            "checkpoint": str(checkpoint),
            "contact_edge": "robot0_piece_1",
            "skip_frame": 1,
        },
    }

    calls = []
    monkeypatch.setattr(
        wrapper,
        "_get_runtime",
        lambda det: {"camera_names": ["cam_robot0"]},
    )
    monkeypatch.setattr(
        wrapper,
        "_predict_label",
        lambda runtime, det: calls.append(det["contact_edge"]) or True,
    )

    wrapper.begin_monitoring([skill_meta])
    wrapper.observe_frame()
    wrapper.observe_frame()
    wrapper.end_monitoring()

    assert wrapper.predict_effect(skill_meta) is True
    assert calls == ["robot0_piece_1", "robot0_piece_1"]


def test_contact_predictor_wrapper_reuses_last_prediction_between_sampled_calls(monkeypatch, tmp_path):
    module = import_robosuite_base(monkeypatch)
    checkpoint = tmp_path / "contact.ckpt"
    checkpoint.write_bytes(b"")
    wrapper = module.ContactPredictorWrapper(
        lfd=types.SimpleNamespace(raw_obs={"robot0_eye_in_hand_image": [[[0, 0, 0]]]}),
        contact_repo_root=_CONTACT_PRED_ROOT,
    )
    skill_meta = {
        "skill_name": "robot0_grasp_piece_1",
        "effect_detection": {
            "backend": "contact_predictor",
            "checkpoint": str(checkpoint),
            "contact_edge": "robot0_piece_1",
            "skip_frame": 3,
        },
    }

    predictions = iter([True, False])
    calls = []
    monkeypatch.setattr(
        wrapper,
        "_get_runtime",
        lambda det: {"camera_names": ["cam_robot0"]},
    )
    monkeypatch.setattr(
        wrapper,
        "_predict_label",
        lambda runtime, det: calls.append(det["contact_edge"]) or next(predictions),
    )

    wrapper.begin_monitoring([skill_meta])
    wrapper.observe_frame()
    wrapper.observe_frame()
    wrapper.observe_frame()
    assert wrapper.predict_effect(skill_meta) is True
    wrapper.observe_frame()
    wrapper.end_monitoring()

    assert wrapper.predict_effect(skill_meta) is False
    assert calls == ["robot0_piece_1", "robot0_piece_1"]


def test_contact_effect_monitor_group_requires_all_predictions(monkeypatch):
    module = import_robosuite_base(monkeypatch)
    skill_metas = [
        {"skill_name": "left", "effect_detection": {"contact_edge": "left_edge"}},
        {"skill_name": "right", "effect_detection": {"contact_edge": "right_edge"}},
    ]
    calls = []

    class FakePredictorWrapper:
        def begin_monitoring(self, metas):
            calls.append(("begin", [meta["skill_name"] for meta in metas]))

        def observe_frame(self):
            calls.append(("observe",))

        def predict_effect(self, skill_meta):
            calls.append(("predict", skill_meta["skill_name"]))
            return skill_meta["skill_name"] == "left"

        def end_monitoring(self):
            calls.append(("end",))

        def close(self):
            pass

    monitor = module.ContactEffectMonitorGroup(FakePredictorWrapper(), skill_metas)

    assert monitor.update() is False
    monitor.close()
    assert calls == [
        ("begin", ["left", "right"]),
        ("observe",),
        ("predict", "left"),
        ("predict", "right"),
        ("end",),
    ]


def test_contact_predictor_wrapper_builds_edge_text_obs(monkeypatch, tmp_path):
    module = import_robosuite_base(monkeypatch)
    wrapper = module.ContactPredictorWrapper(
        lfd=types.SimpleNamespace(
            raw_obs={
                "robot0_eye_in_hand_image": [[[0, 0, 0]]],
                "robot1_eye_in_hand_image": [[[255, 255, 255]]],
            }
        ),
        contact_repo_root=_CONTACT_PRED_ROOT,
    )
    runtime = {
        "camera_names": ["cam_robot0", "cam_robot1"],
        "shape_meta": {
            "obs": {
                "cam_robot0": {"type": "rgb", "shape": [3, 1, 1]},
                "cam_robot1": {"type": "rgb", "shape": [3, 1, 1]},
                "edge_text": {"type": "low_dim", "shape": [768]},
            }
        },
    }
    det = {
        "contact_edge": "robot1_needle_obj",
        "camera_names": ["cam_robot0", "cam_robot1"],
    }
    monkeypatch.setattr(
        wrapper,
        "_get_edge_text_embedding",
        lambda runtime_arg, det_arg: np.ones(768, dtype=np.float32),
        raising=False,
    )

    obs_dict = wrapper._build_obs_dict(runtime, det)

    assert set(obs_dict) == {"cam_robot0", "cam_robot1", "edge_text"}
    assert obs_dict["edge_text"].shape == (1, 768)
    assert obs_dict["edge_text"].dtype == np.float32
    assert obs_dict["edge_text"][0, 0] == 1.0


def test_contact_predictor_wrapper_prefers_processed_ts_observation_over_raw_obs(monkeypatch, tmp_path):
    module = import_robosuite_base(monkeypatch)
    raw = np.array(
        [
            [[255, 0, 0]],
            [[0, 0, 255]],
        ],
        dtype=np.uint8,
    )
    processed = raw[::-1].astype(np.float32) / 255.0
    wrapper = module.ContactPredictorWrapper(
        lfd=types.SimpleNamespace(
            ts=types.SimpleNamespace(
                observation={
                    "robot0_eye_in_hand_image": np.moveaxis(processed, -1, 0),
                }
            ),
            raw_obs={"robot0_eye_in_hand_image": raw},
        ),
        contact_repo_root=_CONTACT_PRED_ROOT,
    )
    runtime = {
        "camera_names": ["cam_robot0"],
        "shape_meta": {
            "obs": {
                "cam_robot0": {"type": "rgb", "shape": [3, 2, 1]},
            }
        },
    }

    obs_dict = wrapper._build_obs_dict(runtime, {"contact_edge": "robot0_piece_1"})

    np.testing.assert_allclose(obs_dict["cam_robot0"][0, :, 0, 0], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(obs_dict["cam_robot0"][0, :, 1, 0], [1.0, 0.0, 0.0])


def test_contact_predictor_wrapper_flips_raw_obs_when_processed_obs_missing(monkeypatch, tmp_path):
    module = import_robosuite_base(monkeypatch)
    raw = np.array(
        [
            [[255, 0, 0]],
            [[0, 0, 255]],
        ],
        dtype=np.uint8,
    )
    wrapper = module.ContactPredictorWrapper(
        lfd=types.SimpleNamespace(
            raw_obs={"robot0_eye_in_hand_image": raw},
        ),
        contact_repo_root=_CONTACT_PRED_ROOT,
    )
    runtime = {
        "camera_names": ["cam_robot0"],
        "shape_meta": {
            "obs": {
                "cam_robot0": {"type": "rgb", "shape": [3, 2, 1]},
            }
        },
    }

    obs_dict = wrapper._build_obs_dict(runtime, {"contact_edge": "robot0_piece_1"})

    np.testing.assert_allclose(obs_dict["cam_robot0"][0, :, 0, 0], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(obs_dict["cam_robot0"][0, :, 1, 0], [1.0, 0.0, 0.0])


def test_contact_predictor_wrapper_uses_single_selected_wrist_camera(monkeypatch, tmp_path):
    module = import_robosuite_base(monkeypatch)
    wrapper = module.ContactPredictorWrapper(
        lfd=types.SimpleNamespace(
            raw_obs={
                "robot0_eye_in_hand_image": [[[0, 0, 0]]],
                "robot1_eye_in_hand_image": [[[255, 255, 255]]],
            }
        ),
        contact_repo_root=_CONTACT_PRED_ROOT,
    )
    runtime = {
        "camera_names": ["cam_robot0", "cam_robot1"],
        "single_camera_per_edge": True,
        "selected_camera_key": "cam_wrist",
        "robot_camera_map": {
            "robot0": "cam_robot0",
            "robot1": "cam_robot1",
        },
        "shape_meta": {
            "obs": {
                "cam_wrist": {"type": "rgb", "shape": [3, 1, 1]},
                "edge_text": {"type": "low_dim", "shape": [768]},
            }
        },
    }
    det = {"contact_edge": "robot1_piece_2"}
    monkeypatch.setattr(
        wrapper,
        "_get_edge_text_embedding",
        lambda runtime_arg, det_arg: np.ones(768, dtype=np.float32),
        raising=False,
    )

    obs_dict = wrapper._build_obs_dict(runtime, det)

    assert set(obs_dict) == {"cam_wrist", "edge_text"}
    assert obs_dict["cam_wrist"].shape == (1, 3, 1, 1)
    assert obs_dict["cam_wrist"][0, 0, 0, 0] == 1.0


def test_contact_predictor_wrapper_uses_robot_camera_map_for_arbitrary_robot_name(monkeypatch, tmp_path):
    module = import_robosuite_base(monkeypatch)
    wrapper = module.ContactPredictorWrapper(
        lfd=types.SimpleNamespace(
            raw_obs={
                "left_wrist_image": [[[0, 0, 0]]],
                "right_wrist_image": [[[255, 255, 255]]],
            }
        ),
        contact_repo_root=_CONTACT_PRED_ROOT,
    )
    runtime = {
        "camera_names": ["cam_left_wrist", "cam_right_wrist"],
        "single_camera_per_edge": True,
        "selected_camera_key": "cam_wrist",
        "robot_camera_map": {
            "left_arm": "cam_left_wrist",
            "right_arm": "cam_right_wrist",
        },
        "shape_meta": {
            "obs": {
                "cam_wrist": {"type": "rgb", "shape": [3, 1, 1]},
                "edge_text": {"type": "low_dim", "shape": [768]},
            }
        },
    }
    det = {
        "contact_edge": "right_arm_cup",
        "obs_key_map": {
            "cam_left_wrist": "left_wrist_image",
            "cam_right_wrist": "right_wrist_image",
        },
    }
    monkeypatch.setattr(
        wrapper,
        "_get_edge_text_embedding",
        lambda runtime_arg, det_arg: np.ones(768, dtype=np.float32),
        raising=False,
    )

    obs_dict = wrapper._build_obs_dict(runtime, det)

    assert set(obs_dict) == {"cam_wrist", "edge_text"}
    assert obs_dict["cam_wrist"][0, 0, 0, 0] == 1.0


def test_contact_predictor_wrapper_unknown_lowdim_key_fails_before_model(monkeypatch, tmp_path):
    module = import_robosuite_base(monkeypatch)
    wrapper = module.ContactPredictorWrapper(
        lfd=types.SimpleNamespace(raw_obs={"robot0_eye_in_hand_image": [[[0, 0, 0]]]}),
        contact_repo_root=_CONTACT_PRED_ROOT,
    )
    runtime = {
        "camera_names": ["cam_robot0"],
        "shape_meta": {
            "obs": {
                "cam_robot0": {"type": "rgb", "shape": [3, 1, 1]},
                "unexpected_lowdim": {"type": "low_dim", "shape": [4]},
            }
        },
    }

    with pytest.raises(KeyError, match="unexpected_lowdim"):
        wrapper._build_obs_dict(runtime, {"contact_edge": "robot0_piece_1"})


def test_contact_predictor_wrapper_missing_checkpoint_raises(monkeypatch, tmp_path):
    module = import_robosuite_base(monkeypatch)
    wrapper = module.ContactPredictorWrapper(
        lfd=types.SimpleNamespace(raw_obs={"robot0_eye_in_hand_image": [[[0, 0, 0]]]}),
        contact_repo_root=_CONTACT_PRED_ROOT,
    )
    skill_meta = {
        "skill_name": "robot0_grasp_piece_1",
        "effect_detection": {
            "backend": "contact_predictor",
            "checkpoint": "/definitely/missing/contact.ckpt",
            "contact_edge": "robot0_piece_1",
        },
    }

    with pytest.raises(FileNotFoundError, match="contact.ckpt"):
        wrapper.predict_effect(skill_meta)


def test_contact_predictor_wrapper_multi_edge_text_combines_with_any_default(monkeypatch, tmp_path):
    module = import_robosuite_base(monkeypatch)
    checkpoint = tmp_path / "contact.ckpt"
    checkpoint.write_bytes(b"")
    wrapper = module.ContactPredictorWrapper(
        lfd=types.SimpleNamespace(raw_obs={"cam_left_wrist": [[[0, 0, 0]]]}),
        contact_repo_root=_CONTACT_PRED_ROOT,
    )
    skill_meta = {
        "skill_name": "handoff_cup",
        "effect_detection": {
            "backend": "contact_predictor",
            "checkpoint": str(checkpoint),
            "contact_edge": "left_arm_cup",
            "edge_text": "left_arm_cup,right_arm_cup",
            "skip_frame": 1,
        },
    }

    per_edge_calls = []
    monkeypatch.setattr(wrapper, "_get_runtime", lambda det: {})
    monkeypatch.setattr(
        wrapper,
        "_predict_label",
        lambda runtime, det: per_edge_calls.append(det["contact_edge"])
        or det["contact_edge"] == "right_arm_cup",
    )

    assert wrapper.predict_effect(skill_meta) is True
    assert per_edge_calls == ["left_arm_cup", "right_arm_cup"]


def test_contact_predictor_wrapper_multi_edge_rule_all_requires_every_edge(monkeypatch, tmp_path):
    module = import_robosuite_base(monkeypatch)
    checkpoint = tmp_path / "contact.ckpt"
    checkpoint.write_bytes(b"")
    wrapper = module.ContactPredictorWrapper(
        lfd=types.SimpleNamespace(raw_obs={"cam_left_wrist": [[[0, 0, 0]]]}),
        contact_repo_root=_CONTACT_PRED_ROOT,
    )
    skill_meta = {
        "skill_name": "handoff_cup",
        "effect_detection": {
            "backend": "contact_predictor",
            "checkpoint": str(checkpoint),
            "contact_edge": "left_arm_cup",
            "edge_text": ["left_arm_cup", "right_arm_cup"],
            "multi_edge_rule": "all",
            "skip_frame": 1,
        },
    }

    monkeypatch.setattr(wrapper, "_get_runtime", lambda det: {})
    monkeypatch.setattr(
        wrapper,
        "_predict_label",
        lambda runtime, det: det["contact_edge"] == "left_arm_cup",
    )

    assert wrapper.predict_effect(skill_meta) is False


def test_contact_predictor_wrapper_threshold_thresholds_runtime_score(monkeypatch, tmp_path):
    module = import_robosuite_base(monkeypatch)
    checkpoint = tmp_path / "contact.ckpt"
    checkpoint.write_bytes(b"")
    wrapper = module.ContactPredictorWrapper(
        lfd=types.SimpleNamespace(raw_obs={"cam_left_wrist": [[[0, 0, 0]]]}),
        contact_repo_root=_CONTACT_PRED_ROOT,
    )

    class _FakeRuntimeSession:
        def __init__(self):
            self.scores = iter([0.3, 0.9])

        def predict_label(self, runtime, obs_dict):
            return next(self.scores)

    fake_session = _FakeRuntimeSession()
    monkeypatch.setattr(wrapper, "_get_runtime_session", lambda: fake_session)
    monkeypatch.setattr(wrapper, "_build_obs_dict", lambda runtime, det: {})

    # det0 = {"contact_edge": "robot0_piece_1", "contact_label_threshold": 0.5}
    det0 = {"contact_edge": "robot0_piece_1"}
    assert wrapper._predict_label({}, det0) is False
    assert wrapper._predict_label({}, det0) is True


def test_contact_predictor_wrapper_resolve_edges_requires_some_edge_spec(monkeypatch, tmp_path):
    module = import_robosuite_base(monkeypatch)
    checkpoint = tmp_path / "contact.ckpt"
    checkpoint.write_bytes(b"")
    wrapper = module.ContactPredictorWrapper(
        lfd=types.SimpleNamespace(raw_obs={"cam_left_wrist": [[[0, 0, 0]]]}),
        contact_repo_root=_CONTACT_PRED_ROOT,
    )
    skill_meta = {
        "skill_name": "handoff_cup",
        "effect_detection": {
            "backend": "contact_predictor",
            "checkpoint": str(checkpoint),
            "skip_frame": 1,
        },
    }

    with pytest.raises(ValueError, match="contact_edge"):
        wrapper.predict_effect(skill_meta)


def test_contact_predictor_wrapper_resolve_edge_object_name_uses_robot_camera_map():
    from examples.pybullet.aloha_real.openworld_aloha.contact_monitoring import (
        _resolve_edge_object_name,
    )

    robot_camera_map = {
        "left_arm": "cam_left_wrist",
        "right_arm": "cam_right_wrist",
    }
    assert _resolve_edge_object_name("left_arm_cup", robot_camera_map) == "cup"
    assert _resolve_edge_object_name("right_arm_cup", robot_camera_map) == "cup"

    with pytest.raises(ValueError, match="unknown_arm_cup"):
        _resolve_edge_object_name("unknown_arm_cup", robot_camera_map)


def test_merge_primitive_effect_detection_fills_checkpoint_from_defaults(monkeypatch):
    su = _import_symbolic_utils(monkeypatch)
    skill_meta = {
        "skill_name": "g",
        "effect_detection": {"backend": "contact_predictor", "contact_edge": "right_arm_cup"},
    }
    merged = su.merge_primitive_effect_detection(
        skill_meta,
        {"contact_predictor_checkpoint": "/path/to/ckpt"},
    )
    assert merged["checkpoint"] == "/path/to/ckpt"
    assert su.skill_requires_contact_predictor_checkpoint(
        skill_meta,
        {"contact_predictor_checkpoint": "/path/to/ckpt"},
    )


def test_unified_primitive_subgoal_detector_calls_predict_when_ckpt_set(monkeypatch):
    su = _import_symbolic_utils(monkeypatch)

    class FakePred:
        def __init__(self):
            self.calls = []

        def predict_effect(self, skill_meta):
            self.calls.append(skill_meta.get("skill_name"))
            return True

    pred = FakePred()
    skill_meta = {
        "skill_name": "right_arm_grasp_cup",
        "matched_streams": ["LearnedAttach"],
        "effect_detection": {
            "backend": "contact_predictor",
            "checkpoint": "/tmp/fake.ckpt",
            "contact_edge": "right_arm_cup",
        },
    }
    det = su.UnifiedPrimitiveSubgoalDetector(
        {"right_arm_grasp_cup": skill_meta},
        contact_predictor=pred,
        effect_detection_defaults={},
        env_type="real",
        lfd=None,
        robot_entity=None,
    )
    out = det.detect(
        {"left": {"subgoal": [("DoneSkill", "right_arm_grasp_cup")], "target_obj": None}}
    )
    assert out["left"] is True
    assert pred.calls == ["right_arm_grasp_cup"]


def test_unified_primitive_subgoal_detector_skips_predict_without_checkpoint(monkeypatch):
    su = _import_symbolic_utils(monkeypatch)

    class FakePred:
        def __init__(self):
            self.calls = []

        def predict_effect(self, skill_meta):
            self.calls.append(1)
            return True

    pred = FakePred()
    skill_meta = {
        "skill_name": "right_arm_grasp_cup",
        "matched_streams": ["LearnedAttach"],
        "effect_detection": {
            "backend": "contact_predictor",
            "contact_edge": "right_arm_cup",
        },
    }
    det = su.UnifiedPrimitiveSubgoalDetector(
        {"right_arm_grasp_cup": skill_meta},
        contact_predictor=pred,
        effect_detection_defaults={},
        env_type="real",
    )
    out = det.detect(
        {"left": {"subgoal": [("DoneSkill", "right_arm_grasp_cup")], "target_obj": None}}
    )
    assert out["left"] is True
    assert pred.calls == []


def test_unified_primitive_subgoal_detector_raises_without_predictor_when_ckpt_required(
    monkeypatch,
):
    su = _import_symbolic_utils(monkeypatch)
    skill_meta = {
        "skill_name": "right_arm_grasp_cup",
        "matched_streams": ["LearnedAttach"],
        "effect_detection": {
            "backend": "contact_predictor",
            "checkpoint": "/tmp/fake.ckpt",
            "contact_edge": "right_arm_cup",
        },
    }
    det = su.UnifiedPrimitiveSubgoalDetector(
        {"right_arm_grasp_cup": skill_meta},
        contact_predictor=None,
        effect_detection_defaults={},
        env_type="real",
    )
    with pytest.raises(ValueError, match="ContactPredictorWrapper"):
        det.detect(
            {"left": {"subgoal": [("DoneSkill", "right_arm_grasp_cup")], "target_obj": None}}
        )


def test_seg_registry_init_requires_sam3_paths():
    from examples.pybullet.aloha_real.openworld_aloha.contact_monitoring import (
        SegReuseRegistry,
        validate_seg_backend_pairing,
    )

    pairing = validate_seg_backend_pairing({"seg_branch": "sam3"})
    with pytest.raises(ValueError, match="sam3_worker_path"):
        SegReuseRegistry.init_for_pairing(pairing, {"seg_branch": "sam3"})


def test_contact_predictor_wrapper_masked_wrist_masks_frame_via_sam3(monkeypatch):
    _ = import_robosuite_base(monkeypatch)
    from examples.pybullet.aloha_real.openworld_aloha.contact_monitoring import (
        _MaskedLFDProvider,
    )
    from examples.pybullet.aloha_real.openworld_aloha.workers.sam3_mask_client import (
        apply_object_mask,
    )

    class _FakeSam3:
        def __init__(self):
            self.calls = []

        def segment(self, rgb_image, class_names):
            self.calls.append((rgb_image.shape, tuple(class_names)))
            mask = np.zeros(rgb_image.shape[:2], dtype=bool)
            mask[:, 1] = True
            return {class_names[0]: [mask]}

        def close(self):
            pass

    class _FakeBaseProvider:
        def __init__(self, image):
            self._image = image

        def get_image(self, obs_key):
            assert obs_key == "cam_left_wrist"
            return self._image

    image = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    sam3 = _FakeSam3()
    masked = _MaskedLFDProvider(_FakeBaseProvider(image), sam3, "cup", apply_object_mask)
    output = masked.get_image("cam_left_wrist")

    assert output.dtype == np.uint8
    assert output.shape == image.shape
    np.testing.assert_array_equal(output[0, 0], [0, 0, 0])
    np.testing.assert_array_equal(output[0, 1], [40, 50, 60])
    assert sam3.calls == [(image.shape, ("cup",))]


def test_load_runtime_schema_metadata_includes_contact_predictor_effect_detection(monkeypatch):
    module = _import_schema_construction(monkeypatch)
    yaml_path = os.path.join(
        ROOT,
        "examples/pybullet/aloha_real/openworld_aloha/configs/dmg_cfgs/two_arm_three_piece_assembly.yaml",
    )

    metadata = module.load_runtime_schema_metadata(
        [yaml_path],
        env_names=["two_arm_three_piece_assembly"],
        root_path=ROOT,
    )

    skill_meta = metadata["skill_meta_map"]["robot0_grasp_piece_1"]
    assert skill_meta["effect_detection"]["backend"] == "contact_predictor"
    assert skill_meta["effect_detection"]["contact_edge"] == "robot0_piece_1"
    assert skill_meta["effect_detection"]["skip_frame"] == 3
    assert skill_meta["effect_detection"]["camera_names"] == ["cam_robot0", "cam_robot1", "cam_agent"]


def test_threading_schema_uses_single_wrist_contact_predictor_config(monkeypatch):
    module = _import_schema_construction(monkeypatch)
    yaml_path = os.path.join(
        ROOT,
        "examples/pybullet/aloha_real/openworld_aloha/configs/dmg_cfgs/two_arm_threading.yaml",
    )

    metadata = module.load_runtime_schema_metadata(
        [yaml_path],
        env_names=["two_arm_threading"],
        root_path=ROOT,
    )

    expected = {
        "robot0_grasp_tripod_obj": ("robot0_tripod_obj", "robot0_eye_in_hand_image"),
        "robot1_grasp_needle_obj": ("robot1_needle_obj", "robot1_eye_in_hand_image"),
    }
    for skill_name, (contact_edge, obs_key) in expected.items():
        effect_detection = metadata["skill_meta_map"][skill_name]["effect_detection"]
        assert effect_detection["backend"] == "contact_predictor"
        assert effect_detection["checkpoint"] == THREADING_SINGLE_WRIST_CHECKPOINT
        assert effect_detection["contact_edge"] == contact_edge
        assert effect_detection["skip_frame"] == 3
        assert effect_detection["camera_names"] == ["cam_wrist"]
        assert effect_detection["obs_key_map"] == {"cam_wrist": obs_key}
