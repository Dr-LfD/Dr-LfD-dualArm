import importlib
import importlib.util
import os
import sys
import types

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# Tests that exercise the contact-predictor image provider need the external ContactPrediction
# package (`contact_pred`); skip them when it is not importable so the suite stays green on a
# plain checkout. Install it + set CONTACT_PREDICTION_ROOT to run them end-to-end.
_NEEDS_CONTACT_PRED = (
    "builds_edge_text_obs",
    "prefers_processed_ts_observation_over_raw_obs",
    "flips_raw_obs_when_processed_obs_missing",
    "uses_single_selected_wrist_camera",
    "uses_robot_camera_map_for_arbitrary_robot_name",
    "unknown_lowdim_key_fails_before_model",
    "threshold_thresholds_runtime_score",
)
# Tests that need dmg configs carrying contact_predictor effect_detection annotations, which the
# DMG-only configs on this branch do not provide.
_NEEDS_EFFECT_DETECTION_CONFIG = (
    "load_runtime_schema_metadata_includes_contact_predictor_effect_detection",
    "threading_schema_uses_single_wrist_contact_predictor_config",
)


def pytest_collection_modifyitems(config, items):
    has_contact_pred = importlib.util.find_spec("contact_pred") is not None
    skip_cp = pytest.mark.skip(
        reason="requires external ContactPrediction repo (contact_pred package)"
    )
    skip_cfg = pytest.mark.skip(
        reason="requires dmg config with contact_predictor effect_detection annotations"
    )
    for item in items:
        if not has_contact_pred and any(s in item.nodeid for s in _NEEDS_CONTACT_PRED):
            item.add_marker(skip_cp)
        if any(s in item.nodeid for s in _NEEDS_EFFECT_DETECTION_CONFIG):
            item.add_marker(skip_cfg)


def stub_module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class _FakeDetector:
    def __init__(self, *args, **kwargs):
        pass


class _Sequence:
    def __init__(self, commands=None, name=None):
        self.commands = commands or []
        self.name = name


def install_robosuite_base_stubs(monkeypatch):
    monkeypatch.setitem(sys.modules, "examples.pybullet.aloha_real.openworld_aloha.primitives", stub_module(
        "examples.pybullet.aloha_real.openworld_aloha.primitives",
        Graphstate=object,
        GroupConf=object,
        GroupTrajectory=object,
        Sequence=_Sequence,
    ))
    monkeypatch.setitem(sys.modules, "examples.pybullet.aloha_real.openworld_aloha.run_openworld", stub_module(
        "examples.pybullet.aloha_real.openworld_aloha.run_openworld",
        prepare_world=lambda *args, **kwargs: (None, None, None),
        plan_detail_mp=lambda *args, **kwargs: (None, None),
    ))
    monkeypatch.setitem(sys.modules, "examples.pybullet.aloha_real.openworld_aloha.schema_executor", stub_module(
        "examples.pybullet.aloha_real.openworld_aloha.schema_executor",
        build_connector_motion=lambda *args, **kwargs: None,
        extract_static_environment=lambda *args, **kwargs: None,
        execute_schema_skeleton_plan=lambda *args, **kwargs: None,
        materialize_ref_goal_result=lambda *args, **kwargs: None,
    ))
    monkeypatch.setitem(sys.modules, "examples.pybullet.aloha_real.openworld_aloha.simple_worlds", stub_module(
        "examples.pybullet.aloha_real.openworld_aloha.simple_worlds",
        update_mesh=lambda *args, **kwargs: None,
    ))
    monkeypatch.setitem(sys.modules, "examples.pybullet.aloha_real.openworld_aloha.network_loader", stub_module(
        "examples.pybullet.aloha_real.openworld_aloha.network_loader",
        get_lfd_wrapper=lambda *args, **kwargs: None,
        update_equivSkill_wrapper=lambda *args, **kwargs: ({}, {"obs": {}}),
    ))
    monkeypatch.setitem(sys.modules, "examples.pybullet.aloha_real.scripts.standalone_scripts.schema_construction", stub_module(
        "examples.pybullet.aloha_real.scripts.standalone_scripts.schema_construction",
        load_runtime_schema_metadata=lambda *args, **kwargs: {"skill_meta_map": {}},
    ))
    monkeypatch.setitem(sys.modules, "examples.pybullet.aloha_real.scripts.tamp_workflow", stub_module(
        "examples.pybullet.aloha_real.scripts.tamp_workflow",
        ArmSchedulerState=object,
        ExecutionRequest=object,
        build_output_info=lambda *args, **kwargs: None,
        build_lane_checks=lambda *args, **kwargs: {},
        execute_request=lambda *args, **kwargs: None,
        get_action_target_object=lambda *args, **kwargs: None,
        get_batch_target_object=lambda *args, **kwargs: None,
        infer_stop_mode=lambda *args, **kwargs: None,
        merge_batch_sequences=lambda *args, **kwargs: None,
        plan_offline_sequence=lambda *args, **kwargs: None,
        plan_online_session=lambda *args, **kwargs: None,
        preserve_arm_confs=lambda *args, **kwargs: None,
        rollback_batch_execution=lambda state, execution_result: state,
        split_sequence_per_arm=lambda *args, **kwargs: None,
    ))
    monkeypatch.setitem(sys.modules, "examples.pybullet.aloha_real.openworld_aloha.symbolic_utils", stub_module(
        "examples.pybullet.aloha_real.openworld_aloha.symbolic_utils",
        search_facts=lambda *args, **kwargs: [],
        filter_subgoal=lambda *args, **kwargs: [],
        get_contact_action_subgoal=lambda *args, **kwargs: [],
        get_barrier_action_subgoal=lambda *args, **kwargs: [],
        build_scheduler_batches=lambda *args, **kwargs: {},
        convert_plan_to_skeleton=lambda *args, **kwargs: [],
        action_invalidates_perception=lambda *args, **kwargs: False,
        SimPrimitiveSubgoalDetector=_FakeDetector,
        BiopCompletionMonitor=type("BiopCompletionMonitor", (), {}),
        effect_monitor_sensor_data=lambda *args, **kwargs: {},
    ))


def import_robosuite_base(monkeypatch):
    install_robosuite_base_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules,
        "examples.pybullet.aloha_real.scripts.interleaved_dmg_osc_plugin",
        raising=False,
    )
    module = importlib.import_module(
        "examples.pybullet.aloha_real.scripts.interleaved_dmg_osc_plugin"
    )
    # In production SegReuseRegistry is initialized by prepare_world(); unit tests construct
    # ContactPredictorWrapper directly, so seed a default seg pairing here (reset after the test).
    from examples.pybullet.aloha_real.openworld_aloha import contact_monitoring as _cm
    monkeypatch.setattr(
        _cm.SegReuseRegistry,
        "_pairing",
        _cm.SegBackendPairing("sam3", "sam3", reuse_mode="sam3"),
        raising=False,
    )
    return module
