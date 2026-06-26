"""DMG TAMP planner with OSC_POSE absolute-EE execution (single flattened ``DMG_OSC_planner`` class).

Runs an online plan-and-execute loop over the sim/robosuite DMG route.
"""

import argparse
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as _SciRot


def _resolve_workspace_root():
    current = os.path.dirname(os.path.abspath(__file__))
    while not os.path.isfile(os.path.join(current, '.repo_root')):
        parent = os.path.dirname(current)
        if parent == current:
            raise RuntimeError("repo root marker '.repo_root' not found in any parent directory")
        current = parent
    return current


root_path = _resolve_workspace_root()
if root_path not in sys.path:
    sys.path.append(root_path)
os.chdir(root_path)

from examples.pybullet.aloha_real.openworld_aloha.contact_monitoring import (
    ContactEffectMonitorGroup,
    ContactPredictorWrapper,
    build_effect_monitor,
)
from examples.pybullet.aloha_real.openworld_aloha.network_loader import (
    get_lfd_wrapper,
    update_equivSkill_wrapper,
)
from examples.pybullet.aloha_real.openworld_aloha.primitives import (
    Graphstate,
    GroupConf,
    GroupTrajectory,
    Sequence,
)
from examples.pybullet.aloha_real.openworld_aloha.run_openworld import (
    plan_detail_mp,
    prepare_world,
)
from examples.pybullet.aloha_real.openworld_aloha.schema_executor import (
    build_connector_motion,
    execute_schema_skeleton_plan,
    extract_static_environment,
    materialize_ref_goal_result,
)
from examples.pybullet.aloha_real.openworld_aloha.simple_worlds import update_mesh
from examples.pybullet.aloha_real.openworld_aloha.symbolic_utils import (
    BiopCompletionMonitor,
    SimPrimitiveSubgoalDetector,
    build_scheduler_batches,
    convert_plan_to_skeleton,
    effect_monitor_sensor_data,
    get_barrier_action_subgoal,
    get_contact_action_subgoal,
    search_facts,
)
from examples.pybullet.aloha_real.scripts.standalone_scripts.schema_construction import (
    load_runtime_schema_metadata,
)
from examples.pybullet.aloha_real.scripts.tamp_workflow import (
    ArmSchedulerState,
    ExecutionRequest,
    build_output_info,
    execute_request,
    get_action_target_object,
    get_batch_target_object,
    infer_stop_mode,
    merge_batch_sequences,
    plan_online_session,
    preserve_arm_confs,
    split_sequence_per_arm,
)


def _is_frozen(primitive):
    """True if this command holds a grasped object (grip frozen during the move)."""
    return bool(getattr(primitive, 'freeze_gripper', False))


class DMG_OSC_planner(object):
    """DMG bimanual TAMP planner executed under OSC_POSE absolute EE control."""

    #: How many times a single stage is re-planned+executed before aborting.
    MAX_STAGE_RETRIES = 3

    #: Right-multiplied onto every FK'd / learned tool pose to map the pybullet
    #: tool frame onto the robosuite ``grip_site`` frame that OSC_POSE tracks.
    #: For the dual-Panda DMG robot the tool link ``panda*_ee_link`` coincides
    #: with ``right_hand`` and ``grip_site`` is mounted with a -90 deg yaw.
    GRIP_SITE_CORRECTION = _SciRot.from_euler('z', -np.pi / 2)

    #: Local tool-frame translation added to every commanded position. OSC_POSE
    #: servos robosuite's ``grip_site`` (link8 + 0.0965 m) while the pybullet
    #: ``panda*_ee_link`` sits at link8 + 0.100 m; the raw FK position would drive
    #: the hand 3.5 mm past the planned pose along the approach axis.
    GRIP_SITE_POS_OFFSET = np.array([0.0, 0.0, -0.0035])

    #: Closing-ramp retiming (see ``_retime_close_ramps``): played back verbatim
    #: the demo gripper channel closes too early, so detected closing ramps are
    #: moved onto the grasp waypoint where the executor settles, snaps closed, and
    #: dwells until the fingers finish before motion resumes.
    GRIPPER_RAMP_EPS = 1e-3       #: per-waypoint command delta that counts as a ramp
    GRIPPER_RAMP_CLOSE_ONLY = True  #: retime only closing ramps; opening plays live
    GRIPPER_SETTLE_STEPS = 10     #: steps converging on the hold waypoint before closing
    GRIPPER_DWELL_STEPS = 15      #: steps holding after the close so the fingers finish

    def __init__(self, para):
        self.para = para
        self.sg_param = para['sg_params']
        self.task_name = self.sg_param['task_name']
        self.success = False
        self.effect_monitor = None
        # Last commanded gripper joint per side, carried across stage executions so a
        # held grip is re-asserted during transfer instead of re-read from live qpos.
        self._carried_gripper_cmd = {}

        self.env_type = 'mj'
        self.vid_save_path = os.path.join(root_path, para['vid_save_path'])
        os.makedirs(self.vid_save_path, exist_ok=True)

        shape_meta = self._setup_skills()
        self._setup_env(shape_meta)
        belief = self._setup_world_and_perception()
        self._build_tamp_kwargs()
        self.online_tamp(belief)

    # ------------------------------------------------------------------
    # Setup phases
    # ------------------------------------------------------------------

    def _setup_skills(self):
        """Load equivariant-skill wrappers, schema metadata and interested objects."""
        self.equivSkill_info_dict, shape_meta = update_equivSkill_wrapper(
            self.sg_param, self.para, get_sgs_from_hdf5_fn=self.get_sgs_from_hdf5
        )

        # self.skillwise_sgs is the flattened union across equivariant-skill prefixes.
        all_skillwise_sgs = {}
        for prefix_key in self.equivSkill_info_dict.keys():
            all_skillwise_sgs.update(self.equivSkill_info_dict[prefix_key]['skillwise_sgs'])
        self.skillwise_sgs = all_skillwise_sgs

        # env_names must be the runtime skill keys (the same list compute_skill_names
        # feeds), so schema_skill_metas is keyed by the PDDL skill names. The biop
        # checkpoint's bimanual scene graph is merged into skillwise_sgs by
        # update_equivSkill_wrapper, so its keys carry 'bimanual_0'; passing the
        # prefix keys (e.g. ['per_skill']) instead drops the bimanual skill.
        self.schema_skill_metas = load_runtime_schema_metadata(
            self.para.get('skill_yaml_paths') or [],
            env_names=list(self.skillwise_sgs.keys()),
            root_path=root_path,
        )["skill_meta_map"]

        if 'interested_objs' in self.sg_param:
            self.interested_objects = list(self.sg_param['interested_objs'].keys())
        else:
            all_interested_objs = set()
            for skill_info in self.skillwise_sgs.values():
                all_interested_objs |= set(skill_info['related_objs'])
            self.interested_objects = list(all_interested_objs)

        return shape_meta

    def _setup_env(self, shape_meta):
        """Initialize the robosuite env through the LfD wrapper and snapshot joint pose."""
        self.env_options = self.get_env_options(shape_meta)

        self.lfd = get_lfd_wrapper(self.para, with_planning=True)
        if self.lfd is None:
            raise RuntimeError(
                "get_lfd_wrapper returned None; the DMG OSC route requires a live LfD "
                "wrapper (check LfD_params/DP_input config)."
            )
        self.lfd.initialize_env(
            env_name=self.sg_param['task_name'], env_options=self.env_options
        )
        # Live joint pose, used to align the pybullet robot with the robosuite env.
        self.initial_jposes = self.lfd.get_cur_jpose_robosuite()
        os.chdir(root_path)

    def _setup_world_and_perception(self):
        """Build the pybullet world + belief and the perception/effect detectors."""
        self._initial_pc_dict = self.save_mj_observation(npz_path=None)

        if hasattr(self.lfd.env, 'workspace_offset'):
            table_center = self.lfd.env.workspace_offset
        elif hasattr(self.lfd.env, 'table_offset'):
            table_center = self.lfd.env.table_offset
        franka_base_pos = self.lfd.env.robots[0].base_pos

        self.robot_entity, belief, _ = prepare_world(
            self.para, env_type=self.env_type,
            mj_pc_dict=self._initial_pc_dict,
            franka_base_pos=franka_base_pos,
            table_center=table_center,
        )
        if self.initial_jposes is not None:
            self.robot_entity.revise_initial_pose(self.initial_jposes)

        contact_predictor_checkpoint = self.para.get("contact_predictor_checkpoint")
        if contact_predictor_checkpoint:
            self._contact_predictor_wrapper = ContactPredictorWrapper(
                self.lfd,
                output_dir=self.vid_save_path,
                contact_repo_root=self.para.get("contact_prediction_root"),
                effect_detection_defaults={
                    "contact_predictor_checkpoint": contact_predictor_checkpoint,
                    "sam3_worker_path": self.para.get("sam3_worker_path"),
                    "sam3_path": self.para.get("sam3_path"),
                    "sam3_model_dir": self.para.get("sam3_model_dir"),
                    "sam3_checkpoint": self.para.get("sam3_checkpoint"),
                    "sam3_conda_env": self.para.get("sam3_conda_env"),
                    "sam3_conda_bin": self.para.get("sam3_conda_bin"),
                },
            )
        else:
            self._contact_predictor_wrapper = None
        self._primitive_subgoal_detector = SimPrimitiveSubgoalDetector(
            self.schema_skill_metas,
            self.lfd,
            contact_predictor=self._contact_predictor_wrapper,
        )
        return belief

    def _build_tamp_kwargs(self):
        """Assemble the stream kwargs shared by every TAMP call on this route."""
        self.tamp_kwargs_dict = {
            'task_name': self.task_name,
            'skillwise_sgs': self.skillwise_sgs,
            'min_ee_dist': self.sg_param['hand_hand_dist_threshold'],
            'interested_objects': self.interested_objects,
            'max_tamp_time': self.sg_param['max_tamp_time'],
            'collision_distance': self.sg_param['attachment_collision_distance'],
            'equivSkill_info_dict': self.equivSkill_info_dict,
            'initial_pc_dict': self._initial_pc_dict,
            # Route OSC EE-trajectory generation to the streams so
            # get_imitate_traj_fn / get_plan_motion_fn take their ee_traj_mode branch.
            'ee_traj_mode': True,
        }

    # ------------------------------------------------------------------
    # DMG environment / skill-graph specialization
    # ------------------------------------------------------------------

    def get_env_options(self, shape_meta):
        cam_keys = [obs_key for obs_key in shape_meta['obs'].keys() if "_image" in obs_key]
        img_shape = shape_meta['obs'][cam_keys[0]].shape
        env_options = {
            "env_configuration": "single-arm-parallel",
            "robots": ["Panda", "Panda"],
            "camera_names": [cam_key.replace("_image", "") for cam_key in cam_keys],
            "camera_heights": img_shape[1],
            "camera_widths": img_shape[2],
            "camera_segmentations": "instance",
        }
        # SDP consumes a fused scene point_cloud obs, which the env only emits when
        # output_all_pcds is set. DP is image-only, so leave it off there.
        if self.para['LfD_params']['lfd_alg'] == 'SDP':
            env_options["output_all_pcds"] = True
        return env_options

    def get_sgs_from_hdf5(self, hdf5_path):
        import json

        import h5py
        import networkx as nx

        skillwise_sg = {}
        with h5py.File(hdf5_path, 'r') as f:
            sg_info = f['sg_info']
            for skill_name, skill_info in sg_info.items():
                # Keep only skills whose name matches an interested primitive.
                if not any(kw in skill_name for kw in self.sg_param['interested_primitives']):
                    continue
                skillwise_sg[skill_name] = {}
                for sg_phase_key in ['pre']:
                    sg_str = skill_info[f'{sg_phase_key}_sg'][()].decode('utf-8')
                    sg = nx.node_link_graph(json.loads(sg_str))
                    sg.graph['obj_names'] = self.sg_param['interested_objs'].keys()
                    sg.graph['hand_names'] = self.sg_param['robots']
                    skillwise_sg[skill_name][sg_phase_key] = sg
        return skillwise_sg

    # ------------------------------------------------------------------
    # Perception
    # ------------------------------------------------------------------

    def _resolve_use_pc_color(self):
        """Read use_pc_color from every equivariant skill wrapper; assert they agree."""
        use_color_by_prefix = {}
        for prefix_key, skill_info in self.equivSkill_info_dict.items():
            assert 'tamp_wrapper' in skill_info, (
                f"Missing tamp_wrapper for equivariant skill prefix {prefix_key!r}"
            )
            wrapper = skill_info['tamp_wrapper']
            assert hasattr(wrapper, 'cfg'), (
                f"Missing cfg on tamp_wrapper for equivariant skill prefix {prefix_key!r}"
            )
            try:
                use_color = bool(wrapper.cfg.data.dataset.get('use_pc_color', False))
            except Exception as exc:
                raise AssertionError(
                    f"Could not read cfg.data.dataset.use_pc_color for "
                    f"equivariant skill prefix {prefix_key!r}"
                ) from exc
            use_color_by_prefix[prefix_key] = use_color

        assert use_color_by_prefix, "No equivariant skill wrappers found in equivSkill_info_dict"
        assert len(set(use_color_by_prefix.values())) == 1, (
            f"All equivariant skill wrappers must agree on use_pc_color; "
            f"got {use_color_by_prefix}"
        )
        return next(iter(use_color_by_prefix.values()))

    def save_mj_observation(self, npz_path=None, cache_image=True):
        """Capture per-instance point clouds (and optionally the agentview image)."""
        mj_pc_dict = {}
        use_color = self._resolve_use_pc_color()
        ncols = 6 if use_color else 3

        def _select_pc(arr, key):
            if use_color:
                assert arr.ndim == 2 and arr.shape[1] >= 6, (
                    f"use_pc_color=True requires Nx6 (xyz+rgb) array for '{key}'; "
                    f"got shape {arr.shape}"
                )
            return arr[:, :ncols]

        pc_instance_obstacles_dict, visible_dict = self.lfd.get_instance_and_obstacles_pcd(
            self.lfd.raw_obs, require_downsample=False,
            get_obstacle=self.para['require_obstacle_pc'],
        )
        if 'obstacle_pcd' in pc_instance_obstacles_dict:
            mj_pc_dict['fixed_obstacles'] = _select_pc(
                pc_instance_obstacles_dict['obstacle_pcd'], 'obstacle_pcd'
            )
        for obj in self.interested_objects:
            if visible_dict[f'{obj}_visible'] is False:
                mj_pc_dict[obj] = None
            else:
                pc_key = f'{obj}_point_cloud'
                mj_pc_dict[obj] = _select_pc(pc_instance_obstacles_dict[pc_key], pc_key)

        if npz_path is not None:
            np.savez(npz_path, **mj_pc_dict)

        if cache_image:
            table_setting_img = self.lfd.raw_obs['agentview_image'][::-1]
            import cv2
            cv2.imwrite(
                os.path.join(self.vid_save_path, f'{self.task_name}_setting.png'),
                table_setting_img,
            )
        return mj_pc_dict

    def _refresh_object_meshes(self, target_objects):
        """Capture a fresh point cloud and update meshes for the given entity objects."""
        if not target_objects:
            return
        mj_pc_dict = self.save_mj_observation(npz_path=None)
        for estimated_obj in target_objects:
            pc = mj_pc_dict.get(estimated_obj.category)
            if pc is None:
                continue
            update_mesh(estimated_obj, pc, category=estimated_obj.category)

    def _build_perception_home_connector(self):
        """Build per-side arm connector motions from the live joint pose to home.

        Returns one Sequence per active arm side. Raises if the live pose is
        unavailable or any connector cannot be planned.
        """
        live_jposes = self.lfd.get_cur_jpose_robosuite()
        if live_jposes is None:
            raise RuntimeError("Cannot build perception-home connector: live joint pose unavailable")
        default_conf = self.robot_entity.get_default_conf()
        sides = ['left'] if len(self.sg_param['robots']) == 1 else ['left', 'right']
        connectors = []
        for side in sides:
            arm_group = f'{side}_arm'
            cur_conf = GroupConf(self.robot_entity, arm_group, live_jposes[arm_group])
            tgt_conf = GroupConf(self.robot_entity, arm_group, default_conf[arm_group])
            connector = build_connector_motion(
                self.robot_entity, arm_group, cur_conf, tgt_conf,
                self.current_state, self.static_environment,
            )
            if connector is None:
                raise RuntimeError(f"Cannot build perception-home connector for {arm_group}")
            connectors.append(connector)
        return connectors

    def _execute_perception_connector(self, connectors):
        """Stream per-arm home connectors through the OSC controller (no replan/BC)."""
        combined_commands = []
        for seq in connectors:
            combined_commands.extend(getattr(seq, 'commands', []))
        combined_seq = Sequence(combined_commands)
        # iterate_process_sequence emits 14-D OSC_POSE actions, so the home motion
        # must run under the same OSC_POSE abs controller execute_robosuite_plan uses.
        self.lfd.update_controllers(controller_name="OSC_POSE", abs_action=True)
        for action in self.iterate_process_sequence(self.robot_entity, combined_seq):
            self.replay_tamp_every(action, duration=0.01)

    def _refresh_affected_object_meshes(self, target_objects, reset_home=False):
        if not target_objects:
            return
        if reset_home:
            self._execute_perception_connector(self._build_perception_home_connector())
        self._refresh_object_meshes(target_objects)

    def perceive_env(self, updated_state, lane_checks):
        """Post-execution subgoal detection; returns (state, achieved, results).

        Refreshes the meshes of the objects the stage moved, detects whether each
        subgoal was achieved, and on a miss retries once with the arms moved home
        (occlusion recovery). Does not commit ``self.current_state``.
        """
        if not lane_checks:
            updated_state = preserve_arm_confs(
                self.current_state, updated_state, self.robot_entity.arms, search_facts
            )
            return updated_state, True, {}

        added = updated_state - self.current_state
        removed = self.current_state - updated_state
        affected_objects = [f[1] for f in search_facts(added | removed, 'atpose')]

        self._refresh_affected_object_meshes(affected_objects, reset_home=False)
        lane_results = self._primitive_subgoal_detector.detect(
            lane_checks, affected_objects=affected_objects
        )
        subgoal_achieved = all(lane_results.values()) if lane_results else True
        if not subgoal_achieved:
            print('Subgoal not achieved after perception, replan locally')
            self._refresh_affected_object_meshes(affected_objects, reset_home=True)
            lane_results = self._primitive_subgoal_detector.detect(
                lane_checks, affected_objects=affected_objects
            )
            subgoal_achieved = all(lane_results.values()) if lane_results else True

        updated_state = preserve_arm_confs(
            self.current_state, updated_state, self.robot_entity.arms, search_facts
        )
        return updated_state, subgoal_achieved, lane_results

    # ------------------------------------------------------------------
    # Contact / effect monitoring
    # ------------------------------------------------------------------

    def _begin_contact_monitoring(self, subgoals):
        """Build the effect monitor for a stage from its per-key subgoal facts."""
        skill_metas = self._select_effect_monitor_skill_metas(subgoals)
        if not skill_metas:
            self.effect_monitor = None
        elif len(skill_metas) == 1:
            self.effect_monitor = build_effect_monitor(
                skill_metas[0], self._contact_predictor_wrapper, 'sim'
            )
        elif all(
            (skill_meta.get("effect_detection") or {}).get("backend") == "contact_predictor"
            for skill_meta in skill_metas
        ):
            self.effect_monitor = ContactEffectMonitorGroup(
                self._contact_predictor_wrapper, skill_metas
            )
        else:
            self.effect_monitor = None

    def _end_contact_monitoring(self):
        if self.effect_monitor is not None and hasattr(self.effect_monitor, "close"):
            self.effect_monitor.close()
        self.effect_monitor = None

    def _select_effect_monitor_skill_metas(self, subgoals):
        if len(self.sg_param.get('robots', [])) < 2:
            return []
        skill_metas = []
        for lane_subgoal in subgoals.values():
            if not lane_subgoal:
                continue
            for fact in lane_subgoal:
                if "doneskill" not in str(fact[0]).lower():
                    continue
                skill_name = getattr(fact[1], "value", fact[1])
                skill_meta = self.schema_skill_metas.get(skill_name)
                if (
                    skill_meta is not None
                    and skill_meta.get("effect_detection")
                    and self._is_bimanual_effect_skill(skill_meta)
                ):
                    skill_metas.append(skill_meta)
        return skill_metas

    def _is_bimanual_effect_skill(self, skill_meta):
        streams = set(skill_meta.get("matched_streams") or [])
        skill_name = str(skill_meta.get("skill_name", "")).lower()
        return (
            "LearnedBiKeyPose" in streams
            or "bimanual" in skill_name
            or (
                skill_meta.get("grounding_arm1") is not None
                and skill_meta.get("grounding_arm2") is not None
            )
        )

    def _effect_monitor_completed(self):
        if self.effect_monitor is None:
            return False
        if isinstance(self.effect_monitor, BiopCompletionMonitor):
            sensor_data = effect_monitor_sensor_data(
                self.lfd, self.robot_entity,
                contact_predictor=self._contact_predictor_wrapper,
            )
            return self.effect_monitor.update(sensor_data)
        return self.effect_monitor.update()

    # ------------------------------------------------------------------
    # Online planning + execution
    # ------------------------------------------------------------------

    def online_tamp(self, belief):
        session = plan_online_session(
            self.para, self.robot_entity, belief, **self.tamp_kwargs_dict
        )
        print(f"Duration of TAMP: {session.duration_tamp}")

        self.problem = session.problem
        self.stream_info = session.stream_info
        self.static_environment = extract_static_environment(session.problem)

        global_solution = session.global_solution
        if global_solution is None or global_solution.plan is None:
            print('\033[31m[online_tamp] No TAMP plan found within budget; '
                  'reporting task_success=False.\033[0m')
            self.output_info = build_output_info(
                task_name=self.task_name,
                task_success=self.success,
                duration_tamp=session.duration_tamp,
                recording_path=None,
            )
            return

        scheduler_batches = build_scheduler_batches(
            global_solution.plan, session.state_history
        )
        scheduler = ArmSchedulerState(
            left_remaining=list(scheduler_batches['left']),
            right_remaining=list(scheduler_batches['right']),
            pending_barriers=list(scheduler_batches['barriers']),
            phase_quotas=list(scheduler_batches['phase_quotas']),
            scene_perceived=True,
        )
        self.current_state = session.state_history[0]

        # Refine + execute one stage at a time. Perceive() is realized
        # by prepare_world() for the first stage and by perceive_env()'s mesh refresh
        # at the end of each stage for the next, so the detailed plan is always
        # grounded in the latest observed object poses.
        while True:
            stage = self._next_stage(scheduler)
            if stage is None:
                break
            kind, payload = stage

            achieved = False
            for attempt in range(self.MAX_STAGE_RETRIES):
                # Schema fast-path first; re-solve a local PDDLStream problem only
                # after a verification miss.
                detailed = self._plan_stage(kind, payload, use_detail_mp=attempt > 0)
                if detailed is None:
                    print(f'\033[33m[online_tamp] {kind} stage planning failed, '
                          f'retry {attempt + 1}/{self.MAX_STAGE_RETRIES}\033[0m')
                    continue
                seq, unit_states, checks = detailed
                status, verified_state, achieved, results = self._execute_and_verify(
                    seq, unit_states, checks
                )
                self.current_state = verified_state
                scheduler.scene_perceived = True
                if status in ('fail', 'success'):
                    achieved = status == 'success'
                    break
                if kind == 'lanes':
                    # Advance the lanes that verified and retry only the rest, so a
                    # successful pick/place is never re-executed for the other arm's
                    # sake. Failed lanes' effects were reverted in _execute_and_verify.
                    for lane in list(payload):
                        if results.get(lane, True):
                            scheduler.pop_lane_batch(lane)
                    payload = {lane: batch for lane, batch in payload.items()
                               if not results.get(lane, True)}
                    if not payload:
                        achieved = True
                        break
                elif achieved:
                    break
                print(f'\033[33m[online_tamp] {kind} stage subgoal not achieved, '
                      f'retry {attempt + 1}/{self.MAX_STAGE_RETRIES}\033[0m')

            if self.success:
                break
            if not achieved:
                print(f'\033[31m[online_tamp] {kind} stage failed after '
                      f'{self.MAX_STAGE_RETRIES} retries, aborting\033[0m')
                break

            # Lane batches were popped incrementally as each lane verified; only the
            # barrier needs its post-stage scheduler advance here.
            if kind == 'barrier':
                scheduler.pending_barriers.pop(0)
                scheduler.advance_phase()

        recording_path = self.lfd.exit()
        self.output_info = build_output_info(
            task_name=self.task_name,
            task_success=self.success,
            duration_tamp=session.duration_tamp,
            recording_path=recording_path,
        )

    def _next_stage(self, scheduler):
        """Return the next executable stage, preserving left/right parallelism.

        A stage is the current-phase left and/or right lane batch, or a single
        barrier once both lanes' phase quotas are exhausted. ``None`` when done.
        """
        lane_batches = {
            lane: scheduler.current_batch(lane)
            for lane in ('left', 'right')
            if scheduler.current_batch(lane) is not None
        }
        if lane_batches:
            return 'lanes', lane_batches
        if scheduler.pending_barriers:
            return 'barrier', scheduler.pending_barriers[0]
        return None

    def _plan_stage(self, kind, payload, use_detail_mp):
        """DetailedTAMP for a stage -> (sequence, unit_states, checks) or None.

        ``unit_states`` maps each unit key (lane name, or ``'barrier'``) to its
        planned post-execution state, so ``_execute_and_verify`` can keep the
        verified units' effects and revert only the failed ones.

        Lanes are planned per-lane and merged so both arms move together. The
        barrier sequence is returned directly (never through merge_batch_sequences)
        so its graphstate_markers survive for BC stop-mode execution; barriers
        always use the schema path (their connector RRT is the stochastic part a
        retry resamples).
        """
        if kind == 'barrier':
            barrier = payload
            seq, updated_state, subgoal, _ = self._run_pending_barrier(barrier)
            if seq is None:
                return None
            if not getattr(seq, 'graphstate_markers', ()):
                raise RuntimeError(
                    "Barrier sequence lost its graphstate_markers; BC stop-mode would break"
                )
            checks = {}
            if subgoal:
                checks['barrier'] = {
                    'subgoal': list(subgoal),
                    'target_obj': get_action_target_object(barrier.action),
                }
            return seq, {'barrier': updated_state}, checks

        plan_fn = (self._replan_lane_batch_with_detail_mp if use_detail_mp
                   else self._plan_lane_batch)
        sequences, unit_states, checks = [], {}, {}
        for lane, batch in payload.items():
            seq, updated_state, lane_subgoal, _ = plan_fn(lane, batch)
            if seq is None:
                return None
            sequences.append(seq)
            unit_states[lane] = updated_state
            if lane_subgoal:
                checks[lane] = {
                    'subgoal': list(lane_subgoal),
                    'target_obj': get_batch_target_object(batch),
                }
        return merge_batch_sequences(sequences), unit_states, checks

    def _execute_and_verify(self, seq, unit_states, checks):
        """ExecuteAndVerify: run the sequence, monitor effects, verify subgoals.

        Returns ``(status, verified_state, achieved, results)``. ``status`` is the
        executor verdict ('replan' on normal completion, 'success'/'fail' terminal);
        ``results`` maps each unit key to its per-unit verification verdict. Each
        unit that misses verification has its symbolic effect reverted so the retry
        re-plans that unit from the un-moved belief, while verified units keep theirs.
        """
        subgoals = {key: chk['subgoal'] for key, chk in checks.items()}
        self._begin_contact_monitoring(subgoals)
        try:
            status = execute_request(
                self.execute_robosuite_plan,
                ExecutionRequest(sequence=seq, stop_mode=infer_stop_mode(seq)),
            ).status
        finally:
            self._end_contact_monitoring()

        # 'fail'/'success' are terminal: online_tamp breaks immediately, so the
        # un-perceived self.current_state is never planned from again.
        if status in ('fail', 'success'):
            return status, self.current_state, status == 'success', {}

        merged_state = self._merge_lane_updated_states(
            self.current_state, unit_states.values()
        )
        verified_state, achieved, results = self.perceive_env(merged_state, checks)
        verified_state = set(verified_state)
        for key, unit_state in unit_states.items():
            if results.get(key, True):
                continue
            added = set(unit_state) - set(self.current_state)
            removed = set(self.current_state) - set(unit_state)
            verified_state = (verified_state - added) | removed
        return status, verified_state, achieved, results

    def _plan_lane_batch(self, lane, batch):
        """Schema fast-path: expand a lane batch into an executable Sequence."""
        lane_subgoal = get_contact_action_subgoal(batch.ref_goal_state, batch.actions)
        result = execute_schema_skeleton_plan(
            self.para, self.robot_entity, self.static_environment,
            batch.actions, self.current_state, **self.tamp_kwargs_dict,
        )
        if result is None:
            return None, self.current_state, lane_subgoal, None
        return result.sequence, set(result.final_state), lane_subgoal, result

    def _replan_lane_batch_with_detail_mp(self, lane, batch):
        """Retry path: re-solve a local PDDLStream problem for this lane's subgoal."""
        lane_subgoal = get_contact_action_subgoal(batch.ref_goal_state, batch.actions)
        seq, updated_state, plan_seg = plan_detail_mp(
            self.para, self.robot_entity, self.problem, self.stream_info,
            self.current_state, lane_subgoal,
            static_environment=self.static_environment,
            skeleton_segment=convert_plan_to_skeleton(batch.actions),
            **self.tamp_kwargs_dict,
        )
        if seq is None:
            return None, self.current_state, lane_subgoal, None
        if plan_seg is not None:
            # Use the freshly grounded actions so a later schema expansion reuses
            # the new stream payloads rather than stale ones from the global plan.
            batch.actions = list(plan_seg)
        return seq, set(updated_state), lane_subgoal, None

    def _merge_lane_updated_states(self, base_state, lane_states):
        merged_state = set(base_state)
        for lane_state in lane_states:
            merged_state -= base_state - lane_state
            merged_state |= lane_state - base_state
        return merged_state

    def _run_pending_barrier(self, barrier):
        """Schema fast-path for a bimanual barrier; materialize its symbolic effect."""
        subgoal = get_barrier_action_subgoal(barrier.ref_goal_state, barrier.action)
        result = execute_schema_skeleton_plan(
            self.para, self.robot_entity, self.static_environment,
            [barrier.action], self.current_state, **self.tamp_kwargs_dict,
        )
        if result is None:
            return None, self.current_state, subgoal, None
        merged = materialize_ref_goal_result(
            base_state=self.current_state,
            ref_goal_state=barrier.ref_goal_state,
            final_confs=result.final_confs,
            sequence=result.sequence,
        )
        return merged.sequence, set(merged.final_state), subgoal, merged

    # ------------------------------------------------------------------
    # OSC_POSE absolute-EE execution
    # ------------------------------------------------------------------

    def replay_tamp_every(self, action, duration):
        import time
        ts = self.lfd.replay_tamp_step(action)
        time.sleep(duration)
        return ts

    def execute_robosuite_plan(self, seq, stop_mode='replan'):
        """Run `seq` under OSC_POSE abs control, optionally falling back to BC."""
        if stop_mode not in {'replan', 'bc'}:
            raise ValueError(f"Unsupported stop_mode: {stop_mode!r}")

        tamp_cmd_gen = self.iterate_process_sequence(self.robot_entity, seq)
        doing_tamp = True
        self.lfd.update_controllers(controller_name="OSC_POSE", abs_action=True)

        for _ in range(self.lfd.max_timesteps):
            if doing_tamp:
                # The TAMP phase is open-loop connector/primitive replay. The effect
                # monitor governs only the *learned policy* (BC) — checking it here
                # would let the connector transit (which brings the two hands close
                # on the way to the bioperation keypose) trip the bimanual monitor
                # and skip the policy entirely, so we never consult it during replay.
                try:
                    action = next(tamp_cmd_gen)
                    self.replay_tamp_every(action, duration=0.01)
                except StopIteration:
                    if stop_mode == 'replan':
                        print("perceive and plan next subgoal")
                        return 'replan'
                    doing_tamp = False
                    self.lfd.set_bc_controller()
            else:
                self.lfd.inference_once()
                if self._effect_monitor_completed():
                    print("Effect monitor completed primitive")
                    return 'replan'

            if self.lfd.handle_rewards():
                print("Task success!")
                self.success = True
                break
        else:
            print("Task failed!")
            self.success = False

        return 'success' if self.success else 'fail'

    def iterate_process_sequence(self, robot_entity, seq):
        """Yield per-step OSC abs actions, 7-D ``[pos(3), axisangle(3), gripper(1)]``
        per active arm (so 7-D single-arm, 14-D bimanual).

        Learned-grasp segments are read from ``GroupTrajectory.ee_path``; all other
        segments are FK'd from their joint waypoints. Idle arms hold their last
        commanded pose until both arms' streams are exhausted.
        """
        # Imported lazily so the module stays importable without robosuite/pybullet.
        from robosuite.utils.transform_utils import quat2axisangle

        from examples.pybullet.utils.pybullet_tools.utils import (
            get_link_pose,
            link_from_name,
            set_joint_positions,
        )

        sides = ['left'] if len(self.sg_param['robots']) == 1 else ['left', 'right']

        init_conf = self.lfd.get_cur_jpose_robosuite()
        arm_joints_len = len(robot_entity.joint_groups['left_arm'])
        gripper_joints_len = len(robot_entity.joint_groups['left_gripper'])

        # -------- waypoint dimension helpers --------

        def normalize_waypoint(waypoint):
            for attr in ('positions', 'values'):
                value = getattr(waypoint, attr, None)
                if value is not None:
                    return value
            return waypoint

        def _split_waypoint(waypoint, want):
            waypoint = normalize_waypoint(waypoint)
            n = len(waypoint)
            if n == arm_joints_len:
                segment = waypoint if want == 'arm' else None
            elif n == gripper_joints_len:
                segment = waypoint if want == 'gripper' else None
            elif n == arm_joints_len + gripper_joints_len:
                segment = (waypoint[:arm_joints_len] if want == 'arm'
                           else waypoint[arm_joints_len:])
            else:
                raise ValueError(
                    f"Unsupported waypoint dimension {n}; expected "
                    f"{arm_joints_len}, {gripper_joints_len}, or "
                    f"{arm_joints_len + gripper_joints_len}"
                )
            return None if segment is None else np.asarray(segment, dtype=float)

        extract_arm_waypoint = lambda wp: _split_waypoint(wp, 'arm')
        extract_gripper_waypoint = lambda wp: _split_waypoint(wp, 'gripper')

        # -------- forward kinematics + OSC slice --------

        # Cache link indices once; link_from_name scans all joints per call.
        ee_link_idx = {
            side: link_from_name(
                robot_entity.robot, robot_entity.manipulators[side][2],
                client=robot_entity.client,
            )
            for side in sides
        }

        def fk_side(side, arm_q):
            arm_joints = robot_entity.get_group_joints(f"{side}_arm")
            set_joint_positions(robot_entity.robot, arm_joints, arm_q, client=robot_entity.client)
            return get_link_pose(robot_entity.robot, ee_link_idx[side], client=robot_entity.client)

        def pose_to_osc_slice(pose, gripper_scalar):
            pos, quat_xyzw = pose
            tool_rot = _SciRot.from_quat(np.asarray(quat_xyzw, dtype=float))
            cmd_pos = np.asarray(pos, dtype=float) + tool_rot.apply(self.GRIP_SITE_POS_OFFSET)
            axisangle = quat2axisangle((tool_rot * self.GRIP_SITE_CORRECTION).as_quat())
            return list(cmd_pos) + list(axisangle) + [float(gripper_scalar)]

        # -------- seed current state per side --------

        current_pose = {side: fk_side(side, init_conf[f'{side}_arm']) for side in sides}
        current_gripper = {
            side: robot_entity.pos2joint_gripper(init_conf[f'{side}_gripper'][0])
            for side in sides
        }
        # A frozen (AtGrasp) move re-asserts the grasp's last commanded grip while
        # holding an object — carried across batches, NOT re-read from the live
        # aperture (a held object holds the fingers apart, relaxing the grip).
        carried_gripper_cmd = getattr(self, '_carried_gripper_cmd', {})
        closed_conf, _ = robot_entity.close_open_conf()
        default_closed = robot_entity.pos2joint_gripper(float(closed_conf[0]))
        frozen_gripper = {
            side: carried_gripper_cmd.get(side, default_closed) for side in sides
        }

        per_arm_queue = split_sequence_per_arm(seq, sides=sides)
        self._bridge_joint_discontinuities(per_arm_queue, sides, extract_arm_waypoint)
        self._append_gripper_transitions(
            per_arm_queue, sides, init_conf, extract_gripper_waypoint
        )

        # -------- per-side step generator --------

        def emit(pose, gripper, steps):
            osc_slice = pose_to_osc_slice(pose, gripper)
            for _ in range(steps):
                yield osc_slice

        def build_side_steps(side):
            last_pose = current_pose[side]
            last_gripper = current_gripper[side]
            for primitive in per_arm_queue[side]:
                path = getattr(primitive, 'refined_qpos', None) or primitive.path
                ee_path = getattr(primitive, 'ee_path', None)
                repeat = max(1, int(getattr(primitive, 'steps_per_waypoint', 1)))
                freeze_gripper = _is_frozen(primitive)

                # Resolve the whole segment's (pose, gripper) targets up front so
                # close ramps can be retimed onto the grasp waypoint.
                poses, grippers = [], []
                pose = last_pose
                gripper = frozen_gripper[side] if freeze_gripper else last_gripper
                for i, waypoint in enumerate(path):
                    cfg = normalize_waypoint(waypoint)
                    gripper_only = len(cfg) == gripper_joints_len
                    if not freeze_gripper:
                        if gripper_only:
                            gripper = robot_entity.pos2joint_gripper(float(cfg[0]))
                        elif len(cfg) == arm_joints_len + gripper_joints_len:
                            gripper = robot_entity.pos2joint_gripper(float(cfg[-1]))
                    if not gripper_only:
                        arm_q = extract_arm_waypoint(cfg)
                        if ee_path is not None:
                            pose = ee_path[min(i, len(ee_path) - 1)]
                        elif arm_q is not None:
                            pose = fk_side(side, arm_q)
                    poses.append(pose)
                    grippers.append(float(gripper))

                if freeze_gripper:
                    holds = {}
                else:
                    grippers, holds = self._retime_close_ramps(
                        poses, grippers, last_gripper, ee_path
                    )

                for i, (next_pose, next_gripper) in enumerate(zip(poses, grippers)):
                    if i in holds:
                        yield from emit(next_pose, holds[i], self.GRIPPER_SETTLE_STEPS)
                    last_pose = next_pose
                    last_gripper = next_gripper
                    yield from emit(last_pose, last_gripper, repeat)
                    if i in holds:
                        yield from emit(last_pose, last_gripper, self.GRIPPER_DWELL_STEPS)

        def initial_hold_gripper(side):
            for primitive in per_arm_queue[side]:
                if isinstance(primitive, Graphstate):
                    continue
                if _is_frozen(primitive):
                    return frozen_gripper[side]
                return current_gripper[side]
            return carried_gripper_cmd.get(side, current_gripper[side])

        side_gens = {side: build_side_steps(side) for side in sides}
        last_slices = {
            side: pose_to_osc_slice(current_pose[side], initial_hold_gripper(side))
            for side in sides
        }

        def record_carried_gripper():
            # Persist the last commanded grip per side for the next batch's frozen
            # move; the OSC slice is [pos(3), axisangle(3), gripper(1)], so [-1] is it.
            self._carried_gripper_cmd = {
                side: float(last_slices[side][-1]) for side in sides
            }

        record_carried_gripper()
        yield [v for side in sides for v in last_slices[side]]
        while True:
            advanced = False
            for side in sides:
                try:
                    last_slices[side] = next(side_gens[side])
                    advanced = True
                except StopIteration:
                    pass  # idle arm holds its last slice
            if not advanced:
                break
            record_carried_gripper()
            yield [v for side in sides for v in last_slices[side]]

    def _retime_close_ramps(self, poses, grippers, prev_gripper, ee_path):
        """Move each closing ramp's actuation onto the segment's grasp waypoint.

        Returns the rewritten commands and ``{hold_index: pre-ramp command}`` for
        the execution loop's settle/dwell holds.
        """
        grippers = list(grippers)
        holds = {}
        prev = float(prev_gripper)
        i = 0
        while i < len(grippers):
            delta = grippers[i] - prev
            ramping = (delta > self.GRIPPER_RAMP_EPS
                       if self.GRIPPER_RAMP_CLOSE_ONLY
                       else abs(delta) > self.GRIPPER_RAMP_EPS)
            if not ramping:
                prev = grippers[i]
                i += 1
                continue
            onset = i
            sign = 1.0 if delta > 0.0 else -1.0
            # Absorb the whole non-decreasing ramp (plateaus included) so the
            # snapped command is the ramp's final value, not a partial one.
            end = onset
            while (end + 1 < len(grippers)
                   and sign * (grippers[end + 1] - grippers[end]) >= -self.GRIPPER_RAMP_EPS):
                end += 1
            if ee_path is not None:
                # Deepest waypoint along the gripper approach axis at ramp onset.
                approach = _SciRot.from_quat(
                    np.asarray(poses[onset][1], dtype=float)
                ).apply([0.0, 0.0, 1.0])
                hold = onset + int(np.argmax([
                    float(np.dot(np.asarray(poses[j][0], dtype=float), approach))
                    for j in range(onset, len(poses))
                ]))
            else:
                hold = onset
            window_end = max(end, hold)
            window = grippers[onset:window_end + 1]
            target = max(window) if sign > 0.0 else min(window)
            for j in range(onset, hold):
                grippers[j] = prev
            for j in range(hold, window_end + 1):
                grippers[j] = target
            holds[hold] = prev
            prev = target
            i = window_end + 1
        return grippers, holds

    @staticmethod
    def _bridge_joint_discontinuities(per_arm_queue, sides, extract_arm_waypoint):
        """Insert linear arm bridges where consecutive primitives don't meet."""
        for side in sides:
            queue = per_arm_queue[side]
            if not queue:
                continue
            bridged = [queue[0]]
            for primitive in queue[1:]:
                prev_path = getattr(bridged[-1], 'refined_qpos', None) or bridged[-1].path
                cur_path = getattr(primitive, 'refined_qpos', None) or primitive.path
                if len(prev_path) > 0 and len(cur_path) > 0:
                    prev_end = extract_arm_waypoint(prev_path[-1])
                    cur_start = extract_arm_waypoint(cur_path[0])
                else:
                    prev_end = cur_start = None
                if (prev_end is not None and cur_start is not None
                        and not np.allclose(prev_end, cur_start, atol=0.05, rtol=0.0)):
                    n_steps = max(10, int(np.ceil(np.max(np.abs(cur_start - prev_end)) / 0.01)) + 1)
                    bridge_path = np.linspace(prev_end, cur_start, num=n_steps).tolist()
                    bridge = GroupTrajectory(primitive.robot, f'{side}_arm', bridge_path)
                    if _is_frozen(primitive):
                        bridge.freeze_gripper = True
                    bridged.append(bridge)
                bridged.append(primitive)
            per_arm_queue[side] = bridged

    @staticmethod
    def _append_gripper_transitions(per_arm_queue, sides, init_conf, extract_gripper_waypoint):
        """Append explicit gripper segments so commanded grasp/release actuate."""
        current_gripper_positions = {
            side: np.asarray(init_conf[f'{side}_gripper'], dtype=float) for side in sides
        }
        for side in sides:
            queue = per_arm_queue[side]
            if not queue:
                continue
            expanded = []
            for primitive in queue:
                expanded.append(primitive)
                if isinstance(primitive, Graphstate):
                    continue
                path = getattr(primitive, 'refined_qpos', None) or primitive.path
                end_gripper = extract_gripper_waypoint(path[-1]) if len(path) > 0 else None
                target_gripper = getattr(primitive, 'target_gripper_positions', None)
                freeze_gripper = _is_frozen(primitive)
                if target_gripper is not None and not freeze_gripper:
                    target_gripper = np.asarray(target_gripper, dtype=float)
                    if not np.allclose(current_gripper_positions[side], target_gripper):
                        expanded.append(GroupTrajectory(
                            primitive.robot, f'{side}_gripper',
                            [current_gripper_positions[side].tolist(), target_gripper.tolist()],
                        ))
                        current_gripper_positions[side] = target_gripper
                elif end_gripper is not None:
                    current_gripper_positions[side] = end_gripper
            per_arm_queue[side] = expanded

    def __call__(self):
        return self.output_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run interleaved DMG task with OSC_POSE absolute EE execution.'
    )
    parser.add_argument(
        '--task_name', type=str, default='two_arm_three_piece_assembly',
        help='Task name to use (default: two_arm_three_piece_assembly)'
    )
    parser.add_argument(
        '--sg', action='append', default=[], metavar='KEY=VALUE',
        help='Override an sg_params field (repeatable), e.g. '
             '--sg biop_ckpt_name=logs/train/dmg_threading/ckpt_perskill_jpose_near.pth'
    )
    parser.add_argument(
        '--dp_ckpt', type=str, default=None,
        help='Override the diffusion-policy checkpoint (LfD_params.DP_input[<task_name>]).'
    )
    args = parser.parse_args()

    task_name = args.task_name
    yaml_path = os.path.join(
        root_path,
        f'examples/pybullet/aloha_real/openworld_aloha/configs/dmg_cfgs/{task_name}.yaml'
    )

    from repo_paths import load_yaml
    parameters = load_yaml(yaml_path)

    # In-memory checkpoint overrides for sweeps; the DMG route consumes this
    # parameters dict directly (no YAML re-read). Numeric values are coerced to
    # float; a path like biop_ckpt_name stays a string.
    for kv in args.sg:
        key, val = kv.split('=', 1)
        try:
            val = float(val)
        except ValueError:
            pass
        parameters['sg_params'][key] = val

    if args.dp_ckpt is not None:
        parameters['LfD_params']['DP_input'][task_name] = args.dp_ckpt

    planner = DMG_OSC_planner(parameters)
    exe_info = planner()
    print(f'Execution results: {exe_info}')
