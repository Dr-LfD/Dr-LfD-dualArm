

import sys
import os
import numpy as np
import time
import h5py

def _resolve_workspace_root():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parts = current_dir.split(os.sep)
    for i, part in enumerate(parts):
        if part == 'pddlstream_aloha':
            return '/' + os.path.join(*parts[:i + 1]) + os.sep
    return os.path.abspath(os.path.join(current_dir, "../../../.."))

root_path = _resolve_workspace_root()
if root_path not in sys.path:
    sys.path.append(root_path)
os.chdir(root_path)

from examples.pybullet.aloha_real.openworld_aloha.contact_monitoring import (
    ContactEffectMonitorGroup,
    ContactPredictorWrapper,
    build_effect_monitor,
)

from examples.pybullet.aloha_real.openworld_aloha.primitives import GroupConf, Sequence
from examples.pybullet.aloha_real.openworld_aloha.run_openworld import prepare_world, plan_detail_mp
from examples.pybullet.aloha_real.openworld_aloha.schema_executor import (
    build_connector_motion,
    extract_static_environment,
    execute_schema_skeleton_plan,
    materialize_ref_goal_result,
)
from examples.pybullet.aloha_real.openworld_aloha.simple_worlds import update_mesh
from examples.pybullet.aloha_real.openworld_aloha.network_loader import get_lfd_wrapper, update_equivSkill_wrapper
from examples.pybullet.aloha_real.scripts.standalone_scripts.schema_construction import load_runtime_schema_metadata
from examples.pybullet.aloha_real.scripts.tamp_workflow import (
    ArmSchedulerState,
    ExecutionRequest,
    LaneAttemptResult,
    build_output_info,
    build_lane_checks,
    execute_request,
    execute_lane_batches,
    get_action_target_object,
    get_batch_target_object,
    infer_stop_mode,
    plan_online_session,
    preserve_arm_confs,
    rollback_batch_execution,
)
from examples.pybullet.aloha_real.openworld_aloha.symbolic_utils import (
    search_facts,
    get_contact_action_subgoal, get_barrier_action_subgoal, build_scheduler_batches,
    action_invalidates_perception,
    SimPrimitiveSubgoalDetector,
    BiopCompletionMonitor,
    effect_monitor_sensor_data,
)


class robosuite_planner(object):
    def __init__(self,   para):
        self.para = para
        self.sg_param = para['sg_params']
        self.old_tgt = None
        self.task_name = self.sg_param['task_name']
        self.executing_lfd = False
        self.effect_monitor = None

        self.success = False
        # Last commanded gripper joint per side, carried across subgoal executions so a
        # held grip is re-asserted during transfer instead of re-read from live qpos.
        self._carried_gripper_cmd = {}

        self.env_type = 'mj'
        self.vid_save_path = os.path.join(root_path,  para['vid_save_path'])
        if not os.path.exists(self.vid_save_path):
            os.makedirs(self.vid_save_path)
        
        self.equivSkill_info_dict, shape_meta = update_equivSkill_wrapper(self.sg_param, self.para, get_sgs_from_hdf5_fn=self.get_sgs_from_hdf5)

        ## TODO: self.skillwise_sgs is flattened, not unified with real_traj plugin
        all_skillwise_sgs = {}
        for prefix_key in self.equivSkill_info_dict.keys():
            skillwise_sgs = self.equivSkill_info_dict[prefix_key]['skillwise_sgs']
            all_skillwise_sgs.update(skillwise_sgs)
        self.skillwise_sgs = all_skillwise_sgs

        self.schema_skill_metas = load_runtime_schema_metadata(
            self.para.get('skill_yaml_paths') or [],
            env_names=list(self.equivSkill_info_dict.keys()),
            root_path=root_path,
        )["skill_meta_map"]

        if 'interested_objs' in self.sg_param:
            self.interested_objects = self.sg_param['interested_objs'].keys()
        else:
            all_interested_objs = set()
            for skill_name, skill_info in self.skillwise_sgs.items():
                all_interested_objs = all_interested_objs.union(set(skill_info['related_objs']))
            self.interested_objects = list(all_interested_objs)

        self.env_options =   self.get_env_options(shape_meta)

        # ## debug: the camera name is incorrect specified when learning assembly grasp. 
        # if len(self.env_options['robots']) ==2:
        #     self.env_options['camera_names'] = ['agentview', 'sideview', 'birdview', 'robot0_eye_in_hand', 'robot1_eye_in_hand']

        self.lfd = get_lfd_wrapper(para, with_planning=True)
        if self.lfd is None:
            self.lfd = self.manually_initialize_env(self.sg_param)
        else:
            self.lfd.initialize_env(\
                env_name = self.sg_param['task_name'], \
                env_options = self.env_options)

        ## use env jpose to revise robot percept_arm_pose
        initial_jposes= self.lfd.get_cur_jpose_robosuite()
        os.chdir(root_path)

        # hdf5_path = os.path.join(para['env_dir'], self.sg_param['example_hdf5_path'])

        mj_pc_dict = self.save_mj_observation(npz_path = None)
        # mj_pc_dict = self.tmp_pc_from_hdf5(hdf5_path)


        # load pybullet world. For single-robot scene, input base pose
        if hasattr(self.lfd.env, 'workspace_offset'):
            table_center = self.lfd.env.workspace_offset
        elif hasattr(self.lfd.env, 'table_offset'):
            table_center = self.lfd.env.table_offset
            
        franka_base_pos = self.lfd.env.robots[0].base_pos

        self.robot_entity, belief, _ = prepare_world(\
            self.para, env_type = self.env_type,  \
             mj_pc_dict = mj_pc_dict,\
             franka_base_pos = franka_base_pos, \
             table_center = table_center)

        if initial_jposes is not None:
            self.robot_entity.revise_initial_pose(initial_jposes)

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

        # Create a list of keyword arguments for TAMP functions
        tamp_kwargs = [
            ('task_name', self.task_name),
            ('skillwise_sgs', self.skillwise_sgs),
            ('min_ee_dist', self.sg_param['hand_hand_dist_threshold']),
            ('interested_objects', self.interested_objects),
            ('max_tamp_time', self.sg_param['max_tamp_time']),
            ('collision_distance', self.sg_param['attachment_collision_distance']),
            ('equivSkill_info_dict', self.equivSkill_info_dict),
            ('initial_pc_dict', mj_pc_dict),
        ]
        self.tamp_kwargs_dict = dict(tamp_kwargs)
        # Subclasses inject extra stream kwargs (e.g. the M2T2 grasp backend)
        # before the streams are built inside online_tamp.
        self.tamp_kwargs_dict.update(self.get_extra_tamp_kwargs())
        # Opt-in neural generic grasp (M2T2/GPD), gated by sg_params.has_generic_pick
        # so non-generic tasks build nothing and never import the predictor. The
        # backend then serves the ATTACH skill via the grasp_backend dispatch.
        if self.sg_param.get('has_generic_pick', False):
            self.tamp_kwargs_dict.update(self._build_generic_grasp_kwargs())

        self.online_tamp(belief)
            
        # # ## for threading debug
        # self.reset_to_hdf5_init(hdf5_path, time_idx= 110)  
        # self.test_dp()
    
    def reset_to_hdf5_init(self, hdf5_path, demo_idx = 0, time_idx = 0):
        f = h5py.File(hdf5_path, 'r')
        init_state = f[f'data/demo_{demo_idx}/states'][()]
        self.lfd.reset_to(init_state[time_idx])

    def get_env_options(self, shape_meta):
        raise NotImplementedError("get_env_options is not implemented.")

    def get_extra_tamp_kwargs(self):
        """Extra stream kwargs merged into tamp_kwargs_dict before planning.

        Default is a no-op; subclasses override to add e.g. an alternative
        grasp backend.
        """
        return {}

    def _build_generic_grasp_kwargs(self):
        """Stream kwargs for the neural generic-grasp backend (M2T2 or GPD).

        Enabled per task via sg_params.has_generic_pick. The grasp source is
        chosen by sg_params.grasp_backend ('m2t2' default, or 'gpd'); placement is
        served geometrically (place_backend='generic'). The heavy M2T2 predictor is
        imported and constructed only here, so tasks without the flag pay nothing.
        """
        grasp_backend = self.sg_param.get('grasp_backend', 'm2t2')
        if grasp_backend == 'gpd':
            kwargs = self._gpd_grasp_kwargs()
        else:
            kwargs = self._m2t2_grasp_kwargs()
        kwargs['place_backend'] = 'generic'
        return kwargs

    def _m2t2_grasp_kwargs(self):
        from examples.pybullet.aloha_real.openworld_aloha.estimation.m2t2_grasp import (
            M2T2GraspPredictor,
        )
        m2t2_cfg = self.sg_param['m2t2']
        predictor = M2T2GraspPredictor(
            checkpoint=m2t2_cfg['checkpoint'],
            config_path=m2t2_cfg['config_path'],
            repo_root=m2t2_cfg['repo_root'],
            mask_thresh=m2t2_cfg.get('mask_thresh'),
            num_points=m2t2_cfg.get('num_points'),
            num_runs=m2t2_cfg.get('num_runs', 1),
            visualize=m2t2_cfg.get('visualize', False),
            viz_top_k=m2t2_cfg.get('viz_top_k', 20),
            viz_pause=m2t2_cfg.get('viz_pause', True),
        )
        return {
            'grasp_backend': 'm2t2',
            'm2t2_grasp_wrapper': predictor,
            'm2t2_contact_radius': m2t2_cfg.get('contact_radius', 0.03),
            'm2t2_grasp_depth': m2t2_cfg.get('grasp_depth', 0.10),
        }

    def _gpd_grasp_kwargs(self):
        # GPD orients grasp approach directions relative to a viewpoint; reuse the
        # robosuite scene camera extrinsic so the viewpoint matches the frame the
        # object clouds were built in.
        from robosuite.utils.camera_utils import get_camera_extrinsic_matrix

        gpd_cfg = self.sg_param.get('gpd', {})
        sim = self.lfd.env.sim
        camera_names = list(self.lfd.env.camera_names)
        cam_name = 'agentview' if 'agentview' in camera_names else camera_names[0]
        cam_point = get_camera_extrinsic_matrix(sim, cam_name)[:3, 3]
        return {
            'grasp_backend': 'gpd',
            'gpd_camera_point': cam_point,
            'gpd_grasp_depth': gpd_cfg.get('grasp_depth', 0.10),
            'gpd_max_candidates': gpd_cfg.get('max_candidates', 10),
        }

    # def save_mj_observation(self, npz_path=None):
    #     raise NotImplementedError("save_mj_observation is not implemented. Please use a valid LfD wrapper.")

    def _resolve_use_pc_color(self):
        """Read use_pc_color from every equivariant skill wrapper and assert they all agree."""
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

    def save_mj_observation(self, npz_path = None, cache_image = True):
        """
        Save the instance pc and agentview image
        """
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

        # pc_instance_dict = self.lfd.get_instance_pcd(self.lfd.raw_obs)
        pc_instance_obstacles_dict, visible_dict = self.lfd.get_instance_and_obstacles_pcd(self.lfd.raw_obs, require_downsample = False, get_obstacle = self.para['require_obstacle_pc'])
        if 'obstacle_pcd' in pc_instance_obstacles_dict:
            mj_pc_dict['fixed_obstacles'] = _select_pc(pc_instance_obstacles_dict['obstacle_pcd'], 'obstacle_pcd')
        for obj in self.interested_objects:
            if visible_dict[f'{obj}_visible'] ==False:
                mj_pc_dict[obj] = None
            else:
                pc_key = f'{obj}_point_cloud'
                mj_pc_dict[obj] = _select_pc(pc_instance_obstacles_dict[pc_key], pc_key)

        if npz_path is not None:
            np.savez(npz_path, **mj_pc_dict)

        if cache_image:
            table_setting_img = self.lfd.raw_obs['agentview_image'][::-1]
            import cv2
            cv2.imwrite(os.path.join(self.vid_save_path, f'{self.task_name}_setting.png'), table_setting_img)
        return mj_pc_dict

    def get_sgs_from_hdf5(self, hdf5_path):
        raise NotImplemented

    def manually_initialize_env(self, sg_param):
        raise NotImplementedError("Manual environment initialization is not implemented. Please use a valid LfD wrapper.")
    
    # 7+1 + 7 + 1, negtive x rbt to positive x rbt
    def replay_tamp_every(self, action, duration):
        ts = self.lfd.replay_tamp_step(action)
        time.sleep(duration)
        return ts
   
    def test_dp(self):
        for i in range(self.lfd.max_timesteps):
            ts = self.lfd.inference_once()
            task_success = self.lfd.handle_rewards()
            if task_success:
                print("Task success!")
                self.success = True
                break
        else:
            print('Task failed!')
            self.success = False
        recording_path = self.lfd.exit()
        self.output_info = {
            'recording_path': recording_path,
            'task_success': self.success,
            'task_name': self.task_name
        }
    
    def _plan_lane_batch(self, lane, batch):
        lane_subgoal = get_contact_action_subgoal(batch.ref_goal_state, batch.actions)
        result = execute_schema_skeleton_plan(
            self.para,
            self.robot_entity,
            self.static_environment,
            batch.actions,
            self.current_state,
            **self.tamp_kwargs_dict,
        )
        if result is None:
            return None, self.current_state, lane_subgoal, None
        return result.sequence, set(result.final_state), lane_subgoal, result

    def _replan_lane_batch_with_detail_mp(self, lane, batch):
        lane_subgoal = get_contact_action_subgoal(batch.ref_goal_state, batch.actions)
        seq, updated_state, plan_seg = plan_detail_mp(
            self.para,
            self.robot_entity,
            self.problem,
            self.stream_info,
            self.current_state,
            lane_subgoal,
            **self.tamp_kwargs_dict,
        )
        if seq is None:
            return None, self.current_state, lane_subgoal, None
        if plan_seg is not None:
            batch.actions = list(plan_seg)
        return seq, updated_state, lane_subgoal, None

    def _merge_lane_updated_states(self, base_state, lane_states):
        merged_state = set(base_state)
        for lane_state in lane_states:
            removed_literals = base_state - lane_state
            added_literals = lane_state - base_state
            merged_state -= removed_literals
            merged_state |= added_literals
        return merged_state

    def _run_pending_barrier(self, barrier):
        subgoal = get_barrier_action_subgoal(barrier.ref_goal_state, barrier.action)
        result = execute_schema_skeleton_plan(
            self.para,
            self.robot_entity,
            self.static_environment,
            [barrier.action],
            self.current_state,
            **self.tamp_kwargs_dict,
        )
        if result is None:
            return None, self.current_state, subgoal, None
        merged_result = materialize_ref_goal_result(
            base_state=self.current_state,
            ref_goal_state=barrier.ref_goal_state,
            final_confs=result.final_confs,
            sequence=result.sequence,
        )
        return merged_result.sequence, set(merged_result.final_state), subgoal, merged_result

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
        """Build per-side arm connector motions from the live joint pose to the default home pose.

        Returns a list of Sequence objects, one per active arm side.
        Raises RuntimeError if the live pose is unavailable or any connector cannot be planned.
        """
        live_jposes = self.lfd.get_cur_jpose_robosuite()
        if live_jposes is None:
            raise RuntimeError(
                "Cannot build perception-home connector: live joint pose unavailable"
            )
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
                raise RuntimeError(
                    f"Cannot build perception-home connector for {arm_group}"
                )
            connectors.append(connector)
        return connectors

    def _execute_perception_connector(self, connectors):
        """Execute a list of Sequence connectors by streaming through the robosuite controller.

        Arms are moved smoothly to the home pose; gripper state is preserved.
        Does not route through execute_robosuite_plan() and never triggers replan/BC mode.
        """
        combined_commands = []
        for seq in connectors:
            combined_commands.extend(getattr(seq, 'commands', []))
        combined_seq = Sequence(combined_commands)
        # iterate_process_sequence (OSC mixin) only ever emits 14-D OSC_POSE
        # actions, so the perception-home motion must run under OSC_POSE — the
        # same controller execute_robosuite_plan switches to. Using
        # planning_controller (JOINT_POSITION, 16-D) here mismatches the producer.
        self.lfd.update_controllers(
            controller_name="OSC_POSE", abs_action=True
        )
        for action in self.iterate_process_sequence(self.robot_entity, combined_seq):
            self.replay_tamp_every(action, duration=0.01)

    def _refresh_affected_object_meshes(self, target_objects, reset_home=False):
        if not target_objects:
            return
        if reset_home:
            connectors = self._build_perception_home_connector()
            self._execute_perception_connector(connectors)
        self._refresh_object_meshes(target_objects)

    def _verify_action_checks(self, updated_state, action_checks, execution_results=None):
        verified_state, subgoal_achieved, action_results = self.perceive_env(
            updated_state, action_checks
        )
        if execution_results:
            for key, execution_result in execution_results.items():
                if action_results.get(key, True):
                    continue
                verified_state = rollback_batch_execution(
                    verified_state, execution_result
                )
        return verified_state, subgoal_achieved, action_results

    def _run_lane_attempt(self, scheduler, batches, plan_batch_fn=None):
        if plan_batch_fn is None:
            plan_batch_fn = self._plan_lane_batch
        bundle = execute_lane_batches(
            batches=batches,
            plan_batch_fn=plan_batch_fn,
            merge_state_fn=lambda lane_states: self._merge_lane_updated_states(
                self.current_state, lane_states
            ),
            executor=self.execute_robosuite_plan,
            before_execute=self._begin_contact_monitoring,
            after_execute=self._end_contact_monitoring,
        )
        if bundle.planning_failed_lane is not None:
            return LaneAttemptResult(
                planning_failed_lane=bundle.planning_failed_lane,
                failed_lanes=set(batches),
            )
        if bundle.exec_status == 'fail':
            return LaneAttemptResult(
                exec_status=bundle.exec_status,
                failed_lanes=set(batches),
            )

        lane_checks = build_lane_checks(
            batches, bundle.subgoals, get_batch_target_object
        )
        updated_state, _subgoal_achieved, lane_results = self._verify_action_checks(
            bundle.merged_updated_state, lane_checks, bundle.execution_results
        )
        self.current_state = updated_state
        scheduler.scene_perceived = True

        successful_lanes = {
            lane for lane in batches
            if lane_results.get(lane, True)
        }
        return LaneAttemptResult(
            exec_status=bundle.exec_status,
            successful_lanes=successful_lanes,
            failed_lanes=set(batches) - successful_lanes,
        )

    def _begin_contact_monitoring(self, batches, sequences, subgoals, execution_results):
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

    def _end_contact_monitoring(self, batches, sequences, subgoals, execution_results):
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
                self.lfd,
                self.robot_entity,
                contact_predictor=self._contact_predictor_wrapper,
            )
            return self.effect_monitor.update(sensor_data)
        return self.effect_monitor.update()

    def online_tamp(self, belief):
        planning_session = plan_online_session(
            self.para, self.robot_entity, belief, **self.tamp_kwargs_dict)
        global_solution = planning_session.global_solution
        state_history = planning_session.state_history
        problem = planning_session.problem
        duration_tamp = planning_session.duration_tamp
        print(f"Duration of TAMP: {duration_tamp}")

        self.problem = problem
        self.stream_info = planning_session.stream_info
        self.static_environment = extract_static_environment(problem)

        scheduler_batches = build_scheduler_batches(global_solution.plan, state_history)
        scheduler = ArmSchedulerState(
            left_remaining=list(scheduler_batches['left']),
            right_remaining=list(scheduler_batches['right']),
            pending_barriers=list(scheduler_batches['barriers']),
            phase_quotas=list(scheduler_batches['phase_quotas']),
            scene_perceived=True,
        )

        self.current_state = state_history[0]

        MAX_LOCAL_RETRIES = 3

        while scheduler.left_remaining or scheduler.right_remaining or scheduler.pending_barriers:
            stage_batches = {
                lane: scheduler.current_batch(lane)
                for lane in ('left', 'right')
                if scheduler.current_batch(lane) is not None
            }

            ## execute per-arm motions
            if stage_batches:
                attempt = self._run_lane_attempt(
                    scheduler, stage_batches, plan_batch_fn=self._plan_lane_batch
                )
                ## Lane planning is stochastic (grasp sampling, RRT/IK restarts) and
                ## nothing has executed yet on a planning failure, so resample the
                ## whole stage before giving up.
                plan_retries = 0
                while (attempt.planning_failed_lane is not None
                       and plan_retries < MAX_LOCAL_RETRIES):
                    plan_retries += 1
                    print(f'\033[33m[online_tamp] {attempt.planning_failed_lane} lane planning failed, '
                          f'retrying ({plan_retries}/{MAX_LOCAL_RETRIES})\033[0m')
                    attempt = self._run_lane_attempt(
                        scheduler, stage_batches, plan_batch_fn=self._plan_lane_batch
                    )
                if attempt.planning_failed_lane is not None:
                    print(f'\033[31m[online_tamp] {attempt.planning_failed_lane} lane planning failed, aborting\033[0m')
                    break
                exec_status = attempt.exec_status
                if exec_status == 'fail':
                    break

                failed_lanes = set(attempt.failed_lanes)
                for lane in attempt.successful_lanes:
                    scheduler.pop_lane_batch(lane)

                local_retries = 0
                while failed_lanes and local_retries < MAX_LOCAL_RETRIES:
                    retry_batches = {
                        lane: scheduler.current_batch(lane)
                        for lane in failed_lanes
                        if scheduler.current_batch(lane) is not None
                    }
                    attempt = self._run_lane_attempt(
                        scheduler,
                        retry_batches,
                        plan_batch_fn=self._replan_lane_batch_with_detail_mp,
                    )
                    if attempt.planning_failed_lane is not None:
                        break
                    exec_status = attempt.exec_status
                    if exec_status == 'fail':
                        failed_lanes = set(retry_batches)
                        break

                    failed_lanes = set(attempt.failed_lanes)
                    for lane in attempt.successful_lanes:
                        scheduler.pop_lane_batch(lane)
                    if exec_status == 'success' and not failed_lanes:
                        break
                    local_retries += 1

                if failed_lanes:
                    print(f'\033[31m[online_tamp] Subgoal not achieved after {MAX_LOCAL_RETRIES} retries, aborting\033[0m')
                    break
                if exec_status == 'success':
                    break
                continue

            if scheduler.pending_barriers:
                barrier = scheduler.pending_barriers.pop(0)
                seq, updated_state, barrier_subgoal, barrier_execution_result = self._run_pending_barrier(barrier)
                if seq is None:
                    print('\033[31m[online_tamp] Explicit barrier planning failed, aborting\033[0m')
                    break
                exec_status = execute_request(
                    self.execute_robosuite_plan,
                    ExecutionRequest(sequence=seq, stop_mode=infer_stop_mode(seq)),
                ).status
                if exec_status == 'fail':
                    break
                barrier_checks = {
                    'barrier': {
                        'subgoal': list(barrier_subgoal),
                        'target_obj': get_action_target_object(barrier.action),
                    }
                } if barrier_subgoal else {}
                if action_invalidates_perception(barrier.action) or barrier_checks:
                    updated_state, _subgoal_achieved, barrier_results = self._verify_action_checks(
                        updated_state,
                        barrier_checks,
                        {'barrier': barrier_execution_result},
                    )
                    self.current_state = updated_state
                    scheduler.scene_perceived = True
                    if barrier_checks and not barrier_results.get('barrier', True):
                        print('\033[31m[online_tamp] Barrier effect not achieved after perception, aborting\033[0m')
                        break
                else:
                    self.current_state = updated_state
                scheduler.advance_phase()
                if exec_status in ('fail', 'success'):
                    break
                continue

            break

        recording_path = self.lfd.exit()
        self.output_info = build_output_info(
            task_name=self.task_name,
            task_success=self.success,
            duration_tamp=duration_tamp,
            recording_path=recording_path,
        )

      
    def perceive_env(self, updated_state, lane_checks):
        if not lane_checks:
            updated_state = preserve_arm_confs(
                self.current_state, updated_state, self.robot_entity.arms, search_facts
            )
            return updated_state, True, {}

        added = updated_state - self.current_state
        removed = self.current_state - updated_state
        affected_objects = [f[1] for f in search_facts(added | removed, 'atpose')]

        self._refresh_affected_object_meshes(affected_objects, reset_home=False)
        lane_results = self._primitive_subgoal_detector.detect(lane_checks, affected_objects=affected_objects)
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

    def execute_robosuite_plan(self, seq, stop_mode='replan'):
        """Execution contract supplied by OSCExecutionMixin in the concrete planner.

        The release build executes every route under OSC_POSE absolute control, so the
        bare base planner is never instantiated; subclassing without the mixin is a bug.
        """
        raise NotImplementedError(
            "execute_robosuite_plan must be provided by OSCExecutionMixin "
            f"(instantiate an *_OSC_planner, not {type(self).__name__})."
        )

    def iterate_process_sequence(self, robot_entity, seq):
        """EE-trajectory generator supplied by OSCExecutionMixin (see execute_robosuite_plan)."""
        raise NotImplementedError(
            "iterate_process_sequence must be provided by OSCExecutionMixin "
            f"(instantiate an *_OSC_planner, not {type(self).__name__})."
        )

    def __call__(self):
        return self.output_info

