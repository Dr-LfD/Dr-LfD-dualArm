

import sys
import os
root_path = next(p for p in (os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *([os.pardir] * k))) for k in range(16)) if os.path.isfile(os.path.join(p, '.repo_root')))
sys.path.append(root_path) if root_path not in sys.path else None
os.chdir(root_path)

from examples.pybullet.aloha_real.openworld_aloha.contact_monitoring import (
    ContactPredictorWrapper,
    SegReuseRegistry,
    build_effect_monitor,
    validate_seg_backend_pairing,
)


import time
import rospy
import numpy as np
import click
import signal
import atexit
import json
import threading

from examples.pybullet.aloha_real.openworld_aloha.run_openworld import (
    get_camera_mappings,
    plan_detail_mp,
    prepare_world,
)
from examples.pybullet.aloha_real.openworld_aloha.open_world_utils import get_skillwise_sgs, load_yaml_params
from examples.pybullet.aloha_real.openworld_aloha.primitives_test import test_primitive
from examples.pybullet.utils.pybullet_tools.aloha_primitives import   BodyPath, Command, compute_absolute_differences, refine_path
from examples.pybullet.aloha_real.openworld_aloha.primitives import (
    GroupConf,
    GroupTrajectory,
    Graphstate,
    Sequence,
    step_curve,
    execute_command,
)
from examples.pybullet.utils.pybullet_tools.utils import wait_for_user, CLIENT
from examples.pybullet.aloha_real.openworld_aloha.symbolic_utils import (
    get_contact_action_subgoal,
    get_barrier_action_subgoal,
    build_scheduler_batches,
    search_facts,
    DummyPrimitiveSubgoalDetector,
    UnifiedPrimitiveSubgoalDetector,
    _extract_skill_name_from_action,
)
from examples.pybullet.aloha_real.openworld_aloha.schema_executor import (
    ActionExecutionResult,
    execute_schema_skeleton_plan,
    extract_static_environment,
    materialize_ref_goal_result,
)
from examples.pybullet.aloha_real.scripts.ros_openworld_base import observation_to_file, obtain_demo_data
from examples.pybullet.aloha_real.scripts.standalone_scripts.schema_construction import load_runtime_schema_metadata
from examples.pybullet.aloha_real.scripts.tamp_workflow import (
    ArmSchedulerState,
    build_lane_checks,
    get_action_target_object,
    get_batch_target_object,
    plan_offline_sequence,
    plan_online_session,
    preserve_arm_confs,
    rollback_batch_execution,
)
from examples.pybullet.aloha_real.openworld_aloha.network_loader import get_lfd_wrapper, update_alohaMultiSkill_wrapper, update_alohaMultiEquivSkill_wrapper
from examples.pybullet.aloha_real.openworld_aloha.skill_naming import resolve_skill_env_key
from examples.pybullet.aloha_real.openworld_aloha.simple_worlds import update_mesh
from examples.pybullet.aloha_real.openworld_aloha.estimation.pc_utils import merge_cluster_by_label

class ExecutionAbort(RuntimeError):
    """Raised after a fatal execution error has already triggered shutdown."""


class RecoverablePerceptionError(Exception):
    """Mesh refresh / perception mismatch; caller should stop this run without shutting down ROS."""


class real_openworld(object):
    MAX_LOCAL_RETRIES = 10

    def __init__(self,   para):
        belief = self._setup_pipeline(para)
        if self._compute_tamp_and_populate_queues(belief):
            self._start_execution_loop()

    def _setup_pipeline(self, para, estimate_belief=True):
        self.para = para
        self.holding = {'left': -1, 'right': -1}
        self.task_name = para['task_name']

        self.executing_lfd = None
        self.effect_monitor = None

        self._online_tamp = para.get('use_online_tamp', False)
        self.scheduler = None
        self.current_state = None
        self.static_environment = None
        self._pending_verification = {}   # side/barrier -> pending verification payload
        self._waiting_for_barrier = {'left': False, 'right': False}
        self.local_retries = {'left': 0, 'right': 0}

        self.env_type = para['env_type']
        self.skill_names = para['skill_names']
        self._skillwise_sgs = get_skillwise_sgs(para, self.skill_names)

        # self.equivSkill_info_dict= update_alohaMultiSkill_wrapper( self.para, self.skill_names, skillwise_sgs)
        self.equivSkill_info_dict = update_alohaMultiEquivSkill_wrapper(
            self.para, self.skill_names, self._skillwise_sgs
        )

        self.schema_skill_metas = load_runtime_schema_metadata(
            self.para.get('skill_yaml_paths') or [],
            env_names=list(self.equivSkill_info_dict.keys()),
            root_path=root_path,
        )["skill_meta_map"]

        self._seg_pairing = validate_seg_backend_pairing(self.para)
        SegReuseRegistry.init_for_pairing(self._seg_pairing, self.para)

        self._effect_detection_defaults = {
            "contact_predictor_checkpoint": self.para.get("contact_predictor_checkpoint"),
            "seg_branch": self.para.get("seg_branch"),
            "contact_wrist_seg_backend": self.para.get("contact_wrist_seg_backend"),
            "text_prompt": self.para.get("text_prompt"),
            "sam_path": self.para.get("sam_path"),
            "sam3_worker_path": self.para.get("sam3_worker_path"),
            "sam3_path": self.para.get("sam3_path"),
            "sam3_model_dir": self.para.get("sam3_model_dir"),
            "sam3_checkpoint": self.para.get("sam3_checkpoint"),
            "sam3_conda_env": self.para.get("sam3_conda_env"),
            "sam3_conda_bin": self.para.get("sam3_conda_bin"),
        }

        if para['real_execute'] == False:
            print("Warn: Only for TAMP testing!")
            lfd_env = None
        else:
            self.lfd = get_lfd_wrapper(para,  with_planning = True)
            lfd_env = self.lfd.env
        self._contact_predictor_wrapper = (
            ContactPredictorWrapper(
                self.lfd,
                output_dir=os.path.join(root_path, para['vid_save_path']),
                contact_repo_root=self.para.get('contact_prediction_root'),
                workspace_root=root_path,
                effect_detection_defaults=self._effect_detection_defaults,
                seg_pairing=self._seg_pairing,
            )
            if getattr(self, 'lfd', None) is not None
            else None
        )
        os.chdir(root_path)

        cam_dir_mapping, cam_extparam_mapping, calibrate_mapping = get_camera_mappings(self.para)
        perception_fn = self.get_perception_fn(
            cam_dir_mapping,
            lfd_env=lfd_env,
            clear_dir=(self.env_type == 'real'),
        )
        self._perception_fn = perception_fn
        self._cam_dir_mapping = cam_dir_mapping
        self._cam_extparam_mapping = cam_extparam_mapping
        self._calibrate_mapping = calibrate_mapping

        debug_save_name = os.path.join(root_path, para['output_dir'], 'real')

        belief = None
        while belief is None:
            self.robot_entity, belief, self._estimator = prepare_world(
                para,
                env_type=self.env_type,
                lfd_env=lfd_env,
                perception_fn=perception_fn,
                sam_path=para['sam_path'],
                sam3_path=para.get('sam3_path'),
                sam3_model_dir=para.get('sam3_model_dir'),
                sam3_checkpoint=para.get('sam3_checkpoint'),
                sam3_conda_env=para.get('sam3_conda_env', 'sam3'),
                use_server=para['use_sam_server'],
                debug_save_name=debug_save_name,
                estimate_belief=estimate_belief,
            )

        self.arm_dof = self.robot_entity.arm_dof
        # Prefer first-round belief objects; fall back to world fixtures when needed.
        self._movable_objects_snapshot = list(belief.estimated_objects)
        if not self._movable_objects_snapshot:
            self._movable_objects_snapshot = list(
                getattr(self._estimator, '_world_movable_objects', [])
            )
        # Snapshot the fully prepared first-round belief surfaces as source-of-truth.
        # This preserves the perceived table together with static pads across replans.
        self._known_surfaces_snapshot = list(belief.known_surfaces)

        if self._contact_predictor_wrapper is not None:
            self._primitive_subgoal_detector = UnifiedPrimitiveSubgoalDetector(
                self.schema_skill_metas,
                contact_predictor=self._contact_predictor_wrapper,
                effect_detection_defaults=self._effect_detection_defaults,
                env_type="real",
                lfd=getattr(self, "lfd", None),
                robot_entity=self.robot_entity,
            )
        else:
            self._primitive_subgoal_detector = DummyPrimitiveSubgoalDetector(self.schema_skill_metas)

        return belief

    def _restore_world_objects_on_belief(self, belief):
        merged_objects = list(belief.estimated_objects)
        existing_object_categories = {obj.category for obj in merged_objects}
        for obj in self._movable_objects_snapshot:
            if obj.category in existing_object_categories:
                continue
            merged_objects.append(obj)
            existing_object_categories.add(obj.category)
        belief.estimated_objects = merged_objects
        # Refresh object snapshot each cycle so perceived objects persist across replans.
        self._movable_objects_snapshot = list(merged_objects)

        merged_surfaces = list(belief.known_surfaces)
        existing_categories = {surface.category for surface in merged_surfaces}
        for surface in self._known_surfaces_snapshot:
            category = surface.category
            if category in existing_categories:
                continue
            merged_surfaces.append(surface)
            existing_categories.add(category)
        belief.known_surfaces = merged_surfaces
        # Refresh snapshot from the latest merged belief surfaces so keyboard-trigger
        # mode (estimate_belief=False bootstrap) can retain a perceived table later.
        self._known_surfaces_snapshot = list(merged_surfaces)
        return belief

    def _compute_tamp_and_populate_queues(self, belief):
        para = self.para
        if self._online_tamp:
            tamp_kwargs = dict(
                equivSkill_info_dict=self.equivSkill_info_dict,
                skillwise_sgs=self._skillwise_sgs,
                env_type=self.env_type,
                task_name=self.task_name,
                primitive_learning_path=para['primitive_learning_path'],
                existing_domain_path=para['existing_domain_path'],
                existing_stream_path=para['existing_stream_path'],
            )
            _t0 = time.time()
            session = plan_online_session(self.para, self.robot_entity, belief, **tamp_kwargs)
            duration_tamp = time.time() - _t0
            print(f"Duration of TAMP (online, coarse): {duration_tamp}")
            self._tamp_kwargs = tamp_kwargs

            self.problem = session.problem
            self.stream_info = session.stream_info
            self.static_environment = extract_static_environment(self.problem)
            self.current_state = session.state_history[0]

            scheduler_batches = build_scheduler_batches(
                session.global_solution.plan, session.state_history
            )
            self.scheduler = ArmSchedulerState(
                left_remaining=list(scheduler_batches['left']),
                right_remaining=list(scheduler_batches['right']),
                pending_barriers=list(scheduler_batches['barriers']),
                phase_quotas=list(scheduler_batches['phase_quotas']),
                scene_perceived=True,
            )
            # Start with empty queues — lazily filled by _refill_lane()
            self.per_arm_queue = {'left': [], 'right': []}
        else:
            self.per_arm_queue, duration_tamp = self.calc_tamp_cmd(
                belief,
                text_prompt=para['text_prompt'],
                env_type=self.env_type,
                task_name=self.task_name,
                skillwise_sgs=self._skillwise_sgs,
                do_testing=para['do_testing'],
                primitive_learning_path=para['primitive_learning_path'],
                equivSkill_info_dict=self.equivSkill_info_dict,
                existing_domain_path=para['existing_domain_path'],
                existing_stream_path=para['existing_stream_path'],
            )
            print(f"Duration of TAMP: {duration_tamp}")

        if para['profile_plan_time']:
            import csv
            num_obj = len(belief.estimated_objects)
            rebuttal_csv = os.path.join(root_path, self.para['vid_save_path'], f'{num_obj}_obj_rebuttal.csv')
            file_exists = os.path.isfile(rebuttal_csv)
            with open(rebuttal_csv, 'a') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Duration of TAMP", "Number of Objects"])
                writer.writerow([duration_tamp, num_obj])
            return False

        if self._online_tamp:
            for side in ('left', 'right'):
                self._refill_lane(side)
            self._schedule_prefetch_for_next_barrier()
        else:
            while len(self.per_arm_queue['left']) == 0 and len(self.per_arm_queue['right']) == 0:
                print("No Command to execute")
                rospy.sleep(1)
            self._schedule_prefetch_first_graphstate_in_queues()

        self._sync_dp_skill_with_upcoming_graphstate()

        return True

    def _init_execution_state(self):
        self.pub_timer = None
        self.cur_body_path = {'left': None, 'right': None}
        self.wp_gen = {'left': None, 'right': None}
        self.old_tgt = {'left': None, 'right': None}
        self.arm_path_finished = {'left': False, 'right': False}
        self.max_diff_info = None
        self.video_saved = False
        self._video_save_lock = threading.Lock()

    def _register_exit_handlers(self):
        # Ensure video saves on all exit paths (graceful, debugger, exception)
        atexit.register(self.save_video_on_exit)
        signal.signal(signal.SIGINT, lambda sig, frame: self.handle_exit())
        signal.signal(signal.SIGTERM, lambda sig, frame: self.handle_exit())

    def _spin_until_shutdown(self):
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("\nReceived KeyboardInterrupt, saving video...")
            self.save_video_on_exit()
        except Exception as e:
            print(f"\nException occurred: {e}, saving video...")
            self.save_video_on_exit()
            raise
        finally:
            self.save_video_on_exit()

    def _start_execution_loop(self):
        self._init_execution_state()
        self.pub_timer = rospy.Timer(rospy.Duration(0.05), self.rbt_routine)
        self._register_exit_handlers()
        self._spin_until_shutdown()

    def get_perception_fn(self, cam_dir_mapping, **kwargs):
        def fn():
            if self.env_type == 'real':
                # save real-time img data to folder
                observation_to_file(
                    cam_dir_mapping,
                    robot_name=self.para['robot_name'],
                    cam_config=self.para,
                    **kwargs,
                )
            else:
                if self.task_name == 'screwdriver_franka':
                    ## read demo data and get initial observation
                    obtain_demo_data(self.para, cam_dir_mapping)
                print('Using cached img data!')
        return fn
    
    def calc_tamp_cmd(self, belief, do_testing = True,  **kwargs):
        CMD_PATH = os.path.join(self.para['output_dir'], 'cmd.pkl')
        if do_testing:
            import time
            start_time = time.time()
            sequence = test_primitive(self.para, self.robot_entity, belief,     **kwargs)
            duration_tamp = time.time() - start_time
        else:
            planning_session = plan_offline_sequence(
                self.para,
                self.robot_entity,
                belief,
                interested_objects=self.para['pre_obj_names'],
                **kwargs,
            )
            sequence = planning_session.sequence
            duration_tamp = planning_session.duration_tamp
        
        per_arm_queue = self.process_cmd(sequence)
        return per_arm_queue, duration_tamp


    def _split_sequence_to_queues(self, sequence: Sequence) -> dict:
        """Split a Sequence into raw per-arm command queues without bridging."""
        self.gripper_joints = {'left': [], 'right': []}
        per_arm_queue = {'left': [], 'right': []}
        marker_iter = iter(getattr(sequence, 'graphstate_markers', ()))
        next_marker = next(marker_iter, None)

        def queue_graphstate(graphstate):
            print(f"Command {graphstate} is imitation learning")
            if graphstate.commands is None:
                graphstate._build_commands()
            per_arm_queue['left'].append(graphstate)
            per_arm_queue['right'].append(graphstate)

        for command_index, cmd in enumerate(sequence.commands):
            while next_marker is not None and next_marker[0] == command_index:
                queue_graphstate(next_marker[1])
                next_marker = next(marker_iter, None)
            if isinstance(cmd, GroupTrajectory):
                arm_side = cmd.group.split('_')[0]
                per_arm_queue[arm_side].append(cmd)
            elif isinstance(cmd, Graphstate):
                raise RuntimeError("Graphstate must be stored in sequence.graphstate_markers")
            else:
                print(f"Command {cmd} is not included")

        while next_marker is not None:
            queue_graphstate(next_marker[1])
            next_marker = next(marker_iter, None)

        return per_arm_queue

    def process_cmd(self, sequence: Sequence) -> dict:
        """Split a Sequence into per-arm queues, bridging from the live robot config."""
        return self.percept_adjust(self._split_sequence_to_queues(sequence))

    def process_cmd_no_adjust(self, sequence: Sequence) -> dict:
        """Split a Sequence into per-arm queues without bridging.

        Use when the sequence already contains the required connector motions
        (e.g. online TAMP batches rebuilt from symbolic AtConf during planning).
        """
        return self._split_sequence_to_queues(sequence)
    

    
    def percept_adjust(self, per_arm_queue, prev_ends=None):
        """Bridge any discontinuity between consecutive trajectories.

        ``prev_ends`` optionally seeds the bridge with a known previous
        per-side configuration. When omitted, the current robot configuration
        is read from the controller. Missing live state is treated as a fatal
        invariant violation instead of silently preserving the old queue.
        """
        if prev_ends is None:
            prev_ends = {'left': None, 'right': None}

        def get_target_gripper_positions(cmd):
            target = getattr(cmd, "target_gripper_positions", None)
            return None if target is None else list(target)

        def should_freeze_gripper(cmd):
            return bool(getattr(cmd, "freeze_gripper", False))

        adjusted_per_arm_queue = {'left': [], 'right': []}
        for side in ['left', 'right']:
            adjusted = []
            # Start from the actual robot position; bridging closes gaps to planned start.
            # After a Graphstate (LfD), position is unknown — reset and let post_lfd()
            # call percept_adjust() again with a fresh actual config.
            prev_conf = None if prev_ends[side] is None else list(prev_ends[side])

            for cmd in per_arm_queue[side]:
                if isinstance(cmd, Graphstate):
                    adjusted.append(cmd)
                    prev_conf = None
                    continue

                # Skip gripper-only commands — their path dimension < arm_dof,
                # so comparing against arm positions would crash conf_close().
                target_group = cmd.group
                qpos_tgt = list(cmd.path[0])
                target_dim = len(qpos_tgt)
                target_gripper = get_target_gripper_positions(cmd)
                needs_full_state = (target_dim > self.arm_dof) or (target_gripper is not None)

                if prev_conf is None:
                    live_group = f'{side}_robot' if needs_full_state else f'{side}_arm'
                    prev_conf = self._require_live_group_conf(live_group)

                if target_dim < self.arm_dof:
                    adjusted.append(cmd)
                    if len(prev_conf) > self.arm_dof:
                        prev_conf = list(prev_conf[:self.arm_dof]) + list(cmd.path[-1])
                    continue

                if target_dim > self.arm_dof:
                    if len(prev_conf) <= self.arm_dof:
                        raise RuntimeError(
                            f'Cannot bridge {target_group} from arm-only state {prev_conf}. '
                            'A full arm+gripper source configuration is required.'
                        )
                    if len(prev_conf) != target_dim:
                        raise RuntimeError(
                            f'Configuration dimension mismatch for {target_group}: '
                            f'source has {len(prev_conf)} values, target has {target_dim}.'
                        )

                if target_dim > self.arm_dof:
                    needs_bridge = not np.allclose(prev_conf, qpos_tgt)
                    bridge_start = list(prev_conf)
                else:
                    needs_bridge = not self.conf_close(prev_conf[:self.arm_dof], qpos_tgt)[0]
                    bridge_start = list(prev_conf[:self.arm_dof])

                if needs_bridge:
                    bridge = cmd.copy()
                    bridge.path = [bridge_start, qpos_tgt]
                    adjusted.append(bridge)

                adjusted.append(cmd)
                if target_dim <= self.arm_dof and len(prev_conf) > self.arm_dof:
                    # arm-only cmd following full state: preserve tracked gripper positions
                    prev_conf = list(cmd.path[-1]) + list(prev_conf[self.arm_dof:])
                else:
                    prev_conf = list(cmd.path[-1])

                if (
                    (target_dim == self.arm_dof)
                    and (target_gripper is not None)
                    and not should_freeze_gripper(cmd)
                ):
                    gripper_cmd = GroupTrajectory(
                        cmd.robot,
                        f'{side}_gripper',
                        [target_gripper, target_gripper],
                        client=cmd.client,
                    )
                    adjusted.append(gripper_cmd)
                    prev_conf = list(cmd.path[-1]) + list(target_gripper)

            adjusted_per_arm_queue[side] = adjusted

        return adjusted_per_arm_queue


    def motion_plan(self, cmd: Command):
        def stabilize_traj(path, append_num = 10):
            first_conf = path[0]
            head = np.array([first_conf for _ in range(append_num)])
            last_conf = path[-1]
            tail = np.array([last_conf for _ in range(append_num)])
            stabilized_path = np.concatenate([head, path, tail])
            return stabilized_path

        if hasattr(cmd, 'refined_qpos') and cmd.refined_qpos is not None:
            refined_qpos = cmd.refined_qpos
        else:
            total_steps = self.para['refine_num']
            num_steps = np.ceil(total_steps / len(cmd.path)-1).astype(int)
            refined_qpos = refine_path(self.robot_entity, cmd.joints, cmd.path, num_steps=num_steps)

        # Stabilize trajectory for gripper or combined robot groups (which include gripper)
        if 'gripper' in cmd.group or 'robot' in cmd.group:
            refined_qpos = stabilize_traj(refined_qpos)

        ## debug
        for qpos in refined_qpos:
            if np.any(np.isnan(qpos)):
                print("nan in refined_qpos")
                import pdb; pdb.set_trace()

        return refined_qpos
    
    
    def get_wp_gen(self, cmd):
        is_gripper_cmd = 'gripper' in cmd.group  # or cmd.group.endswith('_gripper')

        if (not is_gripper_cmd) and len(cmd.path) < self.para['refine_num']:
            cmd.path = self.motion_plan(cmd)

        if len(cmd.path) == 0:
            raise ValueError('No path to execute')

        def gen():
            for conf in cmd.path:
                yield conf
        return gen
        
    def conf_close(self, jpos1, jpos2, tolerance=0.05):
        abs_diff = compute_absolute_differences(jpos1[:self.arm_dof], jpos2[:self.arm_dof])
        max_diff = max(abs_diff)
        if max_diff > tolerance:
            return False, max_diff
        else:
            return True, max_diff
        
    def can_proceed(self, group = None, tolerance=0.05):

        arm_side = group.split('_')[0]
        cmd = self.cur_body_path[arm_side]

        ## current generator is not exhausted
        if cmd.get_executed() == False:
            return False
  
        # # Use looser tolerance for gripper or combined robot groups (which include gripper)
        # if 'gripper' in group or 'robot' in group:
        #     tolerance = 0.02  ## as obj can be bigger, do not be strict

        # # see if reached the end of the path
        group_jpos = self.robot_entity.controller.get_group_conf(group)

        can_proc, max_diff = self.conf_close(jpos1 = cmd.path[-1][:self.arm_dof], jpos2 = group_jpos[:self.arm_dof], tolerance=tolerance)

        if not can_proc:
            self.max_diff_info = {\
                'group_name':cmd.group,\
                # 'joint_id': max_diff.index(max_diff),\
                    'max_diff': max_diff,
                    }
            return False
        else:
            self.max_diff_info = None
            return True

        
    def warn_unreach(self):
        print_info = 'Target qpose not reached, details:'
        if self.max_diff_info:
            print_info += str(self.max_diff_info)
        print(print_info)

    def _abort_execution(self, message):
        print(f'\033[31m{message}\033[0m')
        self.handle_exit()
        raise ExecutionAbort(message)

    def _shutdown_pub_timer(self):
        if self.pub_timer is not None:
            self.pub_timer.shutdown()
            self.pub_timer = None

    def _on_recoverable_perception_failure(self, exc):
        print(f'\033[31m{exc}\033[0m')
        self._shutdown_pub_timer()
        self.executing_lfd = None
        if self.effect_monitor is not None and hasattr(self.effect_monitor, 'close'):
            self.effect_monitor.close()
        self.effect_monitor = None

    def _require_live_group_conf(self, group):
        live_conf = self.robot_entity.controller.get_group_conf(group)
        if live_conf is None:
            raise RuntimeError(f'Live configuration unavailable for "{group}"')
        return list(live_conf)

    def _require_live_dual_arm_config(self):
        real_config = self.robot_entity.controller.get_current_config()
        if real_config is None:
            raise RuntimeError('Live robot config unavailable during AtConf synchronization')

        real_config = list(real_config)
        expected_dim = 2 * self.arm_dof
        if len(real_config) < expected_dim:
            raise RuntimeError(
                f'Live robot config has dim {len(real_config)} < {expected_dim} during AtConf synchronization'
            )
        return real_config

    # def _apply_verified_state(self, verified_state, actions):
    #     self.current_state = verified_state

    @staticmethod
    def _state_delta_from_base(base_state, updated_state):
        base = set(base_state)
        updated = set(updated_state)
        return base - updated, updated - base

    @staticmethod
    def _apply_state_delta(base_state, removed_literals, added_literals):
        merged = set(base_state)
        merged -= set(removed_literals)
        merged |= set(added_literals)
        return merged

    @staticmethod
    def _sync_refreshed_object_pose_facts(state, obj_to_pose):
        """Call ``update_pose(world_pose)`` on the pose object already in ``AtPose``/``Pose``.

        After mesh refresh, *state* must still carry the same ``RelativePose`` (or
        equivalent) used in planning; only its internal pose is updated — no new
        pose objects are created here.
        """
        if not obj_to_pose:
            return set(state)
        merged = set(state)
        for obj, world_pose in obj_to_pose.items():
            if world_pose is None:
                raise RecoverablePerceptionError(
                    f'[_sync_refreshed_object_pose_facts] Missing observed_pose after mesh refresh '
                    f'for object {obj!r}'
                )

            pose_handle = None
            for fact in merged:
                if not isinstance(fact, tuple) or len(fact) < 3:
                    continue
                if str(fact[0]).lower() not in ('atpose', 'pose'):
                    continue
                if fact[1] is not obj:
                    continue
                param = fact[2]
                if hasattr(param, 'update_pose') and callable(getattr(param, 'update_pose')):
                    pose_handle = param
                    break

            if pose_handle is None:
                raise RecoverablePerceptionError(
                    f'[_sync_refreshed_object_pose_facts] No pose with update_pose() in state for '
                    f'{obj!r}; expected AtPose or Pose fact whose pose parameter is the same '
                    f'RelativePose (or equivalent) as in the planning problem init.'
                )
            pose_handle.update_pose(world_pose)
        return merged

    def _finalize_verified_state(self, updated_state, action_checks, execution_results, actions, sync_arms=None):
        verified, _achieved, results = self._verify_action_checks(
            updated_state, action_checks, execution_results
        )
        # if sync_arms:
        #     verified = self._sync_live_arm_atconfs(verified, arms=sync_arms)
        # self._apply_verified_state(verified, actions)
        self.current_state = verified
        return results

    @staticmethod
    def _peek_graphstate_skill_from_sequence(sequence):
        markers = getattr(sequence, 'graphstate_markers', None) or ()
        if not markers:
            return None
        _marker_index, gs = markers[-1]
        return getattr(gs, 'skill_name', None)

    def _schedule_prefetch_for_next_barrier(self):
        if getattr(self, 'lfd', None) is None:
            return
        if not self._online_tamp or self.scheduler is None:
            return
        if not self.scheduler.pending_barriers:
            return
        barrier = self.scheduler.pending_barriers[0]
        sk = _extract_skill_name_from_action(
            barrier.action, ref_goal_state=barrier.ref_goal_state
        )
        if not sk:
            raise RuntimeError(
                f'Cannot resolve skill for pending barrier action {barrier.action}'
            )
        self.lfd.prefetch_skill(sk)

    def _schedule_prefetch_first_graphstate_in_queues(self):
        if getattr(self, 'lfd', None) is None:
            return
        if self._online_tamp:
            return
        for side in ('left', 'right'):
            for cmd in self.per_arm_queue[side]:
                if isinstance(cmd, Graphstate):
                    sk = cmd.skill_name
                    if not sk:
                        raise RuntimeError(
                            'Graphstate in offline queue is missing skill_name for DP prefetch'
                        )
                    self.lfd.prefetch_skill(sk)
                    return

    def _sync_dp_skill_with_upcoming_graphstate(self):
        """Before motion, align DP weights/buffers with the first queued Graphstate (offline)."""
        if getattr(self, 'lfd', None) is None:
            return
        skill = None
        for side in ('left', 'right'):
            for cmd in self.per_arm_queue[side]:
                if isinstance(cmd, Graphstate):
                    skill = cmd.skill_name
                    break
            if skill:
                break
        if not skill:
            return
        if skill != self.lfd.cur_skill:
            self.lfd.consume_prefetched_weights(skill)
            self.lfd.reset_loaded_skill(reset_grippers=False, reinit_buffers=True)
        else:
            self.lfd.cancel_prefetch()

    # ------------------------------------------------------------------
    # Online-TAMP helper methods
    # ------------------------------------------------------------------

    def _refresh_object_meshes(self, target_objects):
        """Refresh meshes for *target_objects* using fresh multiview SAM segmentation.

        Returns:
            dict mapping each refreshed object to its post-refresh ``observed_pose``
            (same key order as iteration over *target_objects*). Empty dict if
            *target_objects* is empty.
        """
        if not target_objects:
            return {}
        # Move arms to home so they don't occlude camera views before capture
        # self.robot_entity.reset(reset_pybullet=False)
        self._perception_fn()
        camera_images = self._capture_camera_images()
        surface = None
        known_surfaces = self._estimator.belief.known_surfaces
        if known_surfaces and hasattr(known_surfaces[0], 'surface'):
            surface = known_surfaces[0].surface

        observed_clusters = {}
        for camera_image in camera_images:
            labeled_clusters = self._estimator.belief.get_clusters_w(
                camera_image,
                surface=surface,
            )
            observed_clusters = merge_cluster_by_label(observed_clusters, labeled_clusters)

        refreshed = 0
        poses_by_obj = {}
        for obj in target_objects:
            matched_cluster = None
            for label, cluster in observed_clusters.items():
                if label.category == obj.category:
                    matched_cluster = cluster
                    break
            if matched_cluster is None:
                raise RecoverablePerceptionError(
                    f'[_refresh_object_meshes] Missing refreshed cluster for "{obj.category}"'
                )
            pc = np.array([lp.point for lp in matched_cluster])
            if pc.size == 0:
                raise RecoverablePerceptionError(
                    f'[_refresh_object_meshes] Empty refreshed cluster for "{obj.category}"'
                )
            update_mesh(obj, pc, category=obj.category)
            observed = getattr(obj, 'observed_pose', None)
            if observed is None:
                raise RecoverablePerceptionError(
                    f'[_refresh_object_meshes] observed_pose unavailable after refresh for '
                    f'"{obj.category}"'
                )
            poses_by_obj[obj] = observed
            refreshed += 1
        print(f'[_refresh_object_meshes] Refreshed {refreshed}/{len(target_objects)} objects')
        return poses_by_obj

    def _affected_objects_for_checks(self, updated_state, lane_checks):
        """Objects whose meshes should be refreshed before subgoal detection.

        Combines symbolic ``AtPose`` deltas with explicit per-lane ``target_obj`` from
        ``lane_checks`` so verification does not depend on pose facts changing.
        """
        added = set(updated_state) - set(self.current_state)
        removed = set(self.current_state) - set(updated_state)
        objects = []
        seen = set()

        def add_obj(obj):
            if obj is None:
                return
            oid = id(obj)
            if oid in seen:
                return
            seen.add(oid)
            objects.append(obj)

        for fact in search_facts(added | removed, 'atpose'):
            if isinstance(fact, tuple) and len(fact) >= 2:
                add_obj(fact[1])
        for lane_check in (lane_checks or {}).values():
            add_obj(lane_check.get('target_obj'))
        return objects

    def _capture_camera_images(self):
        import cv2

        camera_images = []
        for rs_cam, cam_dir in self._cam_dir_mapping.items():
            color_path = os.path.join(cam_dir, 'color_image.png')
            depth_path = os.path.join(cam_dir, 'depth_image.png')
            info_path = os.path.join(cam_dir, 'depth_info.json')
            color_img = cv2.imread(color_path)
            depth_img_mm = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            if color_img is None or depth_img_mm is None:
                raise RecoverablePerceptionError(
                    f'[_capture_camera_images] Missing image data for camera "{rs_cam}"'
                )
            try:
                with open(info_path, 'r') as f:
                    camera_info = json.load(f)
            except FileNotFoundError:
                raise RecoverablePerceptionError(
                    f'[_capture_camera_images] Missing camera info for "{rs_cam}"'
                )

            cam_image = self._estimator.get_image_direct(
                color_img,
                depth_img_mm.astype(np.float32) / 1000.0,
                camera_info,
                self._cam_extparam_mapping[rs_cam],
                camlink2optical=self._calibrate_mapping[rs_cam],
            )
            if cam_image is None:
                raise RecoverablePerceptionError(
                    f'[_capture_camera_images] Failed to compose camera image for "{rs_cam}"'
                )
            camera_images.append(cam_image)
        return camera_images

    def perceive_env(self, updated_state, lane_checks):
        """Post-execution subgoal detection.

        Returns (verified_state, subgoal_achieved, lane_results).
        Does NOT mutate self.current_state — caller owns state writes.

        Runs ``detect`` first without mesh refresh. If any lane fails and
        affected objects are known, moves arms home, recaptures images, and
        ``update_mesh`` for those objects. Then replaces ``AtPose`` / ``Pose``
        facts for those objects with ``observed_pose`` from the refreshed mesh
        before ``preserve_arm_confs`` / rollback / detail-MP replan.
        """
        if not lane_checks:
            updated_state = preserve_arm_confs(
                self.current_state, updated_state, self.robot_entity.arms, search_facts
            )
            return updated_state, True, {}

        affected_objects = self._affected_objects_for_checks(updated_state, lane_checks)
        lane_results = self._primitive_subgoal_detector.detect(
            lane_checks, affected_objects=affected_objects
        )
        subgoal_achieved = all(lane_results.values()) if lane_results else True
        poses_by_obj = {}
        if not subgoal_achieved and affected_objects:
            print('[online_tamp] Subgoal not achieved, recapturing perception before detail-MP retry')
            poses_by_obj = self._refresh_object_meshes(affected_objects) or {}

        updated_state = self._sync_refreshed_object_pose_facts(updated_state, poses_by_obj)
        updated_state = preserve_arm_confs(
            self.current_state, updated_state, self.robot_entity.arms, search_facts
        )
        return updated_state, subgoal_achieved, lane_results

    def _verify_action_checks(self, updated_state, lane_checks, execution_results=None):
        verified, achieved, results = self.perceive_env(updated_state, lane_checks)
        if execution_results:
            for key, exec_result in execution_results.items():
                if not results.get(key, True):
                    verified = rollback_batch_execution(verified, exec_result)
        return verified, achieved, results

    def _plan_lane_batch(self, lane, batch):
        """Expand a lane batch into an executable Sequence via the schema executor."""
        lane_subgoal = get_contact_action_subgoal(batch.ref_goal_state, batch.actions)
        result = execute_schema_skeleton_plan(
            self.para,
            self.robot_entity,
            self.static_environment,
            batch.actions,
            self._planning_current_state(),
            **self._tamp_kwargs,
        )
        if result is None:
            return None, self.current_state, lane_subgoal, None

        return result.sequence, set(result.final_state), lane_subgoal, result


    def _replan_with_detail_mp(self, subgoal, label, batch=None):
        """Shared retry logic: run detail MP and return the standard 4-tuple.

        After a successful replan, if *batch* is provided, its ``actions`` are
        replaced in-place with the new ``plan_seg`` symbolic actions.  This ensures
        subsequent ``_plan_lane_batch`` / schema-executor calls use fresh stream
        payloads rather than stale ones from the original global plan.
        """
        print(f'[online_tamp] {label} schema planning failed, retrying with detail MP')
        seq, updated_state, plan_seg = plan_detail_mp(
            self.para,
            self.robot_entity,
            self.problem,
            self.stream_info,
            self._planning_current_state(),
            subgoal,
            static_environment=self.static_environment,
            **self._tamp_kwargs,
        )
        if seq is None:
            return None, self.current_state, subgoal, None
        if batch is not None and plan_seg is not None:
            batch.actions = list(plan_seg)
        return seq, set(updated_state), subgoal, None

    def _replan_lane_batch_with_detail_mp(self, lane, batch):
        subgoal = get_contact_action_subgoal(batch.ref_goal_state, batch.actions)
        return self._replan_with_detail_mp(subgoal, label=f'{lane} lane', batch=batch)

    def _enqueue_lane_retry_from_subgoal(self, side, batch, lane_subgoal):
        """Retry a failed lane from the verified current state without reusing old action args.

        ``scheduler.current_batch(side)`` still contains the original global-plan action,
        including its old stream objects (e.g. the original learned grasp payload).  A
        perception-driven retry must instead re-run detail MP from the subgoal so new
        streams produce fresh payloads.
        """
        if self.per_arm_queue[side]:
            raise RuntimeError(
                f'[online_tamp] Cannot enqueue {side} retry while its queue is non-empty'
            )

        seq, updated_state, _subgoal, exec_result = self._replan_with_detail_mp(
            lane_subgoal,
            label=f'{side} lane',
            batch=batch,
        )
        # if seq is None:
        #     self._abort_execution(f'[online_tamp] {side} lane detail-MP retry failed, aborting')
        #     return

        removed_literals, added_literals = self._state_delta_from_base(
            self.current_state,
            updated_state,
        )
        self._pending_verification[side] = (
            batch,
            lane_subgoal,
            exec_result,
            removed_literals,
            added_literals,
        )

        split = self.process_cmd_no_adjust(seq)
        self.per_arm_queue[side].extend(split[side])

    def _run_pending_barrier(self, barrier):
        """Expand a barrier action and materialise its symbolic state."""
        subgoal = get_barrier_action_subgoal(barrier.ref_goal_state, barrier.action)
        result = execute_schema_skeleton_plan(
            self.para,
            self.robot_entity,
            self.static_environment,
            [barrier.action],
            self._planning_current_state(),
            **self._tamp_kwargs,
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



    def _sync_live_arm_atconfs(self, state, arms=None):
        """Overwrite symbolic arm AtConf facts from the live controller state."""
        real_config = self._require_live_dual_arm_config()
        live_arm_confs = {
            'left_arm': GroupConf(self.robot_entity, 'left_arm', positions=real_config[:self.arm_dof]),
            'right_arm': GroupConf(
                self.robot_entity,
                'right_arm',
                positions=real_config[self.arm_dof:2 * self.arm_dof],
            ),
        }
        target_arms = tuple(arms or self.robot_entity.arms)
        synced_state = set(state)
        for arm in target_arms:
            arm_group = self.robot_entity.get_arm_group(arm)
            live_conf = live_arm_confs.get(arm_group)
            if live_conf is None:
                raise RuntimeError(f'Cannot map symbolic arm "{arm}" to a live arm configuration')
            synced_state = {
                fact for fact in synced_state
                if not (len(fact) >= 3 and str(fact[0]).lower() == 'atconf' and fact[1] == arm)
            }
            synced_state.add(('AtConf', arm, live_conf))
        return synced_state

    def _planning_current_state(self):
        """Return current_state with AtConf synced from live robot joints.

        In real-execution mode (real_execute=True) the sync is mandatory;
        any failure propagates as an error — do not swallow it silently.
        In simulation (real_execute=False) the live controller is absent,
        so self.current_state is returned unchanged.
        """
        if not self.robot_entity.real_execute:
            return self.current_state
        return self._sync_live_arm_atconfs(self.current_state)

    def _refill_lane(self, side, plan_fn=None):
        """Plan the next batch for *side* and push it onto per_arm_queue.

        Called when the lane's queue is exhausted.  In online mode this
        transparently schedules the next unit of work without a global
        replan, matching the robosuite online_tamp() semantics.
        ``plan_fn`` defaults to ``_plan_lane_batch`` (schema skeleton expansion).
        Pass ``_replan_lane_batch_with_detail_mp`` only when retrying after a
        verified execution failure (see ``_enqueue_lane_retry_from_subgoal``).
        If expansion returns no sequence, raises ``ValueError``.
        """
        if not self._online_tamp:
            return

        if side in self._pending_verification:
            self._on_lane_batch_complete(side)
            if self.per_arm_queue[side]:
                return

        batch = self.scheduler.current_batch(side)
        if batch is None:
            if self.scheduler.pending_barriers:
                # Barrier already dispatched — wait for its completion callback.
                if 'barrier' in self._pending_verification:
                    return
                self._waiting_for_barrier[side] = True
                if all(self._waiting_for_barrier.values()):
                    self._waiting_for_barrier = {'left': False, 'right': False}
                    self._dispatch_pending_barrier()
            else:
                self.arm_path_finished[side] = True
            return

        if plan_fn is None:
            plan_fn = self._plan_lane_batch
        seq, updated_state, lane_subgoal, exec_result = plan_fn(side, batch)
        if seq is None:
            raise ValueError('No solution for primitive online plan. Is MP blocked?')

        removed_literals, added_literals = self._state_delta_from_base(
            self.current_state,
            updated_state,
        )
        self._pending_verification[side] = (
            batch,
            lane_subgoal,
            exec_result,
            removed_literals,
            added_literals,
        )

        split = self.process_cmd_no_adjust(seq)
        self.per_arm_queue[side].extend(split[side])

    def _on_lane_batch_complete(self, side):
        """Verify the just-completed lane batch; retry or advance the scheduler."""
        if side not in self._pending_verification:
            return
        (
            batch,
            lane_subgoal,
            exec_result,
            removed_literals,
            added_literals,
        ) = self._pending_verification.pop(side)
        updated_state = self._apply_state_delta(
            self.current_state,
            removed_literals,
            added_literals,
        )
        lane_check = {
            side: {
                'subgoal': list(lane_subgoal or []),
                'target_obj': get_batch_target_object(batch),
            }
        }
        if exec_result is not None:
            exec_results_map = {side: exec_result}
        else:
            # detail-MP 路径无 exec_result；用已存的状态差量合成，保证 rollback 能正常运行
            synthetic = ActionExecutionResult(
                sequence=None,
                final_confs={},
                final_state=set(updated_state),
                added_facts=frozenset(added_literals),
                removed_facts=frozenset(removed_literals),
            )
            exec_results_map = {side: synthetic}
        results = self._finalize_verified_state(
            updated_state,
            lane_check,
            exec_results_map,
            batch.actions,
        )

        if results.get(side, True):
            self.scheduler.pop_lane_batch(side)
            self.local_retries[side] = 0
        else:
            self.local_retries[side] += 1
            # if self.local_retries[side] >= self.MAX_LOCAL_RETRIES:
            #     self._abort_execution(
            #         f'[online_tamp] {side} lane verification failed after '
            #         f'{self.MAX_LOCAL_RETRIES} retries, aborting'
            #     )
            #     return
            print(f'[online_tamp] {side} lane verification failed, '
                  f'retry {self.local_retries[side]}/{self.MAX_LOCAL_RETRIES}')
            self._enqueue_lane_retry_from_subgoal(side, batch, lane_subgoal)

    def _dispatch_pending_barrier(self):
        """Plan the first pending barrier and push its connector+Graphstate to both queues."""
        if not self.scheduler.pending_barriers:
            return
        barrier = self.scheduler.pending_barriers[0]
        seq, updated_state, subgoal, exec_result = self._run_pending_barrier(barrier)
        if seq is None:
            ## _replan_barrier_with_detail_mp
            subgoal = get_barrier_action_subgoal(barrier.ref_goal_state, barrier.action)
            seq, updated_state, subgoal, exec_result =  self._replan_with_detail_mp(subgoal, label='barrier')

            if seq is None:
                self._abort_execution('[online_tamp] Barrier planning failed, aborting')
                return

        self._pending_verification['barrier'] = (barrier, subgoal, exec_result, updated_state)

        skill_gs = self._peek_graphstate_skill_from_sequence(seq)
        if getattr(self, 'lfd', None) is not None:
            if not skill_gs:
                raise RuntimeError(
                    'Barrier skeleton plan must expose a Graphstate (graphstate_markers) for DP skill alignment'
                )
            if skill_gs != self.lfd.cur_skill:
                self.lfd.consume_prefetched_weights(skill_gs)
                self.lfd.reset_loaded_skill(reset_grippers=False, reinit_buffers=True)
            else:
                self.lfd.cancel_prefetch()

        split = self.process_cmd_no_adjust(seq)
        for s in ('left', 'right'):
            self.per_arm_queue[s].extend(split[s])

    def _on_barrier_complete(self):
        """Verify the just-completed barrier and advance the scheduler."""
        if 'barrier' not in self._pending_verification:
            return
        barrier, subgoal, exec_result, updated_state = self._pending_verification.pop('barrier')
        self.current_state = updated_state  ## for real biop, no need to check eff

        # barrier_check = {}
        # if subgoal:
        #     barrier_check['barrier'] = {
        #         'subgoal': list(subgoal),
        #         'target_obj': get_action_target_object(barrier.action),
        #     }
        # exec_results_map = {'barrier': exec_result} if exec_result is not None else {}
        # results = self._finalize_verified_state(
        #     updated_state,
        #     barrier_check,
        #     exec_results_map,
        #     [barrier.action],
        #     sync_arms=barrier.action.args[:2],
        # )

        # if barrier_check and not results.get('barrier', True):
        #     self._abort_execution('[online_tamp] Barrier effect not achieved after perception, aborting')
        #     return

        # Advance scheduler past this barrier
        if self.scheduler.pending_barriers:
            self.scheduler.pending_barriers.pop(0)
        self.scheduler.advance_phase()
        self._schedule_prefetch_for_next_barrier()

    # ------------------------------------------------------------------

    def post_lfd(self):
        self.executing_lfd = None
        if hasattr(self.effect_monitor, "close"):
            self.effect_monitor.close()
        self.effect_monitor = None
        if self._online_tamp:
            self._on_barrier_complete()
        if getattr(self, 'lfd', None) is not None:
            self.lfd.reset_loaded_skill(reset_grippers=False, reinit_buffers=True)
        # Re-bridge remaining queue from the actual post-LfD robot configuration
        self.per_arm_queue = self.percept_adjust(self.per_arm_queue)
        self.cur_body_path['left'] = None
        self.cur_body_path['right'] = None
        print("The LFD has finished")

    def handle_exit(self):
        print("\nReceived exit signal, saving video...")
        if getattr(self, 'lfd', None) is not None and hasattr(self.lfd, 'cancel_prefetch'):
            self.lfd.cancel_prefetch()
        self.save_video_on_exit()
        if not rospy.is_shutdown():
            rospy.signal_shutdown("Exit signal received")

    def save_video_on_exit(self):
        with self._video_save_lock:
            if self.video_saved:
                return
            self.video_saved = True

        if hasattr(self, 'lfd') and self.lfd is not None:
            if hasattr(self.lfd, 'image_list') and len(self.lfd.image_list) > 0:
                try:
                    save_dir = os.path.join(root_path, self.para['vid_save_path'])
                    self.lfd.exit(save_dir=save_dir)
                    print("Video saved successfully on exit")
                except Exception as e:
                    print(f"Error saving video on exit: {e}")
            else:
                print("No frames recorded, skipping video save")

    def _build_effect_monitor_sensor_data(self):
        ee_dist = self.robot_entity.get_ee_dist()
        return {
            'gripper_vals': {
                'left_arm': self.robot_entity.controller.l_gripper_val,
                'right_arm': self.robot_entity.controller.r_gripper_val,
            },
            # Encode ee_dist as synthetic positions so norm(left_arm-right_arm)==ee_dist
            'eef_xyz': {
                'left_arm': np.array([0.0, 0.0, 0.0]),
                'right_arm': np.array([ee_dist, 0.0, 0.0]),
            },
        }

    def _tick_lfd_execution(self):
        if self.executing_lfd is None:
            return False

        self.lfd.append_image(process_name="lfd")
        reached_lfd_end = self.lfd.inference()

        if self.effect_monitor is not None:
            if self.effect_monitor.update(self._build_effect_monitor_sensor_data()):
                self.post_lfd()
                return True

        if not reached_lfd_end:
            return True

        # if self.task_name in ('screw_handoff_clean', 'screw_handoff_clean_traj'):
        #     self.post_lfd()
        #     return True

        print("LFD has reached the end")
        self.save_video_on_exit()
        self.pub_timer.shutdown()
        return True

    def _pop_next_side_command(self, side):
        if not self.per_arm_queue[side]:
            raise RuntimeError(f'Expected a queued command for side "{side}" but the queue was empty')
        try:
            return self.per_arm_queue[side].pop(0)
        except IndexError as exc:
            raise RuntimeError(f'Failed to pop the next command for side "{side}"') from exc

    def _ensure_side_command_ready(self, side):
        current_cmd = self.cur_body_path[side]
        if isinstance(current_cmd, Graphstate):
            return

        if current_cmd is not None and not self.can_proceed(
            group=current_cmd.group,
            tolerance=self.para['motion_tolerance'],
        ):
            return

        if current_cmd is not None and not self.per_arm_queue[side]:
            return

        if current_cmd is None and not self.per_arm_queue[side]:
            if self._online_tamp:
                self._refill_lane(side)
                if not self.per_arm_queue[side]:
                    return
            else:
                self.arm_path_finished[side] = True
                return

        self.cur_body_path[side] = self._pop_next_side_command(side)
        if isinstance(self.cur_body_path[side], Graphstate):
            self.wp_gen[side] = None
            return

        self.wp_gen[side] = self.get_wp_gen(self.cur_body_path[side])()

    def _prepare_ready_commands(self):
        for side in ('left', 'right'):
            self._ensure_side_command_ready(side)

    def _finalize_if_finished(self):
        if not self.arm_path_finished['left'] or not self.arm_path_finished['right']:
            return False
        print("All paths have been executed!")
        self.save_video_on_exit()
        self.pub_timer.shutdown()
        return True

    def _active_traj_sides(self):
        sides = []
        for side in ('left', 'right'):
            cmd = self.cur_body_path[side]
            if cmd is not None and not isinstance(cmd, Graphstate):
                sides.append(side)
        return sides

    def _get_pending_graphstate(self):
        graphstates = [
            cmd for cmd in (self.cur_body_path['left'], self.cur_body_path['right'])
            if isinstance(cmd, Graphstate)
        ]
        if not graphstates:
            return None
        graphstate = graphstates[0]
        for other in graphstates[1:]:
            if other is not graphstate:
                raise RuntimeError('Expected at most one shared Graphstate command across both arms')
        return graphstate

    def _ensure_lfd_skill_loaded(self, sg):
        if getattr(self, 'lfd', None) is None:
            raise RuntimeError('LfD evaluator is not initialized (real_execute=False?)')
        if not sg.skill_name:
            raise RuntimeError('Graphstate missing skill_name before LfD')
        if sg.skill_name != self.lfd.cur_skill:
            raise RuntimeError(
                f'Graphstate skill {sg.skill_name!r} != loaded DP skill {self.lfd.cur_skill!r} '
                '(expected prefetch + barrier dispatch to align weights)'
            )

    def _start_graphstate_lfd(self, sg):
        print("Start LfD")
        self._ensure_lfd_skill_loaded(sg)
        self.lfd.mark_bc_segment_start()
        self.executing_lfd = resolve_skill_env_key(sg.skill_name, self.equivSkill_info_dict)
        skill_meta = self.schema_skill_metas.get(sg.skill_name)
        self.effect_monitor = build_effect_monitor(skill_meta, self._contact_predictor_wrapper, 'real')


    def _handle_idle_or_graphstate(self):
        sg = self._get_pending_graphstate()
        if sg is not None:
            self._start_graphstate_lfd(sg)
            return

        if self._online_tamp and 'barrier' in self._pending_verification:
            self._on_barrier_complete()
            return

        raise RuntimeError(
            'Execution tick reached an idle state with no active trajectory, no Graphstate, '
            'and no pending online barrier'
        )

    def _handle_side_trajectory_complete(self, side):
        self.wp_gen[side] = None
        if self._online_tamp:
            self.cur_body_path[side] = None
            return None

        self.arm_path_finished[side] = True
        return None

    def _step_side_trajectory(self, side):
        try:
            single_arm_tgt = next(self.wp_gen[side])
            self.old_tgt[side] = single_arm_tgt
        except StopIteration:
            if not self.per_arm_queue[side]:
                return self._handle_side_trajectory_complete(side)

            if self.old_tgt[side] is None:
                raise RuntimeError(
                    f'Trajectory generator for side "{side}" ended before producing any waypoint'
                )

            self.warn_unreach()
            single_arm_tgt = self.old_tgt[side]
            self.cur_body_path[side].set_executed(True)

        self.robot_entity.set_arm_gripper(side, single_arm_tgt)
        return single_arm_tgt

    def _step_active_trajectories(self, traj_sides):
        for side in traj_sides:
            self._step_side_trajectory(side)
        if getattr(self, 'lfd', None) is not None:
            self.lfd.collect_obs_for_tamp_step()
            self.lfd.append_image(process_name="tamp")

    def rbt_routine(self, event):
        self.robot_entity.controller.get_current_config()  # keep-alive poll; result unused
        if self._tick_lfd_execution():
            return

        self._prepare_ready_commands()
        if self._finalize_if_finished():
            return

        traj_sides = self._active_traj_sides()
        if not traj_sides:
            self._handle_idle_or_graphstate()
            return

        self._step_active_trajectories(traj_sides)


###########################################################################

@click.command()
@click.option('--mode', 
              type=click.Choice(['sim', 'file', 'real'], case_sensitive=False),
              default='real',
              help='Execution mode: sim (simulation), file (cached data), or real (real-world)')
@click.option('--testing', 
              default=False,
              help='Enable testing mode (primitive testing vs full TAMP)')
@click.option('--task-name',
              type=str,
              default='screwdriver_noisy',
              help='Task identifier (used for plan monitoring logic)')
@click.option('--skill-yaml-path',
              multiple=True,
              default=[],
              help='Path to a per-skill YAML (relative to project root or absolute). '
                   'Repeat to compose multiple skills.')
@click.option('--pyb-env-addon',
              multiple=True,
              default=[],
              help='PyBullet environment additions (can be specified multiple times)')
@click.option('--profile-plan-time/--no-profile-plan-time',
              default=False,
              help='Enable planning time profiling')
@click.option('--cfg-path',
              default='examples/pybullet/aloha_real/openworld_aloha/configs/sgBase.yaml',
              help='Path to the configuration YAML file')
@click.option('--online-tamp/--offline-tamp',
              default=False,
              help='Use online TAMP (coarse plan + lazy per-batch refinement) instead of offline one-shot planning')
def main(mode, testing, task_name, skill_yaml_path, pyb_env_addon, profile_plan_time, cfg_path, online_tamp):
    """
    ALOHA Real-World Robot Control System

    Examples:
        # Single skill
        python interleaved_real_traj_plugin.py --mode real --task-name screwdriver_noisy \\
            --skill-yaml-path examples/pybullet/aloha_real/openworld_aloha/configs/skill_cfg/screwdriver_per_skill.yaml

        # Composed task (screwdriver + handoff + clean_cup)
        python interleaved_real_traj_plugin.py --mode file --task-name screw_handoff_clean \\
            --skill-yaml-path examples/pybullet/aloha_real/openworld_aloha/configs/skill_cfg/handoff_cup_per_skill.yaml \\
            --skill-yaml-path examples/pybullet/aloha_real/openworld_aloha/configs/skill_cfg/clean_cup_per_skill.yaml \\
            --skill-yaml-path examples/pybullet/aloha_real/openworld_aloha/configs/skill_cfg/screwdriver_per_skill.yaml
    """
    skill_yaml_paths = list(skill_yaml_path)
    pyb_env_addon = list(pyb_env_addon)

    print(f"Starting ALOHA with mode: {mode}, testing: {testing}")
    print(f"Task: {task_name}, Skills: {skill_yaml_paths}")
    print(f"Environment additions: {pyb_env_addon}")

    # Load parameters from YAML configuration
    parameters = load_yaml_params(cfg_path, skill_yaml_paths=skill_yaml_paths, task_name=task_name, mode=mode, is_testing=testing)
    
    # Set additional parameters
    parameters['pyb_env_addon'] = pyb_env_addon
    parameters['profile_plan_time'] = profile_plan_time
    parameters['use_online_tamp'] = online_tamp
    
    # Initialize and run the real openworld system
    real_openworld(parameters)

if __name__ == '__main__':
    main()
