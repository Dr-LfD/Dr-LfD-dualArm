import math
import random
import time
from itertools import cycle, islice

import numpy as np
import pybullet as p
import sys
import os
root_path = next(('/'+ os.path.join(*os.path.dirname(os.path.abspath(__file__)).split(os.sep)[:i+1]) + os.sep
                    for i in range(len(os.path.dirname(os.path.abspath(__file__)).split(os.sep))) 
                    if os.path.dirname(os.path.abspath(__file__)).split(os.sep)[i] == 'pddlstream_aloha'), None)
sys.path.append(root_path) if root_path not in sys.path else None

from pddlstream.language.constants import get_args, get_prefix
from pddlstream.utils import lowercase
from examples.pybullet.aloha_real.openworld_aloha.skill_naming import (
    build_skill_to_env_map,
    policy_skill_name,
    resolve_policy_skill_name,
)
from examples.pybullet.aloha_real.openworld_aloha.network_loader import categorize_skill


def _wrapper_skill_keys(wrapper):
    """Best-effort set of skill-embedding keys the loaded per-skill model carries.

    Used to resolve the obj/surf order of place-skill names against what the model
    was actually trained with (see resolve_policy_skill_name). Returns None when the
    keys can't be read, in which case the canonical name is used unchanged.
    """
    try:
        embs = wrapper.agent.actor.statistics.get('skill_embs_all_tasks')
        if embs:
            return set(embs.keys())
    except Exception:
        pass
    try:
        return set(wrapper.get_skill_names())
    except Exception:
        return None

from examples.pybullet.utils.pybullet_tools.transformations import quaternion_matrix, euler_from_quaternion, quaternion_slerp
from examples.pybullet.utils.pybullet_tools.utils import (
    remove_all_debug,
    INF,
    OOBB,
    PI,
    BodySaver,
    get_joint_names,
    Euler,
    Point,
    Pose,
    PoseSaver,
    Tuple,
    aabb_from_extent_center,
    any_link_pair_collision,
    buffer_aabb,
    get_aabb,
    compute_jacobian,
    convex_area,
    convex_combination,
    draw_pose,
    elapsed_time,
    get_aabb,
    get_aabb_center,
    get_aabb_extent,
    get_center_extent,
    aabb_from_points,
    get_closest_points,
    scale_aabb,
    get_extend_fn,
    get_joint_positions,
    get_length,
    get_link_pose,
    get_moving_links,
    get_point,
    get_pose,
    get_unit_vector,
    get_wrapped_pairs,
    inf_generator,
    invert,
    find_kw_in_skill,
    movable_from_joints,
    multiply,
    pairwise_collision,
    pairwise_collisions,
    plan_2d_joint_motion,
    plan_joint_motion,
    point_from_pose,
    pose_from_tform,
    tform_from_pose,
    quat_from_pose,
    randomize,
    recenter_oobb,
    remove_handles,
    sample_placement_on_aabb,
    set_joint_positions,
    set_pose,
    stable_z_on_aabb,
    tform_point,
    safe_zip,
    link_from_name,
    get_collision_fn,
    approximate_as_prism,
    WorldSaver,
)
from examples.pybullet.utils.pybullet_tools.ikfast.ikfast import (
    closest_inverse_kinematics,
    get_ik_joints,
    ikfast_inverse_kinematics,
    get_ik_fn,
)
# from grasp.utils import gpd_predict_grasps, graspnet_predict_grasps
from examples.pybullet.aloha_real.openworld_aloha.estimation.geometry import trimesh_from_body
from examples.pybullet.aloha_real.openworld_aloha.estimation.surfaces import z_plane
from examples.pybullet.aloha_real.openworld_aloha.primitives import (
    BaseSwitch,
    Grasp,
    GroupConf,
    GroupTrajectory,
    LearnedGrasp,
    RelativePose,
    Sequence,
    Switch,
    Trajectory,
    Graphstate,
)
from examples.pybullet.aloha_real.openworld_aloha.aloha_samplers import (
    COLLISION_DISTANCE,
    MOVABLE_DISTANCE,
    SELF_COLLISIONS,
    compute_gripper_path,
    plan_prehensile,
    plan_workspace_motion,
    sample_attachment_base_confs,
    sample_prehensive_base_confs,
    sample_visibility_base_confs,
    # set_open_positions,
    workspace_collision,
    DISABLE_ALL_COLLISIONS

)

from examples.pybullet.aloha_real.scripts.aloha_tamp_constants import trans2eepose, xyzquat2trans


from examples.pybullet.aloha_real.openworld_aloha.stacking import slice_mesh
from examples.pybullet.aloha_real.openworld_aloha.entities import WORLD_BODY, ParentBody

from examples.pybullet.aloha_real.openworld_aloha.grasp.utils import gpd_predict_grasps


SWITCH_BEFORE = "grasp"  # contact | grasp | pregrasp | arm | none # TODO: tractor
BASE_COST = 1
PROXIMITY_COST_TERM = False
REORIENT = False



TELEPORT = [False]



def get_learned_pick_fn(robot, environment=None, **kwargs):
    robot_saver = BodySaver(robot, client=robot.client)
    environment = [] if environment is None else list(environment)

    def fn(arm, obj, pose, learned_grasp):
        if not hasattr(learned_grasp, "_aq_start") or learned_grasp._aq_start is None:
            return None

        aq_start = learned_grasp._aq_start
        aq_end = learned_grasp._aq_end
        traj_seq = learned_grasp._traj_seq

        side = robot.side_from_arm(arm)
        arm_group = arm
        arm_joints = robot.get_group_joints(arm_group)
        _, _, tool_name = robot.manipulators[side]
        attachment = learned_grasp.create_attachment(
            robot, link=robot.link_from_name(tool_name)
        )

        robot_saver.restore()
        with WorldSaver(client=robot.client):
            aq_end.assign()
            attachment.assign()
            return_path = plan_joint_motion(
                robot,
                arm_joints,
                aq_start.positions[: len(arm_joints)],
                attachments=[attachment],
                obstacles=environment,
                self_collisions=True,
                disabled_collisions=robot.disabled_collisions,
                custom_limits=robot.custom_limits,
                restarts=2,
                iterations=50,
                smooth=100,
                disable_collisions=DISABLE_ALL_COLLISIONS,
            )

        if return_path is None:
            return None

        return_traj = GroupTrajectory(
            robot,
            arm_group,
            return_path,
            client=robot.client,
        )
        full_seq = Sequence(
            commands=list(traj_seq.commands) + [return_traj],
            name="learned-pick-closed",
        )
        full_seq._source_learned_grasp = learned_grasp
        print(
            f"[plan-learned-pick] lg={learned_grasp!r} id={id(learned_grasp)} "
            f"at={full_seq!r} id={id(full_seq)} traj_seq_id={id(traj_seq)}"
        )
        return (aq_start, full_seq)

    return fn

def get_parent_body(skill_type, robot, tool_name):
    if skill_type == 'ATTACH':
        parent=ParentBody(
            body=robot, link=robot.link_from_name(tool_name), client=robot.client
        )
    else:
        parent = WORLD_BODY ## release
    return parent



## gripper_jpath output by net should be in joint angles. It will be converted to position in this function. 
def get_gripper_path(robot, gripper_jpath_1d, arm_gripper_associate_ids = None, skill_type = 'ATTACH', smooth = False):

    if skill_type == 'DETACH':
        _, open_pos = robot.close_open_conf()
        gripper_jpath_1d = np.array(gripper_jpath_1d, copy=True)
        gripper_jpath_1d[-1] = robot.pos2joint_gripper(open_pos[0])  # force full open on release

    gripper_path_2d = [robot.joint2pos_gripper(gripper_jpose) for gripper_jpose in gripper_jpath_1d]
     # manually add lift motion. TODO: revise the hand_obj_dist when segment the motion. 
    gripper_path_2d.insert(0, gripper_path_2d[0])  
    gripper_path_2d.append(gripper_path_2d[-1]) 
    
    if arm_gripper_associate_ids is not None:
        N = len(arm_gripper_associate_ids)
        M = len(gripper_path_2d)
        gripper_arr = np.array(gripper_path_2d)  # shape (M, 2)
        ids = np.array(arm_gripper_associate_ids)

        # Find center of each waypoint's run in out_ids as anchor positions.
        # This preserves arm-gripper synchronization: the gripper transitions
        # at the same pace as the arm, not at uniform speed.
        anchor_positions = []
        anchor_wp_indices = []
        for j in range(M):
            positions = np.where(ids == j)[0]
            if len(positions) > 0:
                anchor_positions.append(positions.mean())
                anchor_wp_indices.append(j)

        anchor_positions = np.array(anchor_positions)
        anchor_wp_indices = np.array(anchor_wp_indices)

        x_query = np.arange(N, dtype=float)
        gripper_path_2d = np.column_stack([
            np.interp(x_query, anchor_positions, gripper_arr[anchor_wp_indices, col])
            for col in range(gripper_arr.shape[1])
        ])
    return np.array(gripper_path_2d)

def pregrasp_pose_from_waypoint(robot, obj, obj_pose_w, wp_mat, tool_dist=0.12):
    """World-frame EE pregrasp pose for a grasp waypoint.

    `wp_mat` is a world-frame EE transform (4x4); the returned pose is that grasp
    backed off `tool_dist` along the tool approach axis (`Grasp.pregrasp`:
    panda +z / aloha +x). Shared by the joint-space (`get_jspace_path`) and
    EE-traj learned-attach paths so the approach/retreat is computed identically.
    """
    wp_o, _ = get_grasp_from_mat(wp_mat, obj)
    grasp = Grasp(obj, wp_o, robot_name=robot.name, tool_dist=tool_dist, obj_dist=0)
    return multiply(obj_pose_w, invert(grasp.pregrasp))


def get_jspace_path(robot, obj_pose, gpose_traj, side, hand_obj_dist_fn, offset_trans = None,
                    obj = None, **kwargs):
    # gpose_traj = group_actions['grasp']
    # gpose_traj = group_actions[side + '_grasp']

    if offset_trans is not None:
        gpose_traj = [np.dot(gtrans, offset_trans) for gtrans in gpose_traj]

    # gpose_waypoints = [objcentric_trans_to_w_gpose(obj_pose, gtrans) for gtrans in gpose_traj]
    gpose_waypoints = [trans2eepose(gtrans) for gtrans in gpose_traj] ## gpose_traj is already in world frame

    # Pre-grasp approach (start) and retreat (end), via the shared helper.
    obj_pose_w = obj_pose.get_pose()
    gpose_waypoints.insert(0, pregrasp_pose_from_waypoint(robot, obj, obj_pose_w, gpose_traj[0]))
    gpose_waypoints.append(pregrasp_pose_from_waypoint(robot, obj, obj_pose_w, gpose_traj[-1]))

    # if TELEPORT[0]:
    #     gpose_waypoints = [gpose_waypoints[0], gpose_waypoints[-1]]

    obj_pose.assign()
    for i in range(len(gpose_waypoints)):
        draw_pose(gpose_waypoints[i], length=0.05, **kwargs)

    arm_path_and_out_ids = plan_workspace_motion(
        robot, side, gpose_waypoints, **kwargs
    )



    # Skip the inserted first/last pregrasp waypoints so switch_id - 1 maps
    # _lowest_learned_waypoint_index
    switch_id = min(range(1, len(gpose_waypoints) - 1), key=lambda i: gpose_waypoints[i][0][2])
    return arm_path_and_out_ids, switch_id


def densify_ee_path(sparse_poses, step_m=0.01, step_rad=0.05):
    """Densify a list of pybullet-convention (pos, quat_xyzw) poses via lerp + SLERP.

    Returns (dense_poses, source_indices) where source_indices[i] is the
    *target* sparse waypoint index that dense_poses[i] moves toward, shifted by
    +1 for get_gripper_path's prepended start waypoint. This mirrors
    interpolate_joint_waypoints' target-index semantics (utils.py), so the
    gripper schedule built by get_gripper_path anchors every grasp waypoint —
    including the final close — exactly as in joint-space mode. (Using the
    segment-source index instead drops the close anchor and shuts the gripper
    before the EE reaches the target.)
    """
    if not sparse_poses:
        return [], []

    def _quat_angle(q0, q1):
        q0 = np.asarray(q0, dtype=float)
        q1 = np.asarray(q1, dtype=float)
        q0 = q0 / (np.linalg.norm(q0) + 1e-12)
        q1 = q1 / (np.linalg.norm(q1) + 1e-12)
        dot = np.clip(abs(np.dot(q0, q1)), 0.0, 1.0)
        return 2.0 * math.acos(dot)

    dense = [(tuple(sparse_poses[0][0]), tuple(sparse_poses[0][1]))]
    source_ids = [1]
    for i in range(len(sparse_poses) - 1):
        p0 = np.asarray(sparse_poses[i][0], dtype=float)
        q0 = np.asarray(sparse_poses[i][1], dtype=float)
        p1 = np.asarray(sparse_poses[i + 1][0], dtype=float)
        q1 = np.asarray(sparse_poses[i + 1][1], dtype=float)
        dist = float(np.linalg.norm(p1 - p0))
        angle = _quat_angle(q0, q1)
        steps = max(1, int(math.ceil(max(dist / step_m, angle / step_rad))))
        for k in range(1, steps + 1):
            frac = float(k) / float(steps)
            pos = (1.0 - frac) * p0 + frac * p1
            quat = quaternion_slerp(q0, q1, frac)
            dense.append((tuple(pos), tuple(quat)))
            # Target sparse waypoint is (i + 1); +1 for get_gripper_path's
            # prepended start padding -> gripper index (i + 2).
            source_ids.append(i + 2)
    return dense, source_ids


def get_grasp_from_mat(grasp_trans_predeff, obj, **kwargs):
    pred_grasp_trans = grasp_trans_predeff[:4, :]
    pred_grasp_pose = trans2eepose(pred_grasp_trans)
    pred_grasp_relative = multiply(invert(pred_grasp_pose), obj.initial_pose)

    if grasp_trans_predeff.shape[0] == 8:
        eff_grasp_trans = grasp_trans_predeff[4:, :]
        eff_grasp_pose = trans2eepose(eff_grasp_trans)
        eff_grasp_relative = multiply(invert(eff_grasp_pose), obj.initial_pose)
    else:
        eff_grasp_relative = None

    return pred_grasp_relative, eff_grasp_relative

def get_grasp_from_trans_w(trans_w, obj_pose, offset_trans = None):
    if offset_trans is not None:
        trans_w = np.dot(trans_w,offset_trans)
    grasp_pose_w = trans2eepose(trans_w)
    obj_pose_w = obj_pose.get_pose()
    grasp_relative = multiply(invert(grasp_pose_w), obj_pose_w)
    return grasp_relative

def get_place_pose_from_trans_w(trans_w, grasp_relative, offset_trans = None):
    if offset_trans is not None:
        trans_w = np.dot(trans_w, offset_trans)
    grasp_pose_w = trans2eepose(trans_w)
    place_pose_w = multiply(grasp_pose_w, grasp_relative)
    return place_pose_w


def get_object_obs(obj_dict, pose = None, use_rgb = False):
    agent_obs = {}
    for pc_name, obj in obj_dict.items():
        if pose is not None:
            pose.assign()
            obj2world_trans = xyzquat2trans(pose.get_pose())
            points_o = np.array([lp.point for lp in obj.labeled_points])
            points_w = np.dot(
                np.concatenate([points_o, np.ones((points_o.shape[0], 1))], axis=1),
                obj2world_trans.T,
            )
        else:
            points_w = np.array(obj.get_world_points())

        if use_rgb:
            rgb = np.array([lp.color for lp in obj.labeled_points])
            agent_obs.update({pc_name: np.concatenate([points_w[:, :3], rgb], axis=1)})
        else:
            agent_obs.update({pc_name: points_w[:, :3]})

    return agent_obs

def get_agent_obs(obj_dict):
    agent_obs = {}
    for pc_name, obj in obj_dict.items():
        set_pose(obj, obj.initial_pose)
        points_w = np.array(obj.get_world_points())
        assert points_w.shape[1] == 3
        agent_obs.update({pc_name: points_w[:, :3]})
    
    return agent_obs

def get_per_skill_infos(equivSkill_info_dict, skill_name):
    # load the diffusion model
    tamp_wrapper = equivSkill_info_dict[skill_name]['tamp_wrapper']
    skillwise_sgs = equivSkill_info_dict[skill_name]['skillwise_sgs']

    pre_obj_names = skillwise_sgs[skill_name]['pre_sg'].graph['obj_names']
    num_grasps = len(pre_obj_names)

    return tamp_wrapper, skillwise_sgs, num_grasps

def get_imitate_traj_fn(robot, equivSkill_info_dict, prefix_key, skill_name,  fixed_obj = [],  is_dmg = True, initial_pc_dict = None,  eff_grasps = None, **kwargs):
    ee_traj_mode = bool(kwargs.pop("ee_traj_mode", False))

    def get_bicond_objs(pre_skill_sg_dict, is_fixed = True):
        pre_obj_names = pre_skill_sg_dict.get('related_objs', [])
        if is_fixed:
            conditioned_objs = set(pre_obj_names).intersection(set(fixed_obj))
        else:
            conditioned_objs = set(pre_obj_names)
        return list(conditioned_objs)
    
        
    robot_saver = BodySaver(robot, client=robot.client)
    # # load the diffusion model
    # env_keys = list(equivSkill_info_dict.keys())
    # ## multiple sets of demos included
    # if len(env_keys) > 1:
    #     assert skill_name in env_keys, f"Skill {skill_name} not found in equivSkill_info_dict keys: {env_keys}"
    #     prefix_key = skill_name
    # else: ## for most simulation tasks
    #     prefix_key = env_keys[0]

    tamp_wrapper = equivSkill_info_dict[prefix_key]['tamp_wrapper']
    obj_centric_mode = equivSkill_info_dict[prefix_key]['obj_centric_mode']
    task_name = equivSkill_info_dict[prefix_key]['task_name']

    # Optional grasp backend: 'diffusion' (default) keeps the learned policy,
    # 'm2t2' swaps the ATTACH generator for an M2T2-predicted grasp pose while
    # leaving DETACH/place on the diffusion policy.
    grasp_backend = kwargs.get('grasp_backend', 'diffusion')
    # Optional place backend: 'diffusion' (default) keeps the learned policy,
    # 'generic' swaps the DETACH generator for a geometric placement sampled on
    # the goal surface (Region) and planned with get_plan_place_fn.
    place_backend = kwargs.get('place_backend', 'diffusion')
    # Bound on how many sampled placements the generic DETACH generator tries
    # before giving up (StopIteration) so PDDLStream can move on.
    generic_place_max_attempts = int(kwargs.get('generic_place_max_attempts', 20))
    # Bound on CONSECUTIVE failed learned-trajectory samples (unimanual_traj_gen
    # returning None, e.g. endpoint IK never solvable for a degenerate scene)
    # before the generator exhausts via StopIteration. Without this, a single
    # next() call spins forever and PDDLStream's max_tamp_time — checked only
    # between stream calls — cannot interrupt it. The streak resets on every
    # success, so healthy bindings keep their unbounded-sampling semantics.
    learned_traj_max_attempts = int(kwargs.get('learned_traj_max_attempts', 40))
    m2t2_grasp_wrapper = kwargs.get('m2t2_grasp_wrapper')
    m2t2_contact_radius = kwargs.get('m2t2_contact_radius', 0.03)
    # Forward shift (m) of the M2T2 grasp pose along its own approach axis
    # (local +z) before IK. M2T2 reports the pose origin at the panda hand base,
    # but the IK target `tool_link` is the fingertip/TCP (~0.1 m forward), so the
    # raw pose lands the fingers short of the contact. Tune this so the gripper
    # closes on the object (see panda_arm_hand.urdf tool_joint z=0.1 and the M2T2
    # control points at 0.1053 m).
    m2t2_grasp_depth = kwargs.get('m2t2_grasp_depth', 0.10)
    # Bound on how many fresh M2T2 prediction batches to draw before the grasp
    # generator gives up (StopIteration) so PDDLStream can move on instead of
    # spinning on expensive inference when no candidate is graspable.
    m2t2_max_prediction_batches = kwargs.get('m2t2_max_prediction_batches', 5)

    # GPD grasp backend ('gpd'): generic grasp poses from the GPD detector run on
    # the target object cloud. gpd_camera_point is the world-frame viewpoint GPD
    # uses to orient approach directions (sourced from the real LIBERO scene camera
    # extrinsic in the plugin). gpd_grasp_depth mirrors m2t2_grasp_depth, and
    # gpd_max_candidates caps how many ranked GPD grasps we try IK on.
    gpd_camera_point = kwargs.get('gpd_camera_point')
    gpd_grasp_depth = kwargs.get('gpd_grasp_depth', 0.10)
    gpd_max_candidates = kwargs.get('gpd_max_candidates', 10)

    # Stationary dwell (replay steps) at the grasp pose during which the m2t2/gpd
    # pick closes the gripper; the arm holds still so the fingers wrap before the
    # retreat starts instead of closing mid-descent.
    grasp_close_dwell_steps = int(kwargs.get('grasp_close_dwell_steps', 10))

    skillwise_sgs_flattend = {
        sk_name: skill_info
        for task_dict in equivSkill_info_dict.values()
        for sk_name, skill_info in task_dict["skillwise_sgs"].items()
    }

    skill_to_env = build_skill_to_env_map(equivSkill_info_dict)


    # Capture the eff_grasps mapping for the unified biop-keypose stream.
    # Each entry is (schema_arm_name, schema_obj_name) in the same order as
    # the extra (obj, pose) arguments the stream will pass to the generator.
    _eff_grasps_mapping = list(eff_grasps) if eff_grasps else []

    ## NOTE: the learnt bimanual kp is inaccurate.
    def decode_bimanual_kp(group_actions, eef_key):
    
        eef_trans_seq = group_actions[eef_key]
        seq_len_half = eef_trans_seq.shape[0] //2
        left_trans_seq = eef_trans_seq[:seq_len_half]
        right_trans_seq = eef_trans_seq[seq_len_half:]

        left_eef_trans = np.mean(left_trans_seq, axis=0)
        right_eef_trans = np.mean(right_trans_seq, axis=0)

        left_eepose = trans2eepose(left_eef_trans)
        right_eepose = trans2eepose(right_eef_trans)

        draw_pose(left_eepose, length=0.1, **kwargs)
        draw_pose(right_eepose, length=0.1,  **kwargs)

        def solve_arm_jpose(side, eepose):
            ik_info = robot.ik_info[side]
            arm_group, _, tool_name = robot.manipulators[side]
            tool_link = link_from_name(robot, tool_name, **kwargs)
            ik_joints = get_ik_joints(robot, ik_info, tool_link, **kwargs)
            fixed_joints = set(ik_joints) - set(robot.get_group_joints(arm_group))
            extract_arm_conf = lambda q: [p for j, p in safe_zip(ik_joints, q) if j not in fixed_joints]
            ik_fn = get_ik_fn(ik_info, method=robot.ik_method, fixed_joints=fixed_joints)
            robot.reset()
            for full_conf in closest_inverse_kinematics(
                ik_fn,
                robot,
                ik_info,
                tool_link,
                eepose,
                max_candidates=INF,
                max_attempts=200,
                max_time=INF,
                max_distance=INF,
                verbose=False,
                **kwargs
            ):
                arm_conf = extract_arm_conf(full_conf)
                return np.array(arm_conf)
            return None

        arm1_jpose = solve_arm_jpose('left', left_eepose)
        arm2_jpose = solve_arm_jpose('right', right_eepose)
        return arm1_jpose, arm2_jpose



    def get_bimanual_jposes(arm1, arm2, sk, *obj_pose_pairs):
        """Generate bimanual keyposes and optionally post-grasp contact poses.

        Extra arguments are (obj, pose) pairs for each eff_hand_obj_edge.
        PDDLStream calls this with 0, 1, or 2 (obj, pose) pairs depending
        on the unified stream's input signature.
        """
        # is_fixed = 'per_skill' not in prefix_key
        is_fixed = True
        bicond_objs = get_bicond_objs(skillwise_sgs_flattend[sk], is_fixed=is_fixed)

        # Parse variadic (obj, pose) pairs
        objs_and_poses = []
        for i in range(0, len(obj_pose_pairs), 2):
            objs_and_poses.append((obj_pose_pairs[i], obj_pose_pairs[i + 1]))

        # Loop-invariant: resolve arm sides and eff_sg once.
        arm1_side = robot.side_from_arm(arm1)
        arm2_side = robot.side_from_arm(arm2)
        eff_sg = skillwise_sgs_flattend[sk].get('eff_sg')
        env_key_for_sk = skill_to_env.get(sk, prefix_key)
        current_skill_info = equivSkill_info_dict[env_key_for_sk]
        current_tamp_wrapper = current_skill_info['tamp_wrapper']
        current_biop_wrapper = current_skill_info.get('biop_wrapper', current_tamp_wrapper)
        current_task_name = current_skill_info['task_name']

        _seed_counter = 42
        while True:
            if len(bicond_objs) == 0:
                # #for aloha, output unconditional jpose.
                dual_jpose = current_biop_wrapper.gen_uncond_jposes(arm1, arm2, sk, seed=_seed_counter)
                arm1_jpose, arm2_jpose = robot.get_valid_dualpose(dual_jpose)

            else:
                ## for dexmimicgen, output jpose conditioned on all objects. e.g., bimanual lift.
                related_pc_dict = {obj_name:initial_pc_dict[obj_name] for obj_name in bicond_objs}

                obs_key = 'pc'
                eef_key = 'eefpos'
                group_actions = current_tamp_wrapper.gen_bimanual_kp(
                    related_pc_dict,
                    skill_name=policy_skill_name(sk),
                    task_name=current_task_name,
                    seed=_seed_counter,
                )

                arm1_jpose, arm2_jpose = decode_bimanual_kp(group_actions, eef_key)

            if len(arm1_jpose) > robot.arm_dof + 1 and len(arm2_jpose) > robot.arm_dof + 1: ## full jpose
                arm1_conf = GroupConf(robot, arm1_side + '_robot', positions=arm1_jpose)
                arm2_conf = GroupConf(robot, arm2_side + '_robot', positions=arm2_jpose)
            else:
                arm1_conf = GroupConf(robot, arm1, positions = arm1_jpose)
                arm2_conf = GroupConf(robot, arm2, positions = arm2_jpose)
            arm1_conf.assign()
            arm2_conf.assign()
            if fixed_obj:
                obstacle_bodies = [
                    o.body if hasattr(o, 'body') else o
                    for o in fixed_obj
                ]
                if pairwise_collisions(robot, obstacle_bodies, **kwargs):
                    print(
                        f"[get_bimanual_jposes] collision with fixed objects for skill {sk}, resampling"
                    )
                    _seed_counter += 1
                    continue
            graph_state = Graphstate(robot, skillwise_sgs_flattend[sk], skill_name = sk)

            # Compute post-grasp for each held object using Phi_tau.
            # _eff_grasps_mapping aligns 1:1 with the (obj, pose) stream args.
            grasps = []
            for idx, (obj, pose) in enumerate(objs_and_poses):
                schema_arm = _eff_grasps_mapping[idx][0]
                place_sk = _eff_grasps_mapping[idx][2]
                is_left = ("0" in schema_arm.lower() or "left" in schema_arm.lower())
                side = "left" if is_left else "right"
                arm_for_obj = arm1 if side == arm1_side else arm2

                env_key_for_sk = skill_to_env.get(place_sk, prefix_key)
                current_skill_info_for_place = equivSkill_info_dict[env_key_for_sk]
                current_tamp_wrapper = current_skill_info_for_place['tamp_wrapper']

                obs_key_g, agent_obs_g, _, _, _, _, _, _, _ = \
                    unimanual_traj_prep(arm_for_obj, obj, obj, pose, place_sk)

                group_actions_g = current_tamp_wrapper.gen_objcentric_traj(
                    obs_key_g,
                    agent_obs_g,
                    skill_name=resolve_policy_skill_name(place_sk, _wrapper_skill_keys(current_tamp_wrapper)),
                    task_name=current_skill_info_for_place['task_name'],
                    seed = 42
                )
                gtrans_objcentric = group_actions_g['grasp'].mean(axis=0)
                contact_pose_relative, _ = get_grasp_from_mat(gtrans_objcentric, obj)

                out_grasp = Grasp(obj, contact_pose_relative, phase='pre', robot_name=robot.name)
                # Write grasp to eff_sg for execution-time Graphstate
                if eff_sg is not None and eff_sg.has_edge(side, obj.category):
                    eff_sg.edges[side, obj.category]['grasp'] = out_grasp

                grasps.append(out_grasp)

            yield Tuple(arm1_conf, arm2_conf, graph_state, *grasps)

    def _build_arm_gripper_traj(arm_gripper_group, arm_path, gripper_jvals, out_ids, sk, pose):
        """Sync gripper joints to the arm waypoints and pack into a GroupTrajectory."""
        gripper_path = get_gripper_path(
            robot, gripper_jvals,
            arm_gripper_associate_ids=out_ids,
            skill_type=categorize_skill(sk),
        )
        arm_gripper_path = np.concatenate((arm_path, gripper_path), axis=1)
        return GroupTrajectory(
            robot, arm_gripper_group, arm_gripper_path,
            contexts=[pose], velocity_scale=0.25, client=robot.client,
        )

    def unimanual_traj_prep(arm, inv_obj, equiv_obj, pose, sk):
        side = robot.side_from_arm(arm)
        arm_group, gripper_group, tool_name = robot.manipulators[side]
        arm_gripper_group = side + '_robot'

        env_key_for_sk = skill_to_env.get(sk, prefix_key)
        current_tamp_wrapper = equivSkill_info_dict[env_key_for_sk]['tamp_wrapper']

        use_rgb = current_tamp_wrapper.cfg.data.dataset.get('use_pc_color', False)

        ## prefix_key is obtained in network_loader.py.  TODO: keep only per_skill, remve the old version
        if 'per_skill' in prefix_key:
            obs_key = 'pc'
            eef_key = 'grasp'
            gripper_key = 'gripper'
        else:
            obs_key = f'{sk}:pc' if is_dmg else f'{side}_pc'
            eef_key = f'{side}_grasp'
            gripper_key = f'{side}_gripper'

        obj_dict = {obs_key: equiv_obj} 
        agent_obs = get_object_obs(obj_dict, pose=pose, use_rgb=use_rgb)

        in_hand_obs = get_object_obs({f'in_hand_pc': inv_obj}, use_rgb = use_rgb)
        agent_obs.update(in_hand_obs)

        obj_aabb = aabb_from_points(agent_obs[obs_key])
        
        def hand_obj_dist_fn(hand_center):
            obj_center = get_aabb_center(obj_aabb)[:3]
            return np.linalg.norm(hand_center - obj_center)
        
        if robot.category == 'pandasinglerobot':
            ## rotate eef 180, as the panda_arm_hand has a tool_link that rotated 180
            offset_trans = np.array([[-1, 0, 0, 0],
                                    [0, -1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
        else:
            offset_trans = None

        return obs_key, agent_obs, eef_key, gripper_key, side, arm_gripper_group, tool_name, hand_obj_dist_fn, offset_trans
        
    def unimanual_traj_gen(obs_key, agent_obs, eef_key, gripper_key, side, arm_gripper_group,  hand_obj_dist_fn, pose, sk, equiv_obj, offset_trans = None):

        env_key_for_sk = skill_to_env.get(sk, prefix_key)
        current_skill_info = equivSkill_info_dict[env_key_for_sk]
        current_tamp_wrapper = current_skill_info['tamp_wrapper']
        learned_grasp_traj_resolution_deg = current_skill_info.get(
            'learned_grasp_traj_resolution_deg',
            1.2,
        )
        ee_traj_step_m          = float(current_skill_info.get('ee_traj_step_m',          0.01))
        ee_traj_step_rad        = float(current_skill_info.get('ee_traj_step_rad',        0.05))
        ee_traj_steps_per_wp    = int(current_skill_info.get('ee_traj_steps_per_waypoint') or 1)

        # Debug log to verify wrapper-skill consistency during execution
        print(f"[unimanual_traj_gen] skill={sk}, env_key={env_key_for_sk}, "
              f"wrapper_id={id(current_tamp_wrapper)}, "
              f"ee_traj_step_m={ee_traj_step_m}, ee_traj_step_rad={ee_traj_step_rad}, "
              f"ee_traj_steps_per_wp={ee_traj_steps_per_wp}")


        group_actions = current_tamp_wrapper.gen_objcentric_traj(
            obs_key,
            agent_obs,
            skill_name=resolve_policy_skill_name(sk, _wrapper_skill_keys(current_tamp_wrapper)),
            task_name=current_skill_info['task_name'],
            seed = 42,
        )

        shifted_eef_traj = np.array(group_actions[eef_key], copy=True)

        if ee_traj_mode:
            # --- EE-trajectory mode (OSC execution) ---
            sparse_mats = list(shifted_eef_traj)
            if offset_trans is not None:
                sparse_mats = [np.dot(gtrans, offset_trans) for gtrans in sparse_mats]
            sparse_poses = [trans2eepose(gtrans) for gtrans in sparse_mats]

            skill_type = categorize_skill(sk)

            # Bracket the learned grasp with an approach (start) + retreat (end)
            # pregrasp, mirroring joint-space get_jspace_path. ATTACH only.
            gripper_1d = group_actions[gripper_key].reshape(-1)
            n_pregrasp = 0
            if skill_type == 'ATTACH' and sparse_poses:
                obj_pose_w = pose.get_pose()
                pre = pregrasp_pose_from_waypoint(robot, equiv_obj, obj_pose_w, sparse_mats[0])
                post = pregrasp_pose_from_waypoint(robot, equiv_obj, obj_pose_w, sparse_mats[-1])
                sparse_poses = [pre] + sparse_poses + [post]
                # Pad the gripper schedule to match: open through approach, closed
                # through retreat. n_pregrasp counts the START prepend only.
                gripper_1d = np.concatenate([gripper_1d[:1], gripper_1d, gripper_1d[-1:]])
                n_pregrasp = 1

            # Densify in Cartesian space (replaces joint-space interpolation)
            ee_dense, out_ids_list = densify_ee_path(
                sparse_poses, step_m=ee_traj_step_m, step_rad=ee_traj_step_rad
            )
            if not ee_dense:
                print("No valid EE path found in Cartesian densification")
                return None, None, None, None
            out_ids = np.array(out_ids_list)

            # Grasp-contact waypoint for the attach transform (trans_w). For ATTACH,
            # use the waypoint closest to the object, not the lowest-z one (lowest-z
            # can be an over-deep approach dip that locks in a too-low grasp). Place/
            # release keep lowest-z (they reuse trans_w for pose reconstruction).
            if skill_type == 'ATTACH':
                switch_id_dense = int(np.argmin(
                    [hand_obj_dist_fn(np.asarray(p[0], dtype=float)) for p in ee_dense]
                ))
            else:
                switch_id_dense = int(np.argmin([p[0][2] for p in ee_dense]))
            # out_ids maps a dense pose -> augmented sparse index + 1; subtract the
            # prepended pregrasp (n_pregrasp) to land back on the demo waypoint index.
            sparse_switch_id = int(out_ids_list[switch_id_dense]) - 1 - n_pregrasp
            sparse_switch_id = min(max(sparse_switch_id, 0), len(shifted_eef_traj) - 1)
            trans_w = np.array(shifted_eef_traj[sparse_switch_id], copy=True)

            # Endpoint IK only — required for schema executor's _aq_start/_aq_end
            arm_group, _, tool_name = robot.manipulators[side]
            tool_link = link_from_name(robot, tool_name, **kwargs)
            ik_info = robot.ik_info[side]
            ik_joints = get_ik_joints(robot, ik_info, tool_link, **kwargs)
            fixed_joints = set(ik_joints) - set(robot.get_group_joints(arm_group))
            extract_arm_conf = lambda q: [
                p for j, p in safe_zip(ik_joints, q) if j not in fixed_joints
            ]
            ik_fn = get_ik_fn(ik_info, method=robot.ik_method, fixed_joints=fixed_joints)

            def _solve_endpoint_ik(eepose):
                robot_saver.restore()
                for full_conf in closest_inverse_kinematics(
                    ik_fn, robot, ik_info, tool_link, eepose,
                    max_candidates=INF, max_attempts=200, max_time=INF,
                    max_distance=INF, verbose=False, **kwargs,
                ):
                    return np.asarray(extract_arm_conf(full_conf), dtype=float)
                return None

            conf_start_arm = _solve_endpoint_ik(ee_dense[0])
            conf_end_arm = _solve_endpoint_ik(ee_dense[-1])
            if conf_start_arm is None or conf_end_arm is None:
                print("No valid endpoint IK found for EE trajectory — cannot build LearnedGrasp Conf")
                return None, None, None, None

            # Gripper path aligned to densified ee path indices (gripper_1d carries
            # the prepended open grip for the approach when a pregrasp was added).
            gripper_path = get_gripper_path(
                robot,
                gripper_1d,
                arm_gripper_associate_ids=out_ids,
                skill_type=skill_type,
            )

            # .path uses endpoint joint confs for symbolic/bridging compatibility
            # (sparse placeholder — NOT used for OSC execution, which reads ee_path)
            n = len(ee_dense)
            arm_placeholder = np.linspace(conf_start_arm, conf_end_arm, num=n)
            arm_gripper_path = np.concatenate((arm_placeholder, gripper_path), axis=1)
            arm_gripper_traj = GroupTrajectory(
                robot,
                arm_gripper_group,
                arm_gripper_path,
                contexts=[pose],
                velocity_scale=0.25,
                client=robot.client,
                ee_path=ee_dense,
                ee_link=tool_name,
                steps_per_waypoint=ee_traj_steps_per_wp,
            )
        else:
            # --- Joint-space mode (original behavior) ---
            arm_path_and_out_ids, switch_id = get_jspace_path(
                robot, pose, shifted_eef_traj, side, hand_obj_dist_fn,
                offset_trans=offset_trans, obj=equiv_obj,
                eef_key=eef_key, obstacles=fixed_obj,
                waypoint_resolution_deg=learned_grasp_traj_resolution_deg,
                return_ids=True,
            )
            if arm_path_and_out_ids is None:
                print("No valid path found in joint space")
                return None, None, None, None

            trans_w = np.array(shifted_eef_traj[switch_id-1], copy=True)  ## if add a wp at begining, need to -1

            arm_path, out_ids = arm_path_and_out_ids
            arm_path = np.array(arm_path)
            out_ids = np.array(out_ids)

            arm_gripper_traj = _build_arm_gripper_traj(
                arm_gripper_group, arm_path,
                group_actions[gripper_key].reshape(-1), out_ids, sk, pose,
            )
        conf_start = arm_gripper_traj.first()
        conf_end = arm_gripper_traj.last()
        return trans_w, arm_gripper_traj, conf_start, conf_end

    def _bounded_learned_trajs(label, sample_fn):
        """Yield successful ``(trans_w, arm_gripper_traj, conf_start, conf_end)``
        tuples from ``sample_fn`` (a ``unimanual_traj_gen`` call), retrying when it
        returns a None trajectory. Exhausts via StopIteration after
        ``learned_traj_max_attempts`` CONSECUTIVE failures (the streak resets on every
        success) so a degenerate scene — endpoint IK never solvable — cannot spin
        uninterruptibly inside one ``next()`` call, which would defeat PDDLStream's
        ``max_tamp_time`` (checked only between stream calls). See the
        ``learned_traj_max_attempts`` comment above."""
        fail_streak = 0
        while True:
            robot_saver.restore()
            result = sample_fn()
            if result[1] is None:  # arm_gripper_traj
                fail_streak += 1
                if fail_streak >= learned_traj_max_attempts:
                    print(f"[{label}] no feasible learned trajectory after "
                          f"{fail_streak} attempts; exhausting stream")
                    robot_saver.restore()
                    return
                continue
            fail_streak = 0
            yield result

    def gen_attach_traj(arm, obj, pose, sk):
        obs_key, agent_obs, eef_key, gripper_key, side, arm_gripper_group, tool_name, hand_obj_dist_fn, offset_trans = unimanual_traj_prep(arm, obj, obj, pose, sk)

        if 'grasp' not in sk:
            raise ValueError(f"Skill {sk} is not an attach skill!")

        for trans_w, arm_gripper_traj, conf_start, conf_end in _bounded_learned_trajs(
                f"gen_attach_traj skill={sk} arm={arm} obj={obj}",
                lambda: unimanual_traj_gen(obs_key, agent_obs, eef_key, gripper_key, side, arm_gripper_group, hand_obj_dist_fn, pose, sk, obj, offset_trans)):

            contact_pose_relative = get_grasp_from_trans_w(trans_w, pose, offset_trans=offset_trans)

            parent_body = get_parent_body(categorize_skill(sk), robot, tool_name)
            switch = Switch(obj, parent=parent_body)

            commands = [arm_gripper_traj, switch]
            seq = Sequence(commands=commands, name="{}-{}-{}".format(sk, side, obj))

            learned_grasp = LearnedGrasp(
                obj,
                contact_pose_relative,
                aq_start=conf_start,
                aq_end=conf_end,
                traj_seq=seq,
                phase='pre',
                client=robot.client,
                robot_name=robot.name,
            )
            yield Tuple(learned_grasp,)


    def _assemble_scene_xyz():
        """Concatenate every world-frame cloud that built the pybullet scene."""
        clouds = []
        for cloud in (initial_pc_dict or {}).values():
            if cloud is None:
                continue
            clouds.append(np.asarray(cloud)[:, :3])
        if not clouds:
            raise ValueError(
                "grasp_backend='m2t2' requires initial_pc_dict scene points"
            )
        return np.concatenate(clouds, axis=0)

    # Constant rotations that map a predictor's grasp frame onto the robosuite/LIBERO
    # panda EE convention get_jspace_path/IK targets: approach along local +Z, fingers
    # close along local +Y. Right-multiplying a world grasp pose by one of these
    # leaves the translation untouched and lands the approach on the +Z column (the
    # axis _apply_ee_correction shifts along). offset_trans (180 deg about z) preserves
    # that column downstream.
    #
    # M2T2: contact-graspnet convention (fingers separate along local X, approach +Z);
    # a +90 deg roll about Z maps its closing axis X -> robot Y (fixes the "twisted
    # pi/2" symptom) and keeps approach on +Z.
    _M2T2_TO_EE_ROLL = np.array([
        [0.0, -1.0, 0.0, 0.0],
        [1.0,  0.0, 0.0, 0.0],
        [0.0,  0.0, 1.0, 0.0],
        [0.0,  0.0, 0.0, 1.0],
    ])
    # GPD: aloha tool convention (approach +X, close +Y binormal, hand axis +Z) -- the
    # same poses gen_fn/get_grasp_gen_fn feeds raw to the *aloha* ee_gripper_link (also
    # +X approach, hence no remap there). For the panda tool_link a 90 deg rotation
    # about Y swings GPD's +X approach onto +Z while leaving the +Y closing axis fixed:
    #   panda +Z <- GPD +X (approach)   panda +Y <- GPD +Y (closing)   panda -X <- GPD +Z
    _GPD_TO_EE = np.array([
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    def _grasp_calibration(sk):
        """Per-skill grasp-trajectory waypoint resolution for the learned-grasp streams."""
        info = equivSkill_info_dict[skill_to_env.get(sk, prefix_key)]
        return info.get('learned_grasp_traj_resolution_deg', 1.2)

    def _apply_ee_correction(world_from_pred, ee_correction, grasp_depth):
        """Rotate a predicted world grasp onto the panda EE convention and advance
        along its approach axis (now local +Z) so the TCP reaches the contact."""
        grasp_w = world_from_pred @ ee_correction
        grasp_w[:3, 3] += grasp_depth * grasp_w[:3, 2]
        return grasp_w

    def _emit_learned_grasp(grasp_w, obj, pose, sk, prep, calib):
        """EE-frame world grasp -> LearnedGrasp, or None if IK fails. Duplicates the
        grasp pose so get_jspace_path inserts approach/retreat standoffs; the gripper
        stays fully open through the descent and closes during a stationary dwell at
        the grasp pose (the anchor-interpolated schedule of get_gripper_path starts
        closing mid-descent, which clips the object before the fingers reach it)."""
        (_obs_key, _agent_obs, eef_key, _gripper_key, side, arm_gripper_group,
         tool_name, hand_obj_dist_fn, offset_trans) = prep
        resolution_deg = calib

        arm_path_and_out_ids, _switch_id = get_jspace_path(
            robot, pose, [grasp_w, grasp_w], side, hand_obj_dist_fn,
            offset_trans=offset_trans, obj=obj,
            eef_key=eef_key, obstacles=fixed_obj,
            waypoint_resolution_deg=resolution_deg,
            return_ids=True,
        )
        if arm_path_and_out_ids is None:
            return None
        arm_path, out_ids = arm_path_and_out_ids

        arm_path = np.array(arm_path)
        out_ids = np.array(out_ids)
        # Dense steps targeting the final (retreat pregrasp) waypoint form the
        # retreat; everything before it is descent + arrival at the grasp pose.
        retreat_mask = out_ids == out_ids.max()
        retreat_start = max(int(np.argmax(retreat_mask)), 1) if retreat_mask.any() else len(arm_path)
        approach_arm = arm_path[:retreat_start]
        retreat_arm = arm_path[retreat_start:]
        dwell_arm = np.repeat(approach_arm[-1:], grasp_close_dwell_steps, axis=0)

        open_row = robot.joint2pos_gripper(robot.max_finger_joint)
        closed_row = robot.joint2pos_gripper(robot.min_finger_joint)
        # Command full close from the first dwell step and hold: the close target
        # must lead the physical fingers, which need the whole dwell to seat
        # before the retreat starts (a ramped command only reaches "closed" on
        # the last dwell step, so the fingers finish closing mid-retreat).
        gripper_rows = np.array(
            [open_row] * len(approach_arm)
            + [closed_row] * (grasp_close_dwell_steps + len(retreat_arm))
        )

        arm_gripper_path = np.concatenate(
            (np.concatenate((approach_arm, dwell_arm, retreat_arm), axis=0), gripper_rows),
            axis=1,
        )
        arm_gripper_traj = GroupTrajectory(
            robot, arm_gripper_group, arm_gripper_path,
            contexts=[pose], velocity_scale=0.25, client=robot.client,
        )
        contact_pose_relative = get_grasp_from_trans_w(
            grasp_w, pose, offset_trans=offset_trans
        )
        parent_body = get_parent_body(categorize_skill(sk), robot, tool_name)
        seq = Sequence(
            commands=[arm_gripper_traj, Switch(obj, parent=parent_body)],
            name="{}-{}-{}".format(sk, side, obj),
        )
        return LearnedGrasp(
            obj,
            contact_pose_relative,
            aq_start=arm_gripper_traj.first(),
            aq_end=arm_gripper_traj.last(),
            traj_seq=seq,
            phase='pre',
            client=robot.client,
            robot_name=robot.name,
        )

    def gen_m2t2_attach_traj(arm, obj, pose, sk):
        from scipy.spatial import cKDTree

        if 'grasp' not in sk:
            raise ValueError(f"Skill {sk} is not an attach skill!")
        if m2t2_grasp_wrapper is None:
            raise ValueError(
                "grasp_backend='m2t2' but no 'm2t2_grasp_wrapper' was provided "
                "in tamp_kwargs"
            )

        prep = unimanual_traj_prep(arm, obj, obj, pose, sk)
        obs_key, agent_obs = prep[0], prep[1]
        calib = _grasp_calibration(sk)

        # The same world-frame cloud that built pybullet gives M2T2 its context.
        scene_xyz = _assemble_scene_xyz()
        # Per-object segmented cloud at the queried pose (from unimanual_traj_prep);
        # used to keep only grasps whose contact lands on this target object.
        target_kdtree = cKDTree(np.asarray(agent_obs[obs_key])[:, :3])

        for _batch in range(m2t2_max_prediction_batches):
            robot_saver.restore()

            candidates = m2t2_grasp_wrapper.predict(scene_xyz)
            target_grasps = []
            if candidates:
                contacts = np.array([c['contact'] for c in candidates])
                dists, _ = target_kdtree.query(contacts)
                target_grasps = [
                    cand for cand, dist in zip(candidates, dists)
                    if dist <= m2t2_contact_radius
                ]

            if not target_grasps:
                print(f"[gen_m2t2_attach_traj] no M2T2 grasp on target {obj}; resampling")
                continue

            produced_any = False
            for cand in target_grasps:
                robot_saver.restore()
                grasp_w = _apply_ee_correction(
                    cand['pose'].copy(), _M2T2_TO_EE_ROLL, m2t2_grasp_depth
                )
                learned_grasp = _emit_learned_grasp(grasp_w, obj, pose, sk, prep, calib)
                if learned_grasp is None:
                    continue
                produced_any = True
                yield Tuple(learned_grasp,)

            if not produced_any:
                print(
                    f"[gen_m2t2_attach_traj] all {len(target_grasps)} M2T2 grasps "
                    f"failed IK for {obj}; resampling"
                )

        robot_saver.restore()
        print(
            f"[gen_m2t2_attach_traj] exhausted {m2t2_max_prediction_batches} "
            f"M2T2 prediction batches for {obj}; no graspable candidate found"
        )
        return


    def gen_gpd_attach_traj(arm, obj, pose, sk):
        if 'grasp' not in sk:
            raise ValueError(f"Skill {sk} is not an attach skill!")
        if gpd_camera_point is None:
            raise ValueError(
                "grasp_backend='gpd' but no 'gpd_camera_point' was provided "
                "in tamp_kwargs"
            )

        prep = unimanual_traj_prep(arm, obj, obj, pose, sk)
        obs_key, agent_obs = prep[0], prep[1]
        calib = _grasp_calibration(sk)

        # World-frame cloud of the target object (from unimanual_traj_prep). GPD
        # detects grasps directly on this object cloud, so no scene assembly or
        # contact filtering is needed (unlike M2T2). The camera point only orients
        # GPD's approach directions.
        target_pts_world = np.asarray(agent_obs[obs_key])[:, :3]
        camera_pose = Pose(point=Point(*gpd_camera_point))

        robot_saver.restore()
        grasps_world, _scores = gpd_predict_grasps(
            target_pts_world, camera_pose, use_tool=True
        )  # world_from_tool, sorted by score desc
        if len(grasps_world) == 0:
            print(f"[gen_gpd_attach_traj] GPD returned no grasp for {obj}")
            return

        produced_any = False
        for grasp in grasps_world[:gpd_max_candidates]:
            robot_saver.restore()
            grasp_w = _apply_ee_correction(
                tform_from_pose(grasp), _GPD_TO_EE, gpd_grasp_depth
            )
            # Calibration aid: visualize the EE-frame grasp before IK so
            # gpd_grasp_depth can be tuned (see plan verification step 3).
            draw_pose(pose_from_tform(grasp_w), length=0.1, **kwargs)

            learned_grasp = _emit_learned_grasp(grasp_w, obj, pose, sk, prep, calib)
            if learned_grasp is None:
                continue
            produced_any = True
            yield Tuple(learned_grasp,)

        robot_saver.restore()
        if not produced_any:
            print(
                f"[gen_gpd_attach_traj] all {len(grasps_world[:gpd_max_candidates])} "
                f"GPD grasps failed IK for {obj}; no graspable candidate found"
            )
        return


    def gen_detach_traj(arm, inv_obj, equiv_obj, pose, sk, inhand_grasp):
        obs_key, agent_obs, eef_key, gripper_key, side, arm_gripper_group, tool_name, hand_obj_dist_fn, offset_trans = unimanual_traj_prep(arm, inv_obj, equiv_obj, pose, sk)

        for trans_w, arm_gripper_traj, conf_start, conf_end in _bounded_learned_trajs(
                f"gen_detach_traj skill={sk} arm={arm} obj={equiv_obj}",
                lambda: unimanual_traj_gen(obs_key, agent_obs, eef_key, gripper_key, side, arm_gripper_group, hand_obj_dist_fn, pose, sk, equiv_obj, offset_trans)):

            place_pose_w = get_place_pose_from_trans_w(trans_w, \
                    inhand_grasp.grasp, offset_trans=offset_trans)
            set_pose(inv_obj, place_pose_w)
            place_pose_out = RelativePose(
                inv_obj,
                parent=ParentBody(equiv_obj, **kwargs),
                parent_state=pose,
                **kwargs
            )
            parent_body= get_parent_body(categorize_skill(sk), robot, tool_name)
            switch = Switch(equiv_obj, parent=parent_body)

            commands = [arm_gripper_traj, switch]
            seq = Sequence(commands=commands, name="{}-{}-{}".format(sk, side, equiv_obj))

            learned_place_payload = LearnedGrasp(
                equiv_obj,
                inhand_grasp.grasp,
                aq_start=conf_start,
                aq_end=conf_end,
                traj_seq=seq,
                phase='post',
                client=robot.client,
                robot_name=robot.name,
            )
            yield Tuple(place_pose_out, learned_place_payload)
            ##  'push'  'insertion' 'open/close drawer'

    # Generic DETACH backend: geometric placement on the goal surface instead of
    # the diffusion policy. Factories are built at stream-binding time, matching
    # how create_hybrid_streams binds sample-placement / plan-place.
    # buffer=0 / percent=1.0: keep the placed object's AABB fully inside the
    # surface AABB (no rim overhang on small surfaces like plates).
    _placement_kwargs = {
        k: v for k, v in kwargs.items()
        if k not in ('buffer', 'percent')
    }
    generic_placement_gen = get_placement_gen_fn(
        robot, fixed_obj, environment=fixed_obj,
        buffer=0.0, percent=1.0, **_placement_kwargs
    )
    generic_plan_place = get_plan_place_fn(robot, environment=fixed_obj, **kwargs)

    def gen_generic_detach_traj(arm, obj, surface, surface_pose, sk, inhand_grasp):
        if categorize_skill(sk) != 'DETACH':
            raise ValueError(f"Skill {sk} is not a detach skill!")

        robot_saver.restore()
        # The placement sampler reads the surface OOBB from the current pybullet
        # state (its own surface_pose.assign() is commented out), so sync it.
        surface_pose.assign()

        attempts = 0
        for (rel_pose,) in islice(
            generic_placement_gen(obj, surface, surface_pose),
            generic_place_max_attempts,
        ):
            attempts += 1
            robot_saver.restore()
            output = generic_plan_place(arm, obj, rel_pose, inhand_grasp)
            if output is None:
                continue
            arm_conf, seq = output
            # The place sequence starts and ends at the same arm conf, so the
            # payload's endpoint confs coincide (same as the coarse generic
            # place path in schema_executor._execute_generic_place).
            learned_place_payload = LearnedGrasp(
                obj,
                inhand_grasp.grasp,
                aq_start=arm_conf,
                aq_end=arm_conf,
                traj_seq=seq,
                phase='post',
                client=robot.client,
                robot_name=robot.name,
            )
            yield Tuple(rel_pose, learned_place_payload)

        robot_saver.restore()
        print(
            f"[gen_generic_detach_traj] exhausted {attempts} placement "
            f"attempts for {obj} on {surface}; no feasible generic place found"
        )
        return

    def gen_nonprehensile_traj(arm, inv_obj, equiv_obj, pose, sk, inhand_grasp = None):
        obs_key, agent_obs, eef_key, gripper_key, side, arm_gripper_group, tool_name, hand_obj_dist_fn, offset_trans = unimanual_traj_prep(arm, inv_obj, equiv_obj, pose, sk)

        for trans_w, arm_gripper_traj, conf_start, conf_end in _bounded_learned_trajs(
                f"gen_nonprehensile_traj skill={sk} arm={arm} obj={equiv_obj}",
                lambda: unimanual_traj_gen(obs_key, agent_obs, eef_key, gripper_key, side, arm_gripper_group, hand_obj_dist_fn, pose, sk, equiv_obj, offset_trans)):

            commands = [arm_gripper_traj]
            seq = Sequence(commands=commands, name="{}-{}-{}".format(sk, robot.side_from_arm(arm), equiv_obj))
            yield Tuple(conf_start, conf_end, seq)

    def gen_unimanual_grasp(arm, obj, pose, sk):
        # side = robot.side_from_arm(arm)
        # # pose.assign()
        # obs_key = sk if is_dmg else side
        # obj_dict = {f'{sk}:pc': obj} if is_dmg else {side+'_pc': obj}
        # agent_obs = get_agent_obs(obj_dict)  

        obs_key, agent_obs, eef_key, gripper_key, side, arm_gripper_group, tool_name, hand_obj_dist_fn, offset_trans = unimanual_traj_prep(arm, obj, obj, pose, sk)

        while True:
            robot_saver.restore()

            env_key_for_sk = skill_to_env.get(sk, prefix_key)
            current_tamp_wrapper = equivSkill_info_dict[env_key_for_sk]['tamp_wrapper']

            print(f"[gen_unimanual_grasp] skill={sk}, env_key={env_key_for_sk}, "
                  f"wrapper_id={id(current_tamp_wrapper)}")

            group_actions = current_tamp_wrapper.gen_objcentric_traj(
                obs_key, agent_obs, skill_name=sk, task_name=task_name, seed=42
            )
            
            gtrans_objcentric = group_actions['grasp'].mean(axis=0)

            contact_pose_relative, _ = get_grasp_from_mat(gtrans_objcentric, obj)

            # pred_grasp_ok = is_gripper_free(robot, obj, side, contact_pose_relative, obstacles = fixed_obj, **kwargs)     

            out_grasp = Grasp(obj, contact_pose_relative, phase = 'pre', robot_name = robot.name)
            # Write grasp onto bimanual skill's eff_sg edge for lazy Graphstate
            bi_eff_sg = skillwise_sgs_flattend[sk]['pre_sg']  ## note the eff_sg of biop is the pre_sg of the place
            if bi_eff_sg is not None and bi_eff_sg.has_edge(side, obj.category):
                bi_eff_sg.edges[side, obj.category]['grasp'] = out_grasp
            else:
                print(f"No bimanual skill found for {sk} or {side} -> {obj.category} edge")
            yield Tuple(out_grasp)

    # exception_case = 'handoff' in task_name and 'place' in skill_name

    initial_skill_type = categorize_skill(skill_name)
    if "bimanual" in initial_skill_type:
        return get_bimanual_jposes
    # elif obj_centric_mode == 'grasp' or exception_case:
    #     return gen_unimanual_grasp
    elif initial_skill_type == 'ATTACH':
        if grasp_backend == 'gpd':
            return gen_gpd_attach_traj
        if grasp_backend == 'm2t2':
            return gen_m2t2_attach_traj
        return gen_attach_traj
    elif initial_skill_type == 'DETACH':
        if place_backend == 'generic':
            return gen_generic_detach_traj
        return gen_detach_traj
    else:
        return gen_nonprehensile_traj


def get_test_cfree_pose_pose(obj_obj_collisions=True, **kwargs):
    def test_cfree_pose_pose(obj1, pose1, obj2, pose2):
        if (obj1 == obj2): # or (pose2 is None): # TODO: skip if in the environment
           return True
        if obj2 in pose1.ancestors():
            return True
        pose1.assign()
        pose2.assign()
        return not pairwise_collision(obj1, obj2, max_distance=MOVABLE_DISTANCE)
    return test_cfree_pose_pose

def get_cfree_pregrasp_pose_test(robot, **kwargs):

    def test(arm, obj1, pose1, grasp1, obj2, pose2):
        side = robot.side_from_arm(arm)
        if (obj1 == obj2): # or (pose2 is None):
            return True
        if obj2 in pose1.ancestors():
            return True
        # if (pose1.important and not obj1.is_fragile) and (pose2.important and not obj2.is_fragile):
        #     return True
        pose2.assign()
        gripper_path = compute_gripper_path(pose1, grasp1)
        grasp = None if (pose1.important and pose2.important) else grasp1
        return not workspace_collision(robot, side, gripper_path, grasp, obstacles=[obj2], max_distance=MOVABLE_DISTANCE)
    return test

def get_cfree_traj_pose_test(robot, **kwargs):
    def test(arm, sequence, obj2, pose2):
        if (pose2 is None): 
            return True

        grasp_kw_list = ['grasp', 'pick', 'place']
        if find_kw_in_skill(sequence.name, grasp_kw_list) is not None:
        # if sequence.name.startswith('pick'):
            return True
        if obj2 in sequence.context_bodies:
            return True
        pose2.assign()
        # set_open_positions(robot, arm)
        robot.set_open_gripper(arm)
        #state = State() # TODO: apply to the context

        for traj in sequence.commands:
            if not isinstance(traj, GroupTrajectory):
                continue
            if obj2 in traj.context_bodies: # TODO: check the grasp
                continue
            moving_links = get_moving_links(traj.robot, traj.joints)
            #for _ in command.iterate(state=None):
            for _ in traj.traverse():
                #wait_if_gui()
                if any_link_pair_collision(traj.robot, moving_links, obj2, max_distance=MOVABLE_DISTANCE):# \
                    #or any_link_pair_collision(traj.robot, moving_links, other_target_link, max_distance=MOVABLE_DISTANCE):
                    return False
        return True
    return test


#######################################################



def compute_stable_poses(obj, weight=0.5, min_prob=0.0, min_area=None):
    # TODO: filter similar orientations (if only a change in yaw)
    # from trimesh.path.packing import paths, polygons, rectangles
    default_pose = Pose()
    yield default_pose, weight
    if weight >= 1:
        return
    history = [default_pose]
    obj_trimesh = trimesh_from_body(obj)
    set_pose(obj, default_pose)

    pose_mats, poses_prob = obj_trimesh.compute_stable_poses(
        center_mass=None, sigma=0.0, n_samples=1, threshold=min_prob
    )
    # pose_mats = pose_mats[poses_prob>.1]
    for pose_mat, pose_score in list(zip(pose_mats, poses_prob)):  # reversed
        area = np.nan
        if min_area is not None:
            new_trimesh = obj_trimesh.copy().apply_transform(
                pose_mat
            )  # apply_transform modifies the input mesh
            # print(new_trimesh.bounds)
            surfaces = slice_mesh(new_trimesh, plane=z_plane(z=1e-2))
            if not surfaces:
                continue
            surface = surfaces[0]
            area = 0.0 if surface is None else convex_area(surface.vertices)
            if area <= min_area:
                continue

        print(
            "Num: {} | Prob: {:.3f} | Area: {:.3f}".format(
                len(history), pose_score, area
            )
        )
        top_pose = pose_from_tform(pose_mat)
        set_pose(obj, top_pose)
        offset_center1, offset_extent1 = get_center_extent(obj)
        offset_center2 = get_point(obj)
        dz = (offset_center1[2] - offset_extent1[2] / 2) - offset_center2[2]
        top_pose = (top_pose[0] + (0, 0, dz), top_pose[1])
        set_pose(obj, top_pose)

        # print(len(history), euler_from_quat(top_pose[1]), pose_score, scores)
        yield top_pose, (1 - weight) * pose_score
        history.append(top_pose)
        # TODO: compare orientation similarity in order to prune similar

### output the pose in the object frame
def generate_stable_poses(obj, deterministic=False, **kwargs):
    # TODO: place on the most stable face
    # TODO: placement that maximizes height
    # TODO: place cost dependent on the quality of the placement
    start_time = time.time()
    weight = 0.0 if REORIENT else 1.0
    # weight = 0.5 if REORIENT else 1.
    poses, scores = zip(*compute_stable_poses(obj, weight=weight, **kwargs))
    print(
        "Poses: {} | Scores: {} | Time: {:.3f}".format(
            len(poses), np.round(scores, 3).tolist(), elapsed_time(start_time)
        )
    )
    if deterministic:
        generator = cycle(iter(poses))
    else:
        # TODO: unweighted version of this if above a threshold
        generator = (
            random.choices(poses, weights=scores, k=1)[0] for _ in inf_generator()
        )  # TODO: python2
    return generator


#######################################################


def get_placement_gen_fn(
    robot,
    other_obstacles,
    environment=[],
    buffer=2e-2,
    z_epsilon = 1e-2,
    max_distance=INF,
    max_attempts=10,
    percent=0.1,
    **kwargs
):  # max_distance=PR2_WINGSPAN
    base_pose = get_link_pose(robot, robot.base_link)
    robot_saver = BodySaver(robot, client=robot.client)

    def gen_fn(obj, surface, surface_pose):
        # surface_pose.assign()
        surface_oobb = surface.get_shape_oobb()  # TODO: change to as long as the COM is on
        # draw_oobb(surface_oobb)
        obstacles = set(environment) - {obj, surface}  # TODO: surface might have walls

        aabb = surface_oobb.aabb
        aabb = buffer_aabb(aabb, buffer)
        # aabb = aabb_from_extent_center(
        #     2 * buffer * np.array([1, 1, 0]) + get_aabb_extent(aabb),
        #     get_aabb_center(aabb),
        # )
        for top_pose in generate_stable_poses(obj):  # cycle
            pose = sample_placement_on_aabb(
                obj,
                aabb,
                max_attempts=max_attempts,
                top_pose=top_pose,  # TODO: reference pose instead?
                percent=percent,
                epsilon=z_epsilon, #1e-3
                **kwargs
            )  # TODO: Z_EPSILON
            if pose is None:
                # yield None
                continue
            pose = multiply(surface_oobb.pose, pose)
            set_pose(obj, pose, **kwargs)
            rel_pose = RelativePose(
                obj,
                parent=ParentBody(surface, **kwargs),
                parent_state=surface_pose,
                **kwargs
            )  # , relative_pose=pose)
            base_distance = get_length(
                point_from_pose(multiply(invert(base_pose), rel_pose.get_pose()))[:2]
            )
            if (surface in other_obstacles) and (base_distance > max_distance):
                continue
            if pairwise_collisions(
                obj, obstacles - set(rel_pose.ancestors()), max_distance=0.0, **kwargs
            ):
                # TODO: max_attempts here as well
                continue
            robot_saver.restore()
            yield Tuple(rel_pose)
        else:
            print("No stable placements found for {}".format(obj))
        # yield None

    return gen_fn


def get_plan_drop_fn(robot, environment=[], z_offset=2e-2, shrink=0.25, **kwargs):
    robot_saver = BodySaver(robot, client=robot.client)

    def fn(arm, obj, grasp, bin, bin_pose, base_conf = None):
        # TODO: don't necessarily need the grasp
        robot_saver.restore()
        if base_conf is not None:
            base_conf.assign()

        bin_pose.assign()
        obstacles = list(environment)

        side = robot.side_from_arm(arm)
        _, gripper_group, _ = robot.manipulators[side]
        gripper = robot.get_component(gripper_group)
        parent_from_tool = robot.get_parent_from_tool(side)

        bin_aabb = get_aabb(bin)
        # _, (_, _, z) = bin_aabb
        # x, y, _ = get_aabb_center(bin_aabb)
        # gripper_pose = Pose(point=Point(x, y, z + 0.1), euler=DOWNWARD_EULER)

        # reference_pose = unit_pose()
        reference_pose = multiply(
            Pose(euler=Euler(pitch=PI / 2, yaw=random.uniform(0, 2 * PI))), grasp.value
        )
        # obj_pose = sample_placement_on_aabb(obj, bin_aabb, top_pose=reference_pose, percent=shrink, epsilon=1e-2)
        # _, extent = approximate_as_prism(obj, reference_pose=reference_pose)
        with PoseSaver(obj):
            set_pose(obj, reference_pose)
            obj_pose = (
                np.append(
                    get_aabb_center(bin_aabb)[:2],
                    [stable_z_on_aabb(obj, bin_aabb) + z_offset],
                ),
                quat_from_pose(reference_pose),
            )  # TODO: get_aabb_top, get_aabb_bottom

        if obj_pose is None:
            return None
        gripper_pose = multiply(obj_pose, invert(grasp.value))
        set_pose(gripper, multiply(gripper_pose, invert(parent_from_tool)))
        set_pose(obj, multiply(gripper_pose, grasp.value))
        if any(
            pairwise_collisions(body, environment, max_distance=0.0)
            for body in [obj, gripper]
        ):
            return None

        _, _, tool_name = robot.manipulators[robot.side_from_arm(arm)]
        attachment = grasp.create_attachment(
            robot, link=robot.link_from_name(tool_name)
        )

        arm_path = plan_workspace_motion(
            robot, side, [gripper_pose], attachment=attachment, obstacles=obstacles
        )
        if arm_path is None:
            return None
        arm_conf = GroupConf(robot, arm, positions=arm_path[0], **kwargs)
        switch = Switch(obj, parent=WORLD_BODY)

        closed_conf, open_conf = robot.close_open_conf()
        # gripper_joints = robot.get_group_joints(gripper_group)
        # closed_conf = grasp.closed_position * np.ones(len(gripper_joints))
        gripper_traj = GroupTrajectory(
            robot,
            gripper_group,
            path=[closed_conf, open_conf],
            contexts=[],
            client=robot.client,
        )

        commands = [switch, gripper_traj]
        sequence = Sequence(
            commands=commands, name="drop-{}-{}".format(robot.side_from_arm(arm), obj)
        )
        return Tuple(arm_conf, sequence)

    return fn

def get_pose_cost_fn(robot, cost_per_m=1.0, **kwargs):
    # TODO(caelan): refactor
    base_pose = get_link_pose(robot, robot.base_link, **kwargs)

    def cost_fn(obj, pose):
        cost = BASE_COST
        if PROXIMITY_COST_TERM:  # Closest is least costly
            point_base = tform_point(
                invert(base_pose), point_from_pose(pose.get_pose())
            )
            distance = get_length(point_base[:2])
            cost += cost_per_m * distance
        return cost

    return cost_fn




def get_plan_pick_fn(robot,   **kwargs):
    robot_saver = BodySaver(robot, client=robot.client)
    # environment = environment
    
    def fn(arm, obj, pose, grasp, base_conf = None):

        
        robot_saver.restore()
        pose.assign()
        if base_conf is not None:
            base_conf.assign()


        arm_path = plan_prehensile(robot, arm, obj, pose, grasp, **kwargs)
        
        ## TODO: lift grasp z and retry
        if arm_path is None:
            # arm_path = plan_prehensile(robot, arm, obj, pose, grasp, **kwargs)
            return None
        
        arm_group, gripper_group, tool_name = robot.manipulators[
            robot.side_from_arm(arm)
        ]
        arm_traj = GroupTrajectory(
            robot,
            arm_group,
            arm_path[::-1],
            context=[pose],
            velocity_scale=0.25,
            client=robot.client,
        )
        arm_conf = arm_traj.first()

        # closed_conf = grasp.closed_position * np.ones(
        #     len(robot.get_group_joints(gripper_group))
        # )
        closed_conf, open_conf = robot.close_open_conf()

        gripper_traj = GroupTrajectory(
            robot,
            gripper_group,
            path=[open_conf, closed_conf],
            contexts=[pose],
            contact_links=robot.get_finger_links(robot.get_group_joints(gripper_group)),
            time_after_contact=1e-1,
            client=robot.client,
            attachments = [obj],
        )
        switch = Switch(
            obj,
            parent=ParentBody(
                body=robot, link=robot.link_from_name(tool_name), client=robot.client
            ),
        )

        # TODO: close the gripper a little bit before pregrasp
        if SWITCH_BEFORE == "contact":
            commands = [arm_traj, switch, arm_traj.reverse()]
        elif SWITCH_BEFORE == "grasp":
            commands = [arm_traj, switch, gripper_traj, arm_traj.reverse()]
        elif SWITCH_BEFORE == "pregrasp":
            commands = [arm_traj, gripper_traj, switch, arm_traj.reverse()]
        elif SWITCH_BEFORE == "arm":
            commands = [arm_traj, gripper_traj, arm_traj.reverse(), switch]
        elif SWITCH_BEFORE == "none":
            commands = [arm_traj, gripper_traj, arm_traj.reverse()]
        else:
            raise NotImplementedError(SWITCH_BEFORE)

        sequence = Sequence(
            commands=commands, name="pick-{}-{}".format(robot.side_from_arm(arm), obj)
        )
        return Tuple(arm_conf, sequence)

    return fn




#######################################################
   

def get_plan_place_fn(robot, **kwargs):
    robot_saver = BodySaver(robot, client=robot.client)

    def fn(arm, obj, pose, grasp, base_conf = None):

        robot_saver.restore()
        if base_conf is not None:
            base_conf.assign()

        arm_path = plan_prehensile(robot, arm, obj, pose, grasp, is_placing = True,  **kwargs)

        if arm_path is None:
            plan_prehensile(robot, arm, obj, pose, grasp, is_placing = True, **kwargs)
            return None


        arm_group, gripper_group, tool_name = robot.manipulators[
            robot.side_from_arm(arm)
        ]
        arm_traj = GroupTrajectory(
            robot,
            arm_group,
            arm_path[::-1],
            context=[grasp],
            velocity_scale=0.25,
            client=robot.client,
        )
        arm_conf = arm_traj.first()

        # closed_conf, open_conf = robot.get_group_limits(gripper_group)
        closed_conf, open_conf = robot.close_open_conf()
        gripper_traj = GroupTrajectory(
            robot,
            gripper_group,
            path=[closed_conf, open_conf],
            contexts=[grasp],
            client=robot.client,
            attachments =[],
        )
        switch = Switch(obj, parent=WORLD_BODY)

        # TODO: wait for a bit and remove colliding objects
        if SWITCH_BEFORE == "contact":
            commands = [arm_traj, switch, arm_traj.reverse()]
        elif SWITCH_BEFORE == "grasp":
            commands = [arm_traj, gripper_traj, switch, arm_traj.reverse()]
        elif SWITCH_BEFORE == "pregrasp":
            commands = [arm_traj, switch, gripper_traj, arm_traj.reverse()]
        elif SWITCH_BEFORE == "arm":
            commands = [switch, arm_traj, gripper_traj, arm_traj.reverse()]
        elif SWITCH_BEFORE == "none":
            commands = [arm_traj, gripper_traj, arm_traj.reverse()]
        else:
            raise NotImplementedError(SWITCH_BEFORE)
        sequence = Sequence(
            commands=commands, name="place-{}-{}".format(robot.side_from_arm(arm), obj)
        )
        return Tuple(arm_conf, sequence)

    return fn




#######################################################
def parse_fluents(fluents, environment, robot, objs_to_contact = []):
    obstacles = list(environment)
    attachments = []
    base_attachments = []
    for fluent in fluents:
        predicate = str(get_prefix(fluent)).lower()
        args = get_args(fluent)
        if predicate == "atconf":
            args[-1].assign()
        elif predicate == "atpose":
            body, pose = args
            if pose is None:
                continue

            if(body.get_shape_oobb().aabb.upper[2]<0.01):
                # Filter out the floor
                continue
            # ## TODO: in plan_pick, the obj is not yet atgrasp. We should also filter out that. Now for unsafe, we filter out colObs
            if body.category in objs_to_contact:
                continue

            obstacles.append(body)
            pose.assign() ### IMPORTANT!!
        elif predicate == "atgrasp":
            arm, body, grasp = args

            # side = robot.get_arbitrary_side()
            side = arm.split("_")[0]
            _, _, tool_name = robot.manipulators[side]
            tool_link = robot.link_from_name(tool_name)
            attachment = grasp.create_attachment(robot, link=tool_link)
            attachment.assign()
            attachments.append(attachment)
        elif predicate == "atattachmentgrasp":
            body, grasp = args
            # Get object pose in base frame
            base_attachments.append((get_aabb(body, client=robot.client), grasp))
        else:
            raise NotImplementedError(predicate)
    attached = {attachment.child for attachment in attachments}

    obstacles = set(obstacles) - attached

    return obstacles, attachments, base_attachments


def get_plan_motion_fn(
    robot, environment=[], collision_distance = -1,   **kwargs
):
    ee_traj_mode = bool(kwargs.pop("ee_traj_mode", False))
    robot_saver = BodySaver(robot, client=robot.client)
    robot_aabb = scale_aabb(recenter_oobb(robot.get_shape_oobb()).aabb, 0.5)

    def _conf_positions(conf):
        return tuple(getattr(conf, "positions", conf))

    def _extract_target_gripper_positions(group, conf):
        if "arm" not in group:
            return None
        arm_joints = robot.get_group_joints(group)
        positions = tuple(getattr(conf, "positions", conf))
        if len(positions) <= len(arm_joints):
            return None
        return positions[len(arm_joints):]

    def _annotate_motion_command(command, group, target_gripper_positions, freeze_gripper):
        command.freeze_gripper = bool(freeze_gripper)
        if ("arm" in group) and (target_gripper_positions is not None) and not freeze_gripper:
            command.target_gripper_positions = tuple(target_gripper_positions)

    def _planning_arm_is_grasping(group, fluents):
        group_name = str(group).lower()
        for fluent in fluents:
            if not fluent:
                continue
            if str(get_prefix(fluent)).lower() != "atgrasp":
                continue
            args = get_args(fluent)
            if len(args) < 3:
                continue
            if str(args[0]).lower() == group_name:
                return True
        return False

    def fn(group, q1, q2, fluents=[]):
        objs_to_contact = ['plate_1', 'flat_stove_1', 'basket_1', 'cup']  ### TODO: remove me, as moving to a pregrasp is collision-free 
        obstacles, attachments, base_attachments = parse_fluents(
            fluents,
            environment,
            robot,
            objs_to_contact=objs_to_contact,
        )
        target_gripper_positions = _extract_target_gripper_positions(group, q2)
        freeze_gripper = _planning_arm_is_grasping(group, fluents)

        if TELEPORT[0]:
            arm_joints = robot.get_group_joints(group)
            path = [
                _conf_positions(q1)[: len(arm_joints)],
                _conf_positions(q2)[: len(arm_joints)],
            ]
            command = GroupTrajectory(
                robot,
                group,
                path,
                attachments=attachments,
                client=robot.client,
            )
            _annotate_motion_command(command, group, target_gripper_positions, freeze_gripper)
            sequence = Sequence(
                commands=[command],
                name="move-{}".format(group),
            )
            return Tuple(sequence)
        
        robot_saver.restore()
        print("Plan motion fn {}->{}".format(q1, q2))

        q1.assign()
        for attachment in attachments:
            attachment.assign()
            
        joints = robot.get_group_joints(group)
        plan_joints = joints
        plan_positions = q2.positions[:len(joints)]

        # TODO: separate collision resolution for movable
        if group==robot.base_group:
            resolutions = 0.1 * np.ones(len(q2.joints))
            min_vals, max_vals = robot.get_group_limits(robot.base_group)
            path = plan_2d_joint_motion(
                robot,
                robot_aabb,
                q2.joints,
                min_vals,
                max_vals,
                q1.positions,
                q2.positions,
                resolutions=resolutions,
                obstacle_oobbs=[obstacle.get_shape_oobb() for obstacle in obstacles],
                restarts=0,
                max_iterations=100,
                smooth=100,
                attachments=base_attachments,
                disable_collisions=DISABLE_ALL_COLLISIONS,
                **kwargs
            )
            print("Output path: "+str(path))
        else:
            reso_deg = 3  # 10
            resolutions = math.radians(reso_deg) * np.ones(len(plan_joints))

            ## for moka_pot, collision distance = 0.05, use aabb
            if abs(collision_distance) > 0.5:
                use_aabb = False
            else:
                use_aabb = True

            # Penalize proximal/wide-range joints to reduce redundant swings when
            # the target is a small EE displacement (active in ee_traj_mode only).
            weights = None
            if ee_traj_mode and "arm" in group:
                base_w = [1.5, 1.5, 1.0, 1.0, 0.8, 0.8, 0.8]
                weights = (base_w + [1.0] * max(0, len(plan_joints) - len(base_w)))[:len(plan_joints)]

            path = plan_joint_motion(
                robot,
                plan_joints,
                plan_positions,
                resolutions=resolutions,
                weights=weights,
                obstacles=obstacles,
                attachments=attachments,
                self_collisions=SELF_COLLISIONS,
                disabled_collisions=robot.disabled_collisions,
                # max_distance=COLLISION_DISTANCE,
                max_distance=collision_distance,
                use_aabb=use_aabb,
                custom_limits=robot.custom_limits,
                restarts=1,
                iterations=5,
                smooth=100,
                # disable_collisions=DISABLE_ALL_COLLISIONS,
                radius = 20,
                **kwargs
            )

        if path is None:  ## preview the fail reason in gui
            for conf in [q1, q2]:
                conf.assign()
                for attachment in attachments:
                    attachment.assign()
            return None
        
        if len(path) <=2:
            print('path too short')

        command = GroupTrajectory(
            robot,
            group,
            path,
            attachments=attachments,
            client=robot.client,
        )
        _annotate_motion_command(command, group, target_gripper_positions, freeze_gripper)
        sequence = Sequence(
            commands=[command],
            name="move-{}".format(group),
        )
        return Tuple(sequence)

    return fn


#######################################################


def get_similarGrasp_test(robot, **kwargs):
    def test(arm, obj, grasp1, grasp2, sk1, sk2):
        if grasp1.phase is None or grasp2.phase is None:
            return False
        # if grasp1.skill_name is None or grasp2.skill_name is None:
        #     return False
        
        if grasp1.phase != grasp2.phase and sk1 != sk2:
            return True
        else:
            return False
    return test
