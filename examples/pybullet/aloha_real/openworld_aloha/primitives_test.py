# test pick and place

# if __name__ == '__main__':
#     from pre_import import *
    
import sys
import os
import json
from types import SimpleNamespace

import numpy as np

from .open_world_utils import load_insertion_param, load_yaml_params, get_camera_mappings, EXE_FOLDER
sys.path.append(EXE_FOLDER) if EXE_FOLDER not in sys.path else None

os.chdir(EXE_FOLDER)

from examples.pybullet.aloha_real.openworld_aloha.openworld_streams import (
    get_imitate_traj_fn,
    get_plan_motion_fn,
    get_plan_pick_fn,
    get_plan_place_fn,
    get_placement_gen_fn,
)
# get_grasp_gen_fn is the learned-grasp stream restored in Phase 4; it is imported lazily
# inside the functions that use it so this module imports without the Phase-4 grasp stack.


from examples.pybullet.aloha_real.openworld_aloha.primitives import (
    GroupConf,
    GroupTrajectory,
    RelativePose,
    Sequence,
    execute_command,
)

from examples.pybullet.aloha_real.openworld_aloha.simple_worlds import  load_aloha_world_flexible, load_dual_franka_world_flexible
from examples.pybullet.utils.pybullet_tools.utils import wait_for_user, WorldSaver, connect,  CLIENT, flatten, read_pickle, write_pickle
from examples.pybullet.aloha_real.openworld_aloha.problem_construction import get_fixed
from pddlstream.utils import  get_file_path, write_dill
from examples.pybullet.aloha_real.openworld_aloha.entities import Object
from examples.pybullet.aloha_real.openworld_aloha.policy_simp import estimation_policy


from examples.pybullet.aloha_real.openworld_aloha.robot_entities import ALOHARobot
from examples.pybullet.aloha_real.openworld_aloha.run_openworld import prepare_world

# # TODO: use saver to restore the belief
# def test_grasp_old(para,  env_type = "file",   use_estimation = True):
#     robot_name = para['robot_name']
#     use_gui = para['use_gui']
#     text_prompt = para['text_prompt']

#     connect(use_gui = use_gui)
    
#     table = None
#     tgtobj = None

#     additional_list = []
#     obj_info_ls = []
#     if env_type == 'sim':
#         additional_list = ["floor", "socket"]
#         CFG_PATH = os.path.join(EXE_FOLDER, 'config/aloha_scene.yaml')
#         CMD_PATH, obj_info_ls, is_record = load_insertion_param(CFG_PATH)
    
#     if robot_name == 'dualfranka':
#         robot_entity, names, movable_bodies, stackable_bodies = load_dual_franka_world_flexible( additional_list=additional_list, obj_info_list=obj_info_ls)
#     elif robot_name == 'aloha':
#         robot_entity, names, movable_bodies, stackable_bodies = load_aloha_world_flexible(additional_list=additional_list, obj_info_list=obj_info_ls)

#     fixed_objects = get_fixed([robot_entity], movable_bodies)

#     estimator = estimation_policy(robot_entity, env_type = env_type,  teleport=False, client=CLIENT, seg_branch='sam', text_prompt=text_prompt)

#     cam_dir_mapping, cam_extparam_mapping = get_camera_mappings(para)
#     if env_type == 'sim' and   not use_estimation:
#         belief = estimator.belief
#         belief.estimated_objects = movable_bodies
#         belief.known_surfaces = [Object(stackable_bodies[i], category=names[stackable_bodies[i]]) for i in range(len(stackable_bodies))]

#         table = belief.known_surfaces[0]
#         tgtobj = belief.estimated_objects[0]

#         grasp_mode = 'top'
#         robot = belief.robot

#     else:

#         belief = estimator.estimate_state_multiview_file(cam_dir_mapping, cam_extparam_mapping)
#         #===============

#         table = estimator.estimates[-1]["surfaces"][0]
#         tgtobj = estimator.estimates[-1]["objects"][0]

#         # save obj as ply
#         # lab_pts = tgtobj.points
#         # points = [lp.point for lp in lab_pts]
#         # # points = filter_pc(points)
#         # import open3d as o3d
#         # pcd = o3d.geometry.PointCloud()
#         # pcd.points = o3d.utility.Vector3dVector(points)
#         # o3d.io.write_point_cloud("mug_OOD3.ply", pcd)

#         grasp_mode = 'gpd'
#         robot = belief.robot

#     assert table is not None
#     assert tgtobj is not None

def test_primitive(para, robot_entity, belief, env_type = 'sim', use_estimation = False, **kwargs):
    robot_name = para['robot_name']
    movable_bodies = belief.estimated_objects
    fixed_objects = get_fixed([robot_entity], movable_bodies)
    table = belief.known_surfaces[0]
    tgtobj = belief.estimated_objects[0]
    
    if env_type == 'sim' and   not use_estimation:
        grasp_mode = 'top'
    else:
        grasp_mode = 'gpd'

    # learned-grasp stream is restored in Phase 4; imported lazily so this test module loads
    # without the Phase-4 grasp stack.
    from examples.pybullet.aloha_real.openworld_aloha.openworld_streams import get_grasp_gen_fn
    grasp_gen = get_grasp_gen_fn(
        robot=robot_entity,
        other_obstacles=_grasp_obstacles(fixed_objects, robot_entity, table),
        grasp_mode=grasp_mode,
        max_time=5,
        robot_name=robot_name,
    )
    pick_gen = get_plan_pick_fn(robot_entity, environment=fixed_objects)
    place_gen = get_plan_place_fn(robot_entity, environment = fixed_objects )
    placement_gen = get_placement_gen_fn(robot_entity, fixed_objects, environment=fixed_objects)
    motion_gen = get_plan_motion_fn(robot_entity, environment=fixed_objects)

    
    # plan grasp
    plan_group = 'right_arm'
    grasp_gen = grasp_gen(plan_group, tgtobj)

    # plan placement
    table_pose = RelativePose(table, important=True)
    place_pose_gen  = placement_gen(tgtobj, table, table_pose)


    pose0 = RelativePose(tgtobj, important=True)
    init_confs = {
        group: GroupConf(robot_entity, group, important=True)
        for group in robot_entity.groups
    }
    conf0 = init_confs[plan_group]

    saver = WorldSaver()

    id = -1
    for grasp in grasp_gen:
        id +=1
        seq_list = []
        saver.restore()
        pick_result = pick_gen(plan_group, tgtobj, pose0, grasp[0], init_confs["base"])

        if pick_result is None:
            print("---No pick plan for grasp ", id)
            continue

        grasp_conf = pick_result[0]
        seq_list.append(pick_result[1])
        saver.restore()

        ################### plan placement
        while True:
            place_pose = next(place_pose_gen)[0]
            # for left/right eachable
            if 'left' in  plan_group and place_pose.value[0][0] > 0:
                continue
            elif 'right' in plan_group and place_pose.value[0][0] < 0:
                continue

            place_result = place_gen(plan_group, tgtobj, place_pose, grasp[0], init_confs["base"])

            if place_result is not None:
                place_conf = place_result[0]
                if place_conf is None:
                    continue

                placemotion_result = motion_gen(plan_group, grasp_conf, place_conf)

                if placemotion_result is not None:
                    print("Got placement ", id)
                    seq_list.append(placemotion_result[0])
                    seq_list.append(place_result[1])
                    break

        # switch_cmd = place_result[1].commands[1:3]
        # place_result[1].commands = switch_cmd
        

        ####################

        # go back to the initial configuration
        result = motion_gen(plan_group, place_conf, conf0)

        if result is None:
            print("No motion plan for grasp ", id)
            continue

        seq_list.append(result[0])


        sequence = Sequence(
            flatten(cmd.commands for cmd in seq_list)
        )
        sequence.dump()

        # below will make the estimated objects disappear
        # belief.reset()
        # p.removeAllUserDebugItems()

        saver.restore()

        ans = wait_for_user('Execute?')

        aborted = execute_command(sequence, teleport=False, client=CLIENT)

        ans = wait_for_user('Finish?')
        if ans == 'q':
            # cmd_path = os.path.join(EXE_FOLDER, 'temp/sequence_pick.pkl')
            # write_pickle(cmd_path, sequence)
            return sequence
        
    raise RuntimeError('No plan found')


def _require_key(mapping, key, context):
    if key not in mapping:
        raise KeyError(f'missing {context}[{key!r}]')
    return mapping[key]


def _normalize_arm(arm):
    arm = str(arm)
    if arm.endswith('_arm'):
        return arm
    if arm in ('left', 'right'):
        return f'{arm}_arm'
    raise ValueError(f'unsupported arm name: {arm!r}')


def _load_task_schema(para, prefix_key):
    sg = _require_key(para, prefix_key, 'para')
    schema_cfg = _require_key(sg, 'schema', f'para[{prefix_key}]')
    schema_rel = _require_key(schema_cfg, 'path', 'schema')
    skill_yaml_paths = _require_key(para, 'skill_yaml_paths', 'para')
    if len(skill_yaml_paths) != 1:
        raise ValueError(
            f'--primitive-test requires exactly one --skill-yaml-path, got {len(skill_yaml_paths)}'
        )
    schema_path = os.path.normpath(
        os.path.join(os.path.dirname(skill_yaml_paths[0]), schema_rel)
    )
    if not os.path.isfile(schema_path):
        raise FileNotFoundError(schema_path)
    with open(schema_path, 'r') as f:
        return json.load(f)


def _biop_schema_spec(schema):
    streams = _require_key(schema, 'instantiated_streams', 'schema')
    matches = [s for s in streams if s.get('template') == 'sample-biop-keypose']
    if len(matches) != 1:
        raise ValueError(
            f'expected exactly one sample-biop-keypose stream, got {len(matches)}'
        )
    return matches[0]


def _resolve_primitive_test_config(para, equivSkill_info_dict):
    skill_names = _require_key(para, 'skill_names', 'para')
    if len(skill_names) != 1:
        raise ValueError(
            f'--primitive-test requires exactly one --skill-yaml-path, got {len(skill_names)}'
        )
    prefix_key = skill_names[0]
    if prefix_key not in equivSkill_info_dict:
        raise KeyError(f'prefix_key {prefix_key!r} not in equivSkill_info_dict')
    sg = _require_key(para, prefix_key, 'para')
    pre_arms = _require_key(sg, 'pre_arms', f'para[{prefix_key}]')
    pre_obj_names = _require_key(sg, 'pre_obj_names', f'para[{prefix_key}]')
    if len(pre_arms) != 1:
        raise ValueError(f'pre_arms must have exactly one entry, got {pre_arms}')
    if len(pre_obj_names) != 1:
        raise ValueError(f'pre_obj_names must have exactly one entry, got {pre_obj_names}')
    biop_spec = _biop_schema_spec(_load_task_schema(para, prefix_key))
    biop_skill = f'bimanual_{prefix_key}'
    skillwise_sgs = _require_key(
        equivSkill_info_dict[prefix_key], 'skillwise_sgs', f'equivSkill_info_dict[{prefix_key}]'
    )
    if biop_skill not in skillwise_sgs:
        raise KeyError(f'{biop_skill!r} not in skillwise_sgs for {prefix_key!r}')
    return SimpleNamespace(
        prefix_key=prefix_key,
        biop_skill=biop_skill,
        grasp_arm=_normalize_arm(pre_arms[0]),
        biop_arm1=_require_key(biop_spec, 'arm1', 'sample-biop-keypose'),
        biop_arm2=_require_key(biop_spec, 'arm2', 'sample-biop-keypose'),
        eff_grasps=[tuple(g) for g in _require_key(biop_spec, 'eff_grasps', 'sample-biop-keypose')],
        target_category=pre_obj_names[0],
    )


def _object_by_category(objects, category):
    matches = [o for o in objects if getattr(o, 'category', None) == category]
    if len(matches) != 1:
        raise ValueError(
            f'expected exactly one object with category {category!r}, got {len(matches)}'
        )
    return matches[0]


def _default_conf(default_confs, arm_group):
    robot_group = _robot_group(arm_group)
    if robot_group in default_confs:
        return default_confs[robot_group]
    if arm_group in default_confs:
        return default_confs[arm_group]
    raise KeyError(f'no default conf for arm group {arm_group!r}')


def _obstacle_key(obj):
    body = obj.body if hasattr(obj, 'body') else obj
    try:
        hash(body)
    except TypeError:
        return id(obj)
    return body


def _grasp_obstacles(fixed_objects, robot, table):
    """Grasp collision check includes table and fixed environment."""
    obstacles = list(fixed_objects) + [robot, table]
    seen = set()
    deduped = []
    for obs in obstacles:
        key = _obstacle_key(obs)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(obs)
    return deduped


def _biop_obstacles(fixed_objects, table):
    """Biop keypose sampling ignores table collisions."""
    table_key = _obstacle_key(table)
    return [obs for obs in fixed_objects if _obstacle_key(obs) != table_key]


def _robot_group(arm_group):
    return arm_group.replace('_arm', '_robot')


def _extract_arm_conf(robot, arm_group, conf):
    n = len(robot.get_group_joints(arm_group))
    return GroupConf(
        robot,
        arm_group,
        positions=tuple(conf.positions[:n]),
        important=getattr(conf, 'important', False),
    )


def _default_group_confs(robot):
    if not hasattr(robot, 'get_default_conf'):
        return {
            group: GroupConf(robot, group, important=True)
            for group in robot.groups
        }
    defaults = robot.get_default_conf()
    return {
        group: GroupConf(robot, group, positions=positions, important=True)
        for group, positions in defaults.items()
        if group in robot.groups and positions
    }


def _plan_arm_motion(motion_gen, robot, arm_group, start_conf, goal_conf, fluents=None):
    q_start = _extract_arm_conf(robot, arm_group, start_conf)
    q_goal = _extract_arm_conf(robot, arm_group, goal_conf)
    if np.allclose(q_start.positions, q_goal.positions, atol=1e-3):
        print(f'[motion] skip {arm_group}: start equals goal {q_start.positions}')
        return None
    print(f'[motion] {arm_group}: start={q_start.positions} -> goal={q_goal.positions}')
    if fluents is None:
        return _unwrap_motion_sequence(motion_gen(arm_group, q_start, q_goal))
    return _unwrap_motion_sequence(
        motion_gen(arm_group, q_start, q_goal, fluents=fluents)
    )


def _unwrap_motion_sequence(output):
    if output is None:
        return None
    if isinstance(output, tuple):
        candidate = output[0]
        if hasattr(candidate, 'commands'):
            return candidate
        if len(output) > 1 and hasattr(output[1], 'commands'):
            return output[1]
    if hasattr(output, 'commands'):
        return output
    return None


def _last_arm_conf(sequence, arm_group):
    for cmd in reversed(sequence.commands):
        if not isinstance(cmd, GroupTrajectory):
            continue
        if cmd.group != arm_group:
            continue
        return cmd.last()
    return None


def test_primitive_pipeline(
    para,
    robot_entity,
    belief,
    equivSkill_info_dict,
    env_type='file',
    **kwargs,
):
    """GPD grasp + pick + bimanual keypose + dual-arm motion + LfD Graphstate entry."""
    cfg = _resolve_primitive_test_config(para, equivSkill_info_dict)
    robot_name = para['robot_name']
    movable_bodies = belief.estimated_objects
    fixed_objects = get_fixed([robot_entity], movable_bodies)
    table = belief.known_surfaces[0]
    tgtobj = _object_by_category(belief.estimated_objects, cfg.target_category)

    if env_type == 'sim' and not kwargs.get('use_estimation', False):
        grasp_mode = 'top'
    else:
        grasp_mode = 'gpd'

    grasp_arm = cfg.grasp_arm
    pose0 = RelativePose(tgtobj, important=True)
    init_confs = {
        group: GroupConf(robot_entity, group, important=True)
        for group in robot_entity.groups
    }
    default_confs = _default_group_confs(robot_entity)

    # learned-grasp stream is restored in Phase 4; imported lazily (see test_primitive).
    from examples.pybullet.aloha_real.openworld_aloha.openworld_streams import get_grasp_gen_fn
    grasp_gen_fn = get_grasp_gen_fn(
        robot=robot_entity,
        other_obstacles=_grasp_obstacles(fixed_objects, robot_entity, table),
        grasp_mode=grasp_mode,
        max_time=5,
        robot_name=robot_name,
    )
    pick_gen = get_plan_pick_fn(robot_entity, environment=fixed_objects)
    motion_gen = get_plan_motion_fn(robot_entity, environment=fixed_objects)

    biop_gen = get_imitate_traj_fn(
        robot_entity,
        equivSkill_info_dict=equivSkill_info_dict,
        prefix_key=cfg.prefix_key,
        skill_name=cfg.biop_skill,
        fixed_obj=_biop_obstacles(fixed_objects, table),
        eff_grasps=list(cfg.eff_grasps),
        **kwargs,
    )

    saver = WorldSaver()
    for grasp_id, grasp in enumerate(grasp_gen_fn(grasp_arm, tgtobj)):
        saver.restore()
        pick_result = pick_gen(grasp_arm, tgtobj, pose0, grasp[0], init_confs['base'])
        if pick_result is None:
            print(f'---No pick plan for GPD grasp {grasp_id}')
            continue

        pregrasp_conf = pick_result[0]
        pick_seq = pick_result[1]
        post_pick_conf = _last_arm_conf(pick_seq, grasp_arm)
        if post_pick_conf is None:
            print(f'---No post-pick conf in pick sequence for grasp {grasp_id}')
            continue

        approach_motion = _plan_arm_motion(
            motion_gen,
            robot_entity,
            grasp_arm,
            _default_conf(default_confs, grasp_arm),
            pregrasp_conf,
        )
        if approach_motion is None:
            print(f'---No approach motion to pregrasp for grasp {grasp_id}')
            continue

        saver.restore()
        biop_result = next(
            biop_gen(cfg.biop_arm1, cfg.biop_arm2, cfg.biop_skill, tgtobj, pose0)
        )
        arm1_conf, arm2_conf, graph_state, *_ = biop_result

        post_pick_fluents = [('AtGrasp', grasp_arm, tgtobj, grasp[0])]
        arm_goals = {'left_arm': arm1_conf, 'right_arm': arm2_conf}
        motions = {
            arm: _plan_arm_motion(
                motion_gen,
                robot_entity,
                arm,
                post_pick_conf if arm == grasp_arm else _default_conf(default_confs, arm),
                arm_goals[arm],
                fluents=post_pick_fluents if arm == grasp_arm else None,
            )
            for arm in ('left_arm', 'right_arm')
        }
        left_motion = motions['left_arm']
        right_motion = motions['right_arm']

        if left_motion is None:
            print(f'---No left motion to bimanual jpose for grasp {grasp_id}')
            continue
        if right_motion is None:
            print(f'---No right motion to bimanual jpose for grasp {grasp_id}')
            continue

        commands = list(approach_motion.commands)
        commands.extend(pick_seq.commands)
        commands.extend(left_motion.commands)
        commands.extend(right_motion.commands)
        sequence = Sequence(commands, name='primitive-test')
        sequence.graphstate_markers = [(len(commands), graph_state)]
        print(
            f'Primitive test plan ready: task={cfg.prefix_key}, grasp={grasp_id}, '
            f'commands={len(commands)}, graphstate={graph_state.skill_name}'
        )
        wait_for_user('Execute?')
        return sequence

    raise RuntimeError(f'No primitive test plan found for {cfg.prefix_key}')


if __name__ == '__main__':
    from .run_openworld import prepare_world

    # ## screwdriver franka 
    # cfg_path = 'examples/pybullet/aloha_real/openworld_aloha/configs/sg_screwdriver_franka.yaml'  #
    # parameters = load_yaml_params(cfg_path)
    # parameters['execution']['pyb_env_addon'] = ["floor", "socket"]
    # robot_entity, belief = prepare_world(parameters, env_type = "sim")
    # command = test_primitive(parameters, robot_entity, belief, env_type = "sim", use_estimation=False)

    ## screwdriver aloha, using img stored in temp_vis/realrobot
    cfg_path = 'examples/pybullet/aloha_real/openworld_aloha/configs/sg_transfer_cup.yaml'  #
    parameters = load_yaml_params(cfg_path)
    parameters['real_execute'] = False
    # test_grasp(parameters, env_type = "file", use_estimation=True)\
    robot_entity, belief, _ = prepare_world(parameters, env_type = "file")
    command = test_primitive(parameters, robot_entity, belief, env_type = "file", use_estimation=False)
