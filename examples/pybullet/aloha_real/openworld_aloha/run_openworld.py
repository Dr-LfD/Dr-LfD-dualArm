#!/usr/bin/env python

# if __name__ == '__main__':
#     from pre_import import *


import sys
import os

EXE_FOLDER = next(p for p in (os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *([os.pardir] * k))) for k in range(16)) if os.path.isfile(os.path.join(p, '.repo_root')))
sys.path.insert(0, EXE_FOLDER)
os.chdir(EXE_FOLDER)

from examples.pybullet.aloha_real.openworld_aloha.open_world_utils import get_camera_mappings

# CMD_PATH = os.path.join(EXE_FOLDER, '../statistics/cmd_logs/transfer_both_record.pkl') 

from pddlstream.algorithms.algorithm import reset_globals
from pddlstream.algorithms.serialized import solve_all_goals, solve_next_goal
from examples.pybullet.utils.pybullet_tools.utils import wait_for_user, WorldSaver, connect, LockRenderer, CLIENT, disconnect_all, remove_all_debug
from pddlstream.utils import INF, Profiler, str_from_object
from pddlstream.language.constants import print_solution, Solution


from examples.pybullet.aloha_real.openworld_aloha.problem_construction import (
    pddlstream_from_schema_problem,
)
from examples.pybullet.aloha_real.openworld_aloha.simple_worlds import load_aloha_world_flexible, load_mesh, load_dual_franka_world_flexible, load_panda_dual_world_flexible, load_panda_world_flexible
from examples.pybullet.aloha_real.openworld_aloha.primitives import post_process, execute_command, map_schema_plan_args
from examples.pybullet.aloha_real.openworld_aloha.policy_simp import estimation_policy
from examples.pybullet.aloha_real.openworld_aloha.openworld_streams import TELEPORT

from examples.pybullet.aloha_real.openworld_aloha.entities import Object

from examples.pybullet.aloha_real.openworld_aloha.symbolic_utils import update_problem, simulate_plan_execution
from examples.pybullet.aloha_real.openworld_aloha.schema_executor import (
    execute_schema_skeleton_plan,
)

def prepare_world(para, env_type = "sim", obj_info_ls = [], mj_pc_dict = None, estimate_belief=True, **kwargs):

    robot_name = para['robot_name']
    pybullet_use_gui = para['use_gui']
    pyb_env_addon = para['pyb_env_addon']
    text_prompt = para['text_prompt']

    disconnect_all() ## incase SAM fails and perception restarts
    connect(use_gui=pybullet_use_gui)

    if env_type =='mj':
        # assert len(obj_info_ls) > 0
        assert mj_pc_dict is not None
        seg_branch = None
    elif env_type == 'sim':
        assert len(obj_info_ls) > 0
        assert mj_pc_dict is  None
        seg_branch = None    
    elif env_type == 'file' or env_type == 'real':
        assert mj_pc_dict is None
        assert len(obj_info_ls) == 0
        random_rot_camera = para['random_rot_camera'] ## Note: Only for equivariance test. Do not use it on real robot!
        from examples.pybullet.aloha_real.openworld_aloha.contact_monitoring import (
            SegReuseRegistry,
            init_seg_branch_for_canonical,
            validate_seg_backend_pairing,
        )
        pairing = validate_seg_backend_pairing(para)
        SegReuseRegistry.init_for_pairing(pairing, para)
        seg_branch = init_seg_branch_for_canonical(pairing.perception)

    if robot_name == 'dualfranka':
        robot_entity, names, movable_bodies, stackable_bodies = load_dual_franka_world_flexible( additional_list=pyb_env_addon, obj_info_list=obj_info_ls, real_execute = para['real_execute'], **kwargs)
    elif robot_name == 'aloha':
        robot_entity, names, movable_bodies, stackable_bodies = load_aloha_world_flexible(additional_list=pyb_env_addon, obj_info_list=obj_info_ls, real_execute = para['real_execute'], **kwargs)
    elif robot_name == 'pandadual':
        robot_entity, names, movable_bodies, stackable_bodies = load_panda_dual_world_flexible( additional_list=pyb_env_addon,  real_execute = False, **kwargs)
    elif robot_name == 'panda':
        robot_entity, names, movable_bodies, stackable_bodies = load_panda_world_flexible( additional_list=pyb_env_addon,  real_execute = False, **kwargs)

    estimator_kwargs = dict(kwargs)
    if seg_branch == 'sam3':
        for key in (
            'sam3_path',
            'sam3_model_dir',
            'sam3_checkpoint',
            'sam3_conda_env',
            'sam3_conda_bin',
        ):
            if key in para:
                estimator_kwargs.setdefault(key, para[key])
    elif seg_branch == 'sam':
        for key in ('sam_path', 'use_server'):
            if key in para:
                estimator_kwargs.setdefault(key, para[key])
    if not estimate_belief:
        estimator_kwargs['perception_fn'] = None
    estimator = estimation_policy(
        robot_entity,
        teleport=False,
        client=CLIENT,
        seg_branch=seg_branch,
        text_prompt=text_prompt,
        env_type=env_type,
        **estimator_kwargs,
    )

    estimator._world_movable_objects = list(movable_bodies)
    estimator._world_surface_objects = [
        Object(stackable_bodies[i], category=names[stackable_bodies[i]])
        for i in range(len(stackable_bodies))
    ]

    if not estimate_belief:
        belief = estimator.belief
        belief.known_surfaces = list(estimator._world_surface_objects)
        return robot_entity, belief, estimator
    if env_type == 'sim':
        belief = estimator.belief
        belief.estimated_objects = [Object(movable_bodies[i], category=names[movable_bodies[i]]) if type(movable_bodies[i]) == str else movable_bodies[i] for i in range(len(movable_bodies))]
        belief.known_surfaces = list(estimator._world_surface_objects)
    elif  env_type == 'mj':
        estimator.esmate_mj_state(load_mesh, mj_pc_dict)
        belief = estimator.belief
        belief.known_surfaces = list(estimator._world_surface_objects)
    elif env_type == 'file' or env_type == 'real':
        cam_dir_mapping, cam_extparam_mapping, calibrate_mapping = get_camera_mappings(para)

        belief = estimator.estimate_state_multiview_file(cam_dir_mapping, cam_extparam_mapping, calibrate_mapping = calibrate_mapping, filter_surface = para['filter_surface'],  random_rot_camera = random_rot_camera, **kwargs)

        belief.estimated_objects.extend(estimator._world_movable_objects)
        belief.known_surfaces.extend(estimator._world_surface_objects)
    else:
        raise ValueError("Invalid image source")
    
    # ## ensure all objects are detected. Note: 1 category per obj
    # if type(text_prompt) == str:
    #     tgt_obj_list = [obj for obj in text_prompt.split('.') if obj != '']
    # else:
    #     tgt_obj_list = text_prompt
    # if  len(belief.estimated_objects)  != len(tgt_obj_list):
    #     raise ValueError(f'Detected objects {len(belief.estimated_objects)} do not match the expected {len(tgt_obj_list)}')
    
    return robot_entity, belief, estimator


def _plan_schema_problem(para, robot_entity, belief, *, use_perceived, planning_mode,
                         teleport, real_time_render=True, **kwargs):
    """Build a schema PDDLStream problem, solve it, and map plan args to arm groups.

    Returns ``(problem, stream_info, solution)`` where ``solution.plan`` has been
    rewritten by :func:`map_schema_plan_args`. Raises if schema planning is not
    configured (``use_schema`` unset or no ``skill_yaml_paths``).
    """
    skill_yaml_paths = [
        p if os.path.isabs(p) else os.path.join(EXE_FOLDER, p)
        for p in (para.get('skill_yaml_paths') or [])
    ]
    if not (para.get('use_schema') and skill_yaml_paths):
        raise ValueError(
            "Schema planning is required: set use_schema=true and provide skill_yaml_paths"
        )
    tmp_pddl_dir = para.get('schema_tmp_pddl_dir')
    if tmp_pddl_dir and not os.path.isabs(tmp_pddl_dir):
        tmp_pddl_dir = os.path.join(EXE_FOLDER, tmp_pddl_dir)

    problem, stream_info = pddlstream_from_schema_problem(
        robot_entity, belief,
        skill_yaml_paths=skill_yaml_paths,
        tmp_pddl_dir=tmp_pddl_dir,
        object_mapping=para.get('object_mapping'),
        match_by_category=para.get('match_by_category', True),
        use_perceived=use_perceived,
        planning_mode=planning_mode,
        use_constraints=para.get('use_constraints', False),
        teleport=teleport, real_time_render=real_time_render, client=CLIENT, **kwargs,
    )

    _, _, _, stream_map, init, goal = problem
    print('Init:', init)
    print('Goal:', goal)
    print('Streams:', str_from_object(set(stream_map)))

    saver = WorldSaver()
    solution = plan_pddlstream_restart(
        problem, real_time_render=real_time_render,
        stream_info=stream_info, saver=saver, **kwargs,
    )
    plan, cost, certificate, action_instances = solution
    if hasattr(robot_entity, 'get_arm_group'):
        plan = map_schema_plan_args(plan, robot_entity)
    return problem, stream_info, Solution(plan, cost, certificate, action_instances)


def _gui_execute(sequence, robot_entity):
    """Interactive GUI execution: prompt, drop grasp-gen grippers, run, prompt."""
    wait_for_user('Execute?')
    robot_entity.remove_components()  # drop grippers spawned by grasp generation
    execute_command(sequence, teleport=False, client=CLIENT, record_refined=True)
    wait_for_user('Finish?')


def compute_TAMP_online(para, robot_entity, belief, teleport=False, **kwargs):
    prev_teleport = TELEPORT[0]
    TELEPORT[0] = True
    try:
        problem, stream_info, solution = _plan_schema_problem(
            para, robot_entity, belief,
            use_perceived=para.get('use_perceived', True),
            planning_mode="coarse",
            teleport=teleport, **kwargs,
        )
        plan, _, certificate, action_instances = solution
        if plan is None:
            print("No plan to simulate")
            return None, None, problem, stream_info

        print(f"Simulating plan with {len(plan)} actions")
        state_history = simulate_plan_execution(
            problem.init, plan, action_instances, certificate.preimage_facts
        )

        sequence = post_process(plan)
        if para['use_gui']:
            _gui_execute(sequence, robot_entity)
        return solution, state_history, problem, stream_info
    finally:
        TELEPORT[0] = prev_teleport


def compute_TAMP_cmd(para, robot_entity, belief, teleport=False, **kwargs):
    """Schema-mode TAMP returning a post-processed command ``Sequence`` for real-robot execution.

    Consumed by the real-robot ROS plugin; schema planning is mandatory. Unlike
    :func:`compute_TAMP_online` (which forces ``planning_mode="coarse"`` for the
    interleaved online loop), this offline command path lets the caller choose the
    planning mode, defaulting to ``"detailed"``.
    """
    planning_mode = kwargs.pop('planning_mode', None) or para.get('planning_mode', 'detailed')
    _, _, solution = _plan_schema_problem(
        para, robot_entity, belief,
        use_perceived=para.get('use_perceived', False),
        planning_mode=planning_mode,
        teleport=para['teleport'], **kwargs,
    )

    sequence = post_process(solution.plan)
    if para['use_gui']:
        _gui_execute(sequence, robot_entity)
    return sequence


def plan_detail_mp(para, robot_entity, problem, stream_info, current_state, subgoal,
                   static_environment=None, skeleton_segment=None, **kwargs):
    remove_all_debug()

    from pddlstream.algorithms.constraints import PlanConstraints
    pybullet_use_gui = para['use_gui']
    TELEPORT[0] = False

    new_problem = update_problem(problem, current_state, subgoal)

    saver = WorldSaver()

    constraints = PlanConstraints(skeletons=[skeleton_segment], exact=True) \
        if skeleton_segment is not None else None

    solution_seg = plan_pddlstream_restart(new_problem, real_time_render=True,
                                           stream_info=stream_info, saver=saver,
                                           constraints=constraints, **kwargs)

    plan_seg, _, new_evaluations, new_action_instances = solution_seg
    if plan_seg is None and constraints is not None:
        print('\033[33m[plan_detail_mp] Skeleton-constrained planning failed, retrying unconstrained\033[0m')
        solution_seg = plan_pddlstream_restart(new_problem, real_time_render=True,
                                               stream_info=stream_info, saver=saver,
                                               constraints=None, **kwargs)
        plan_seg, _, new_evaluations, new_action_instances = solution_seg

    if plan_seg is None:
        return None, current_state, None

    # GUI debug execution uses post_process path
    sequence_seg = post_process(plan_seg, robot_entity=robot_entity)

    if pybullet_use_gui:
        _gui_execute(sequence_seg, robot_entity)

    # Schema-executor expansion: same path as _plan_lane_batch, authoritative for real execution.
    if static_environment is not None:
        schema_result = execute_schema_skeleton_plan(
            para, robot_entity, static_environment, plan_seg, current_state, **kwargs
        )
        if schema_result is not None:
            sequence_seg = schema_result.sequence

    # NOTE: the facts generated by streams are not included in simulate_plan_execution()
    state_history_seg = simulate_plan_execution(current_state, plan_seg, new_action_instances)
    ## add the literals gerneated by streams

    #  the facts generated by streams 
    literals_from_stream = set(new_evaluations[1]) - current_state 
    ## as we want to union the literals from streams and the literals from the plan, we need to remove the literals that are removed in the plan
    removed_literals = current_state - state_history_seg[-1]
    literals_from_stream -= removed_literals

    updated_state = state_history_seg[-1].union(literals_from_stream)

    return sequence_seg, updated_state, plan_seg

def plan_pddlstream_restart(problem, real_time_render=False, serialize=False,
                            constraints=None, saver=None, stream_info=None,
                            max_tamp_time=INF, **kwargs):
    reset_globals()


    profiler = Profiler()
    profiler.save()
    with LockRenderer(lock=not real_time_render, **kwargs):
        solve_fn = solve_next_goal if serialize else solve_all_goals
        extra_kwargs = {} if constraints is None else {'constraints': constraints}
        solve_kwargs = dict(
            problem=problem,
            **extra_kwargs,
            stream_info=stream_info,
            replan_actions={"perceive"},
            initial_complexity=0,
            planner="ff-astar2",
            max_planner_time=5,
            unit_costs=True,
            success_cost=INF,
            max_tamp_time=max_tamp_time,
            max_memory=INF,
            max_restarts=INF,
            max_skeletons=3,
            iteration_time=50,
            unit_efforts=True,
            max_effort=INF,
            effort_weight=1,
            search_sample_ratio=2,
            verbose=True,
            debug=False,
            visualize=True,
        )
        solve_kwargs["initial_problem"] = solve_kwargs.pop("problem")
        solution = solve_fn(**solve_kwargs)
    # belief.robot.remove_components()
    profiler.restore()
    saver.restore()
    print_solution(solution)
    return solution

