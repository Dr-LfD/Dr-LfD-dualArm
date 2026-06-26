
import os
import tempfile
from repo_paths import load_yaml
from pddlstream.utils import read
from examples.pybullet.aloha_real.openworld_aloha.primitives import GroupConf, RelativePose
from pddlstream.language.constants import (
    Equal,
    PDDLProblem,
)
from examples.pybullet.utils.pybullet_tools.utils import is_placement, get_bodies,  find_kw_in_skill

from examples.pybullet.aloha_real.openworld_aloha.openworld_streams import get_placement_gen_fn, get_plan_place_fn, get_plan_drop_fn, get_plan_motion_fn, get_pose_cost_fn, get_test_cfree_pose_pose, get_cfree_pregrasp_pose_test, get_cfree_traj_pose_test, BASE_COST, get_imitate_traj_fn, get_learned_pick_fn, get_reachability_test, get_mdf_clear_test
from pddlstream.language.generator import from_gen_fn, from_fn, from_test
from pddlstream.language.stream import StreamInfo, PartialInputs
from pddlstream.language.function import FunctionInfo

import networkx as nx

def get_fixed(robots, movable):
    rigid = [body for body in get_bodies() if body not in  robots]
    fixed = [body for body in rigid if body not in movable]
    return fixed

def get_skillwise_literals(robot_entity, skillwise_sg_info, skill_name, perceived_objects):
    # ## compatible for real_aloha
    # if 'skill_type' not in skillwise_sg_info:
    #     return [('DoneSkill', skill_name)]
    
    skill_type = skillwise_sg_info['skill_type']
    if 'bimanual' in skill_type:
        sg_nx = skillwise_sg_info['pre_sg']
    elif skill_type == 'ATTACH':
        sg_nx = skillwise_sg_info['cur_sg']
    elif skill_type == 'DETACH':
        sg_nx = skillwise_sg_info['pre_sg']
    else:
        raise NotImplementedError("Unknown skill type: ", skill_type)
    
    ## TODO: currently it is order-sensitive. Requrie to be specified in yaml
    related_rbts = skillwise_sg_info['related_rbts']
    ## find related objects from sg_nx

    tgt_obj_literals = []
    doneskill_literal_args = []
    related_arm_names = [f"{ robot_entity.rbt_ids_to_side[rbt]}_arm" for rbt in related_rbts]
    doneskill_literal_args.extend(related_arm_names)
    for id,rbt in enumerate(related_rbts):
        # # arm_name = related_arm_names[id]
        # if skill_type == 'ATTACH': 
        #     related_obj_name = skillwise_sg_info['related_objs'][id]
        #     related_obj = [obj for obj in perceived_objects if obj.category == related_obj_name][0]

        ## get objects that attached to left, right
        obj_nbrs = set(nx.neighbors(sg_nx, rbt))
        if len(obj_nbrs) ==0:
            continue
        inhand_obj_name = list(obj_nbrs - set(related_rbts))[0]
        inhand_obj = [obj for obj in perceived_objects if obj.category == inhand_obj_name][0]
        tgt_obj_literals.append(('tgt_obj', skill_name, inhand_obj))
        doneskill_literal_args.append(inhand_obj)
        
        # ## else only append related_obj
        # doneskill_literal_args.append(related_obj)

    doneskill_literal = (f'DoneSkill{skill_type}', skill_name, *doneskill_literal_args)

    return  [doneskill_literal], tgt_obj_literals

def _schema_obj_to_body(obj):
    if obj is None:
        return None
    if hasattr(obj, 'body'):
        return obj.body
    return obj

def _load_schema_input_from_yaml(yaml_path):
    """Return ((initial_graph, skills, objects_dict), source_label) from a per-skill YAML file.

    Reads the 'schema' key inside 'sg_params' (path form only; the inline
    initial_graph+skills form is reserved but not yet wired up).
    """
    from examples.pybullet.aloha_real.scripts.standalone_scripts.schema_construction import (
        parse_config,
    )
    yaml_data = load_yaml(yaml_path)

    schema_cfg = yaml_data.get("sg_params", {}).get("schema")
    if schema_cfg is None:
        raise ValueError(f"Missing sg_params.schema in per-skill YAML: {yaml_path}")
    if not isinstance(schema_cfg, dict):
        raise ValueError(f"sg_params.schema must be a dict in: {yaml_path}")

    if "path" in schema_cfg:
        schema_path = schema_cfg["path"]
        if not os.path.isabs(schema_path):
            schema_path = os.path.normpath(
                os.path.join(os.path.dirname(yaml_path), schema_path)
            )
        object_mapping = yaml_data.get("object_mapping")
        return parse_config(schema_path, object_mapping=object_mapping), schema_path

    # if "initial_graph" in schema_cfg and "skills" in schema_cfg:
    #     return parse_config_from_dict(schema_cfg), yaml_path

    raise ValueError(
        f"sg_params.schema must contain either 'path' or inline "
        f"'initial_graph' + 'skills' in: {yaml_path}"
    )


def _compose_all_schema_inputs(inputs_with_sources):
    """Compose ((graph, skills, objects_dict), source) pairs into one (graph, skills, dict).

    Needs at least one entry.
    """
    from examples.pybullet.aloha_real.scripts.standalone_scripts.schema_construction import _compose_raw

    (composed_graph, composed_skills, composed_objects_dict), composed_source = inputs_with_sources[0]
    for (graph_b, skills_b, objects_dict_b), source_b in inputs_with_sources[1:]:
        composed_graph, composed_skills = _compose_raw(
            composed_graph, composed_skills, graph_b, skills_b,
            source_a=composed_source, source_b=source_b,
        )
        composed_objects_dict = {**composed_objects_dict, **objects_dict_b}
        composed_source = "composed"
    return composed_graph, composed_skills, composed_objects_dict


def _make_domain_name(task_name, skill_yaml_paths):
    """Derive a filename stem for debug PDDL output."""
    if task_name:
        return str(task_name).replace(os.sep, "_").replace(" ", "_")
    if not skill_yaml_paths:
        return "composed"
    return "_".join(os.path.splitext(os.path.basename(p))[0] for p in skill_yaml_paths)


def pddlstream_from_schema_problem(
    robot_entity, belief,
    skill_yaml_paths, tmp_pddl_dir=None,
    existing_domain_path = None,
    existing_stream_path = None,
    object_mapping=None,
    match_by_category=True, use_perceived=False,
    task_name=None, skillwise_sgs=None, planning_mode="detailed", **kwargs
):
    from examples.pybullet.aloha_real.scripts.standalone_scripts.schema_construction import (
        get_schema_metadata_from_data, compute_skill_names, build_action_schema_from_data,
    )
    robot_entity.reset()
    # Per-task toggle (default off): emit reachability + MDF policy-safety constraints only
    # for tasks that opt in via use_constraints (aloha real robot, mj-insertion-unsafe). DMG
    # threading/assembly leave it off, so generated PDDL stays constraint-free.
    enable_constraints = kwargs.pop("use_constraints", False)
    perceived_objects = list(belief.estimated_objects)
    stackable = list(belief.known_surfaces)

    if not skill_yaml_paths:
        raise ValueError("skill_yaml_paths must be a non-empty list of per-skill YAML paths")
    env_names = list(skillwise_sgs.keys()) if skillwise_sgs else []
    schema_inputs = [_load_schema_input_from_yaml(p) for p in skill_yaml_paths]
    composed_graph, composed_skills, composed_objects_dict = _compose_all_schema_inputs(schema_inputs)
    schema_metadata = get_schema_metadata_from_data(
        composed_graph, composed_skills, env_names=env_names, planning_mode=planning_mode,
        objects_dict=composed_objects_dict,
        enable_constraints=enable_constraints,
    )
    debug_dir = tmp_pddl_dir or os.path.join(tempfile.gettempdir(), "pddlstream_aloha_debug")
    ## debug mode: use existing domain and stream
    if existing_domain_path and existing_stream_path:
        domain_pddl = read(existing_domain_path)
        stream_pddl = read(existing_stream_path)
    else:
        domain_pddl, stream_pddl = build_action_schema_from_data(
            composed_graph, composed_skills,
            output_dir=debug_dir,
            domain_name=f"{_make_domain_name(task_name, skill_yaml_paths)}_{planning_mode}",
            env_names=env_names,
            planning_mode=planning_mode,
            objects_dict=composed_objects_dict,
            enable_constraints=enable_constraints,
        )
    # arm_names = schema_metadata["arm_names"]
    # movable_names = schema_metadata["movable_names"]
    # surface_names = schema_metadata["surface_names"]
    # object_names = schema_metadata["object_names"]
    movable_names = [obj.category for obj in belief.estimated_objects]
    surface_names = [surf.category for surf in belief.known_surfaces]
    object_names = movable_names + surface_names
    skill_goals = schema_metadata["skill_goals"]
    classified = schema_metadata.get("classified", [])
    schema_skill_names = compute_skill_names(classified, env_names)
    side_to_rbt = getattr(robot_entity, "side_to_rbt_ids", None) or {}
    rbt_to_side = getattr(robot_entity, "rbt_ids_to_side", None) or {}
    schema_arm_to_group = {rbt: f"{side}_arm" for side, rbt in side_to_rbt.items()}
    # constant_map = {"@base": "base", "@head": "head", "@torso": "torso"}
    constant_map = {}

    def resolve_schema_object(name):
        if object_mapping and name in object_mapping:
            return object_mapping[name]
        if match_by_category:
            for obj in perceived_objects + stackable:
                if getattr(obj, "category", None) == name:
                    return obj
                if name in getattr(obj, "category", ""):
                    return obj
        return None

    schema_name_to_obj = {}
    for name in object_names:
        body_or_surface = resolve_schema_object(name)
        if body_or_surface is not None:
            schema_name_to_obj[name] = body_or_surface
        else:
            import warnings
            warnings.warn(f"Schema object {name} has no matching perceived body; skipping.")

    init_confs = {group: GroupConf(robot_entity, group, important=True, **kwargs) for group in robot_entity.groups}
    init_poses = {}
    for name, pddl_obj in schema_name_to_obj.items():
        body = _schema_obj_to_body(pddl_obj)
        if body is not None:
            init_poses[pddl_obj] = RelativePose(body, important=True, **kwargs)
    for surf in stackable:
        init_poses[surf] = RelativePose(surf, important=True, **kwargs)
    init = []

    for group_key in robot_entity.groups:
        if group_key == "body" or group_key == "base":
            continue
        # group_key = "base" if group == "body" else group
        if "gripper" in group_key or "robot" in group_key: ## only arm conf
            continue
        conf = init_confs[group_key]
        init.extend([("Conf", group_key, conf), ("InitConf", group_key, conf), ("RestConf", group_key, conf), ("AtConf", group_key, conf), ("CanMove", group_key), Equal(("MoveCost", group_key), 1)])
        if "arm" in group_key:
            init.extend([("Arm", group_key), ("ArmEmpty", group_key), ("Controllable", group_key), (group_key, group_key)])

    for k, v in schema_arm_to_group.items():
        init.append((k, v)) ## ('robot0', 'left_arm')

    # for arm_schema in arm_names:
    #     arm_group = schema_arm_to_group.get(arm_schema)
    #     if arm_group is not None:
    #         init.append((arm_schema, arm_group))

    for name in surface_names:
        if name not in schema_name_to_obj:
            continue
        pddl_obj = schema_name_to_obj[name]
        pose = init_poses.get(pddl_obj) or (RelativePose(_schema_obj_to_body(pddl_obj), important=True, **kwargs) if _schema_obj_to_body(pddl_obj) else None)
        if pose and pddl_obj not in init_poses:
            init_poses[pddl_obj] = pose
        if pose:
            init.extend([("Region", pddl_obj), ("AtPose", pddl_obj, pose), ("Pose", pddl_obj, pose)])
        init.append((name, pddl_obj))

    interested_objs = kwargs.get("interested_objs") or {}
    if not interested_objs and skill_yaml_paths:
        for p in skill_yaml_paths:
            _y = load_yaml(p)
            interested_objs.update(_y.get("sg_params", {}).get("interested_objs", {}))
    for name in movable_names:
        if name not in schema_name_to_obj:
            continue
        pddl_obj = schema_name_to_obj[name]
        body = _schema_obj_to_body(pddl_obj)
        pose = init_poses.get(pddl_obj)
        if pose is None and body:
            pose = RelativePose(body, important=True, **kwargs)
            init_poses[pddl_obj] = pose
        obj_attrs = interested_objs.get(name, ["Movable", "Graspable", "CanPick"])
        for attr in obj_attrs:
            init.append((attr, pddl_obj))
        if pose:
            init.extend([("AtPose", pddl_obj, pose), ("Pose", pddl_obj, pose)])
        init.append((name, pddl_obj))

    for name, pddl_obj in schema_name_to_obj.items():
        if name in surface_names:
            continue
        body = _schema_obj_to_body(pddl_obj)
        if body is None:
            continue
        pose = init_poses.get(pddl_obj)
        for surface in stackable:
            surf_name = getattr(surface, "category", None)
            surf_pddl = schema_name_to_obj.get(surf_name, surface)
            if surf_pddl is surface and surface not in init_poses:
                init_poses[surface] = RelativePose(surface, important=True, **kwargs)
            surf_pose = init_poses.get(surf_pddl) or init_poses.get(surface)
            init.extend([("Stackable", pddl_obj, surf_pddl), Equal(("PlaceCost", pddl_obj, surf_pddl), 1),
            #  ("Droppable", pddl_obj, surf_pddl), Equal(("DropCost", pddl_obj, surf_pddl), 1)
            ])
            if pose and surf_pose and is_placement(body, surface):
                init.append(("Supported", pddl_obj, pose, surf_pddl, surf_pose))

    for i, meta in enumerate(classified):
        sk_name = schema_skill_names[i]
        if "LearnedAttach" in meta.get("matched_streams", []):
            init.append(("SkillAttach", sk_name))
        if "LearnedDetach" in meta.get("matched_streams", []):
            init.append(("SkillDetach", sk_name))
        if "LearnedBiKeyPose" in meta.get("matched_streams", []):
            init.append(("Skillbimanual", sk_name))
            if enable_constraints:
                # SkillCheckObj for movable objects NOT manipulated by this skill: scopes the
                # CFreeMDF policy-safety check to potential blocking obstacles only.
                involved = meta.get("involved_objects", set())
                for mname in movable_names:
                    if mname in involved:
                        continue
                    pobj = schema_name_to_obj.get(mname)
                    if pobj is None:
                        continue
                    init.append(("SkillCheckObj", sk_name, pobj))
        init.append((sk_name, sk_name))

    goal = []
    num_arms = len(getattr(robot_entity, "arms", []))
    for sg in skill_goals:
        sk = sg.get("sk")
        if num_arms >= 2 and 'bimanual' not in sk:
            continue
        goal.append(("DoneSkill", sk))
        
    # # # # hardcode goals for screw-handoff-clean
    # goal.append(('On', schema_name_to_obj['cup'], schema_name_to_obj['left_pad']))
    # goal.append(('On', schema_name_to_obj['sponge'], schema_name_to_obj['right_pad']))
    # goal.append(('Holding', schema_name_to_obj['screwdriver']))
    
    table = stackable[0] if stackable else None
    movable_list = [b for b in (_schema_obj_to_body(schema_name_to_obj.get(n)) for n in movable_names if n in schema_name_to_obj) if b is not None]
    fixed_objects = get_fixed([robot_entity], movable_list)
    for name, pddl_obj in schema_name_to_obj.items():
        body = _schema_obj_to_body(pddl_obj)
        if body and "fixed" in getattr(body, "category", ""):
            fixed_objects.append(body)
    for surf in stackable:
        if surf in fixed_objects:
            fixed_objects.remove(surf) ## estimated table will always collide with robot


    # The DMG runner supplies equivSkill_info_dict via kwargs; schema planning
    # cannot bind the learned (diffusion) streams without it. Pop the values we
    # forward explicitly so they don't collide with **stream_kwargs below.
    stream_kwargs = dict(kwargs)
    equivSkill_info_dict = stream_kwargs.pop("equivSkill_info_dict", None)
    posegen_mode = stream_kwargs.pop("posegen_mode", "diffusion")
    if equivSkill_info_dict is None:
        raise ValueError(
            "pddlstream_from_schema_problem requires equivSkill_info_dict "
            "(provided by the DMG runner); none was passed in kwargs."
        )

    stream_map, stream_info = create_hybrid_streams(
        robot_entity, table, obstacles=fixed_objects,
        skill_names=schema_skill_names or [f"sk_{i}" for i in range(len(classified))],
        task_name=task_name or "schema", posegen_mode=posegen_mode,
        instantiated_streams=schema_metadata.get("instantiated_streams", []),
        planning_mode=planning_mode,
        schema_arm_to_group=schema_arm_to_group,
        equivSkill_info_dict=equivSkill_info_dict,
        enable_constraints=enable_constraints,
        **stream_kwargs
    )
    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal), stream_info


def create_hybrid_streams(robot, table, obstacles=[], skill_names = None,  grasp_mode="gpd", task_name = 'screwdriver', posegen_mode = 'diffusion',  verbose = False, equivSkill_info_dict = None, instantiated_streams=None, planning_mode="detailed", enable_constraints=False, **kwargs):

    schema_arm_to_group = kwargs.pop('schema_arm_to_group', None)
    # MDF policy-safety config (only used when enable_constraints): path to the skill's
    # swept-volume field + clearance margin.
    mdf_path = kwargs.pop('mdf_path', None)
    mdf_safety_margin = kwargs.pop('mdf_safety_margin', 0.05)
    static_obstacles = list(obstacles)
    if table is not None and table not in static_obstacles:
        static_obstacles.append(table)

    def _unwrap_schema_args(fn, map_arm=True):
        if schema_arm_to_group is None:
            return fn
        def wrapped(*args, **kw):
            args = list(args)
            for i in range(len(args)):
                if map_arm and schema_arm_to_group and args[i] in schema_arm_to_group:
                    args[i] = schema_arm_to_group[args[i]]
            return fn(*args, **kw)
        return wrapped

    def _wrap_stream_fn(fn, map_arm=True):
        if schema_arm_to_group is None:
            return fn
        return _unwrap_schema_args(fn, map_arm=map_arm)

    stream_map = {
        'test-cfree-pose-pose': from_test(get_test_cfree_pose_pose(**kwargs)),
        'test-cfree-pregrasp-pose': from_test(get_cfree_pregrasp_pose_test(robot)),
        'test-cfree-traj-pose': from_test(get_cfree_traj_pose_test(robot)),

        'sample-placement': from_gen_fn(get_placement_gen_fn(robot, obstacles, environment=obstacles, **kwargs)),
        'plan-learned-pick': from_fn(_wrap_stream_fn(get_learned_pick_fn(robot, environment=static_obstacles, **kwargs))),
        'plan-place': from_fn(_wrap_stream_fn(get_plan_place_fn(robot, environment=obstacles, **kwargs))),
        'plan-drop': from_fn(_wrap_stream_fn(get_plan_drop_fn(robot, environment=obstacles, **kwargs))),
        'plan-motion': from_fn(_wrap_stream_fn(get_plan_motion_fn(robot, environment=static_obstacles, algorithm = "rrt_star", **kwargs))),  #algorithm='lattice', algorithm = "rrt_star",
        'PoseCost': get_pose_cost_fn(robot, **kwargs),
    }



    stream_info = {
        'test-cfree-pose-pose': StreamInfo(p_success=1e-3, eager=False, verbose=verbose),
        'test-cfree-pregrasp-pose': StreamInfo(p_success=1e-2, verbose=verbose),
        'test-cfree-traj-pose':  StreamInfo(p_success=1e-1, verbose=verbose),

        'sample-grasp': StreamInfo(overhead=1e1, opt_gen_fn=PartialInputs(unique=True)),
        'sample-placement': StreamInfo(overhead=1e-1, opt_gen_fn=PartialInputs(unique=True)),

        'plan-pick': StreamInfo(overhead=1e1),
        'plan-learned-pick': StreamInfo(overhead=1e1),
        'plan-place': StreamInfo(overhead=1e1),
        'plan-motion': StreamInfo(overhead=1e2),

        'PoseCost': FunctionInfo(opt_fn=lambda *args: BASE_COST, eager=True),
    }
    if posegen_mode != 'diffusion':
        raise NotImplementedError(
            f"posegen_mode {posegen_mode!r} is no longer supported; only 'diffusion' remains"
        )

    # Reachability + MDF policy-safety bindings, per-task opt-in.
    _mdf_cache = {}

    def _get_mdf_data():
        if 'data' not in _mdf_cache:
            from examples.pybullet.aloha_real.learned_classifier import mdf_construction as _mdfmod
            from pddlstream.utils import get_file_path
            path = mdf_path or 'mdf_data_mj_insertion.pkl'
            if not os.path.isabs(path):
                path = get_file_path(_mdfmod.__file__, path)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"MDF data file not found: {path}. Set sg_params.mdf_path or generate it "
                    f"via mdf_construction.construct_mdf_insertion before enabling use_constraints."
                )
            _mdf_cache['data'] = _mdfmod.load_mdf_dict(path)
        return _mdf_cache['data']

    def _bind_biop_cfree(stream_name):
        # Keypose-free CFreeMDF check; the inlined universal in BiOperation is compiled by
        # universal_to_conditional. Do NOT set eager=True (it suppresses the move-aside skeleton).
        stream_map[stream_name] = from_test(get_mdf_clear_test(_get_mdf_data(), safety_margin=mdf_safety_margin))
        stream_info[stream_name] = StreamInfo(p_success=1e-1, verbose=verbose)

    if enable_constraints:
        stream_map['test-reachable'] = from_test(get_reachability_test(robot))

    ## Learned bimanual / unimanual skills (DMG).
    bi_kw = ['bi', 'bimanual']
    uni_kw = ['pick', 'place', 'grasp']
    bi_skills = []
    uni_skills = []
    for env_key in equivSkill_info_dict.keys():
        skills_learned_in_env = equivSkill_info_dict[env_key]['skill_names']
        for skill_name in skills_learned_in_env:
            skill_kw = find_kw_in_skill(skill_name, uni_kw + bi_kw)
            if skill_kw is None:
                skill_kw = skill_name
            if skill_kw in bi_kw:
                bi_skills.append(skill_name)
                stream_map.update({
                    # qpose and grasp learned from diffusion
                    'sample-biop-keypose': from_gen_fn(get_imitate_traj_fn(
                        robot, equivSkill_info_dict=equivSkill_info_dict, prefix_key=env_key,
                        skill_name=skill_name, fixed_obj=static_obstacles, **kwargs)),
                })
                stream_info.update({
                    'sample-biop-keypose': StreamInfo(overhead=5e1),
                })
            elif skill_kw in uni_kw:
                stream_mode = equivSkill_info_dict[env_key]['obj_centric_mode']
                uni_skills.append(skill_name)
                stream_map.update({
                    f'sample-{skill_kw}-{stream_mode}': from_gen_fn(get_imitate_traj_fn(
                        robot,
                        equivSkill_info_dict=equivSkill_info_dict,
                        prefix_key=env_key,
                        skill_name=skill_name,
                        fixed_obj=static_obstacles,
                        **kwargs,
                    )),
                })
                stream_info.update({
                    f'sample-{skill_kw}-{stream_mode}': StreamInfo(overhead=1e1),
                })
            else:
                raise ValueError(f"Unknown skill type for skill {skill_name!r}")

    def _bind_instantiated_imitate_stream(name, template, skill_name, spec):
        if template not in {"sample-grasp-traj", "sample-place-traj", "sample-biop-keypose"}:
            return False
        env_key = next(
            (
                ek
                for ek, ev in (equivSkill_info_dict or {}).items()
                if skill_name in ev.get("skill_names", [])
            ),
            next(iter(equivSkill_info_dict or {}), None),
        )
        if env_key is None:
            return False
        stream_map[name] = from_gen_fn(get_imitate_traj_fn(
            robot,
            equivSkill_info_dict=equivSkill_info_dict,
            prefix_key=env_key,
            skill_name=skill_name,
            fixed_obj=static_obstacles,
            eff_grasps=spec.get("eff_grasps", []),
            **kwargs,
        ))
        source = template
        if source in stream_info:
            stream_info[name] = stream_info[source]
        else:
            overhead = 5e1 if template == "sample-biop-keypose" else 1e1
            stream_info[name] = StreamInfo(overhead=overhead)
        return True

    # Alias per-skill instantiated stream names to the corresponding registered generators.
    if instantiated_streams:
        grasp_sources = [k for k in stream_map if k.startswith("sample-grasp-") and not k[-1].isdigit()]
        place_sources = [k for k in stream_map if k.startswith("sample-place-") and not k[-1].isdigit()]
        template_to_source = {
            "sample-grasp-traj": grasp_sources[0] if grasp_sources else None,
            "sample-place-traj": place_sources[0] if place_sources else None,
            "sample-biop-keypose": "sample-biop-keypose" if "sample-biop-keypose" in stream_map else None,
        }
        for spec in instantiated_streams:
            name = spec.get("name")
            template = spec.get("template")
            if not name:
                continue

            if enable_constraints and template == "test-cfree-bioperation-pose":
                _bind_biop_cfree(name)
                continue

            if _bind_instantiated_imitate_stream(name, template, spec["skill"], spec):
                continue

            source = template_to_source.get(template)
            if not source or source not in stream_map:
                continue
            stream_map[name] = stream_map[source]
            if source in stream_info:
                stream_info[name] = stream_info[source]

    return stream_map, stream_info
