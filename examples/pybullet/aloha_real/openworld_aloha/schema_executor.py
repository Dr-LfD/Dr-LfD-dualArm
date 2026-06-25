import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from examples.pybullet.aloha_real.openworld_aloha.openworld_streams import (
    TELEPORT,
    get_plan_motion_fn,
    get_plan_pick_fn,
    get_plan_place_fn,
)
from examples.pybullet.aloha_real.openworld_aloha.primitives import Sequence
from examples.pybullet.aloha_real.openworld_aloha.symbolic_utils import search_facts


@dataclass(frozen=True)
class ActionExecutionResult:
    sequence: Sequence
    final_confs: dict
    final_state: set
    added_facts: frozenset = field(default_factory=frozenset)
    removed_facts: frozenset = field(default_factory=frozenset)


_MOTION_FLUENT_PREDICATES = {"atconf", "atpose", "atgrasp", "atattachmentgrasp"}
_TEMPLATE_DIR = Path(__file__).with_name("pddl_templates")
_GENERIC_ACTION_TEMPLATES = {
    "pick": ("action_pick_coarse.pddl", "action_pick.pddl"),
    "place": ("action_place_coarse.pddl", "action_place.pddl"),
}


def _problem_init(problem_or_init):
    if problem_or_init is None:
        return ()
    if isinstance(problem_or_init, (list, tuple)):
        if len(problem_or_init) >= 1 and isinstance(problem_or_init[0], tuple):
            return problem_or_init
        if len(problem_or_init) >= 5:
            return problem_or_init[4] or ()
    if hasattr(problem_or_init, "init"):
        return getattr(problem_or_init, "init") or ()
    return ()


def extract_static_environment(problem_or_init):
    init = _problem_init(problem_or_init)
    movable = {
        fact[1]
        for fact in init
        if len(fact) >= 2 and str(fact[0]).lower() == "movable"
    }
    static_bodies = []
    seen = set()
    for fact in init:
        if len(fact) < 3 or str(fact[0]).lower() != "atpose":
            continue
        body = fact[1]
        if body in movable or body in seen:
            continue
        static_bodies.append(body)
        seen.add(body)
    return static_bodies


def _current_conf_for_arm(current_state, arm):
    matches = [
        fact[2]
        for fact in search_facts(current_state, "atconf", fact_args={arm})
        if len(fact) >= 3 and fact[1] == arm
    ]
    if not matches:
        raise ValueError(f"Missing AtConf for arm {arm!r}")
    return matches[0]


def _replace_arm_conf(state, arm, new_conf):
    updated = {
        fact for fact in state
        if not (len(fact) >= 3 and str(fact[0]).lower() == "atconf" and fact[1] == arm)
    }
    updated.add(("AtConf", arm, new_conf))
    return updated


def _discard_facts(state, facts):
    removal_keys = {
        (str(fact[0]).lower(),) + tuple(fact[1:])
        for fact in facts
        if fact
    }
    return {
        fact
        for fact in state
        if not fact or ((str(fact[0]).lower(),) + tuple(fact[1:])) not in removal_keys
    }



def _project_single_arm_state(current_state, arm, final_conf, added_facts=(), removed_facts=()):
    final_state = _replace_arm_conf(set(current_state), arm, final_conf)
    final_state = _discard_facts(final_state, removed_facts)
    final_state |= set(added_facts)
    return final_state


@lru_cache(maxsize=None)
def _template_parameter_names(template_name):
    template_path = _TEMPLATE_DIR / template_name
    match = re.search(r":parameters\s*\(([^)]*)\)", template_path.read_text())
    if match is None:
        raise ValueError(f"Could not parse :parameters from template {template_path}")
    return tuple(token.lstrip("?") for token in match.group(1).split())


def _generic_action_param_map(action):
    name = action.name.lower()
    template_names = _GENERIC_ACTION_TEMPLATES.get(name)
    if template_names is None:
        raise ValueError(f"Unsupported generic action template lookup for {action}")
    matches = [
        _template_parameter_names(template_name)
        for template_name in template_names
        if len(_template_parameter_names(template_name)) == len(action.args)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Could not match generic action {action} to a unique template by parameter count"
        )
    return dict(zip(matches[0], action.args))


def _generic_action_is_detailed(param_map):
    return "at" in param_map


def _iter_flat_commands(command):
    if isinstance(command, Sequence):
        for child in command.commands:
            yield from _iter_flat_commands(child)
        return
    yield command


def _arm_motion_endpoint_conf(command, arm, endpoint):
    if endpoint not in {"first", "last"}:
        raise ValueError(f"Unsupported motion endpoint {endpoint!r}")
    commands = list(_iter_flat_commands(command))
    if endpoint == "last":
        commands = list(reversed(commands))
    for motion in commands:
        if getattr(motion, "group", None) != arm:
            continue
        getter = getattr(motion, endpoint, None)
        if callable(getter):
            return getter()
    raise ValueError(f"Could not infer {endpoint} arm conf for {arm!r} from motion {command!r}")


def _certified_arm_motion(motion, arm):
    return (
        _arm_motion_endpoint_conf(motion, arm, "first"),
        _arm_motion_endpoint_conf(motion, arm, "last"),
        motion,
    )


def _concat_sequences(*sequences):
    commands = []
    graphstate_markers = []
    for seq in sequences:
        if seq is None:
            continue
        graphstate_markers.extend(
            (len(commands) + marker_index, graphstate)
            for marker_index, graphstate in getattr(seq, "graphstate_markers", ())
        )
        commands.extend(getattr(seq, "commands", []))
    return Sequence(commands, graphstate_markers=graphstate_markers)


def _extract_motion_fluents(current_state, active_arm):
    fluents = []
    for fact in current_state:
        if not fact:
            continue
        predicate = str(fact[0]).lower()
        if predicate not in _MOTION_FLUENT_PREDICATES:
            continue
        if predicate == "atgrasp" and len(fact) >= 4:
            # Preserve both active and passive grasps so collision checking
            # models held objects as attachments rather than world obstacles.
            fluents.append(fact)
            continue
        fluents.append(fact)
    return fluents


def _graphstate_sequence(graph_state):
    if getattr(graph_state, "commands", None) is None:
        graph_state._build_commands()
    commands = list(getattr(graph_state, "commands", ()) or ())
    return Sequence(
        commands,
        name=getattr(graph_state, "skill_name", "bioperation"),
        graphstate_markers=[(0, graph_state)],
    )


def _unwrap_motion_output(output):
    if output is None:
        return None
    if isinstance(output, tuple):
        return output[-1]
    return output


def build_connector_motion(robot_entity, arm, current_conf, target_conf, current_state, static_environment=(), collision_distance=-1):
    if current_conf == target_conf:
        return Sequence([], name=f"idle-{arm}")
    motion_fn = get_plan_motion_fn(
        robot_entity,
        environment=list(static_environment),
        collision_distance=collision_distance,
    )
    motion_fluents = _extract_motion_fluents(current_state, arm)
    prev_teleport = TELEPORT[0]
    TELEPORT[0] = False
    try:
        output = motion_fn(arm, current_conf, target_conf, fluents=motion_fluents)
    finally:
        TELEPORT[0] = prev_teleport
    return _unwrap_motion_output(output)


def _single_arm_result(current_state, arm, final_conf, sequence, added_facts=(), removed_facts=()):
    final_state = _project_single_arm_state(
        current_state,
        arm,
        final_conf,
        added_facts=added_facts,
        removed_facts=removed_facts,
    )
    current_state = set(current_state)
    return ActionExecutionResult(
        sequence=sequence,
        final_confs={arm: final_conf},
        final_state=final_state,
        added_facts=frozenset(final_state - current_state),
        removed_facts=frozenset(current_state - final_state),
    )


def materialize_ref_goal_result(base_state, ref_goal_state, final_confs, sequence):
    final_state = set(ref_goal_state)
    for arm, conf in (final_confs or {}).items():
        final_state = _replace_arm_conf(final_state, arm, conf)
    base_state = set(base_state)
    return ActionExecutionResult(
        sequence=sequence,
        final_confs=dict(final_confs or {}),
        final_state=final_state,
        added_facts=frozenset(final_state - base_state),
        removed_facts=frozenset(base_state - final_state),
    )


def _execute_learned_pick(action, current_state, robot_entity, static_environment, collision_distance=-1):
    arm, obj, skill, pose, payload = action.args[:5]
    connector = build_connector_motion(
        robot_entity, arm, _current_conf_for_arm(current_state, arm),
        payload._aq_start, current_state, static_environment=static_environment,
        collision_distance=collision_distance,
    )
    if connector is None:
        return None
    return _single_arm_result(
        current_state=current_state,
        arm=arm,
        final_conf=payload._aq_end,
        sequence=_concat_sequences(connector, payload._traj_seq),
        added_facts={
            ("AtGrasp", arm, obj, payload),
            ("Holding", obj),
            ("ArmHolding", arm, obj),
            ("HasPicked", obj),
            ("DoneSkill", skill),
        },
        removed_facts={
            ("ArmEmpty", arm),
            ("AtPose", obj, pose),
        },
    )


def _execute_learned_place(action, current_state, robot_entity, static_environment, collision_distance=-1):
    arm, obj, grasp, skill, pose = action.args[:5]
    payload = action.args[-1]
    connector = build_connector_motion(
        robot_entity, arm, _current_conf_for_arm(current_state, arm),
        payload._aq_start, current_state, static_environment=static_environment,
        collision_distance=collision_distance,
    )
    if connector is None:
        return None
    return _single_arm_result(
        current_state=current_state,
        arm=arm,
        final_conf=payload._aq_end,
        sequence=_concat_sequences(connector, payload._traj_seq),
        added_facts={
            ("AtPose", obj, pose),
            ("ArmEmpty", arm),
            ("DoneSkill", skill),
        },
        removed_facts={
            ("AtGrasp", arm, obj, grasp),
            ("ArmHolding", arm, obj),
            ("Holding", obj),
        },
    )


def _execute_generic_place(action, current_state, robot_entity, static_environment, collision_distance=-1):
    param_map = _generic_action_param_map(action)
    arm = param_map["a"]
    grasp = param_map["g"]
    obj = param_map["o"]
    pose = param_map["p"]
    current_conf = _current_conf_for_arm(current_state, arm)
    if _generic_action_is_detailed(param_map):
        start_conf, end_conf, local_seq = _certified_arm_motion(param_map["at"], arm)
    else:
        plan_place = get_plan_place_fn(
            robot_entity,
            environment=list(static_environment),
        )
        output = plan_place(arm, obj, pose, grasp, base_conf=current_conf)
        if output is None:
            return None
        coarse_conf, local_seq = output
        start_conf = coarse_conf
        end_conf = coarse_conf
    pre_connector = build_connector_motion(
        robot_entity,
        arm,
        current_conf,
        start_conf,
        current_state,
        static_environment=static_environment,
        collision_distance=collision_distance,
    )
    if pre_connector is None:
        return None
    added_facts = {
        ("AtPose", obj, pose),
        ("ArmEmpty", arm),
        ("CanMove", arm),
    }
    removed_facts = {
        ("AtGrasp", arm, obj, grasp),
        ("ArmHolding", arm, obj),
        ("Holding", obj),
    }
    placed_state = _project_single_arm_state(
        current_state,
        arm,
        end_conf,
        added_facts=added_facts,
        removed_facts=removed_facts,
    )
    post_connector = build_connector_motion(
        robot_entity,
        arm,
        end_conf,
        current_conf,
        placed_state,
        static_environment=static_environment,
        collision_distance=collision_distance,
    )
    if post_connector is None:
        return None
    return _single_arm_result(
        current_state=current_state,
        arm=arm,
        final_conf=current_conf,
        sequence=_concat_sequences(pre_connector, local_seq, post_connector),
        added_facts=added_facts,
        removed_facts=removed_facts,
    )


def _execute_generic_pick(action, current_state, robot_entity, static_environment, collision_distance=-1):
    param_map = _generic_action_param_map(action)
    arm = param_map["a"]
    grasp = param_map["g"]
    obj = param_map["o"]
    pose = param_map["p"]
    current_conf = _current_conf_for_arm(current_state, arm)
    if _generic_action_is_detailed(param_map):
        start_conf, end_conf, local_seq = _certified_arm_motion(param_map["at"], arm)
    else:
        plan_pick = get_plan_pick_fn(
            robot_entity,
            environment=list(static_environment),
        )
        output = plan_pick(arm, obj, pose, grasp, base_conf=current_conf)
        if output is None:
            return None
        coarse_conf, local_seq = output
        start_conf = coarse_conf
        end_conf = coarse_conf
    connector = build_connector_motion(
        robot_entity,
        arm,
        current_conf,
        start_conf,
        current_state,
        static_environment=static_environment,
        collision_distance=collision_distance,
    )
    if connector is None:
        return None
    return _single_arm_result(
        current_state=current_state,
        arm=arm,
        final_conf=end_conf,
        sequence=_concat_sequences(connector, local_seq),
        added_facts={
            ("AtGrasp", arm, obj, grasp),
            ("CanMove", arm),
            ("Holding", obj),
            ("ArmHolding", arm, obj),
            ("HasPicked", obj),
        },
        removed_facts={
            ("AtPose", obj, pose),
            ("ArmEmpty", arm),
        },
    )


def _execute_bioperation(action, current_state, robot_entity, static_environment, collision_distance=-1):
    arm1 = action.args[0]
    arm2 = action.args[1]
    left_conf = action.args[3]
    right_conf = action.args[4]
    graph_state = action.args[-1]
    left_connector = build_connector_motion(
        robot_entity, arm1, _current_conf_for_arm(current_state, arm1),
        left_conf, current_state, static_environment=static_environment,
        collision_distance=collision_distance,
    )
    right_connector = build_connector_motion(
        robot_entity, arm2, _current_conf_for_arm(current_state, arm2),
        right_conf, current_state, static_environment=static_environment,
        collision_distance=collision_distance,
    )
    if left_connector is None or right_connector is None:
        return None
    final_state = set(current_state)
    final_state = _replace_arm_conf(final_state, arm1, left_conf)
    final_state = _replace_arm_conf(final_state, arm2, right_conf)
    if len(action.args) >= 3:
        final_state.add(("DoneSkill", action.args[2]))
    current_state = set(current_state)
    return ActionExecutionResult(
        sequence=_concat_sequences(left_connector, right_connector, _graphstate_sequence(graph_state)),
        final_confs={arm1: left_conf, arm2: right_conf},
        final_state=final_state,
        added_facts=frozenset(final_state - current_state),
        removed_facts=frozenset(current_state - final_state),
    )


def is_generic_pick_place_action(action):
    return getattr(action, "name", "").lower() in {"pick", "place"}


def contains_generic_pick_place_actions(actions):
    if actions is None:
        return False
    if hasattr(actions, "name"):
        actions = (actions,)
    return any(is_generic_pick_place_action(action) for action in actions)


def execute_schema_action(robot_entity, static_environment, action, current_state, collision_distance=-1):
    name = action.name.lower()
    if name.startswith("learnedpick"):
        return _execute_learned_pick(action, current_state, robot_entity, static_environment, collision_distance=collision_distance)
    if name.startswith("learnedplace"):
        return _execute_learned_place(action, current_state, robot_entity, static_environment, collision_distance=collision_distance)
    if name == "pick":
        return _execute_generic_pick(action, current_state, robot_entity, static_environment, collision_distance=collision_distance)
    if name == "place":
        return _execute_generic_place(action, current_state, robot_entity, static_environment, collision_distance=collision_distance)
    if name.startswith("bioperation_"):
        return _execute_bioperation(action, current_state, robot_entity, static_environment, collision_distance=collision_distance)
    raise ValueError(f"Unsupported schema action for skeleton execution: {action}")


def execute_schema_skeleton_plan(para, robot_entity, static_environment, plan, current_state, collision_distance=-1, **kwargs):
    initial_state = set(current_state)
    state = set(initial_state)
    sequences = []
    final_confs = {}
    for action in plan:
        result = execute_schema_action(
            robot_entity=robot_entity,
            static_environment=static_environment,
            action=action,
            current_state=state,
            collision_distance=collision_distance,
        )
        if result is None:
            return None
        sequences.append(result.sequence)
        final_confs.update(result.final_confs)
        state = set(result.final_state)
    return ActionExecutionResult(
        sequence=_concat_sequences(*sequences),
        final_confs=final_confs,
        final_state=state,
        added_facts=frozenset(state - initial_state),
        removed_facts=frozenset(initial_state - state),
    )
