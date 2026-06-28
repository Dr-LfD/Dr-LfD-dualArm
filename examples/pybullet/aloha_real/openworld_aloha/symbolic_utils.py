import os
import sys
import re
from dataclasses import dataclass
import numpy as np


EXE_FOLDER = next(p for p in (os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *([os.pardir] * k))) for k in range(16)) if os.path.isfile(os.path.join(p, '.repo_root')))
if EXE_FOLDER not in sys.path:
    sys.path.insert(0, EXE_FOLDER)
os.chdir(EXE_FOLDER)
from pddlstream.language.constants import PDDLProblem
from pddlstream.algorithms.constraints import WILD
from examples.pybullet.aloha_real.openworld_aloha.primitives import (
    Conf, GroupConf, Grasp, Trajectory, GroupTrajectory, Sequence, Command)


def normalize_predicate_names(facts):
    """
    Convert all predicate names in facts to lowercase.
    
    Args:
        facts: List or set of facts (tuples where first element is predicate name)
    
    Returns:
        set: Facts with lowercase predicate names
    """
    normalized_facts = set()
    for fact in facts:
        if isinstance(fact, tuple) and len(fact) > 0:
            # Convert predicate name (first element) to lowercase
            predicate_name = fact[0].lower() if isinstance(fact[0], str) else fact[0]
            normalized_fact = (predicate_name,) + fact[1:]
            normalized_facts.add(normalized_fact)
        elif isinstance(fact, str):
            # Handle string facts (single predicates)
            normalized_facts.add(fact.lower())
        else:
            # Handle other formats (like atoms, etc.)
            normalized_facts.add(fact)
    return normalized_facts

def check_preconditions_hold(state, preconditions):
    """
    Check if all preconditions are satisfied in the current state.
    
    Args:
        state: Current state (set of facts)
        preconditions: Set of precondition facts
    
    Returns:
        bool: True if all preconditions are satisfied
    """
    return preconditions.issubset(state)

def ground_predicate(predicate, key_param_mapping):
    """
    Ground a predicate by mapping symbolic parameters to concrete values and normalize predicate name.
    
    Args:
        predicate: The predicate with symbolic parameters
        key_param_mapping: Dictionary mapping symbolic parameters to concrete values
    
    Returns:
        tuple: Grounded predicate with lowercase predicate name
    """
    grounded_args = tuple(key_param_mapping.get(x, x) for x in predicate.args)
    predicate_name = predicate.predicate.lower()  # Normalize to lowercase
    return (predicate_name,) + grounded_args

def map_symb_to_pred(action_instance, action_params):
    """
    Map symbolic action instance to grounded predicates for preconditions and effects.
    
    Args:
        action_instance: The action instance with symbolic parameters
        action_params: The grounded parameters from the action plan
    
    Returns:
        tuple: (precondition_set, added_effect_set, removed_effect_set)
    """
    action_placeholders = list(action_instance.var_mapping.values())
    key_param_mapping = {k: param for k, param in zip(action_placeholders, action_params)}
    precondition = set()
    added_effect = set()
    removed_effect = set()

    # Process preconditions
    for abstract_precond in action_instance.precondition:
        grounded_predicate = ground_predicate(abstract_precond, key_param_mapping)
        precondition.add(grounded_predicate)

    # Process add effects
    for effect in action_instance.add_effects:
        grounded_predicate = ground_predicate(effect[1], key_param_mapping)
        added_effect.add(grounded_predicate)

    # Process delete effects
    for effect in action_instance.del_effects:
        grounded_predicate = ground_predicate(effect[1], key_param_mapping)
        removed_effect.add(grounded_predicate)

    print(f"Precondition: {precondition}")
    # print(f"Added effects: {added_effect}")
    # print(f"Removed effects: {removed_effect}")

    return precondition, added_effect, removed_effect


def simulate_plan_execution(initial_state, action_plan, action_instances, preimage_facts=None):
    """
    Simulate the execution of a plan step by step, tracking state changes.
    
    Args:
        domain: PDDL domain
        initial_state: Initial state facts
        action_plan: List of action instances
        preimage_facts: Optional preimage facts for validation
    
    Returns:
        List of states after each action
    """
    # Start with initial state
    current_state = normalize_predicate_names(initial_state)
    state_history = [set(current_state)]  # Track state after each step
    
    print(f"Initial state: {len(current_state)} facts (normalized to lowercase)")
    if preimage_facts:
        print(f"Preimage facts: {len(preimage_facts)} facts")
        preimage_facts = normalize_predicate_names(preimage_facts)
        print(f"Preimage facts normalized: {len(preimage_facts)} facts")
    
        missing_facts = preimage_facts - current_state
        if missing_facts:
            print(f"Warning: Missing preimage facts: {len(missing_facts)} facts")
    
    # Apply each action step by step
    for step, action_instance in enumerate(action_instances):
        print(f"\nStep {step}: Applying action {action_instance.name}")

        # Map symbolic action to grounded predicates
        precondition, added_effect, removed_effect = map_symb_to_pred(action_instance, action_plan[step].args)
        
        # Check if action is applicable
        if not check_preconditions_hold(current_state, precondition):
            print (f"Missing preconditions: {precondition - current_state}, make sure it is Derived")
        
        # Apply the action effects manually
        # Remove delete effects
        if removed_effect:
            print(f"Removing effects: {removed_effect}")
            assert removed_effect.issubset(current_state), f"Attempting to remove facts not in current state: {removed_effect - current_state}"
            current_state -= removed_effect
        
        # Add add effects
        if added_effect:
            print(f"Adding effects: {added_effect}")
            assert not added_effect.intersection(current_state), f"Attempting to add facts already in current state: {added_effect.intersection(current_state)}"
            current_state |= added_effect
        
        # Record the new state
        state_history.append(set(current_state))
        
        print(f"State after action: {len(current_state)} facts")
    
    return state_history




def update_problem(problem, current_state, subgoal):
    domain_pddl, constant_map, stream_pddl, stream_map, init, goal = problem


    # added_effects = subgoal - current_state
    # # removed_effects = current_state - subgoal
    new_init = list(current_state)
    new_goal = subgoal
 
    print(f"Current initial states: {new_init}")
    print(f"Current goal state: {new_goal}")

    new_problem = PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, new_init, new_goal)
    return new_problem



def search_facts(fact_set, fact_name, fact_args=None):
    """
    Search facts by predicate name (regex) and optional arguments (exact membership).

    - fact_name: string treated as a regex (case-insensitive) or a compiled regex.
    - fact_args: iterable of objects; match if any object appears among the fact's arguments.

    Returns:
      - If fact_args is None: list of matching facts
      - Else: set of matching facts
    """
    # Compile predicate name regex
    if isinstance(fact_name, re.Pattern):
        name_re = fact_name
    else:
        name_re = re.compile(str(fact_name), re.IGNORECASE)

    same_name_facts = [fact for fact in fact_set if isinstance(fact, tuple) and fact and name_re.fullmatch(str(fact[0]).lower())]

    if fact_args is None:
        return same_name_facts

    facts_to_return = set()
    for fact in same_name_facts:
        if set(fact[1:]).intersection(fact_args):
            facts_to_return.add(fact)
    return facts_to_return


def filter_subgoal(ref_goal_state, current_state):
    added_effects = ref_goal_state - current_state
    lfd_subgoal = [fact for fact in ref_goal_state if 'doneskill' in fact[0]]
    return lfd_subgoal


def _arm_to_lane_name(arm_name):
    """Map an arm identifier (string or Object wrapper) to 'left' or 'right'."""
    value = getattr(arm_name, 'value', arm_name)
    if value is None:
        return None
    name = str(value).lower()
    if 'left' in name or 'robot0' in name:
        return 'left'
    if 'right' in name or 'robot1' in name:
        return 'right'
    return None


def action_invalidates_perception(action):
    """Return True for actions that make the cached scene observation stale."""
    name = action.name.lower()
    return name.startswith(
        ('learnedpick', 'learnedplace', 'pick', 'place', 'bioperation_')
    )


def is_perception_reset_action(action):
    """Backward-compatible alias for stale-scene invalidating actions."""
    return action_invalidates_perception(action)


@dataclass
class LaneActionBatch:
    actions: list
    ref_goal_state: set


@dataclass
class GlobalBarrier:
    action: object
    ref_goal_state: set


def split_lane_actions_into_batches(lane_action_refs):
    """Chunk a lane's actions into batches ending at the first contact-changing action."""
    batches = []
    current_actions = []
    current_ref_goal_state = None

    for action, ref_goal_state in lane_action_refs:
        current_actions.append(action)
        current_ref_goal_state = ref_goal_state
        if action_invalidates_perception(action):
            batches.append(
                LaneActionBatch(
                    actions=list(current_actions),
                    ref_goal_state=current_ref_goal_state,
                )
            )
            current_actions = []
            current_ref_goal_state = None

    if current_actions:
        batches.append(
            LaneActionBatch(
                actions=list(current_actions),
                ref_goal_state=current_ref_goal_state,
            )
        )
    return batches


def build_scheduler_batches(global_plan, state_history):
    """Build per-lane batches and explicit global barriers from a solved plan.

    Actions are grouped into *phases* separated by barriers.  Phase *i*
    contains the lane actions between ``barriers[i-1]`` and ``barriers[i]``.
    The returned ``phase_quotas`` list records how many batches each lane is
    allowed to serve in each phase, so the scheduler can block a lane from
    running ahead of its next barrier.
    """
    # Split global plan into segments at barrier boundaries.
    segments = []          # list of {"left": [...], "right": [...]}
    global_barriers = []
    current_segment = {"left": [], "right": []}

    for idx, action in enumerate(global_plan):
        ref_goal_state = state_history[idx + 1]
        if action.name.startswith('bioperation_'):
            segments.append(current_segment)
            global_barriers.append(GlobalBarrier(action=action, ref_goal_state=ref_goal_state))
            current_segment = {"left": [], "right": []}
            continue
        if not getattr(action, 'args', None):
            raise ValueError(f"Unsupported scheduler action without args: {action}")
        lane = _arm_to_lane_name(action.args[0])
        if lane not in current_segment:
            raise ValueError(f"Cannot classify scheduler action into lane: {action.name}, args={action.args}")
        current_segment[lane].append((action, ref_goal_state))

    segments.append(current_segment)  # final segment (after last barrier)

    # Build batches per phase and record per-lane quotas.
    all_left = []
    all_right = []
    phase_quotas = []

    for segment in segments:
        left_batches = split_lane_actions_into_batches(segment["left"])
        right_batches = split_lane_actions_into_batches(segment["right"])
        all_left.extend(left_batches)
        all_right.extend(right_batches)
        phase_quotas.append({"left": len(left_batches), "right": len(right_batches)})

    return {
        "left": all_left,
        "right": all_right,
        "barriers": global_barriers,
        "phase_quotas": phase_quotas,
    }


def _extract_skill_name_from_action(action, ref_goal_state=None):
    """Return the runtime skill name for learned actions, else None.

    Supports both the one-conf and two-conf learned-pick/place signatures by
    matching candidate action arguments against DoneSkill facts when available.
    """
    name = action.name.lower()
    if name.startswith("bioperation_"):
        if len(action.args) <= 2:
            raise ValueError(f"Action missing expected skill argument at index 2: {action}")
        return getattr(action.args[2], 'value', action.args[2])

    if not (name.startswith("learnedpick") or name.startswith("learnedplace")):
        return None

    candidate_skills = {
        getattr(fact[1], 'value', fact[1])
        for fact in (ref_goal_state or [])
        if 'doneskill' in str(fact[0]).lower() and len(fact) >= 2
    }
    for arg in action.args:
        value = getattr(arg, 'value', arg)
        if value in candidate_skills:
            return value

    fallback_indices = (2, 3)
    for index in fallback_indices:
        if len(action.args) > index:
            value = getattr(action.args[index], 'value', action.args[index])
            if isinstance(value, str):
                return value
    return None


def _extract_place_subgoal(ref_goal_state, action):
    """Return the minimal refinement subgoal for a generic place action."""
    if len(action.args) < 5:
        raise ValueError(
            f"Place action missing required args for discrete subgoal extraction: {action}"
        )
    arm = getattr(action.args[0], 'value', action.args[0])
    obj = getattr(action.args[2], 'value', action.args[2])
    support = getattr(action.args[4], 'value', action.args[4])
    return [
        ('On', obj, support),
        ('ArmEmpty', arm),
    ]


def _extract_pick_subgoal(ref_goal_state, action):
    """Return the minimal refinement subgoal for a generic pick action."""
    if len(action.args) < 4:
        return []
    arm = getattr(action.args[0], 'value', action.args[0])
    grasp = getattr(action.args[1], 'value', action.args[1])
    obj = getattr(action.args[2], 'value', action.args[2])
    matching = []
    for fact in ref_goal_state:
        predicate = str(fact[0]).lower()
        if predicate == 'atgrasp' and len(fact) >= 4:
            fact_arm = getattr(fact[1], 'value', fact[1])
            fact_obj = getattr(fact[2], 'value', fact[2])
            fact_grasp = getattr(fact[3], 'value', fact[3])
            if fact_arm == arm and fact_obj == obj and fact_grasp == grasp:
                matching.append(fact)
        elif predicate == 'armholding' and len(fact) >= 3:
            fact_arm = getattr(fact[1], 'value', fact[1])
            fact_obj = getattr(fact[2], 'value', fact[2])
            if fact_arm == arm and fact_obj == obj:
                matching.append(fact)
        elif predicate in {'holding', 'haspicked'} and len(fact) >= 2:
            fact_obj = getattr(fact[1], 'value', fact[1])
            if fact_obj == obj:
                matching.append(fact)
    return matching


def get_contact_action_subgoal(ref_goal_state, actions):
    """Return the DoneSkill* fact for the contact-aware action ending this batch."""
    if not actions:
        return []
    terminal_action = actions[-1]
    if not is_perception_reset_action(terminal_action):
        return []
    if terminal_action.name.lower() == "pick":
        return _extract_pick_subgoal(ref_goal_state, terminal_action)
    if terminal_action.name.lower() == "place":
        return _extract_place_subgoal(ref_goal_state, terminal_action)
    skill_name = _extract_skill_name_from_action(terminal_action, ref_goal_state=ref_goal_state)
    if skill_name is None:
        return []
    matching = [
        fact for fact in ref_goal_state
        if 'doneskill' in str(fact[0]).lower()
        and len(fact) >= 2
        and getattr(fact[1], 'value', fact[1]) == skill_name
    ]
    if not matching:
        raise ValueError(
            f"Could not find DoneSkill fact for lane batch ending at skill '{skill_name}'. "
            f"Ref goal state was: {sorted(ref_goal_state, key=str)}"
        )
    return matching


def get_barrier_action_subgoal(ref_goal_state, barrier_action):
    """Return the DoneSkill* fact for a barrier action such as bioperation_*."""
    skill_name = _extract_skill_name_from_action(barrier_action, ref_goal_state=ref_goal_state)
    if skill_name is None:
        raise ValueError(f"Barrier action missing skill argument: {barrier_action}")
    matching = [
        fact for fact in ref_goal_state
        if 'doneskill' in str(fact[0]).lower()
        and len(fact) >= 2
        and getattr(fact[1], 'value', fact[1]) == skill_name
    ]
    if not matching:
        raise ValueError(
            f"Could not find DoneSkill fact for barrier action skill '{skill_name}'. "
            f"Ref goal state was: {sorted(ref_goal_state, key=str)}"
        )
    return matching


_GEOMETRIC_TYPES = (Conf, GroupConf, Grasp, Trajectory, GroupTrajectory, Sequence, Command)


def _is_pose_tuple(value):
    """Heuristic: a pose is a 2-tuple of (3-float-tuple, 4-float-tuple)."""
    if not (isinstance(value, tuple) and len(value) == 2):
        return False
    pos, quat = value
    if not (isinstance(pos, (tuple, list)) and isinstance(quat, (tuple, list))):
        return False
    return len(pos) == 3 and len(quat) == 4


def _to_skeleton_arg(arg):
    """Convert a plan action arg to a skeleton constant (raw value) or WILD."""
    value = getattr(arg, 'value', arg)
    if isinstance(value, _GEOMETRIC_TYPES) or _is_pose_tuple(value):
        return WILD
    return value


def convert_plan_to_skeleton(plan_actions):
    """
    Convert a list of Action(name, args) to PlanConstraints skeleton format.

    Identity args (arms, objects, skills) become raw values so they survive
    reset_globals() — to_obj() re-registers them fresh each planning session.
    Geometric args (confs, grasps, poses, trajectories) become WILD so they
    are re-sampled from updated perception.
    """
    return [
        (action.name, tuple(_to_skeleton_arg(arg) for arg in action.args))
        for action in plan_actions
    ]


def check_effect_achieved(skill_meta, sensor_data, env_type):
    """
    Determine if a skill's expected contact graph change was achieved.

    Args:
        skill_meta: classified skill dict with 'matched_streams', 'grounding_arm',
                    and 'effect_detection' (required, from DMG config).
        sensor_data: {
            'eef_xyz':      {arm_name: np.array},   # end-effector positions
            'obj_pose':     np.array or None,         # object centre (sim)
            'gripper_vals': {arm_name: float},        # encoder values (real)
            'obj_visible':  bool,
            'contact_predictor': object with predict_effect(skill_meta),
        }
        env_type: 'sim' or 'real'

    Returns:
        bool: True if effect state detected
    """
    streams = skill_meta.get('matched_streams', [])
    det = skill_meta.get('effect_detection') or {}
    skill_label = skill_meta.get('skill_name', str(skill_meta))

    def _require(key):
        if key not in det:
            raise ValueError(
                f"effect_detection['{key}'] missing for skill '{skill_label}'. "
                f"Set it in the DMG config's effect_detection dict."
            )
        return det[key]

    if det.get('backend') == 'contact_predictor':
        predictor = sensor_data.get('contact_predictor')
        if predictor is None:
            raise ValueError(
                f"sensor_data['contact_predictor'] missing for skill '{skill_label}'."
            )
        return predictor.predict_effect(skill_meta)

    if 'LearnedAttach' in streams:
        arm = skill_meta.get('grounding_arm') or _require('arm')
        if env_type == 'real':
            thresh = _require('gripper_close_threshold')
            return sensor_data['gripper_vals'][arm] < thresh
        else:
            thresh = _require('obj_eef_dist_threshold')
            return bool(np.linalg.norm(
                np.asarray(sensor_data['obj_pose']) - np.asarray(sensor_data['eef_xyz'][arm])
            ) < thresh)

    if 'LearnedDetach' in streams:
        # Object not visible after a place/release → detach succeeded
        if not sensor_data.get('obj_visible', True):
            return True
        arm = skill_meta.get('grounding_arm') or _require('arm')
        if env_type == 'real':
            thresh = _require('gripper_open_threshold')
            return sensor_data['gripper_vals'][arm] > thresh
        else:
            thresh = _require('obj_eef_dist_threshold')
            return bool(np.linalg.norm(
                np.asarray(sensor_data['obj_pose']) - np.asarray(sensor_data['eef_xyz'][arm])
            ) > thresh)

    if 'LearnedBiKeyPose' in streams:
        arm1 = skill_meta.get('grounding_arm1')
        arm2 = skill_meta.get('grounding_arm2')
        if not arm1 or not arm2:
            raise ValueError(
                f"grounding_arm1/grounding_arm2 missing for bimanual skill '{skill_label}'."
            )
        def _compare(value, threshold, comparison, key_name):
            if comparison not in ('lt', 'gt'):
                raise ValueError(
                    f"Invalid effect_detection['{key_name}']={comparison} "
                    f"for skill '{skill_label}'. Use 'lt' or 'gt'."
                )
            return value < threshold if comparison == 'lt' else value > threshold

        checks = []
        if det.get('hand_hand_dist_threshold') is not None:
            hand_dist = np.linalg.norm(
                np.asarray(sensor_data['eef_xyz'][arm1]) - np.asarray(sensor_data['eef_xyz'][arm2])
            )
            checks = [
                (
                    hand_dist,
                    det['hand_hand_dist_threshold'],
                    det['hand_hand_dist_comparison'],
                    'hand_hand_dist_comparison',
                ),
            ]

        if det.get('left_gripper_close_threshold') is not None:
            checks.append(
                (
                    sensor_data['gripper_vals'][arm1],
                    det['left_gripper_close_threshold'],
                    det['left_gripper_comparison'],
                    'left_gripper_comparison',
                )
            )
        if det.get('right_gripper_open_threshold') is not None:
            checks.append(
                (
                    sensor_data['gripper_vals'][arm2],
                    det['right_gripper_open_threshold'],
                    det['right_gripper_comparison'],
                    'right_gripper_comparison',
                )
            )

        return all(_compare(value, threshold, cmp, key_name) for value, threshold, cmp, key_name in checks)

    raise ValueError(
        f"Unknown matched_streams {streams} for skill '{skill_label}'. Cannot detect effect."
    )


def effect_monitor_sensor_data(lfd, robot_entity, contact_predictor=None, target_obj=None):
    """Build the observation dict expected by ``check_effect_achieved`` / ``BiopCompletionMonitor.update``."""
    sensor_data = {}
    if lfd is not None and hasattr(lfd, "get_cur_eef_xyz_robosuite"):
        raw = lfd.get_cur_eef_xyz_robosuite()
        left = np.asarray(raw["left_arm"])
        right = np.asarray(raw["right_arm"])
        sensor_data["eef_xyz"] = {
            "left_arm": left,
            "right_arm": right,
            "robot0": left,
            "robot1": right,
        }
    # Gripper values: the real ROS Controller publishes l/r_gripper_val, but the
    # sim SimulatedController never sets them — so for sim read the live robosuite
    # gripper qpos through the LfD wrapper and map it with the same
    # pos2joint_gripper scale the OSC executor commands. (eef_xyz above is sourced
    # the same capability-based way.)
    ctrl = robot_entity.controller if robot_entity is not None else None
    if ctrl is not None and getattr(ctrl, "l_gripper_val", None) is not None:
        sensor_data["gripper_vals"] = {
            "left_arm": ctrl.l_gripper_val,
            "right_arm": ctrl.r_gripper_val,
        }
    elif lfd is not None and hasattr(lfd, "get_cur_jpose_robosuite") and robot_entity is not None:
        jpose = lfd.get_cur_jpose_robosuite()
        sensor_data["gripper_vals"] = {
            "left_arm": float(robot_entity.pos2joint_gripper(jpose["left_gripper"][0])),
            "right_arm": float(robot_entity.pos2joint_gripper(jpose["right_gripper"][0])),
        }
    if target_obj is not None and getattr(target_obj, "observed_pose", None) is not None:
        sensor_data["obj_pose"] = np.asarray(target_obj.observed_pose[0])
    else:
        sensor_data["obj_pose"] = None
    sensor_data["obj_visible"] = (target_obj is not None) and (
        getattr(target_obj, "body", None) is not None
    )
    sensor_data["contact_predictor"] = contact_predictor
    return sensor_data


class BiopCompletionMonitor:
    """
    Real-time bimanual-operation (biop) completion monitor.

    Supports two completion modes:
      1) monitor_switch_t: debounce effect predicate over consecutive ticks.
      2) timeout_t: complete at/after timeout. If extra effect predicates are
         configured, timeout acts as a gate and completion also requires
         check_effect_achieved(...) to be True.
    """
    def __init__(self, skill_meta, env_type):
        det = skill_meta.get('effect_detection') or {}
        skill_label = skill_meta.get('skill_name', str(skill_meta))
        self.switch_t = det.get('monitor_switch_t')
        self.timeout_t = det.get('timeout_t')
        self.timeout_requires_effect = (
            self.timeout_t is not None
            and any(key not in {'timeout_t', 'monitor_switch_t'} for key in det.keys())
        )

        self.skill_meta = skill_meta
        self.env_type = env_type
        if self.switch_t is None and self.timeout_t is None:
            raise ValueError(
                f"effect_detection for skill '{skill_label}' must include either "
                "'monitor_switch_t' or 'timeout_t'."
            )
        if self.switch_t is not None and self.switch_t <= 0:
            raise ValueError(
                f"effect_detection['monitor_switch_t'] must be > 0 for skill '{skill_label}'."
            )
        if self.timeout_t is not None and self.timeout_t <= 0:
            raise ValueError(
                f"effect_detection['timeout_t'] must be > 0 for skill '{skill_label}'."
            )
        self.count = 0
        self.total_updates = 0

    def update(self, sensor_data):
        """Returns True when effect held for switch_t ticks or timeout reached."""
        if self.switch_t is not None:
            if check_effect_achieved(self.skill_meta, sensor_data, self.env_type):
                self.count += 1
            else:
                self.count = 0
            if self.count >= self.switch_t:
                return True
        if self.timeout_t is not None:
            self.total_updates += 1
            if self.total_updates >= self.timeout_t:
                if self.timeout_requires_effect:
                    return check_effect_achieved(self.skill_meta, sensor_data, self.env_type)
                return True
        return False


def _generic_pick_subgoal_achieved(target_obj, eef_xyz, distance_threshold=0.12):
    """Return True if any arm EEF is within distance_threshold of target_obj's pose."""
    if target_obj is None or getattr(target_obj, 'body', None) is None:
        return False
    observed_pose = getattr(target_obj, 'observed_pose', None)
    if observed_pose is None:
        return False
    obj_xyz = np.asarray(observed_pose[0])
    if obj_xyz.shape[-1] != 3:
        return False
    return any(
        np.linalg.norm(obj_xyz - np.asarray(arm_xyz)) <= distance_threshold
        for arm_xyz in eef_xyz.values()
    )


class PrimitiveSubgoalDetector:
    """Abstract base for post-execution learned-primitive success verification.

    Called after a pick/place batch completes to decide whether the expected
    symbolic subgoal was achieved in the physical world.  Returns a per-lane
    bool map; False triggers local replanning.
    """

    def __init__(self, schema_skill_metas):
        self.schema_skill_metas = schema_skill_metas

    def detect(self, lane_checks, **kwargs):
        """Return {lane: achieved} for each lane. Must be overridden."""
        raise NotImplementedError


class SimPrimitiveSubgoalDetector(PrimitiveSubgoalDetector):
    """MuJoCo sim detector using eef-obj distance and check_effect_achieved."""

    def __init__(self, schema_skill_metas, lfd, contact_predictor=None):
        super().__init__(schema_skill_metas)
        self.lfd = lfd
        self.contact_predictor = contact_predictor

    def detect(self, lane_checks, affected_objects=None, **kwargs):
        eef_xyz_raw = self.lfd.get_cur_eef_xyz_robosuite()
        eef_xyz = {'robot0': np.asarray(eef_xyz_raw['left_arm'])}
        if 'right_arm' in eef_xyz_raw:
            eef_xyz['robot1'] = np.asarray(eef_xyz_raw['right_arm'])

        obj_by_category = (
            {getattr(o, 'category', None): o for o in affected_objects}
            if affected_objects else {}
        )

        def resolve_target_obj(candidate):
            if candidate is None or not obj_by_category:
                return candidate
            return obj_by_category.get(getattr(candidate, 'category', None), candidate)

        lane_results = {}
        for lane, lane_check in lane_checks.items():
            subgoal = lane_check.get('subgoal', [])
            doneskill_facts = [f for f in subgoal if 'doneskill' in str(f[0]).lower()]
            target_obj = resolve_target_obj(lane_check.get('target_obj'))
            if not doneskill_facts:
                generic_pick_facts = [
                    fact for fact in subgoal
                    if str(fact[0]).lower() in {'atgrasp', 'armholding', 'holding', 'haspicked'}
                ]
                if generic_pick_facts:
                    lane_results[lane] = _generic_pick_subgoal_achieved(target_obj, eef_xyz)
                else:
                    lane_results[lane] = True
                continue
            lane_achieved = True
            for doneskill_fact in doneskill_facts:
                skill_name = getattr(doneskill_fact[1], 'value', doneskill_fact[1])
                skill_meta = self.schema_skill_metas.get(skill_name) if skill_name else None
                if skill_meta is None or not skill_meta.get('effect_detection'):
                    # No effect_detection configured — treat primitive as achieved.
                    achieved = True
                else:
                    streams = skill_meta.get('matched_streams', [])
                    det = skill_meta.get('effect_detection') or {}
                    requires_target_obj = (
                        ('LearnedBiKeyPose' not in streams)
                        and det.get('backend') != 'contact_predictor'
                    )
                    if target_obj is None and requires_target_obj:
                        lane_achieved = False
                        continue
                    sensor_data = {
                        'eef_xyz': eef_xyz,
                        'obj_pose': np.asarray(target_obj.observed_pose[0]) if target_obj else None,
                        'obj_visible': (target_obj is not None) and (target_obj.body is not None),
                        'contact_predictor': self.contact_predictor,
                    }
                    achieved = check_effect_achieved(skill_meta, sensor_data, env_type='sim')
                lane_achieved = lane_achieved and achieved
            lane_results[lane] = lane_achieved
        return lane_results


class DummyPrimitiveSubgoalDetector(PrimitiveSubgoalDetector):
    """Always-true detector — real robot sensor pipeline not yet validated."""

    def detect(self, lane_checks, **kwargs):
        return {lane: True for lane in lane_checks}


def merge_primitive_effect_detection(skill_meta, effect_detection_defaults):
    """Merge per-skill effect_detection with YAML defaults (same keys as ContactPredictorWrapper)."""
    det = dict(skill_meta.get("effect_detection") or {})
    defaults = dict(effect_detection_defaults or {})
    if "checkpoint" not in det and defaults.get("contact_predictor_checkpoint"):
        det["checkpoint"] = defaults["contact_predictor_checkpoint"]
    for key in (
        "sam3_worker_path",
        "sam3_path",
        "sam3_model_dir",
        "sam3_checkpoint",
        "sam3_conda_env",
        "sam3_conda_bin",
        "sam_path",
        "seg_branch",
        "contact_wrist_seg_backend",
        "text_prompt",
        # "contact_label_threshold",
    ):
        if key not in det and key in defaults:
            det[key] = defaults[key]
    return det


def skill_requires_contact_predictor_checkpoint(skill_meta, effect_detection_defaults):
    """True iff merged config is contact_predictor with a non-empty checkpoint string."""
    det = merge_primitive_effect_detection(skill_meta, effect_detection_defaults)
    if det.get("backend") != "contact_predictor":
        return False
    ckpt = det.get("checkpoint")
    return bool(ckpt and str(ckpt).strip())


class UnifiedPrimitiveSubgoalDetector(PrimitiveSubgoalDetector):
    """Real post-primitive verification: contact_predictor when merged checkpoint exists; else dummy per skill.

    Non-contact ``effect_detection`` is delegated to ``check_effect_achieved`` with live
    ``eef_xyz`` / ``gripper_vals`` when ``lfd`` / ``robot_entity`` are provided.
    """

    def __init__(
        self,
        schema_skill_metas,
        *,
        contact_predictor=None,
        effect_detection_defaults=None,
        env_type="real",
        lfd=None,
        robot_entity=None,
    ):
        super().__init__(schema_skill_metas)
        self.contact_predictor = contact_predictor
        self.effect_detection_defaults = dict(effect_detection_defaults or {})
        self.env_type = env_type
        self.lfd = lfd
        self.robot_entity = robot_entity

    def _build_sensor_data(self, target_obj):
        return effect_monitor_sensor_data(
            self.lfd,
            self.robot_entity,
            contact_predictor=self.contact_predictor,
            target_obj=target_obj,
        )

    def detect(self, lane_checks, affected_objects=None, **kwargs):
        obj_by_category = (
            {getattr(o, "category", None): o for o in affected_objects}
            if affected_objects
            else {}
        )

        def resolve_target_obj(candidate):
            if candidate is None or not obj_by_category:
                return candidate
            return obj_by_category.get(getattr(candidate, "category", None), candidate)

        eef_xyz = (self._build_sensor_data(None).get("eef_xyz") or {})

        lane_results = {}
        for lane, lane_check in lane_checks.items():
            subgoal = lane_check.get("subgoal", [])
            doneskill_facts = [f for f in subgoal if "doneskill" in str(f[0]).lower()]
            target_obj = resolve_target_obj(lane_check.get("target_obj"))
            if not doneskill_facts:
                generic_pick_facts = [
                    fact
                    for fact in subgoal
                    if str(fact[0]).lower() in {"atgrasp", "armholding", "holding", "haspicked"}
                ]
                if generic_pick_facts:
                    lane_results[lane] = _generic_pick_subgoal_achieved(target_obj, eef_xyz)
                else:
                    lane_results[lane] = True
                continue

            lane_achieved = True
            for doneskill_fact in doneskill_facts:
                skill_name = getattr(doneskill_fact[1], "value", doneskill_fact[1])
                skill_meta = self.schema_skill_metas.get(skill_name) if skill_name else None
                if skill_meta is None or not skill_meta.get("effect_detection"):
                    achieved = True
                    lane_achieved = lane_achieved and achieved
                    continue

                merged = merge_primitive_effect_detection(skill_meta, self.effect_detection_defaults)
                streams = skill_meta.get("matched_streams", [])
                requires_target_obj = (
                    ("LearnedBiKeyPose" not in streams)
                    and merged.get("backend") != "contact_predictor"
                )
                if target_obj is None and requires_target_obj:
                    lane_achieved = False
                    continue

                if merged.get("backend") == "contact_predictor":
                    if not skill_requires_contact_predictor_checkpoint(
                        skill_meta, self.effect_detection_defaults
                    ):
                        achieved = True
                    else:
                        if self.contact_predictor is None:
                            raise ValueError(
                                f"Skill '{skill_name}' requires contact_predictor (merged checkpoint is set) "
                                "but no ContactPredictorWrapper was provided to UnifiedPrimitiveSubgoalDetector."
                            )
                        achieved = self.contact_predictor.predict_effect(skill_meta)
                    lane_achieved = lane_achieved and achieved
                    continue

                sensor_data = self._build_sensor_data(target_obj)
                achieved = check_effect_achieved(skill_meta, sensor_data, self.env_type)
                lane_achieved = lane_achieved and achieved

            lane_results[lane] = lane_achieved
        return lane_results
