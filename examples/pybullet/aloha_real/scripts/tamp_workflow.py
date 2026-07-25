"""Shared, route-agnostic planning/execution helpers for the interleaved TAMP routes.

Consumed by the sim/robosuite DMG and real-robot ALOHA plugins.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import time

from examples.pybullet.aloha_real.openworld_aloha.primitives import GroupTrajectory, Sequence
from examples.pybullet.aloha_real.openworld_aloha.run_openworld import (
    compute_TAMP_cmd,
    compute_TAMP_online,
    plan_detail_mp,
)
from examples.pybullet.aloha_real.openworld_aloha.schema_executor import (
    execute_schema_skeleton_plan,
    materialize_ref_goal_result,
)
from examples.pybullet.aloha_real.openworld_aloha.symbolic_utils import (
    get_barrier_action_subgoal,
    get_contact_action_subgoal,
)


@dataclass
class ArmSchedulerState:
    """Tracks remaining per-lane batches and global barriers for the online scheduler."""
    left_remaining: list
    right_remaining: list
    pending_barriers: list
    phase_quotas: list = field(default_factory=list)
    scene_perceived: bool = True

    def current_batch(self, lane):
        remaining = self.left_remaining if lane == 'left' else self.right_remaining
        if not remaining:
            return None
        # Respect phase quota: lane must wait for barrier when its quota is exhausted.
        if self.phase_quotas and self.phase_quotas[0].get(lane, 0) <= 0:
            return None
        return remaining[0]

    def pop_lane_batch(self, lane):
        if lane == 'left':
            if self.left_remaining:
                self.left_remaining.pop(0)
        elif lane == 'right':
            if self.right_remaining:
                self.right_remaining.pop(0)
        else:
            raise ValueError(f"Unknown lane: {lane!r}")
        if self.phase_quotas:
            self.phase_quotas[0][lane] = max(0, self.phase_quotas[0].get(lane, 0) - 1)

    def advance_phase(self):
        """Advance past a completed barrier to the next phase."""
        if self.phase_quotas:
            self.phase_quotas.pop(0)


def split_sequence_per_arm(seq, pause_num=5, sides=None):
    """Split a Sequence into per-arm command queues.

    Appends ``pause_num`` repeated final waypoints so each arm dwells briefly
    at the end of its trajectory before the next command is dispatched.
    """
    if sides is None:
        sides = ['left', 'right']
    per_arm_queue = {side: [] for side in sides}
    for cmd in seq.commands:
        if isinstance(cmd, GroupTrajectory):
            arm_side = cmd.group.split('_')[0]
            if arm_side in per_arm_queue:
                path = list(cmd.path) + [cmd.path[-1]] * pause_num
                ee_path = getattr(cmd, "ee_path", None)
                if ee_path is not None:
                    ee_path = list(ee_path) + [ee_path[-1]] * pause_num
                new_cmd = GroupTrajectory(cmd.robot, cmd.group, path,
                                          attachments=cmd.attachments,
                                          ee_path=ee_path,
                                          ee_link=getattr(cmd, "ee_link", None),
                                          steps_per_waypoint=getattr(cmd, "steps_per_waypoint", 1))
                for attr in ("target_gripper_positions", "freeze_gripper", "joint_space_exec"):
                    if hasattr(cmd, attr):
                        setattr(new_cmd, attr, getattr(cmd, attr))
                per_arm_queue[arm_side].append(new_cmd)
    return per_arm_queue


@dataclass(frozen=True)
class PlanningSession:
    duration_tamp: float
    sequence: Optional[Sequence] = None
    global_solution: Any = None
    state_history: Any = None
    problem: Any = None
    stream_info: Any = None


def preserve_arm_confs(base_state, updated_state, arms, search_facts_fn):
    preserved_state = set(updated_state)
    for arm in arms:
        current_confs = search_facts_fn(preserved_state, 'atconf', fact_args=[arm])
        if current_confs:
            continue
        preserved_state |= search_facts_fn(base_state, 'atconf', fact_args=[arm])
    return preserved_state


def state_delta(base_state, updated_state):
    """Facts (removed, added) between *base_state* and *updated_state*."""
    base, updated = set(base_state), set(updated_state)
    return base - updated, updated - base


def apply_state_delta(state, removed_facts, added_facts):
    """Apply a (removed, added) delta to *state*."""
    return (set(state) - set(removed_facts)) | set(added_facts)


def revert_effects(state, *, added_facts, removed_facts):
    """Undo one unit's symbolic effect: subtract what it added, restore what it removed."""
    return (set(state) - set(added_facts)) | set(removed_facts)


def merge_lane_states(base_state, lane_states):
    """Fold each lane's delta-from-base back onto *base_state*."""
    merged = set(base_state)
    for lane_state in lane_states:
        removed, added = state_delta(base_state, lane_state)
        merged = apply_state_delta(merged, removed, added)
    return merged


def get_action_target_object(action):
    """Return the first object-typed argument of *action*, or None."""
    if action is None or not getattr(action, 'args', None):
        return None
    objs = [arg for arg in action.args if hasattr(arg, 'category')]
    return objs[0] if objs else None


def get_batch_target_object(batch):
    """Return the target object of the last action in *batch*, or None."""
    if not batch.actions:
        return None
    return get_action_target_object(batch.actions[-1])


def plan_offline_sequence(para, robot_entity, belief, **kwargs):
    """Plan a full command ``Sequence`` up front (offline) via schema-mode ``compute_TAMP_cmd``.

    Used by the real-robot plugin's offline path; the online interleaved path uses
    :func:`plan_online_session` instead.
    """
    start_time = time.time()
    sequence = compute_TAMP_cmd(para, robot_entity, belief, **kwargs)
    return PlanningSession(
        duration_tamp=time.time() - start_time,
        sequence=sequence,
    )


def plan_online_session(para, robot_entity, belief, **kwargs):
    start_time = time.time()
    (
        global_solution,
        state_history,
        problem,
        stream_info,
    ) = compute_TAMP_online(
        para, robot_entity, belief, **kwargs)
    return PlanningSession(
        duration_tamp=time.time() - start_time,
        global_solution=global_solution,
        state_history=state_history,
        problem=problem,
        stream_info=stream_info,
    )


def plan_lane_batch_schema(para, robot_entity, static_environment, batch,
                           planning_state, base_state, tamp_kwargs):
    """Schema fast-path: expand a lane batch into an executable Sequence.

    Returns ``(sequence, final_state, subgoal, result)``; ``(None, base_state,
    subgoal, None)`` when schema expansion fails.
    """
    subgoal = get_contact_action_subgoal(batch.ref_goal_state, batch.actions)
    result = execute_schema_skeleton_plan(
        para, robot_entity, static_environment,
        batch.actions, planning_state, **tamp_kwargs,
    )
    if result is None:
        return None, base_state, subgoal, None
    return result.sequence, set(result.final_state), subgoal, result


def replan_subgoal_detail_mp(para, robot_entity, problem, stream_info, static_environment,
                             subgoal, planning_state, base_state, tamp_kwargs,
                             *, batch=None, skeleton_segment=None, label=None):
    """Detail-MP fallback for a subgoal; mutates ``batch.actions`` in place on success.

    ``skeleton_segment`` constrains the solve (DMG passes one, the real route
    passes ``None``). Returns the standard 4-tuple; ``(None, base_state, subgoal,
    None)`` on failure.
    """
    if label is not None:
        print(f'[online_tamp] {label} schema planning failed, retrying with detail MP')
    seq, updated_state, plan_seg = plan_detail_mp(
        para, robot_entity, problem, stream_info, planning_state, subgoal,
        static_environment=static_environment, skeleton_segment=skeleton_segment,
        **tamp_kwargs,
    )
    if seq is None:
        return None, base_state, subgoal, None
    if batch is not None and plan_seg is not None:
        batch.actions = list(plan_seg)
    return seq, set(updated_state), subgoal, None


def plan_barrier_schema(para, robot_entity, static_environment, barrier,
                        planning_state, base_state, tamp_kwargs):
    """Schema fast-path for a bimanual barrier; materialize its effect onto *base_state*."""
    subgoal = get_barrier_action_subgoal(barrier.ref_goal_state, barrier.action)
    result = execute_schema_skeleton_plan(
        para, robot_entity, static_environment,
        [barrier.action], planning_state, **tamp_kwargs,
    )
    if result is None:
        return None, base_state, subgoal, None
    merged = materialize_ref_goal_result(
        base_state=base_state,
        ref_goal_state=barrier.ref_goal_state,
        final_confs=result.final_confs,
        sequence=result.sequence,
    )
    return merged.sequence, set(merged.final_state), subgoal, merged
