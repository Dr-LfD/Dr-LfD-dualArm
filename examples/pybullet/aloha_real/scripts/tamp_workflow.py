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
                for attr in ("target_gripper_positions", "freeze_gripper"):
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


@dataclass(frozen=True)
class ExecutionRequest:
    sequence: Sequence
    stop_mode: str


@dataclass(frozen=True)
class ExecutionResult:
    status: str


def preserve_arm_confs(base_state, updated_state, arms, search_facts_fn):
    preserved_state = set(updated_state)
    for arm in arms:
        current_confs = search_facts_fn(preserved_state, 'atconf', fact_args=[arm])
        if current_confs:
            continue
        preserved_state |= search_facts_fn(base_state, 'atconf', fact_args=[arm])
    return preserved_state


def rollback_batch_execution(updated_state, execution_result):
    if execution_result is None:
        return set(updated_state)
    rolled_back = set(updated_state)
    rolled_back -= set(getattr(execution_result, "added_facts", ()))
    rolled_back |= set(getattr(execution_result, "removed_facts", ()))
    return rolled_back


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


def build_lane_checks(batches, subgoals, get_target_object_fn=None):
    target_fn = get_target_object_fn if get_target_object_fn is not None else get_batch_target_object
    lane_checks = {}
    for lane, batch in batches.items():
        lane_subgoal = subgoals.get(lane)
        if not lane_subgoal:
            continue
        lane_checks[lane] = {
            'subgoal': list(lane_subgoal),
            'target_obj': target_fn(batch),
        }
    return lane_checks


def merge_batch_sequences(sequences):
    return Sequence([cmd for seq in sequences for cmd in seq.commands])


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


def infer_stop_mode(sequence):
    return "bc" if getattr(sequence, 'graphstate_markers', ()) else "replan"


def execute_request(executor, request):
    return ExecutionResult(
        status=executor(request.sequence, stop_mode=request.stop_mode)
    )


def build_output_info(task_name, task_success, duration_tamp, recording_path):
    return {
        "recording_path": recording_path,
        "task_success": task_success,
        "duration_tamp": duration_tamp,
        "task_name": task_name,
    }
