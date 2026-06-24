"""Shared OSC_POSE absolute-EE executor for robosuite TAMP planners.

`OSCExecutionMixin` drives a planned command sequence under a single robosuite
`OSC_POSE` *absolute* controller by emitting per-step Cartesian EE targets:

* learned-grasp segments carry a dense `GroupTrajectory.ee_path` and are
  commanded directly;
* every other segment (motion / bridge / gripper-only / generic place) is
  forward-kinematicsed from its joint waypoints to an EE pose at execution time.

It is robot-agnostic: the per-side EE link is read from the robot entity's
manipulator tool frame, so the same mixin serves the dual-Panda and single-Panda
robots. Mix it in *before* the planner base class so its
execution overrides win the MRO::

    class FooOSCPlanner(OSCExecutionMixin, FooPlanner): ...

The robot-specific quantities are `GRIP_SITE_CORRECTION` and
`GRIP_SITE_POS_OFFSET` (see the attribute docstrings); subclasses override them
when their tool frame differs.
"""

import numpy as np
from scipy.spatial.transform import Rotation as _SciRot

from examples.pybullet.aloha_real.openworld_aloha.primitives import Graphstate, GroupTrajectory
from examples.pybullet.aloha_real.scripts.tamp_workflow import split_sequence_per_arm
from examples.pybullet.utils.pybullet_tools.utils import (
    get_link_pose,
    link_from_name,
    set_joint_positions,
)


def _is_frozen(primitive):
    """True if this command holds a grasped object (grip frozen during the move)."""
    return bool(getattr(primitive, 'freeze_gripper', False))


class OSCExecutionMixin:
    """Execute a TAMP plan as absolute OSC_POSE EE targets (see module docstring)."""

    #: Right-multiplied onto every FK'd / learned tool pose to map the pybullet
    #: tool frame onto the robosuite ``grip_site`` frame that OSC_POSE tracks.
    #:
    #: Default is the dual-Panda DMG value: its tool link ``panda*_ee_link``
    #: coincides with the ``right_hand`` frame, and ``grip_site`` is mounted on it
    #: with a -90 deg yaw (PandaGripper root quat ``0.707107 0 0 -0.707107``).
    #: Robots whose tool frame differs (e.g. the single-Panda ``tool_link``)
    #: must override this with their own calibrated rotation.
    GRIP_SITE_CORRECTION = _SciRot.from_euler('z', -np.pi / 2)

    #: Local tool-frame translation added to every commanded position. OSC_POSE
    #: servos robosuite's ``grip_site`` = link8 + 0.0965 m (right_hand at 0.1065
    #: in robot.xml − 0.107 link8 joint + 0.097 eef body in panda_gripper.xml),
    #: while the pybullet tool frames (dual-Panda ``panda*_ee_link``, single-Panda
    #: ``tool_link``) sit at link8 + 0.100 m. Commanding the raw FK position
    #: drives the hand 3.5 mm past the planned pose along the approach axis.
    GRIP_SITE_POS_OFFSET = np.array([0.0, 0.0, -0.0035])

    #: Played back verbatim, the demo gripper channel closes too early:
    #: get_gripper_path interpolates the close between sparse-keypoint run
    #: centers, so the ramp starts well above the contact pose — and OSC_POSE is
    #: a servo, so the EE also lags the targets. Detected closing ramps
    #: (-1 open ... +1 closed) are therefore retimed onto the grasp waypoint
    #: (see _retime_close_ramps): the gripper stays open until the hold
    #: waypoint, the executor settles there, snaps the command closed, and
    #: dwells so the fingers finish before motion resumes.
    GRIPPER_RAMP_EPS = 1e-3      #: per-waypoint command delta that counts as a ramp
    GRIPPER_RAMP_CLOSE_ONLY = True  #: retime only closing ramps; opening plays live
    GRIPPER_SETTLE_STEPS = 10    #: steps converging on the hold waypoint before closing
    GRIPPER_DWELL_STEPS = 15    #: steps holding after the close so the fingers finish

    # ------------------------------------------------------------------
    # Execution loop
    # ------------------------------------------------------------------

    def execute_robosuite_plan(self, seq, stop_mode='replan'):
        """Run `seq` under OSC_POSE abs control, optionally falling back to BC."""
        if stop_mode not in {'replan', 'bc'}:
            raise ValueError(f"Unsupported stop_mode: {stop_mode!r}")

        tamp_cmd_gen = self.iterate_process_sequence(self.robot_entity, seq)
        doing_tamp = True
        self.lfd.update_controllers(controller_name="OSC_POSE", abs_action=True)

        for _ in range(self.lfd.max_timesteps):
            if doing_tamp:
                try:
                    action = next(tamp_cmd_gen)
                    self.replay_tamp_every(action, duration=0.01)
                    if self._effect_monitor_completed():
                        print("Effect monitor completed primitive")
                        return 'replan'
                except StopIteration:
                    if stop_mode == 'replan':
                        print("perceive and plan next subgoal")
                        return 'replan'
                    doing_tamp = False
                    self.lfd.set_bc_controller()
            else:
                self.lfd.inference_once()
                if self._effect_monitor_completed():
                    print("Effect monitor completed primitive")
                    return 'replan'

            if self.lfd.handle_rewards():
                print("Task success!")
                self.success = True
                break
        else:
            print("Task failed!")
            self.success = False

        return 'success' if self.success else 'fail'

    # ------------------------------------------------------------------
    # Plan -> per-step OSC action generator
    # ------------------------------------------------------------------

    def iterate_process_sequence(self, robot_entity, seq):
        """Yield per-step OSC abs actions, 7-D ``[pos(3), axisangle(3), gripper(1)]``
        per active arm (so 7-D single-arm, 14-D bimanual).

        Learned-grasp segments are read from ``GroupTrajectory.ee_path``; all
        other segments are FK'd from their joint waypoints. Idle arms hold their
        last commanded pose until both arms' streams are exhausted.
        """
        # Imported lazily so the module stays importable without robosuite.
        from robosuite.utils.transform_utils import quat2axisangle

        sides = ['left'] if len(self.sg_param['robots']) == 1 else ['left', 'right']

        init_conf = self.lfd.get_cur_jpose_robosuite()
        arm_joints_len = len(robot_entity.joint_groups['left_arm'])
        gripper_joints_len = len(robot_entity.joint_groups['left_gripper'])

        # -------- waypoint dimension helpers --------

        def normalize_waypoint(waypoint):
            for attr in ('positions', 'values'):
                value = getattr(waypoint, attr, None)
                if value is not None:
                    return value
            return waypoint

        def _split_waypoint(waypoint, want):
            """Return the arm or gripper slice of a waypoint, or None if absent.

            `want` is 'arm' or 'gripper'. A waypoint may hold arm-only,
            gripper-only, or concatenated arm+gripper joint values.
            """
            waypoint = normalize_waypoint(waypoint)
            n = len(waypoint)
            if n == arm_joints_len:
                segment = waypoint if want == 'arm' else None
            elif n == gripper_joints_len:
                segment = waypoint if want == 'gripper' else None
            elif n == arm_joints_len + gripper_joints_len:
                segment = (waypoint[:arm_joints_len] if want == 'arm'
                           else waypoint[arm_joints_len:])
            else:
                raise ValueError(
                    f"Unsupported waypoint dimension {n}; expected "
                    f"{arm_joints_len}, {gripper_joints_len}, or "
                    f"{arm_joints_len + gripper_joints_len}"
                )
            return None if segment is None else np.asarray(segment, dtype=float)

        extract_arm_waypoint = lambda wp: _split_waypoint(wp, 'arm')
        extract_gripper_waypoint = lambda wp: _split_waypoint(wp, 'gripper')

        # -------- forward kinematics + OSC slice --------

        # Cache link indices once; link_from_name scans all joints per call.
        ee_link_idx = {
            side: link_from_name(
                robot_entity.robot, robot_entity.manipulators[side][2],
                client=robot_entity.client,
            )
            for side in sides
        }

        def fk_side(side, arm_q):
            """FK the arm joints of `side` to a (pos, quat_xyzw) tool pose."""
            arm_joints = robot_entity.get_group_joints(f"{side}_arm")
            set_joint_positions(robot_entity.robot, arm_joints, arm_q, client=robot_entity.client)
            return get_link_pose(robot_entity.robot, ee_link_idx[side], client=robot_entity.client)

        def pose_to_osc_slice(pose, gripper_scalar):
            """Map a (pos, quat_xyzw) tool pose + gripper scalar to a 7-D OSC slice.

            Applies GRIP_SITE_CORRECTION / GRIP_SITE_POS_OFFSET so the commanded
            pose is the robosuite grip_site frame the OSC controller tracks.
            """
            pos, quat_xyzw = pose
            tool_rot = _SciRot.from_quat(np.asarray(quat_xyzw, dtype=float))
            cmd_pos = np.asarray(pos, dtype=float) + tool_rot.apply(self.GRIP_SITE_POS_OFFSET)
            axisangle = quat2axisangle((tool_rot * self.GRIP_SITE_CORRECTION).as_quat())
            return list(cmd_pos) + list(axisangle) + [float(gripper_scalar)]

        # -------- seed current state per side --------

        current_pose = {side: fk_side(side, init_conf[f'{side}_arm']) for side in sides}
        current_gripper = {
            side: robot_entity.pos2joint_gripper(init_conf[f'{side}_gripper'][0])
            for side in sides
        }
        # Grip command a frozen (AtGrasp) move re-asserts while holding an object.
        # It must be the grasp's last commanded grip — carried across batches via
        # self._carried_gripper_cmd — NOT the live aperture: a grasped object holds
        # the fingers apart, so re-seeding current_gripper from the live qpos maps to
        # a near-neutral scalar that relaxes the grip and drops the object. The
        # fully-closed default only guards the impossible case of a hold preceding
        # any commanded grip (no grasp has run yet on this executor).
        carried_gripper_cmd = getattr(self, '_carried_gripper_cmd', {})
        closed_conf, _ = robot_entity.close_open_conf()
        default_closed = robot_entity.pos2joint_gripper(float(closed_conf[0]))
        frozen_gripper = {
            side: carried_gripper_cmd.get(side, default_closed) for side in sides
        }

        per_arm_queue = split_sequence_per_arm(seq, sides=sides)
        self._bridge_joint_discontinuities(per_arm_queue, sides, extract_arm_waypoint)
        self._append_gripper_transitions(
            per_arm_queue, sides, init_conf, extract_gripper_waypoint
        )

        # -------- per-side step generator --------

        def emit(pose, gripper, steps):
            osc_slice = pose_to_osc_slice(pose, gripper)
            for _ in range(steps):
                yield osc_slice

        def build_side_steps(side):
            last_pose = current_pose[side]
            last_gripper = current_gripper[side]
            for primitive in per_arm_queue[side]:
                path = getattr(primitive, 'refined_qpos', None) or primitive.path
                ee_path = getattr(primitive, 'ee_path', None)
                repeat = max(1, int(getattr(primitive, 'steps_per_waypoint', 1)))
                freeze_gripper = _is_frozen(primitive)

                # Resolve the whole segment's (pose, gripper) targets up front
                # so close ramps can be retimed onto the grasp waypoint.
                poses, grippers = [], []
                pose = last_pose
                # A frozen move holds the grasp's commanded grip for the whole move
                # (the carried live aperture would otherwise relax it); seed it once
                # and leave it untouched below.
                gripper = frozen_gripper[side] if freeze_gripper else last_gripper
                for i, waypoint in enumerate(path):
                    cfg = normalize_waypoint(waypoint)
                    gripper_only = len(cfg) == gripper_joints_len
                    if not freeze_gripper:
                        if gripper_only:
                            gripper = robot_entity.pos2joint_gripper(float(cfg[0]))
                        elif len(cfg) == arm_joints_len + gripper_joints_len:
                            gripper = robot_entity.pos2joint_gripper(float(cfg[-1]))
                    # Update the pose for any waypoint carrying arm dofs. Calling
                    # extract_arm_waypoint also validates the dimension even when
                    # ee_path supplies the pose, so malformed trajectories raise.
                    if not gripper_only:
                        arm_q = extract_arm_waypoint(cfg)
                        if ee_path is not None:
                            pose = ee_path[min(i, len(ee_path) - 1)]
                        elif arm_q is not None:
                            pose = fk_side(side, arm_q)
                    poses.append(pose)
                    grippers.append(float(gripper))

                if freeze_gripper:
                    # No close ramp to retime — the grip is held closed throughout.
                    holds = {}
                else:
                    grippers, holds = self._retime_close_ramps(
                        poses, grippers, last_gripper, ee_path
                    )

                for i, (next_pose, next_gripper) in enumerate(zip(poses, grippers)):
                    if i in holds:
                        # Converge onto the grasp waypoint with the fingers
                        # still open before the close actuates.
                        yield from emit(next_pose, holds[i],
                                        self.GRIPPER_SETTLE_STEPS)
                    last_pose = next_pose
                    last_gripper = next_gripper
                    yield from emit(last_pose, last_gripper, repeat)
                    if i in holds:
                        # Let the fingers finish closing before motion resumes.
                        yield from emit(last_pose, last_gripper,
                                        self.GRIPPER_DWELL_STEPS)

        def initial_hold_gripper(side):
            # Match the first emitted step: a leading frozen move must hold the
            # grasp's commanded grip from frame one, not the re-seeded live aperture.
            for primitive in per_arm_queue[side]:
                if isinstance(primitive, Graphstate):
                    continue
                if _is_frozen(primitive):
                    return frozen_gripper[side]
                return current_gripper[side]
            # No actionable primitive this batch: an idle arm keeps its last
            # commanded grip (carried) so a still-held object isn't relaxed by the
            # live aperture it would otherwise be commanded all batch.
            return carried_gripper_cmd.get(side, current_gripper[side])

        side_gens = {side: build_side_steps(side) for side in sides}
        last_slices = {
            side: pose_to_osc_slice(current_pose[side], initial_hold_gripper(side))
            for side in sides
        }

        def record_carried_gripper():
            # Persist the last commanded grip per side for the next batch's frozen
            # (held-object) move. Recorded before every yield so an early effect-
            # monitor exit that abandons this generator still leaves it current.
            # The OSC slice is [pos(3), axisangle(3), gripper(1)], so [-1] is the grip.
            self._carried_gripper_cmd = {
                side: float(last_slices[side][-1]) for side in sides
            }

        # Initial holding action, then step until every side is exhausted.
        record_carried_gripper()
        yield [v for side in sides for v in last_slices[side]]
        while True:
            advanced = False
            for side in sides:
                try:
                    last_slices[side] = next(side_gens[side])
                    advanced = True
                except StopIteration:
                    pass  # idle arm holds its last slice
            if not advanced:
                break
            record_carried_gripper()
            yield [v for side in sides for v in last_slices[side]]

    def _retime_close_ramps(self, poses, grippers, prev_gripper, ee_path):
        """Move each closing ramp's actuation onto the segment's grasp waypoint.

        get_gripper_path interpolates the demo close between sparse-keypoint
        run centers, so played back verbatim the fingers start closing well
        before the EE reaches the contact pose (and may finish during the
        lift). Each close ramp is rewritten to stay at its pre-ramp (open)
        command until the hold waypoint, then snap to the ramp's final command
        there. The hold waypoint is the deepest pose along the gripper approach
        axis (local +Z toward the fingertips, evaluated from the onset onward)
        for ee_path segments — orientation-generic, so it covers top-down and
        side grasps alike — and the onset itself for segments without poses
        (gripper-only transitions, where the arm holds anyway).

        Returns the rewritten commands and ``{hold_index: pre-ramp command}``
        for the execution loop's settle/dwell holds.
        """
        grippers = list(grippers)
        holds = {}
        prev = float(prev_gripper)
        i = 0
        while i < len(grippers):
            delta = grippers[i] - prev
            ramping = (delta > self.GRIPPER_RAMP_EPS
                       if self.GRIPPER_RAMP_CLOSE_ONLY
                       else abs(delta) > self.GRIPPER_RAMP_EPS)
            if not ramping:
                prev = grippers[i]
                i += 1
                continue
            onset = i
            sign = 1.0 if delta > 0.0 else -1.0
            # Absorb the whole non-decreasing ramp (plateaus included) so the
            # snapped command is the ramp's final value, not a partial one.
            end = onset
            while (end + 1 < len(grippers)
                   and sign * (grippers[end + 1] - grippers[end])
                   >= -self.GRIPPER_RAMP_EPS):
                end += 1
            if ee_path is not None:
                # Deepest waypoint along the approach axis at ramp onset; for a
                # top-down grasp (approach = −z world) this is the lowest pose.
                approach = _SciRot.from_quat(
                    np.asarray(poses[onset][1], dtype=float)
                ).apply([0.0, 0.0, 1.0])
                hold = onset + int(np.argmax([
                    float(np.dot(np.asarray(poses[j][0], dtype=float), approach))
                    for j in range(onset, len(poses))
                ]))
            else:
                hold = onset
            window_end = max(end, hold)
            # Extremal command over the whole rewritten window, not just the
            # first ramp's end: if the hold lies beyond it, a later ramp's
            # fuller close must win over a partial first one.
            window = grippers[onset:window_end + 1]
            target = max(window) if sign > 0.0 else min(window)
            for j in range(onset, hold):
                grippers[j] = prev
            for j in range(hold, window_end + 1):
                grippers[j] = target
            holds[hold] = prev
            prev = target
            i = window_end + 1
        return grippers, holds

    # ------------------------------------------------------------------
    # Plan preprocessing (shared with the base joint-space executor's intent)
    # ------------------------------------------------------------------

    @staticmethod
    def _bridge_joint_discontinuities(per_arm_queue, sides, extract_arm_waypoint):
        """Insert linear arm bridges where consecutive primitives don't meet."""
        for side in sides:
            queue = per_arm_queue[side]
            if not queue:
                continue
            bridged = [queue[0]]
            for primitive in queue[1:]:
                prev_path = getattr(bridged[-1], 'refined_qpos', None) or bridged[-1].path
                cur_path = getattr(primitive, 'refined_qpos', None) or primitive.path
                if len(prev_path) > 0 and len(cur_path) > 0:
                    prev_end = extract_arm_waypoint(prev_path[-1])
                    cur_start = extract_arm_waypoint(cur_path[0])
                else:
                    prev_end = cur_start = None
                if (prev_end is not None and cur_start is not None
                        and not np.allclose(prev_end, cur_start, atol=0.05, rtol=0.0)):
                    n_steps = max(10, int(np.ceil(np.max(np.abs(cur_start - prev_end)) / 0.01)) + 1)
                    bridge_path = np.linspace(prev_end, cur_start, num=n_steps).tolist()
                    bridge = GroupTrajectory(primitive.robot, f'{side}_arm', bridge_path)
                    # A bridge into a held-object move must also hold the grip closed.
                    if _is_frozen(primitive):
                        bridge.freeze_gripper = True
                    bridged.append(bridge)
                bridged.append(primitive)
            per_arm_queue[side] = bridged

    @staticmethod
    def _append_gripper_transitions(per_arm_queue, sides, init_conf, extract_gripper_waypoint):
        """Append explicit gripper segments so commanded grasp/release actuate."""
        current_gripper_positions = {
            side: np.asarray(init_conf[f'{side}_gripper'], dtype=float) for side in sides
        }
        for side in sides:
            queue = per_arm_queue[side]
            if not queue:
                continue
            expanded = []
            for primitive in queue:
                expanded.append(primitive)
                if isinstance(primitive, Graphstate):
                    continue
                path = getattr(primitive, 'refined_qpos', None) or primitive.path
                end_gripper = extract_gripper_waypoint(path[-1]) if len(path) > 0 else None
                target_gripper = getattr(primitive, 'target_gripper_positions', None)
                freeze_gripper = _is_frozen(primitive)
                if target_gripper is not None and not freeze_gripper:
                    target_gripper = np.asarray(target_gripper, dtype=float)
                    if not np.allclose(current_gripper_positions[side], target_gripper):
                        expanded.append(GroupTrajectory(
                            primitive.robot, f'{side}_gripper',
                            [current_gripper_positions[side].tolist(), target_gripper.tolist()],
                        ))
                        current_gripper_positions[side] = target_gripper
                elif end_gripper is not None:
                    current_gripper_positions[side] = end_gripper
            per_arm_queue[side] = expanded
