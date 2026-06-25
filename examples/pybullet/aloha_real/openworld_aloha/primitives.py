from itertools import chain

import numpy as np
from examples.pybullet.utils.pybullet_tools.retime import interpolate_path, sample_curve
from examples.pybullet.utils.pybullet_tools.utils import (
    BASE_LINK,
    INF,
    Attachment,
    Ray,
    State,
    WorldSaver,
    STATIC_MASS,
    get_mass,
    add_fixed_constraint,
    add_segments,
    adjust_path,
    draw_pose,
    empty_sequence,
    flatten,
    get_bodies,
    get_closest_points,
    get_fixed_constraints,
    get_joint_positions,
    get_pose,
    invert,
    is_fixed_base,
    link_from_name,
    multiply,
    pose_from_pose2d,
    remove_constraint,
    remove_handles,
    set_joint_positions,
    set_pose,
    unit_pose,
    wait_if_gui,
    wait_for_duration,
    waypoints_from_path
)

from examples.pybullet.aloha_real.openworld_aloha.grasping import control_until_contact, get_pregrasp
# from examples.pybullet.aloha_real.openworld_aloha.simulation.control import follow_path, stall_for_duration, step_curve
from examples.pybullet.aloha_real.openworld_aloha.entities import WORLD_BODY, ParentBody

DRAW_Z = 1e-2
USE_CONSTRAINTS = True
LEAD_CONTROLLER = True



class RelativePose(object):  # TODO: BodyState, RigidAttachment
    # Extends RelPose from SS-Replan
    # Attachment
    def __init__(
        self,
        body,
        parent=None,
        parent_state=None,
        relative_pose=None,
        important=False,
        client=None,
        **kwargs
    ):
        self.body = body
        self.client = client
        # if parent is WORLD_BODY:
        #     parent = ParentBody()
        self.parent = parent
        self.parent_state = parent_state
        if not isinstance(self.body, int):
            self.body = int(str(self.body).split("#")[1])
        if relative_pose is None:
            relative_pose = multiply(
                invert(self.get_parent_pose()), get_pose(self.body)
            )
        self.relative_pose = tuple(relative_pose)
        self.important = important  # TODO: plan harder when true
        # self.initial = False # TODO: initial

    @property
    def value(self):
        return self.relative_pose

    def ancestors(self):
        if self.parent_state is None:
            return [self.body]
        return self.parent_state.ancestors() + [self.body]

    # def ancestors(self):
    #     if self.parent_state is None:
    #         return []
    #     return self.parent_state.ancestors() + [self.parent_state.body]
    def get_parent_pose(self):
        if self.parent is WORLD_BODY:
            return unit_pose()
        if self.parent_state is not None:
            self.parent_state.assign()
        return self.parent.get_pose()

    def get_pose(self):
        return multiply(self.get_parent_pose(), self.relative_pose)

    def update_pose(self, world_pose):
        """Recompute ``relative_pose`` so ``get_pose()`` matches *world_pose*."""
        old_pos = self.relative_pose
        parent_pose = self.get_parent_pose()
        self.relative_pose = tuple(multiply(invert(parent_pose), world_pose))
        print(f'old_pos is {old_pos}, updated pose is {self.relative_pose}')

    def assign(self):
        world_pose = self.get_pose()
        set_pose(self.body, world_pose)
        return world_pose

    def draw(self):
        raise NotImplementedError()

    def get_attachment(self):
        assert self.parent is not None
        parent_body, parent_link = self.parent
        # self.assign()
        # return create_attachment(parent_body, parent_link, self.body)
        return Attachment(
            parent_body, parent_link, self.relative_pose, self.body, client=self.client
        )

    def __repr__(self):
        name = "wp" if self.parent is WORLD_BODY else "rp"
        return "{}{}".format(name, id(self) % 1000)





class Grasp(object):  # RelativePose
    def __init__(
        self, body, grasp, pregrasp=None, closed_position=0.0, client=None, phase = None, \
            tool_dist=None, obj_dist=None, **kwargs
    ):
        # TODO: condition on a gripper (or list valid pairs)
        self.body = body
        self.grasp = grasp
        self.client = client
        self.phase = phase
        if tool_dist is None or obj_dist is None: 
            if body.category == "colObs":
                tool_dist = 0.15 #0.09
                obj_dist = 0.05 #0.03
            elif body.category == "cup":
                tool_dist = 0.09
                obj_dist = 0.02
            else:
                tool_dist = 0.15
                obj_dist = 0.05
        if pregrasp is None:
            pregrasp = get_pregrasp(grasp, tool_distance= tool_dist, object_distance=obj_dist, **kwargs)
        self.pregrasp = pregrasp
        self.closed_position = closed_position  # closed_positions

    @property
    def value(self):
        return self.grasp

    @property
    def approach(self):
        return self.pregrasp

    def create_relative_pose(self, robot, link=BASE_LINK):  # create_attachment
        parent = ParentBody(body=robot, link=link, client=self.client)
        return RelativePose(
            self.body, parent=parent, relative_pose=self.grasp, client=self.client
        )

    def create_attachment(self, *args, **kwargs):
        # TODO: create_attachment for a gripper
        relative_pose = self.create_relative_pose(*args, **kwargs)
        return relative_pose.get_attachment()

    def __repr__(self):
        return "g{}".format(id(self) % 1000)


class LearnedGrasp(Grasp):
    """Payload object produced by an imitation/diffusion skill.

    Reuses the grasp/contact container so learned skills can carry their
    execution payload through PDDL as a single object. The payload caches
    start/end confs plus the trajectory sequence for later execution.
    """
    def __init__(self, body, contact_pose, aq_start, aq_end, traj_seq, **kwargs):
        super().__init__(body, contact_pose, **kwargs)
        self._aq_start = aq_start
        self._aq_end = aq_end
        self._traj_seq = traj_seq

    def __repr__(self):
        return "lg{}".format(id(self) % 1000)




class Conf(object):  # TODO: parent class among Pose, Grasp, and Conf
    # TODO: counter
    def __init__(
        self, body, joints, positions=None, important=False, client=None, **kwargs
    ):
        # TODO: named conf
        self.body = body
        self.joints = joints
        self.client = client
        if positions is None:
            positions = get_joint_positions(self.body, self.joints)
        self.positions = tuple(positions)
        self.important = important
        # TODO: parent state?

    @property
    def robot(self):
        return self.body

    @property
    def values(self):
        return self.positions

    def assign(self):
        set_joint_positions(self.body, self.joints, self.positions)

    def iterate(self):
        yield self

    def __repr__(self):
        return "q{}".format(id(self) % 1000)


class GroupConf(Conf):
    def __init__(self, body, group, *args, **kwargs):
        joints = body.get_group_joints(group)
        super(GroupConf, self).__init__(body, joints, *args, **kwargs)
        self.group = group

    def __repr__(self):
        return "{}q{}".format(self.group[0], id(self) % 1000)


#######################################################


class WorldState(State):
    def __init__(self, savers=[], attachments={}, client=None):
        # a part of the state separate from PyBullet
        # TODO: other fluent things
        super(WorldState, self).__init__(attachments)
        self.world_saver = WorldSaver(client=client)
        self.savers = tuple(savers)
        self.client = client

    def assign(self):
        self.world_saver.restore()
        for saver in self.savers:
            saver.restore()
        self.propagate()

    def copy(self):  # update
        return self.__class__(savers=self.savers, attachments=self.attachments)

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, list(self.savers), sorted(self.attachments)
        )


# #######################################################


class Command(object):
    # def __init__(self, state=[]):
    #    self.state = tuple(state)

    def switch_client(self):
        raise NotImplementedError

    @property
    def context_bodies(self):
        return set()

    def iterate(self, state, **kwargs):
        raise NotImplementedError()

    def controller(self, *args, **kwargs):
        raise NotImplementedError()

    def execute(self, controller, *args, **kwargs):
        # raise NotImplementedError()
        return True


class BaseSwitch(Command):
    def __init__(self, body, parent=None, client=None, **kwargs):

        self.body = body
        self.parent = parent
        self.client = client

    def iterate(self, state, **kwargs):
        if self.parent is WORLD_BODY and self.body in state.attachments.keys():
            del state.attachments[self.body]
        elif self.parent is not None:
            relative_pose = RelativePose(
                self.body, parent=self.parent, client=self.client
            )
            state.attachments[self.body] = relative_pose

        return empty_sequence()

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.body)


class Switch(Command):
    def __init__(self, body, parent=None):
        self.body = body
        self.parent = parent

    def switch_client(self, robot):
        return Switch(self.body, parent=self.parent)

    def iterate(self, state, **kwargs):
        if self.parent is WORLD_BODY and self.body in state.attachments.keys():
            del state.attachments[self.body]
        elif self.parent is not None:
            robot, tool_link = self.parent
            gripper_group = None
            for _, (_, gripper_group, tool_name) in robot.manipulators.items():
                if link_from_name(robot, tool_name, client=robot.client) == tool_link:
                    break
            else:
                raise RuntimeError(tool_link)
            gripper_joints = robot.get_group_joints(gripper_group)
            finger_links = robot.get_finger_links(gripper_joints)

            movable_bodies = [
                body for body in get_bodies(client=robot.client) if (body != robot)
            ]
 
            max_distance = 5e-2
            collision_bodies = [
                body
                for body in movable_bodies
                if (all(
                    get_closest_points(
                        robot,
                        body,
                        link1=link,
                        max_distance=max_distance,
                        client=robot.client,
                    )
                    for link in finger_links
                )  and get_mass(body, client=robot.client) != STATIC_MASS )
            ]

            if len(collision_bodies) > 0:
                relative_pose = RelativePose(
                    collision_bodies[0], parent=self.parent, client=robot.client
                )
                state.attachments[self.body] = relative_pose

        return empty_sequence()

    def controller(self, use_constraints=USE_CONSTRAINTS, **kwargs):
        if not use_constraints:
            return  # empty_sequence()
        if self.parent is WORLD_BODY:
            # TODO: record the robot and tool_link
            for constraint in get_fixed_constraints():
                remove_constraint(constraint)
        else:
            robot, tool_link = self.parent
            gripper_group = None
            for group, (
                arm_group,
                gripper_group,
                tool_name,
            ) in robot.manipulators.items():
                if link_from_name(robot, tool_name) == tool_link:
                    break
            else:
                raise RuntimeError(tool_link)
            gripper_joints = robot.get_group_joints(gripper_group)
            finger_links = robot.get_finger_links(gripper_joints)

            movable_bodies = [
                body
                for body in get_bodies(client=self.robot.client)
                if (body != robot) and not is_fixed_base(body, client=self.robot.client)
            ]
            # collision_bodies = [body for body in movable_bodies if any_link_pair_collision(
            #    robot, finger_links, body, max_distance=1e-2)]

            gripper_width = robot.get_gripper_width(gripper_joints)
            max_distance = gripper_width / 2.0
            collision_bodies = [
                body
                for body in movable_bodies
                if all(
                    get_closest_points(
                        robot, body, link1=link, max_distance=max_distance
                    )
                    for link in finger_links
                )
            ]
            for body in collision_bodies:
                # TODO: improve the PR2's gripper force
                add_fixed_constraint(body, robot, tool_link, max_force=None)
        # TODO: yield for longer
        yield

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.body)


class Wait(Command):
    def __init__(self, duration):
        self.duration = duration

    def iterate(self, state, **kwargs):
        return empty_sequence()
        # yield relative_pose

    def controller(self, *args, **kwargs):
        raise NotImplementedError('wait controller is not implemented')
        # return stall_for_duration(duration=self.duration)
        # return hold_for_duration(robot, duration=self.duration)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.duration)

class Trajectory(Command):
    def __init__(
        self,
        robot,
        joints,
        path,
        velocity_scale=1.0,
        contact_links=[],
        time_after_contact=INF,
        contexts=[],
        client=None,
        **kwargs
    ):
        self.robot = robot
        self.client = client
        self.joints = joints
        self.path = tuple(path)  # waypoints_from_path
        self.velocity_scale = velocity_scale
        self.contact_links = tuple(contact_links)
        self.time_after_contact = time_after_contact
        self.contexts = tuple(contexts)


    @property
    def context_bodies(self):
        return {self.robot} | {
            context.body for context in self.contexts if hasattr(context, "body")
        }

    def conf(self, positions):
        return Conf(self.robot, self.joints, positions=positions, client=self.client)

    def arm_conf(self, positions, arm_dof=6):
        return Conf(self.robot, self.joints[:arm_dof], positions=positions[:arm_dof], client=self.client)

    def first(self):
        return self.conf(self.path[0])
    
    def first_arm_conf(self, arm_dof=6):
        return self.arm_conf(self.path[0], arm_dof=arm_dof)

    def last(self):
        return self.conf(self.path[-1])
    
    def last_arm_conf(self, arm_dof=6):
        return self.arm_conf(self.path[-1], arm_dof=arm_dof)

    def reverse(self):
        return self.__class__(
            self.robot,
            self.joints,
            self.path[::-1],
            velocity_scale=self.velocity_scale,
            contact_links=self.contact_links,
            time_after_contact=self.time_after_contact,
            contexts=self.contexts,
        )  # , **self.kwargs)

    def draw(self, only_waypoints=True, **kwargs):
        path = waypoints_from_path(self.path) if only_waypoints else self.path
        handles = []
        if self.group == "base":
            handles.extend(
                draw_pose(pose_from_pose2d(base_conf, z=DRAW_Z), length=5e-2, **kwargs)
                for base_conf in path
            )
        return handles

    def adjust_path(self):
        # if len(self.path) <= 2:
        #     print('debug here')        
        current_positions = get_joint_positions(
            self.robot, self.joints
        )  # Important for adjust_path
        if np.isnan(current_positions).any():
            current_positions = self.path[0]
            print('Warning: pybullet get_joint_positions failed')
        return adjust_path(
            self.robot,
            self.joints,
            [current_positions] + list(self.path),
            client=self.client,
        )  # Accounts for the wrap around

    def compute_waypoints(self):
        return waypoints_from_path(
            adjust_path(self.robot, self.joints, self.path)
        )

    def compute_curve(self, draw=False, verbose=False, **kwargs):
        path = self.adjust_path()
        # path = self.compute_waypoints()
        # TODO: error when fewer than 2 points
        if np.isnan(path).any():
            raise ValueError("Invalid path")

        positions_curve = interpolate_path(
            self.robot, self.joints, path, client=self.client
        )

        if verbose:
            print(
                "Following {} {}-DOF waypoints in {:.3f} seconds".format(
                    len(path), len(self.joints), positions_curve.x[-1]
                )
            )
        if not draw:
            return positions_curve
        handles = []
        if self.group == "base":
            # TODO: color by derivative magnitude or theta
            handles.extend(
                add_segments(
                    np.append(q[:2], [DRAW_Z])
                    for t, q in sample_curve(positions_curve, time_step=10.0 / 60)
                )
            )
        wait_if_gui()
        remove_handles(handles)
        return positions_curve

    def traverse(self):
        # TODO: traverse from an initial conf?
        for positions in self.path:
            set_joint_positions(self.robot, self.joints, positions)
            yield positions

    def iterate(self, state, teleport=False, record_refined = False,  **kwargs):
        if(teleport):
            set_joint_positions(self.robot, self.joints, self.path[-1])
            return self.path[-1]
        else:
            ## add the refined trajectory for mujoco
            jpos_gen =  step_curve(
                self.robot,
                self.joints,
                self.compute_curve(client=self.client, **kwargs),
                client=self.client,
            )
            # # https://chatgpt.com/c/67aefb81-480c-8007-8926-7b242192b914
            # if record_refined:
            #     self.refined_qpos = list(jpos_gen)  # 消耗生成器
            #     return iter(self.refined_qpos)  # 重新创建一个生成器返回
            # else:
            #     return jpos_gen

            ## https://chat.deepseek.com/a/chat/s/37cf9add-f9f9-4942-b3f8-91b4f0df4a97
            import itertools
            jpos_gen, record_gen = itertools.tee(jpos_gen)
            # if record refined traj, the intermediate motion will not be executed, as elements are consumed by the generator
            if record_refined:
                self.refined_qpos = list(record_gen)
            return jpos_gen

    def controller(self, *args, **kwargs):
        raise NotImplementedError("traj controller is not implemented")
        waypoints = self.compute_waypoints()
        if LEAD_CONTROLLER:
            lead_step = 5e-2 * self.velocity_scale
            velocity_scale = None
        else:
            lead_step = None
            velocity_scale = 5e-1  # None | 5e-1
        controller = follow_path(
            self.robot,
            self.joints,
            waypoints,
            lead_step=lead_step,
            velocity_scale=velocity_scale,
            max_force=None,
        )  # None | 1e6
        # **self.kwargs)
        # return controller
        return control_until_contact(
            controller, self.robot, self.contact_links, self.time_after_contact
        )

    def execute(self, controller, *args, **kwargs):
        raise NotImplementedError()

    def __repr__(self):
        return "t{}".format(id(self) % 1000)

class GroupTrajectory(Trajectory):
    def __init__(self, robot, group, path, attachments=[], *args,
                 ee_path=None, ee_link=None, steps_per_waypoint=1, **kwargs):
        joints = robot.get_group_joints(group)
        super(GroupTrajectory, self).__init__(robot, joints, path, *args, **kwargs)
        self.group = group
        self.attachments = attachments
        self.executed = False
        self.ee_path = ee_path
        self.ee_link = ee_link
        self.steps_per_waypoint = int(steps_per_waypoint)

    def set_executed(self, val):
        self.executed = val

    def get_executed(self):
        return self.executed

    def copy(self):
        return GroupTrajectory(self.robot, self.group, self.path,
                               attachments=self.attachments,
                               ee_path=self.ee_path, ee_link=self.ee_link,
                               steps_per_waypoint=self.steps_per_waypoint)

    def switch_client(self, robot):
        return GroupTrajectory(robot, self.group, self.path, client=robot.client,
                               ee_path=self.ee_path, ee_link=self.ee_link,
                               steps_per_waypoint=self.steps_per_waypoint)

    def conf(self, positions):
        return GroupConf(self.robot, self.group, positions=positions, client=self.client)

    def execute(self, controller, *args, **kwargs):
        self.robot.update_conf()
        velocity_fraction = 0.2
        velocity_fraction *= self.velocity_scale
        positions_curve = self.compute_curve(
            velocity_fraction=velocity_fraction
        )  # TODO: assumes the PyBullet robot is up-to-date
        times, positions = zip(*sample_curve(positions_curve, time_step=1e-1))
        # = np.array(times) / self.velocity_scale
        print(
            "\nGroup: {} | Positions: {} | Duration: {:.3f}\nStart: {}\nEnd: {}".format(
                self.group, len(positions), times[-1], positions[0], positions[-1]
            )
        )
        controller.command_group_trajectory(
            self.group, positions, times, blocking=True, **kwargs
        )
        controller.wait(duration=1.0)
        self.robot.update_conf()
        # return True
        if self.group in self.robot.gripper_groups:  # Never abort after gripper movement
            return True
        return not controller.any_arm_fully_closed()

    def reverse(self):
        return self.__class__(
            self.robot,
            self.group,
            self.path[::-1],
            velocity_scale=self.velocity_scale,
            contact_links=self.contact_links,
            time_after_contact=self.time_after_contact,
            contexts=self.contexts,
            client=self.client,
        )  # , **self.kwargs)

    def __repr__(self):
        return "{}t{}".format(self.group[0], id(self) % 1000)
        

#######################################################


class Sequence(Command):  # Commands, CommandSequence
    def __init__(self, commands=[], name=None, graphstate_markers=None):
        self.context = None  # TODO: make a State?
        self.commands = tuple(commands)
        self.graphstate_markers = tuple(graphstate_markers or [])
        self.name = self.__class__.__name__.lower()[:3] if (name is None) else name

    def switch_client(self, robot):
        return Sequence(
            [command.switch_client(robot) for command in self.commands],
            name=self.name,
            graphstate_markers=[
                (marker_index, graphstate.switch_client(robot))
                for marker_index, graphstate in self.graphstate_markers
            ],
        )

    @property
    def context_bodies(self):
        return set(flatten(command.context_bodies for command in self.commands))

    def __len__(self):
        return len(self.commands)

    def iterate(self, *args, **kwargs):
        for command in self.commands:
            print("Executing {} command: {}".format(type(command), str(command)))
            for output in command.iterate(*args, **kwargs):
                yield output

    def controller(self, *args, **kwargs):
        return chain.from_iterable(
            command.controller(*args, **kwargs) for command in self.commands
        )

    def execute(self, *args, return_executed=False, **kwargs):
        executed = []
        for command in self.commands:
            if not command.execute(*args, **kwargs):
                return False, executed if return_executed else False
            executed.append(command)
        return True, executed if return_executed else True

    def reverse(self):
        if self.graphstate_markers:
            raise RuntimeError("Sequence.reverse() does not support graphstate_markers")
        return Sequence(
            [command.reverse() for command in reversed(self.commands)], name=self.name
        )

    def dump(self):
        print("[{}]".format(" -> ".join(map(repr, self.commands))))

    def __repr__(self):
        return "{}({})".format(self.name, len(self.commands))

class GripperAssigner(Command):
    def __init__(self,  parent,  grasp = None, pregrasp=None, **kwargs):
        self.grasp = grasp
        if self.grasp is not None:
            self.body = grasp.body
        else:
            self.body = None
        self.parent = parent

    def iterate(self, state, **kwargs):
        if self.parent is WORLD_BODY and len(state.attachments.keys()):
            assert self.grasp is None

            state.attachments = {}

        elif self.parent is not None:
            robot, tool_link = self.parent
            # gripper_group = None
            # for _, (_, gripper_group, tool_name) in robot.manipulators.items():
            #     if link_from_name(robot, tool_name, client=robot.client) == tool_link:
            #         break
            # else:
            #     raise RuntimeError(tool_link)

 

            relative_pose = RelativePose(
                self.grasp.body, parent=self.parent, client=robot.client
            )
            # revise relative pose to grasp pose
            relative_pose.relative_pose = self.grasp.grasp
            state.attachments[self.body] = relative_pose

        return empty_sequence()
    
    def __repr__(self):
        return "gr{}".format(id(self) % 1000)

class Graphstate(Command):
    def __init__(self, robot, sg_dict, skill_name = None) -> None:
        self.robot = robot
        self.skill_name = skill_name    
        self.sg_dict = sg_dict
        self.commands = None  # built lazily in iterate()

    def switch_client(self, robot):
        return Graphstate(robot, self.sg_dict, skill_name=self.skill_name)

    def copy(self):
        return Graphstate(self.robot, {'pre_sg': None, 'eff_sg': None}, skill_name = self.skill_name)

    def _build_commands(self):
        """Build commands lazily so that grasps written to eff_sg by other streams are available."""
        self.commands = []
        robot = self.robot
        sg_dict = self.sg_dict

        pre_sg = sg_dict['pre_sg']
        eff_sg = sg_dict['eff_sg']
        if eff_sg is None:
            return
        ## check if edges in pre_sg remains in eff_sg
        for rbt_name in sg_dict['related_rbts']:
            hand_name = robot.rbt_ids_to_side[rbt_name]
            pre_out_nodes = set(pre_sg.successors(hand_name))
            eff_out_nodes = set(eff_sg.successors(hand_name))
            graspped_objs = eff_out_nodes - pre_out_nodes
            for graspped_obj in graspped_objs:
                grasp = eff_sg.edges[hand_name, graspped_obj]['grasp']
                obj = grasp.body
                side = hand_name.split('_')[0]
                arm_group, gripper_group, tool_name = robot.manipulators[side]

                closed_conf = grasp.closed_position * np.ones(
                    len(self.robot.get_group_joints(gripper_group))
                )                
                
                gripper_traj = GroupTrajectory(
                    robot,
                    gripper_group,
                    path=[closed_conf],
                    # contexts=[pose],
                    contact_links=robot.get_finger_links(robot.get_group_joints(gripper_group)),
                    time_after_contact=1e-1,
                    client=robot.client,
                    attachments = [obj],
                )
                grasp_cmd = GripperAssigner(
                    grasp = grasp, 
                    parent=ParentBody(
                        body=robot, link=robot.link_from_name(tool_name), client=robot.client
                    ),
                )
                self.commands.append(grasp_cmd)
                self.commands.append(gripper_traj)

            released_objs = pre_out_nodes - eff_out_nodes
            for released_obj in released_objs:
                release_cmd = GripperAssigner(parent=WORLD_BODY)
                self.commands.append(release_cmd)

                side = hand_name.split('_')[0]
                arm_group, gripper_group, tool_name = robot.manipulators[side]

                close_conf, open_conf = robot.close_open_conf()
                gripper_traj = GroupTrajectory(
                    robot,
                    gripper_group,
                    path=[open_conf],
                    # contexts=[pose],
                    contact_links=robot.get_finger_links(robot.get_group_joints(gripper_group)),
                    time_after_contact=1e-1,
                    client=robot.client,
                    attachments = [released_obj],
                )
                self.commands.append(gripper_traj)

    def iterate(self, state, **kwargs):
        if self.commands is None:
            self._build_commands()
        print('Execute LfD!')
        # construct grasp, close gripper, assign the obj pose
        for command in self.commands:
            print("Executing {} command: {}".format(type(command), str(command)))
            for output in command.iterate(state, **kwargs):
                yield output

    def __repr__(self):
        return "sg{}".format(id(self) % 1000)

def map_schema_plan_args(plan, robot):
    """
    Map schema arm constants (robot0, robot1) to group names (left_arm, right_arm) and
    resolve PDDLStream Object wrappers to .value in plan action args. Use when plan comes
    from schema-based problem and execution expects group names and perceived bodies.
    """
    if plan is None or not hasattr(robot, 'get_arm_group'):
        return plan
    from pddlstream.language.constants import Action
    from pddlstream.language.object import Object as PDDLObject
    mapped = []
    for name, args in plan:
        if not args:
            mapped.append((name, args))
            continue
        new_args = []
        for a in args:
            if hasattr(robot, 'get_arm_group') and isinstance(a, str) and getattr(robot, 'rbt_ids_to_side', None) and a in robot.rbt_ids_to_side:
                new_args.append(robot.get_arm_group(a))
            elif isinstance(a, PDDLObject):
                new_args.append(a.value)
            else:
                new_args.append(a)
        mapped.append(Action(name, tuple(new_args)))
    return mapped


def post_process(plan, robot_entity=None):
    if plan is None:
        return None

    def append_command(command, flattened_commands, graphstate_markers):
        if isinstance(command, Graphstate):
            if command.commands is None:
                command._build_commands()
            graphstate_markers.append((len(flattened_commands), command))
            flattened_commands.extend(command.commands)
            return
        if isinstance(command, Sequence):
            for sub_command in command.commands:
                append_command(sub_command, flattened_commands, graphstate_markers)
            return
        flattened_commands.append(command)

    flattened_commands = []
    graphstate_markers = []
    for name, args in plan:
        if len(args) == 0:
            continue
        tail_arg = args[-1]
        if isinstance(tail_arg, Command):
            command = tail_arg
        elif hasattr(tail_arg, "_traj_seq") and isinstance(tail_arg._traj_seq, Command):
            command = tail_arg._traj_seq
        else:
            continue
        append_command(command, flattened_commands, graphstate_markers)

    sequence = Sequence(flattened_commands, graphstate_markers=graphstate_markers)

    # Add extra trajectories as in interleaved_real_plugin.py
    if robot_entity is not None:
        commands_list = list(sequence.commands)
        new_commands = []
        new_graphstate_markers = []
        marker_iter = iter(sequence.graphstate_markers)
        next_marker = next(marker_iter, None)

        for command_index, command in enumerate(commands_list):
            while next_marker is not None and next_marker[0] == command_index:
                graphstate = next_marker[1]
                if getattr(graphstate, 'skill_name', None) == 'screwdriver_aloha':
                    try:
                        closed_conf, open_conf = robot_entity.close_open_conf()
                        _, right_gripper_group, _ = robot_entity.manipulators['right']
                        new_commands.append(
                            GroupTrajectory(
                                robot_entity,
                                right_gripper_group,
                                path=[open_conf, closed_conf],
                            )
                        )
                    except Exception as e:
                        print(f"Warning: failed to insert right gripper close before screwdriver_aloha: {e}")
                new_graphstate_markers.append((len(new_commands), graphstate))
                next_marker = next(marker_iter, None)
            new_commands.append(command)

        while next_marker is not None:
            graphstate = next_marker[1]
            if getattr(graphstate, 'skill_name', None) == 'screwdriver_aloha':
                try:
                    closed_conf, open_conf = robot_entity.close_open_conf()
                    _, right_gripper_group, _ = robot_entity.manipulators['right']
                    new_commands.append(
                        GroupTrajectory(
                            robot_entity,
                            right_gripper_group,
                            path=[open_conf, closed_conf],
                        )
                    )
                except Exception as e:
                    print(f"Warning: failed to insert right gripper close before screwdriver_aloha: {e}")
            new_graphstate_markers.append((len(new_commands), graphstate))
            next_marker = next(marker_iter, None)

        sequence = Sequence(
            new_commands,
            name=sequence.name,
            graphstate_markers=new_graphstate_markers,
        )

    sequence.dump()
    return sequence




#####################


def iterate_sequence(state, sequence, time_step=5e-3, teleport=False, **kwargs):  # None | INF
    assert sequence is not None
    for i, _ in enumerate(sequence.iterate(state, teleport=teleport, **kwargs)):
        state.propagate()
        if time_step is None:
            wait_if_gui()
        else:
            wait_for_duration(time_step)
    return state

def execute_command(command, save=True, client=None, teleport=False, **kwargs):

    aborted = False
    state = WorldState(client=client)
    executed_commands = []
    if command is None:
        aborted = True
    else:
        state.assign()
        iterate_sequence(state, command, teleport=teleport, **kwargs)
        aborted = False
        
    # conf = robot.update_conf()

    return aborted

def step_curve(body, joints, curve, time_step=2e-2, print_freq=None, **kwargs):
    for num_steps, (time_elapsed, positions) in enumerate(
        sample_curve(curve, time_step=time_step)
    ):
        set_joint_positions(body, joints, positions, **kwargs)
        yield positions



############## TODO: use networkx
class SG_node(object):
    def __init__(self, body, parent_nodes = [], child_nodes = [],  position = None) -> None:
        self.body = body
        self.position = position
        self.parent_nodes = parent_nodes
        self.child_nodes = child_nodes

class SG_edge(object):
    def __init__(self, parent_node, child_node, relation = {}) -> None:
        self.parent_node = parent_node
        self.child_node = child_node


#################
