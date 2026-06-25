import os

# from run_estimator import *

from examples.pybullet.utils.pybullet_tools.ikfast.utils import IKFastInfo
from examples.pybullet.utils.pybullet_tools.utils import link_from_name, joints_from_names, get_joints, get_joint_positions, \
    get_joint_names, set_joint_positions, get_link_pose
from examples.pybullet.aloha_real.openworld_aloha.entities import Camera, Manipulator, Robot

from examples.pybullet.aloha_real.openworld_aloha.estimation.lis import CAMERA_MATRIX
from examples.pybullet.aloha_real.scripts.aloha_tamp_constants import PERCEPT_ARM_POSE as ALOHA_PERCEPT_ARM_POSE
from examples.pybullet.aloha_real.scripts.aloha_tamp_constants import START_ARM_POSE as ALOHA_START_ARM_POSE
from examples.pybullet.aloha_real.scripts.aloha_tamp_constants import xyzquat2trans, PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE,JOINT_NAMES, PUPPET_POS2JOINT, PUPPET_JOINT2POS, PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSE, PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN, qpos_to_eepose
from examples.pybullet.aloha_real.scripts.aloha_utils import move_grippers, move_arms

from scipy.spatial.transform import Rotation
from examples.pybullet.utils.pybullet_tools.ikfast.ikfast import (
    ikfast_forward_kinematics,
    get_ik_joints,
)

FRANKA_PERCEPT_ARM_POSE =  [0.0, -0.744, 0.0, -1.9, 0.0, 1.221, 0.0] #[0, -1.1188, 0, -2.6307, 0, 1.6663,0]
FRANKA_GRIPPER_POSITION_OPEN = 0
FRANKA_GRIPPER_POSITION_CLOSE = 200 ## max close: 255
# from robots.panda.panda_controller import PandaController
import numpy as np
import time

CAMERA_FRAME = "camera_frame"
CAMERA_OPTICAL_FRAME = "camera_frame"

LEFT_INFO = IKFastInfo(
    module_name="aloha.ikfastpuppetleft",
    base_link="base",
    ee_link="puppet_left/ee_gripper_link",
    free_joints=[],
)

RIGHT_INFO = IKFastInfo(
    module_name="aloha.ikfastpuppetright",
    base_link="base",
    ee_link="puppet_right/ee_gripper_link",
    free_joints=[],
)

ALOHA_INFOS = {
    "left_arm": LEFT_INFO,
    "right_arm": RIGHT_INFO,
}

from examples.pybullet.utils.pybullet_tools.ikfast.franka_panda.ik import PANDA_INFO

# DUAL_FRANKA_INFOS = {
#     "left_arm": PANDA_INFO,
#     "right_arm": PANDA_INFO,
# }

from examples.pybullet.utils.pybullet_tools.ikfast.franka_dual.ik import PANDA1_INFO, PANDA2_INFO

DUAL_FRANKA_INFOS = {
    "left_arm": PANDA2_INFO,
    "right_arm": PANDA1_INFO,
}

def side_from_arm(arm):
    side = arm.split("_")[0]
    return side


def arm_from_side(side):
    return "{}_arm".format(side)


def gripper_from_arm(arm):  # TODO: deprecate
    side = side_from_arm(arm)
    return "{}_gripper".format(side)


def limit_values(values, boundes):
    mins, maxs = boundes
    # mins, maxs = map(np.array, boundes.copy())
    if not np.all(mins <= values) or not np.all(values <= maxs):
        print("Values out of bounds!")
    return np.clip(values, mins, maxs)



################ controller ################


class Controller(object):
    def __init__(self, *args, **kwargs):
        pass


class SimulatedController(Controller):
    def __init__(self, robot, client=None, **kwargs):
        self.client = client
        self.robot = robot
        
    # def side_from_arm(self, arm):
    #     return arm.replace("_arm", "")

    def open_gripper(self, arm):  # These are mirrored on the pr2

        self.robot.set_open_gripper(arm)

    def close_gripper(self, arm):  # These are mirrored on the pr2

        self.robot.set_close_gripper(arm)

    def get_group_joints(self, group):
        return joints_from_names(self.robot, self.robot.joint_groups[group])

    def set_group_conf(self, group, positions):
        set_joint_positions(self.robot, self.get_group_joints(group), positions)

    def set_group_positions(self, group_positions):
        for group, positions in group_positions.items():
            self.set_group_conf(group, positions)

    def get_current_config(self):
        pass

    @property
    def joint_positions(self):
        joints = get_joints(self.robot)
        joint_positions = get_joint_positions(self.robot, joints)
        joint_names = get_joint_names(self.robot, joints)
        return {k: v for k, v in zip(joint_names, joint_positions)}

    def command_group(self, group, positions, **kwargs):  # TODO: default timeout
        self.set_group_positions({group: positions})

    def command_group_dict(
        self, group, positions_dict, **kwargs
    ):  # TODO: default timeout
        positions = [positions_dict[nm] for nm in self.robot.joint_groups[group]]
        self.command_group(group, positions)

    def command_group_trajectory(
        self, group, positions, times_from_start, dt=0.01, **kwargs
    ):
        for position in positions:
            self.command_group(group, position)
            time.sleep(dt)
            yield

    def wait(self, duration):
        time.sleep(duration)

    def wait_for_clients(self, clients, timeout=0):
        pass

    def any_arm_fully_closed(self):
        return False



class PysicalALOHAController(Controller):
    def __init__(self, robot, joint_names, max_gripper_pose=None, max_gripper_joint = None,  **kwargs):
        self.robot = robot
        self.joint_names = joint_names

    
    def setup_ros_common(self, env):
        self.puppet_bot_left = env.puppet_bot_left
        self.puppet_bot_right = env.puppet_bot_right
        import rospy
        from sensor_msgs.msg import JointState
        rospy.Subscriber('/puppet_left/joint_states', JointState, self.l_joint_state_callback)
        rospy.Subscriber('/puppet_right/joint_states', JointState, self.r_joint_state_callback)

        # self.target_arm_single = None
        
        self.l_joint_qpos = None
        self.r_joint_qpos = None
        while self.l_joint_qpos is None or self.r_joint_qpos is None:
            print('no state received')
            rospy.sleep(1)

    def l_joint_state_callback(self, data):
        index_list = [data.name.index(name) for name in self.joint_names]
        self.l_joint_qpos = [data.position[name_idx] for name_idx in index_list]
        self.l_gripper_pos = data.position[-2]
        self.l_gripper_val = PUPPET_POS2JOINT(self.l_gripper_pos)

        
    def r_joint_state_callback(self, data):
        index_list = [data.name.index(name) for name in self.joint_names]
        self.r_joint_qpos = [data.position[name_idx] for name_idx in index_list]
        self.r_gripper_pos = data.position[-2]
        self.r_gripper_val = PUPPET_POS2JOINT(self.r_gripper_pos)

    def get_env_bot(self, arm):
        return self.puppet_bot_left if arm == 'left' else self.puppet_bot_right
    
    ## NOTE: when closing, do not close too much
    def gripper_cmd_blocking(self, arm, is_open):
        bot = self.get_env_bot(arm)
        cmd = (PUPPET_GRIPPER_JOINT_OPEN -0.2) if is_open else PUPPET_GRIPPER_JOINT_CLOSE
        move_grippers([bot], [cmd], move_time=1.5)

    def arm_cmd_publish(self, arm, target_arm_single):
        bot = self.get_env_bot(arm)
        bot.arm.set_joint_positions(target_arm_single, blocking = False)

    def gripper_cmd_publish(self, side, tgt_val, max_step = 0.5):
        bot = self.get_env_bot(side)
        from interbotix_xs_msgs.msg import JointSingleCommand
        gripper_command = JointSingleCommand(name="gripper")
        gripper_command.cmd = tgt_val
        bot.gripper.core.pub_single.publish(gripper_command)


    def get_side_full_conf(self, side):
        arm_conf = self.l_joint_qpos if side == 'left' else self.r_joint_qpos
        gripper_pos = self.l_gripper_pos if side == 'left' else self.r_gripper_pos
        if arm_conf is None or gripper_pos is None:
            return None
        return np.hstack((arm_conf, gripper_pos * np.array([1, -1])))


    def get_group_conf(self, group):
        if group == 'left_arm':
            return self.l_joint_qpos
        elif group == 'right_arm':
            return self.r_joint_qpos
        elif group == 'left_robot':
            return self.get_side_full_conf('left')
        elif group == 'right_robot':
            return self.get_side_full_conf('right')
        elif group == 'left_gripper':
            return self.l_gripper_pos*np.array([1, -1]) 
        elif group == 'right_gripper':
            return self.r_gripper_pos*np.array([1, -1])

    # def get_side_conf(self, side):
    #     if side == 'left':
    #         return self.l_joint_qpos
    #     else:
    #         return self.r_joint_qpos
        
    ## return jpose in pysical robot
    def get_current_config(self):
        left_conf = self.get_group_conf('left_arm')
        right_conf = self.get_group_conf('right_arm')
        cur_conf = np.hstack((left_conf, right_conf))
        return cur_conf
    
    ## return jpose in pybullet
    @property
    def joint_positions(self):
        joints = get_joints(self.robot)
        joint_positions = get_joint_positions(self.robot, joints)
        joint_names = get_joint_names(self.robot, joints)
        return {k: v for k, v in zip(joint_names, joint_positions)}

    def do_resetting(self):
        all_bots = [self.puppet_bot_left, self.puppet_bot_right]
        move_grippers(all_bots, [PUPPET_GRIPPER_JOINT_CLOSE] * 2, move_time=1.5)
        move_grippers(all_bots, [PUPPET_GRIPPER_JOINT_OPEN-0.2] * 2, move_time=1.5)
        move_arms(all_bots, [ALOHA_PERCEPT_ARM_POSE] * 2, move_time=2)

    def go_to_cfg(self, tgt_left_arm = ALOHA_PERCEPT_ARM_POSE, tgt_right_arm = ALOHA_PERCEPT_ARM_POSE, tgt_left_gripper = PUPPET_GRIPPER_JOINT_OPEN, tgt_right_gripper = PUPPET_GRIPPER_JOINT_OPEN):
        move_arms([self.puppet_bot_left, self.puppet_bot_right], [tgt_left_arm, tgt_right_arm], move_time=1.5)
        move_grippers([self.puppet_bot_left, self.puppet_bot_right], [tgt_left_gripper, tgt_right_gripper], move_time=1.5)


class PysicalDualFrankaController(Controller):
    def __init__(self, robot, **kwargs):
        self.robot = robot
    
    def setup_ros_common(self, env):
        import rospy
        self.env = env
        # self.target_arm_single = None
        
        self.get_current_config()
        while self.l_joint_qpos is None or self.r_joint_qpos is None:
            self.get_current_config()
            print('no state received')
            rospy.sleep(1)


    
    ## NOTE: when closing, do not close too much
    def gripper_cmd_blocking(self, arm, is_open):
        tgt_qpos = self.env.get_qpos()
        cmd = FRANKA_GRIPPER_POSITION_OPEN if is_open else FRANKA_GRIPPER_POSITION_CLOSE
        if arm == 'left':
            tgt_qpos[7] = cmd
        elif arm == 'right':
            tgt_qpos[15] = cmd
        else:
            raise ValueError('Wrong arm')
        self.env.move_arms_grippers(tgt_qpos, move_time=1.0)

    def arm_cmd_publish(self, arm, target_arm_single):
        # bot = self.get_env_bot(arm)
        # bot.arm.set_joint_positions(target_arm_single, blocking = False)
        from franka_msgs.msg import servoj

        target_arm_single_list = list(target_arm_single)
        if 'left' in arm:
            self.env.left_servoj_pub.publish(
                servoj(keepalive=1, cmd_q=target_arm_single_list)
            )
        elif 'right' in arm:
            self.env.right_servoj_pub.publish(
                servoj(keepalive=1, cmd_q=target_arm_single_list)
            )
        else:
            raise ValueError('Wong arm side')

    def get_side_conf(self, side):
        if side == 'left':
            arm_state = self.env.franka_recorder_left.arm_state
        else:
            arm_state = self.env.franka_recorder_right.arm_state
        single_qpos= arm_state["qpos"]
        # print(f"len of arm {side} qpos: ", len(single_qpos))
        return single_qpos
        
    ## return jpose in pysical robot
    def get_current_config(self):
        self.l_joint_qpos = self.get_side_conf('left')
        self.r_joint_qpos = self.get_side_conf('right')
        cur_conf = np.hstack((self.l_joint_qpos, self.r_joint_qpos))
        return cur_conf
    
    ## return jpose in pybullet
    @property
    def joint_positions(self):
        joints = get_joints(self.robot)
        joint_positions = get_joint_positions(self.robot, joints)
        joint_names = get_joint_names(self.robot, joints)
        return {k: v for k, v in zip(joint_names, joint_positions)}

    def do_resetting(self):
        tgt_pose_close = (FRANKA_PERCEPT_ARM_POSE + [FRANKA_GRIPPER_POSITION_CLOSE])*2
        self.env.move_arms_grippers(tgt_pose_close, move_time=3.0)
        tgt_pose_open = (FRANKA_PERCEPT_ARM_POSE + [FRANKA_GRIPPER_POSITION_OPEN])*2
        self.env.move_arms_grippers(tgt_pose_open, move_time=1.0)


    

################ SINGLE ARM robot ################
class SingleArmBase(Robot):
    def __init__(self, body, lfd_env=None, **kwargs):
        if lfd_env is not None and self.real_execute:
            self.controller.setup_ros_common(lfd_env)

        super(SingleArmBase, self).__init__(body, **kwargs)

        self.limit_dict = {}
        for arm in self.arms:
            mins, maxs = map(np.array, self.get_group_limits(arm))
            self.limit_dict.update({arm: (np.array(mins), np.array(maxs))})

    # def arm_from_side(self, side):
    #     return arm_from_side(side)

    # def side_from_arm(self, arm):
    #     return side_from_arm(arm)

    def arm_conf(self, arm, config):
        assert arm in self.arms
        return config[arm]

    @property
    def groups(self):
        return self.joint_groups

    @property
    def default_mobile_base_arm(self):
        return self.get_default_conf()

    @property
    def default_fixed_base_arm(self):
        return self.get_default_conf()

    @property
    def base_link(self):
        return link_from_name(self.robot, self.BASE_LINK)

    def reset(self, reset_pybullet=True, **kwargs):
        if self.real_execute and not reset_pybullet:
            self.controller.do_resetting()
        else:
            conf = self.get_default_conf()
            for group, positions in conf.items():
                self.set_group_positions(group, positions)

    def post_single_arm_reset(self, **kwargs):
        if hasattr(self.controller, 'go_to_cfg'):
            self.controller.go_to_cfg(**kwargs)

    def get_link_trans(self, link_name):
        link_id = link_from_name(self.robot, link_name)
        link_pose = self.get_link_pose(link_id)
        link_mat = xyzquat2trans(link_pose)
        return link_mat
    
    def get_valid_arm_pose(self, arm_pose, arm_name):
        if arm_name in self.limit_dict:
            arm_pose = limit_values(arm_pose[:self.arm_dof], self.limit_dict[arm_name])
        return arm_pose

    def get_ee_pose(self, arm_pose=None, arm_name=None):
        if arm_pose is None:
            current_config = self.controller.get_current_config()
            arm_pose = current_config[:self.arm_dof]
        
        if arm_name is None:
            arm_name = self.arms[0]  # Default to first arm
            
        # This would need to be implemented based on the specific robot's forward kinematics
        # For now, returning a placeholder
        return arm_pose
    
    def set_close_gripper(self, arm):
        return self.set_gripper(arm, is_close=True)
    
    def close_open_conf(self):
        raise NotImplementedError("close_open_conf should be implemented in child class")
    
    def set_open_gripper(self, arm):
        return self.set_gripper(arm, is_close=False)

    def set_gripper(self, arm, is_close):
        side = self.side_from_arm(arm) 
        _, gripper_group, _ = self.manipulators[side]
        close_conf, open_conf = self.close_open_conf()
        if is_close:
            self.set_group_positions(gripper_group, close_conf)
        else:
            self.set_group_positions(gripper_group, open_conf)

    def arm_from_side(self, side):
        return arm_from_side(side)

    def side_from_arm(self, arm):
        return side_from_arm(arm)

    def get_arm_group(self, arm):
        rbt_to_side = getattr(self, "rbt_ids_to_side", None)
        if rbt_to_side and arm in rbt_to_side:
            return f"{rbt_to_side[arm]}_arm"
        return arm
    
class PandaSingleRobot(SingleArmBase):
    def __init__(self, robot_body, ik_method = 'ikfast', link_names={}, client=None, real_camera=False, real_execute=False, \
                 arms = ["left_arm"],  **kwargs):
        self.ik_method = ik_method
        self.link_names = link_names
        self.body = robot_body
        self.client = client
        self.arms = arms
        self.real_camera = real_camera
        self.real_execute = real_execute
        ## reverse of dual franka
        Panda_TOOL_FRAMES = {
            "left_arm": "tool_link", 
        }

        Panda_manipulators = {
            side_from_arm(arm): Manipulator(
                arm, gripper_from_arm(arm), Panda_TOOL_FRAMES[arm]
            )
            for arm in self.arms
        }
        Panda_ik_infos = {'left': PANDA_INFO}


        ## NOTEL id starts from 1
        Panda_GROUPS = {
            "base": [],
            "left_arm": ["panda_joint{}".format(i) for i in range(1, 8)],
            "left_gripper": ["panda_finger_joint1", "panda_finger_joint2"],
        }
        Panda_GROUPS["left_robot"] = Panda_GROUPS["left_arm"] + Panda_GROUPS["left_gripper"]

        cameras = []

        if not self.real_execute:
            self.controller = SimulatedController(self.robot, client=self.client)
        else:
            self.controller = PysicalPandaController(self.robot, client=self.client)


        super(PandaSingleRobot, self).__init__(
            robot_body,
            ik_info=Panda_ik_infos,
            manipulators=Panda_manipulators,
            cameras=cameras,
            joint_groups=Panda_GROUPS,
            link_names=link_names,
            client=client,
            **kwargs
        )
        self.arm_dof = 7
        self.max_depth = 3.0
        self.min_z = 0.0
        self.BASE_LINK = "panda_link0"
        self.max_finger_joint = -1
        self.min_finger_joint = 1
        self.max_finger_pose = 0.04
        self.min_finger_pose = 0.0
        self.max_gripper_width = 2*(self.max_finger_pose-self.min_finger_pose)
        self.gripper_pos_unnormalize_fn = lambda x: x* (self.max_finger_pose - self.min_finger_pose) + self.min_finger_pose
        self.gripper_pos_normalise_fn = lambda x: (x - self.min_finger_pose) / (self.max_finger_pose - self.min_finger_pose)
        self.initial_arm_pose = FRANKA_PERCEPT_ARM_POSE

    def revise_initial_pose(self, initial_jposes):
        self.initial_arm_pose = list(initial_jposes['left_arm'])



    def get_default_conf(self):
        close_conf, open_conf = self.close_open_conf()
        conf = {
            "left_arm": self.initial_arm_pose,
            "left_gripper": open_conf,
        }
        conf['left_robot'] = conf['left_arm'] + conf['left_gripper']
        return conf

    ## if right finger is mimicking left finger, sign will be opposite
    def close_open_conf(self):
        return [self.min_finger_pose, self.min_finger_pose], [self.max_finger_pose, self.max_finger_pose]
    
    ## [-1,1] --> [0.0, 0.05]
    def joint2pos_gripper(self, gripper_jval):
        tgt_pos = self.gripper_pos_unnormalize_fn((gripper_jval - self.min_finger_joint) / (self.max_finger_joint - self.min_finger_joint) )  
        # tgt_pos = self.max_finger_pose * gripper_jval
        return (tgt_pos, tgt_pos)
    
    def pos2joint_gripper(self, tgt_pos):
        gripper_jval = self.gripper_pos_normalise_fn(tgt_pos) * (self.max_finger_joint - self.min_finger_joint) + self.min_finger_joint

        return gripper_jval
    
    
    @property
    def rbt_ids_to_side(self):
        idx_to_side = {
            'robot0': 'left',
        }
        return idx_to_side
    
    @property
    def side_to_rbt_ids(self):
        side_to_idx = {
            'left': 'robot0',
        }
        return side_to_idx
################# DUAL_ARM_ROBOT ################

class DualArmBase(Robot):
    def __init__(self, body, lfd_env = None, **kwargs):
        if lfd_env is not None and self.real_execute:
            self.controller.setup_ros_common(lfd_env)


        super(DualArmBase, self).__init__(body, **kwargs)

        self.limit_dict = {}
        for arm in ['left_arm', 'right_arm']:
            mins, maxs = map(np.array, self.get_group_limits(arm))
            self.limit_dict.update({arm: (np.array(mins), np.array(maxs))})


    def arm_from_side(self, side):
        return arm_from_side(side)

    def side_from_arm(self, arm):
        return side_from_arm(arm)

    def get_arm_group(self, arm):
        rbt_to_side = getattr(self, "rbt_ids_to_side", None)
        if rbt_to_side and arm in rbt_to_side:
            return f"{rbt_to_side[arm]}_arm"
        return arm

    def arm_conf(self, arm, config):
        assert arm in self.arms
        return config[arm]

    @property
    def groups(self):
        return self.joint_groups

    @property
    def default_mobile_base_arm(self):
        return self.get_default_conf()

    @property
    def default_fixed_base_arm(self):
        return self.get_default_conf()

    @property
    def base_link(self):
        return link_from_name(self.robot, self.BASE_LINK)

    def reset(self, reset_pybullet = True, **kwargs):
        if self.real_execute and not reset_pybullet:
            self.controller.do_resetting()
        else:
            conf = self.get_default_conf()
            for group, positions in conf.items():
                self.set_group_positions(group, positions)

    def post_bimanual_reset(self, **kwargs):
        self.controller.go_to_cfg(**kwargs)

    def get_link_trans(self, link_name):
        link_id = link_from_name(self.robot, link_name)
        link_pose =  self.get_link_pose(link_id)
        link_mat = xyzquat2trans(link_pose)
        return link_mat
    
    def get_valid_dualpose(self, dual_jpose, reverse_arms = False, full_jpose = False):
        left_jpose_raw, right_jpose_raw = dual_jpose
        left_jpose = limit_values(left_jpose_raw[:self.arm_dof], self.limit_dict['left_arm'])
        right_jpose = limit_values(right_jpose_raw[:self.arm_dof], self.limit_dict['right_arm'])

        ## check if the jpose is full (arm + gripper)
        full_jpose = len(left_jpose_raw) == self.arm_dof + 1 and len(right_jpose_raw) == self.arm_dof + 1
        if full_jpose:
            if 'aloha' not in self.name:
                raise NotImplementedError("normalization of franka  not implemented")
            unnormalized_left = PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(left_jpose_raw[self.arm_dof])
            unnormalized_right = PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(right_jpose_raw[self.arm_dof])
            left_gripper_pos = self.joint2pos_gripper(unnormalized_left)
            right_gripper_pos = self.joint2pos_gripper(unnormalized_right)
            left_jpose = list(left_jpose) + list(left_gripper_pos)
            right_jpose = list(right_jpose) + list(right_gripper_pos)

        if reverse_arms:
            return [right_jpose, left_jpose]
        else:
            return [left_jpose, right_jpose]
    

    def get_ee_dist(self, left_jpose = None, right_jpose = None):
        if left_jpose is None and right_jpose is None:
            dual_jpose = self.controller.get_current_config() ## NOTE: 12D!
            left_jpose = dual_jpose[:self.arm_dof]
            right_jpose = dual_jpose[self.arm_dof: -1]
        left_ee_pose = qpos_to_eepose(left_jpose, 0)
        right_ee_pose = qpos_to_eepose(right_jpose, 1)
        left_xyz = left_ee_pose[0]
        right_xyz = right_ee_pose[0]
        return np.linalg.norm(left_xyz - right_xyz)
    
    def set_close_gripper(self, arm):
        return self.set_gripper(arm, is_close = True)
    
    def close_open_conf(self):
        raise NotImplementedError("close_open_conf should be implemented in child class")
    
    def set_open_gripper(self, arm):
        return self.set_gripper(arm, is_close = False)

    def set_gripper(self, arm, is_close):
        side = self.side_from_arm(arm)
        _, gripper_group, _ = self.manipulators[side]
        close_conf, open_conf = self.close_open_conf()
        if is_close:
            self.set_group_positions(gripper_group, close_conf)
        else:
            self.set_group_positions(gripper_group, open_conf)  


class ALOHARobot(DualArmBase):
    def __init__(self, robot_body, ik_method = 'ikfast', link_names={}, client=None, real_camera=False, real_execute=False, \
                 arms = ["left_arm", "right_arm"],  **kwargs):

        self.ik_method = ik_method
        self.link_names = link_names
        self.body = robot_body
        self.client = client
        self.arms = arms
        self.real_camera = real_camera
        self.real_execute = real_execute

        ALOHA_GROUPS = {
            "base": [],
            "left_arm": ["puppet_left/waist", "puppet_left/shoulder", "puppet_left/elbow", "puppet_left/forearm_roll", "puppet_left/wrist_angle", "puppet_left/wrist_rotate"],
            "left_gripper": ["puppet_left/left_finger", "puppet_left/right_finger"],
            "right_arm": ["puppet_right/waist", "puppet_right/shoulder", "puppet_right/elbow", "puppet_right/forearm_roll", "puppet_right/wrist_angle", "puppet_right/wrist_rotate"],
            "right_gripper": ["puppet_right/left_finger", "puppet_right/right_finger"],
        }

        ALOHA_GROUPS["left_robot"] = ALOHA_GROUPS["left_arm"] + ALOHA_GROUPS["left_gripper"]
        ALOHA_GROUPS["right_robot"] = ALOHA_GROUPS["right_arm"] + ALOHA_GROUPS["right_gripper"]

        ALOHA_TOOL_FRAMES = {
            "left_arm": "puppet_left/ee_gripper_link", 
            "right_arm": "puppet_right/ee_gripper_link",
        }

        ALOHA_manipulators = {
            side_from_arm(arm): Manipulator(
                arm, gripper_from_arm(arm), ALOHA_TOOL_FRAMES[arm]
            )
            for arm in self.arms
        }
        aloha_ik_infos = {side_from_arm(arm): ALOHA_INFOS[arm] for arm in self.arms}

        if not real_camera:
            cameras = [
                Camera(
                    self,
                    link=link_from_name(self.body, CAMERA_FRAME),
                    optical_frame=link_from_name(
                        self.body, CAMERA_OPTICAL_FRAME
                    ),
                    camera_matrix=CAMERA_MATRIX,
                    client=client,
                )
            ]
        else:
            cameras = []

        if not self.real_execute:
            self.controller = SimulatedController(self.robot, client=self.client)
        else:
            self.controller = PysicalALOHAController(self, joint_names = JOINT_NAMES)


        super(ALOHARobot, self).__init__(
            robot_body,
            ik_info=aloha_ik_infos,
            manipulators=ALOHA_manipulators,
            cameras=cameras,
            joint_groups=ALOHA_GROUPS,
            link_names=link_names,
            client=client,
            **kwargs
        )
        self.arm_dof = 6
        self.max_depth = 3.0
        self.min_z = 0.0
        self.BASE_LINK = "base"
        self.max_finger_pose = PUPPET_GRIPPER_POSITION_OPEN 
        self.min_finger_pose = PUPPET_GRIPPER_POSITION_CLOSE
        self.max_gripper_width = 2*(self.max_finger_pose-self.min_finger_pose)
        self.percept_arm_pose = ALOHA_PERCEPT_ARM_POSE

    @property
    def rbt_ids_to_side(self):
        idx_to_side = {
            'robot0': 'left',
            'robot1': 'right',
            'left': 'left',
            'right': 'right',
        }
        return idx_to_side
    
    @property
    def side_to_rbt_ids(self):
        side_to_idx = {
            'left': 'robot0',
            'right': 'robot1',
        }
        return side_to_idx
    
    def get_default_conf(self):
        close_conf, open_conf = self.close_open_conf()
        conf = {
            "left_gripper": open_conf,
            "right_gripper": open_conf,
        }
        if self.real_execute:
            conf.update({
            "left_arm": self.percept_arm_pose,
            "right_arm": self.percept_arm_pose,
            })
        else:
            conf.update({            
            "left_arm": ALOHA_START_ARM_POSE[:self.arm_dof],
            "right_arm": ALOHA_START_ARM_POSE[self.arm_dof+2: 2*self.arm_dof+2],
            })

        conf['left_robot'] = conf['left_arm'] + conf['left_gripper']
        conf['right_robot'] = conf['right_arm'] + conf['right_gripper']
        return conf

    
    def close_open_conf(self):
        return [self.min_finger_pose, -self.min_finger_pose], [self.max_finger_pose, -self.max_finger_pose]

    def set_arm_gripper(self, side, single_arm_tgt):
        if len(single_arm_tgt) == self.arm_dof:
            ## is arm
            self.controller.arm_cmd_publish(side, single_arm_tgt[:self.arm_dof])
        elif len(single_arm_tgt) == 2:
            ##  gripper cmd pos 2 joint
            tgt_joint = PUPPET_POS2JOINT(single_arm_tgt[0]) 
            self.controller.gripper_cmd_publish(side, tgt_joint)   
        elif len(single_arm_tgt) == (self.arm_dof + 2):
            ## is arm + gripper
            self.controller.arm_cmd_publish(side, single_arm_tgt[:self.arm_dof])
            tgt_joint = PUPPET_POS2JOINT(single_arm_tgt[self.arm_dof])  ## only the politive value for [0.049, -0.049]
            self.controller.gripper_cmd_publish(side, tgt_joint)   

    def joint2pos_gripper(self, gripper_jval):
        tgt_pos = PUPPET_JOINT2POS(gripper_jval)
        return (tgt_pos, -tgt_pos)

class DUALfrankaRobot(DualArmBase):
    def __init__(self, robot_body, ik_method = 'pybullet', link_names={}, client=None, real_camera=False, real_execute=False, \
                 arms = ["left_arm", "right_arm"],  **kwargs):
        self.ik_method = ik_method
        self.link_names = link_names
        self.body = robot_body
        self.client = client
        self.arms = arms
        self.real_camera = real_camera
        self.real_execute = real_execute

        DUALfranka_TOOL_FRAMES = {
            "left_arm": "panda2_ee_link", 
            "right_arm": "panda1_ee_link",
        }

        DUALfranka_manipulators = {
            side_from_arm(arm): Manipulator(
                arm, gripper_from_arm(arm), DUALfranka_TOOL_FRAMES[arm]
            )
            for arm in self.arms
        }
        DUALfranka_ik_infos = {side_from_arm(arm): DUAL_FRANKA_INFOS[arm] for arm in self.arms}


        ## NOTEL id starts from 1
        DUALfranka_GROUPS = {
            "base": [],
            "left_arm": ["panda2_joint{}".format(i) for i in range(1, 8)],
            "left_gripper": ["panda2_hande_left_finger_joint", "panda2_hande_right_finger_joint"],
            "right_arm": ["panda1_joint{}".format(i) for i in range(1, 8)],
            "right_gripper":  ["panda1_hande_left_finger_joint", "panda1_hande_right_finger_joint"],
        }



        if not real_camera:
            cameras = [
                Camera(
                    self,
                    link=link_from_name(self.body, CAMERA_FRAME),
                    optical_frame=link_from_name(
                        self.body, CAMERA_OPTICAL_FRAME
                    ),
                    camera_matrix=CAMERA_MATRIX,
                    client=client,
                )
            ]
        else:
            cameras = []

        if not self.real_execute:
            self.controller = SimulatedController(self.robot, client=self.client)
        else:
            self.controller = PysicalDualFrankaController(self.robot, client=self.client)


        super(DUALfrankaRobot, self).__init__(
            robot_body,
            ik_info=DUALfranka_ik_infos,
            manipulators=DUALfranka_manipulators,
            cameras=cameras,
            joint_groups=DUALfranka_GROUPS,
            link_names=link_names,
            client=client,
            **kwargs
        )
        self.arm_dof = 7
        self.max_depth = 3.0
        self.min_z = 0.0
        self.BASE_LINK = "world"
        self.max_finger_pose = 0.025
        self.min_finger_pose = 0.0
        self.max_gripper_width = 2*(self.max_finger_pose-self.min_finger_pose)
        self.percept_arm_pose = FRANKA_PERCEPT_ARM_POSE
 

    def get_default_conf(self):
        close_conf, open_conf = self.close_open_conf()
        conf = {
            "left_arm": self.percept_arm_pose,
            "right_arm": self.percept_arm_pose,
            "left_gripper": open_conf,
            "right_gripper": open_conf,
        }
        return conf

    def close_open_conf(self):
        return (self.min_finger_pose, self.min_finger_pose), (self.max_finger_pose, self.max_finger_pose)






class PandaDualRobot(DualArmBase):
    def __init__(self, robot_body, ik_method = 'ikfast', link_names={}, client=None, real_camera=False, real_execute=False, \
                 arms = ["left_arm", "right_arm"],  **kwargs):
        self.ik_method = ik_method
        self.link_names = link_names
        self.body = robot_body
        self.client = client
        self.arms = arms
        self.real_camera = real_camera
        self.real_execute = real_execute
        ## reverse of dual franka
        PandaDual_TOOL_FRAMES = {
            "left_arm": "panda2_ee_link", 
            "right_arm": "panda1_ee_link",
        }

        PandaDual_manipulators = {
            side_from_arm(arm): Manipulator(
                arm, gripper_from_arm(arm), PandaDual_TOOL_FRAMES[arm]
            )
            for arm in self.arms
        }
        PandaDual_ik_infos = {side_from_arm(arm): DUAL_FRANKA_INFOS[arm] for arm in self.arms}


        ## NOTEL id starts from 1
        PandaDual_GROUPS = {
            "base": [],
            "left_arm": ["panda2_joint{}".format(i) for i in range(1, 8)],
            "left_gripper": ["panda2_finger_joint1", "panda2_finger_joint2"],
            "right_arm": ["panda1_joint{}".format(i) for i in range(1, 8)],
            "right_gripper":  ["panda1_finger_joint1", "panda1_finger_joint2"],
        }
        PandaDual_GROUPS["left_robot"] = PandaDual_GROUPS["left_arm"] + PandaDual_GROUPS["left_gripper"]
        PandaDual_GROUPS["right_robot"] = PandaDual_GROUPS["right_arm"] + PandaDual_GROUPS["right_gripper"]


        # if not real_camera:
        #     cameras = [
        #         Camera(
        #             self,
        #             link=link_from_name(self.body, CAMERA_FRAME),
        #             optical_frame=link_from_name(
        #                 self.body, CAMERA_OPTICAL_FRAME
        #             ),
        #             camera_matrix=CAMERA_MATRIX,
        #             client=client,
        #         )
        #     ]
        # else:
        cameras = []

        if not self.real_execute:
            self.controller = SimulatedController(self.robot, client=self.client)
        else:
            self.controller = PysicalPandaDualController(self.robot, client=self.client)


        super(PandaDualRobot, self).__init__(
            robot_body,
            ik_info=PandaDual_ik_infos,
            manipulators=PandaDual_manipulators,
            cameras=cameras,
            joint_groups=PandaDual_GROUPS,
            link_names=link_names,
            client=client,
            **kwargs
        )
        self.arm_dof = 7
        self.max_depth = 3.0
        self.min_z = 0.0
        self.BASE_LINK = "world"
        self.max_finger_joint = -1
        self.min_finger_joint = 1
        self.max_finger_pose = 0.04
        self.min_finger_pose = 0.0
        self.max_gripper_width = 2*(self.max_finger_pose-self.min_finger_pose)
        self.gripper_pos_unnormalize_fn = lambda x: x* (self.max_finger_pose - self.min_finger_pose) + self.min_finger_pose
        self.gripper_pos_normalise_fn = lambda x: (x - self.min_finger_pose) / (self.max_finger_pose - self.min_finger_pose)
        self.initial_left_arm_pose = FRANKA_PERCEPT_ARM_POSE
        self.initial_right_arm_pose = FRANKA_PERCEPT_ARM_POSE

    def revise_initial_pose(self, initial_jposes):
        self.initial_left_arm_pose = list(initial_jposes['left_arm'])
        self.initial_right_arm_pose = list(initial_jposes['right_arm'])


    def get_default_conf(self):
        close_conf, open_conf = self.close_open_conf()
        conf = {
            "left_arm": self.initial_left_arm_pose,
            "right_arm": self.initial_right_arm_pose,
            "left_gripper": open_conf,
            "right_gripper": open_conf,
        }
        conf['left_robot'] = conf['left_arm'] + conf['left_gripper']
        conf['right_robot'] = conf['right_arm'] + conf['right_gripper']
        return conf

    ## if right finger is mimicking left finger, sign will be opposite
    def close_open_conf(self):
        return [self.min_finger_pose, self.min_finger_pose], [self.max_finger_pose, self.max_finger_pose]
    
    ## [-1,1] --> [0.0, 0.05]
    def joint2pos_gripper(self, gripper_jval):
        tgt_pos = self.gripper_pos_unnormalize_fn((gripper_jval - self.min_finger_joint) / (self.max_finger_joint - self.min_finger_joint) )  
        # tgt_pos = self.max_finger_pose * gripper_jval
        return (tgt_pos, tgt_pos)
    
    def pos2joint_gripper(self, tgt_pos):
        gripper_jval = self.gripper_pos_normalise_fn(tgt_pos) * (self.max_finger_joint - self.min_finger_joint) + self.min_finger_joint
        # gripper_jval = tgt_pos / self.max_finger_pose
        return gripper_jval
    
    # def hand_name_map(self, node_name):
    #     hand_name_dict = {
    #         'robot0': 'left',
    #         'robot1': 'right',
    #     }
    #     return hand_name_dict[node_name]

    @property
    def rbt_ids_to_side(self):
        idx_to_side = {
            'robot0': 'left',
            'robot1': 'right',
        }
        return idx_to_side
    
    @property
    def side_to_rbt_ids(self):
        side_to_idx = {
            'left': 'robot0',
            'right': 'robot1',
        }
        return side_to_idx

    def get_arm_group(self, arm):
        """Map schema arm constant (e.g. robot0, robot1) to group name (left_arm, right_arm)."""
        rbt_to_side = getattr(self, "rbt_ids_to_side", None)
        if rbt_to_side and arm in rbt_to_side:
            return f"{rbt_to_side[arm]}_arm"
        return arm


    
    def compute_fk(self, qpos_dual):
        qpos_dim_half = int(0.5*len(qpos_dual))
        qpos = {'left': qpos_dual[:self.arm_dof], 'right': qpos_dual[qpos_dim_half:qpos_dim_half+self.arm_dof]}


        ee_pos = {}
        for side in ['left', 'right']:
            world_from_tool = ikfast_forward_kinematics(self.robot, self.ik_info[side],self.get_tool_link(side), conf =  qpos[side])

            # tool_link = self.get_tool_link(side)
            # ik_joints = get_ik_joints(self.robot, self.ik_info[side], tool_link)
            # conf = qpos[side]
            # set_joint_positions(self.robot, ik_joints, conf)
            # world_from_tool = get_link_pose(self.robot, tool_link)
            rotvec = Rotation.from_quat(world_from_tool[1]).as_rotvec()
            ee_pos[side] = np.concatenate((world_from_tool[0], rotvec))
            
        eef_14d = np.concatenate((\
            ee_pos['left'], \
            np.array([qpos_dual[self.arm_dof]]), \
            ee_pos['right'], \
            np.array([qpos_dual[-1]]) ))
        return eef_14d
    
    def qpos_to_DMG_controller(self, joint_14d):
        eef_14d = self.compute_fk(joint_14d)
        return eef_14d


    
