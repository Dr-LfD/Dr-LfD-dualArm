
import pathlib
XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets/' # note: absolute path


### Customized constants
RBT_ID = {0: "left", 1: "right"}


### ALOHA fixed constants
DT = 0.02
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
PERCEPT_ARM_POSE = [0, -1.2015, 0.5727, 0.0838,1.1418,0]
START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]
# START_ARM_POSE = (PERCEPT_ARM_POSE + [0.02239, -0.02239])*2
SINGLE_READY_POSE = [0, -1.7, 1.55, 0.12, 0.65, 0]


# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.052## 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844


# # Gripper joint limits (qpos[6])
# MASTER_GRIPPER_JOINT_OPEN = 0.8835 #0.3083
# MASTER_GRIPPER_JOINT_CLOSE = -0.0598 #-0.6842
# PUPPET_GRIPPER_JOINT_OPEN = 1.9 ## 0.18 #1.4910
# PUPPET_GRIPPER_JOINT_CLOSE = -0.1 #-1.2 #-0.6213, -1.0

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = 0.8835
MASTER_GRIPPER_JOINT_CLOSE = -0.0598 
PUPPET_GRIPPER_JOINT_OPEN = 0.18
PUPPET_GRIPPER_JOINT_CLOSE = -1.1 #-1.2 
############################ Helper functions ############################

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE)/2

### Robot constants
import numpy as np
# from tf.transformations import  euler_from_matrix  #Return quaternion from rotation matrix.
from scipy.spatial.transform import Rotation 

import sys
import os
root_path = next(('/'+ os.path.join(*os.path.dirname(os.path.abspath(__file__)).split(os.sep)[:i+1]) + os.sep
                    for i in range(len(os.path.dirname(os.path.abspath(__file__)).split(os.sep))) 
                    if os.path.dirname(os.path.abspath(__file__)).split(os.sep)[i] == 'pddlstream_aloha'), None)
sys.path.append(root_path) if root_path not in sys.path else None
from examples.pybullet.utils.pybullet_tools.utils import  Euler, Pose, Point


class vx300s:
    Slist = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, -0.12705, 0.0, 0.0],
                      [0.0, 1.0, 0.0, -0.42705, 0.0, 0.05955],
                      [1.0, 0.0, 0.0, 0.0, 0.42705, 0.0],
                      [0.0, 1.0, 0.0, -0.42705, 0.0, 0.35955],
                      [1.0, 0.0, 0.0, 0.0, 0.42705, 0.0]]).T

    M = np.array([[1.0, 0.0, 0.0, 0.536494],
                  [0.0, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.42705],
                  [0.0, 0.0, 0.0, 1.0]])
    
LEFT_BASE_POSE = np.array(
    [[1.0, 0.0, 0.0, -0.469],
     [0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0, 0.0],
     [0.0, 0.0, 0.0, 1.0]]
)

RIGHT_BASE_POSE = np.array(
    [[-1.0, 0.0, 0.0, 0.469],
     [ 0.0, -1.0, 0.0, 0.0],
     [ 0.0, 0.0, 1.0, 0.0],
     [ 0.0, 0.0, 0.0, 1.0]]
)

### Judgement thresholds
GRIPPER_EPSILON = 0.2
EE_VEL_EPSILONE = 0.03
EE_DIST_BOUND = 0.075

REAL_GRIPPER_EPSILON = 0.2
REAL_EE_DIST_BOUND = 0.12

def qpos_to_eetrans(
    joint_qpos, robot
):
    if robot not in ["left", "right"]:
        robot = RBT_ID[robot]  # convert robot id to string
    import modern_robotics as mr

    relative_eepose =mr.FKinSpace(vx300s.M, vx300s.Slist, joint_qpos[:6])

    if robot == "left":
        eemat = np.matmul(LEFT_BASE_POSE, np.array(relative_eepose))
    else:
        eemat = np.matmul(RIGHT_BASE_POSE, np.array(relative_eepose))
    
    return eemat


# rotation matrix to quatanion and position
def trans2eepose(tgt_eepose_mat):
    rot_mat = tgt_eepose_mat[:3, :3]
    # tgt_orientation = euler_from_matrix(rot_mat)
    tgt_orientation = Rotation.from_matrix(rot_mat).as_euler('xyz')
    tgt_position = tgt_eepose_mat[:3, -1]
    tgt_eepose = Pose(Point(x=tgt_position[0], y=tgt_position[1], z=tgt_position[2]), \
        Euler(tgt_orientation[0], tgt_orientation[1], tgt_orientation[2]))
    return tgt_eepose


def qpos_to_eepose(joint_qpos, robot):
    eepose_mat = qpos_to_eetrans(joint_qpos, robot)
    return trans2eepose(eepose_mat)

def trans2xyzrpy(tgt_eepose_mat):
    rot_mat = tgt_eepose_mat[:3, :3]
    tgt_orientation = Rotation.from_matrix(rot_mat).as_euler('xyz')
    tgt_position = tgt_eepose_mat[:3, -1]
    return np.array(list(tgt_position) +  list(tgt_orientation))

def xyzquat2trans(xyzquat):
    xyz = xyzquat[0]
    quat = xyzquat[1]
    rot_mat = Rotation.from_quat(quat).as_matrix()
    return np.concatenate([np.concatenate([rot_mat, np.array([xyz]).T], axis=1), np.array([[0, 0, 0, 1]])], axis=0)

def qpos_to_xyzrpy(joint_qpos, robot):
    eepose_mat = qpos_to_eetrans(joint_qpos, robot)
    return trans2xyzrpy(eepose_mat)

def read_qpos_mat(file_name, robot_id):
    qpos_mat = np.loadtxt(file_name)
    # for each row, convert qpos to eepose
    eepose_list = []
    for i in range(qpos_mat.shape[0]):
        eepose = qpos_to_eepose(qpos_mat[i][:6], robot_id)
        ee_xyz = eepose[0]
        eepose_list.append(ee_xyz)
    return eepose_list