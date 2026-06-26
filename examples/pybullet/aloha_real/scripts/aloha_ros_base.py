
import rospy
# from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
import numpy as np
import sys
import os
root_path = next(p for p in (os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *([os.pardir] * k))) for k in range(16)) if os.path.isfile(os.path.join(p, '.repo_root')))
sys.path.append(root_path) if root_path not in sys.path else None
from examples.pybullet.utils.pybullet_tools.aloha_primitives import BodyPath, Command, GRIP_CMD, \
    Attach, Detach, compute_absolute_differences
from examples.pybullet.aloha_real.openworld_aloha.run_openworld import compute_TAMP_cmd
from examples.pybullet.aloha_real.scripts.aloha_tamp_constants import PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE, \
     RBT_ID, JOINT_NAMES, SINGLE_READY_POSE, qpos_to_eetrans

###################################################
# JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate",\
#     "left_finger", "right_finger"]

# # in case the grasp force is large
# PUPPET_GRIPPER_JOINT_CLOSE = 0.0
######################################################
  

# find the first element in the body_list that is left arm and right arm
def init_conf_tamp_idx( body_paths: BodyPath):
    left_id = None
    right_id = None
    group_list = [body_path.group_id for body_path in body_paths]
    
    if 0 in group_list:
        left_id = group_list.index(0)
    if 1 in group_list:
        right_id = group_list.index(1)

    return left_id, right_id

def get_gripper_cmd_sim( body_path: BodyPath):
    if len(body_path.attachments) == 0:
        return list(GRIP_CMD["open"])
    else:
        return list(GRIP_CMD["closed"] ) 

def get_gripper_cmd_real( body_path: BodyPath):
    if len(body_path.attachments) == 0:
        return PUPPET_GRIPPER_JOINT_OPEN
    else:
        return PUPPET_GRIPPER_JOINT_CLOSE 
    
def get_bodypath_type( body_path: BodyPath):
    if type(body_path) == Attach:
        return 0
    elif type(body_path) == Detach:
        return 1
    else:
        return 2
    

# def all_ee_pose(l_joint_qpos, r_joint_qpos):
#     left_eepose = qpos_to_eetrans(l_joint_qpos, 0)
#     right_eepose = qpos_to_eetrans(r_joint_qpos, 1)
#     return left_eepose, right_eepose

########################################
class aloha_base(object):
    def __init__(self, para, init_ros = True):
        if init_ros:
            rospy.init_node('aloha_tamp', anonymous=True)
        self.node_init_common(para)
        
        self.body_path_list = self.calc_tamp_cmd(text_prompt =para['text_prompt'])

        while len(self.body_path_list) ==0:
            print("No Command to execute")
            # sleep for 1 second
            rospy.sleep(1)

        self.body_path_gen = self.get_body_path_gen_skip()()
        self.cur_body_path =None
        self.refined_path = None
        self.wp_gen= None
        
        self.pub_timer = rospy.Timer(rospy.Duration(0.08), self.rbt_routine)
        
        rospy.spin()


    
    def conf_close(self, tolerance=0.05):
        # see if reached the end of the path
        arm_name = RBT_ID[self.cur_body_path.group_id]
        single_arm_pos = self.l_joint_qpos if arm_name == "left" else self.r_joint_qpos

        abs_diff = compute_absolute_differences(self.cur_body_path.path[-1][:6], single_arm_pos[:6])
        max_diff = max(abs_diff)
        if max_diff > tolerance:
            self.max_diff_info = {'robot_id':self.cur_body_path.group_id, \
                'joint_id': abs_diff.index(max_diff),\
                    'max_diff': max_diff,
                    }
            return False
        else:
            self.max_diff_info = None
            return True
        

    def warn_unreach(self):
        print_info = 'Target qpose not reached, details:'
        if self.max_diff_info:
            print_info += str(self.max_diff_info)
        print(print_info)
        
    def get_body_path_gen_skip(self):
        path_list = self.body_path_list
        def gen():
            for body_path in path_list:
                if hasattr(body_path, 'path'):
                    yield body_path
        return gen
                
        
    # self.wp_gen= self.get_wp_gen(self.cur_body_path); a = self.wp_gen(); wp = next(a)
    def get_wp_gen(self, body_path:BodyPath):
        self.refined_path = self.motion_plan(body_path)
        if len(self.refined_path.path) == 0:
            raise ValueError('No path to execute')
        
        def gen():
            for conf in self.refined_path.path:
                    yield conf
                
        return gen
        
    #########################  should be revised in openworld-tamp
    def calc_tamp_cmd(self, **kwargs):
        tamp_cmd = compute_TAMP_cmd(self.para)
        body_path_list = self.process_cmd(tamp_cmd)        
        return body_path_list
                   
        # add Bodypath from cur_conf to the 1st conf in the command
    def process_cmd(self, command: Command):
        cur_conf = self.get_current_config()
        l_cur_conf = list(cur_conf[:8])
        r_cur_conf = list(cur_conf[8:])
        left_id, right_id = init_conf_tamp_idx(command.body_paths)
        l_pre_path = BodyPath(0, [l_cur_conf, command.body_paths[left_id].path[0]], \
            joints=command.body_paths[left_id].joints, group_id=0)
        r_pre_path = BodyPath(1, [r_cur_conf, command.body_paths[right_id].path[0]], \
            joints=command.body_paths[right_id].joints, group_id=1)
        

        command.body_paths.insert(0, l_pre_path)
        command.body_paths.insert(0, r_pre_path)
        return command.body_paths
    

##########################################

 
    def node_init_common(self, para):
        self.para = para
        self.max_diff_info = None
        
        self.controller_pub_init()
        
        rospy.Subscriber('/puppet_left/joint_states', JointState, self.l_joint_state_callback)
        rospy.Subscriber('/puppet_right/joint_states', JointState, self.r_joint_state_callback)
        
        sleep_conf = (SINGLE_READY_POSE + list(GRIP_CMD['closed']))*2 
        self.target_arm_pose = np.array(sleep_conf)
        self.target_arm_single = None
        
        self.l_joint_qpos = None
        self.r_joint_qpos = None
        while self.l_joint_qpos is None or self.r_joint_qpos is None:
            print('no state received')
            rospy.sleep(1)

       
    
    def l_joint_state_callback(self, data):
        index_list = [data.name.index(name) for name in JOINT_NAMES]
        self.l_joint_qpos = [data.position[name_idx] for name_idx in index_list]
        self.l_gripper_val = data.position[-2]

        
    def r_joint_state_callback(self, data):
        index_list = [data.name.index(name) for name in JOINT_NAMES]
        self.r_joint_qpos = [data.position[name_idx] for name_idx in index_list]
        self.r_gripper_val = data.position[-2]
        
    def get_current_config(self):
        # concatenate the left and right arm joint positions
        cur_conf = np.hstack((self.l_joint_qpos, self.r_joint_qpos))
        return cur_conf

    def rbt_routine(self, event):
        raise NotImplementedError('rbt_routine method is not implemented')
    

    def controller_pub_init(self):
        raise NotImplementedError('controller_pub_init method is not implemented')
  
    def motion_plan(self, bodypath: BodyPath):
        raise NotImplementedError('motion_plan method is not implemented')
    
    def rbt_routine(self, event):
        raise NotImplementedError('rbt_routine method is not implemented')
    
    def arm_traj_control(self):
        raise NotImplementedError('arm_traj_control method is not implemented')
    
