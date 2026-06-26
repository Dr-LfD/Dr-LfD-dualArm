
import rospy
from std_msgs.msg import Float64
import numpy as np
import sys
import os
root_path = next(p for p in (os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *([os.pardir] * k))) for k in range(16)) if os.path.isfile(os.path.join(p, '.repo_root')))
sys.path.append(root_path) if root_path not in sys.path else None
os.chdir(root_path)
from examples.pybullet.utils.pybullet_tools.aloha_primitives import  Command, BodyPath

from examples.pybullet.aloha_real.openworld_aloha.primitives import GroupTrajectory, Graphstate

from examples.pybullet.aloha_real.scripts.aloha_ros_base import aloha_base, JOINT_NAMES, init_conf_tamp_idx, PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE

from pddlstream.utils import get_file_path

from examples.pybullet.aloha_real.openworld_aloha.primitives_test import  Sequence

from examples.pybullet.aloha_real.openworld_aloha.run_openworld import compute_TAMP_cmd
from examples.pybullet.aloha_real.scripts.aloha_tamp_constants import PERCEPT_ARM_POSE

from pddlstream.utils import get_file_path, read_pickle
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import json

ARM_GRIPPER_NAMES = JOINT_NAMES + ['left_finger', 'right_finger']


####################### perception part
def obtain_demo_data(cfg, cam_dir_mapping):
    h5_parent_dir = cfg['h5_parent_dir']
    input_hdf5_path = os.path.join(h5_parent_dir, f"episode_0.hdf5")
    # ensure path exist
    if not os.path.exists(input_hdf5_path):
        raise NameError("File not found!")
    
    import h5py
    with h5py.File(input_hdf5_path, "r") as f:
        print("Keys: {}".format(f.keys()))

        data_dict = {}

        obs_grp = f['observations']
        obs_qpos = obs_grp['qpos'][()]

        data_dict['qpos'] = obs_qpos

        # get color and depth
        for rs_cam in cam_dir_mapping.keys():
            color_key = f'color_img_{rs_cam}'
            depth_key = f'depth_img_{rs_cam}'
            camera_info_key = f'camera_info_{rs_cam}'
            data_dict[color_key] = f[color_key][()]
            data_dict[depth_key] = f[depth_key][()]
            data_dict[camera_info_key] = f[camera_info_key][()]


            save_dir = cam_dir_mapping[rs_cam]
            rgb_name = os.path.join(save_dir, 'color_image.png')
            depth_name = os.path.join(save_dir, 'depth_image.png')
            color_info_name = os.path.join(save_dir, 'color_info.json')
            depth_info_name = os.path.join(save_dir, 'depth_info.json')
            import cv2, json
            init_rgb = cv2.cvtColor(data_dict[color_key][0], cv2.COLOR_RGB2BGR)
            cv2.imwrite(rgb_name, init_rgb)
            init_depth_mm = data_dict[depth_key][0]*1000
            cv2.imwrite(depth_name, init_depth_mm.astype(np.uint16))
            decoded_data = data_dict[camera_info_key].decode('utf-8')
            json_data = json.loads(decoded_data)                
            with open(color_info_name, 'w') as f:
                json.dump(json_data, f, indent=4)
            with open(depth_info_name, 'w') as f:
                json.dump(json_data, f, indent=4)
        
        return save_dir
        
class observation_to_file(object):
    def __init__(self, cam_dir_mapping, lfd_env=None, clear_dir=True, robot_name='aloha', cam_config=None):
        if clear_dir:
            # empty the directory
            for cam, temp_img_dir in cam_dir_mapping.items():
                for file in os.listdir(temp_img_dir):
                    os.remove(os.path.join(temp_img_dir, file))

        self.cam_dir_mapping = cam_dir_mapping
        # Optional full YAML configuration (e.g., sgBase.yaml)
        self.cam_config = cam_config
        # Per-camera data storage populated from ROS callbacks
        self.cam_data = {}

        if robot_name == 'aloha':
            self.obtain_aloha_sensor_data()
        else:
            self.obtain_franka_sensor_data(lfd_env.image_recorder, cam_dir_mapping)

    def obtain_franka_sensor_data(self, img_recorder, cam_dir_mapping):
        all_obs_dict = img_recorder.get_current_data()
        for cam in cam_dir_mapping.keys():
            save_dir = cam_dir_mapping[cam]
            cam_rgb = all_obs_dict[cam]['color']
            cam_bgr = cv2.cvtColor(cam_rgb, cv2.COLOR_RGB2BGR)
            cam_depth = all_obs_dict[cam]['depth']
            cam_depth_mm = (cam_depth * 1000).astype(np.uint16)
            cam_info = all_obs_dict[cam]['camera_info']
            cv2.imwrite(os.path.join(save_dir, 'color_image.png'), cam_bgr)
            cv2.imwrite(os.path.join(save_dir, 'depth_image.png'), cam_depth_mm)
            with open(os.path.join(save_dir, 'depth_info.json'), 'w') as f:
                json.dump(cam_info, f, indent=4)
            with open(os.path.join(save_dir, 'color_info.json'), 'w') as f:
                json.dump(cam_info, f, indent=4)

        print("All sensor data saved")

    # As back camera is not in the img_recorder, we use a ROS-based version of the perception fn.
    def _empty_cam_data(self):
        """Return an empty camera data dict."""
        return {
            'cur_color_img': None,
            'cur_depth_img': None,
            'color_info': None,
            'depth_info': None,
        }

    def _get_depth_topic_mapping(self):
        """
        Build a mapping from ROS camera name (e.g., 'camera_2') to the depth topic to subscribe to.

        If cam_config (YAML) is provided and contains <camX>_depth_topic fields, use those.
        Each value can be:
          - a full topic (starting with '/') like '/camera/depth/image_raw', or
          - a suffix like 'depth/image_raw', which will be expanded to '/<cam_name>/depth/image_raw'.

        Otherwise, fall back to the original hard-coded behavior:
          - 'camera_2' -> '/camera_2/aligned_depth_to_color/image_raw'
          - others     -> '/<name>/depth/image_raw'
        """
        mapping = {}

        if self.cam_config is not None:
            active_cams = self.cam_config.get('active_cams', [])
            for cam in active_cams:
                rs_name = self.cam_config.get(f'{cam}_name')
                if rs_name is None:
                    continue
                depth_spec = self.cam_config.get(f'{cam}_depth_topic', None)
                if depth_spec is None:
                    # Fallback same as below
                    if rs_name == 'camera_2':
                        topic = f'/{rs_name}/aligned_depth_to_color/image_raw'
                    else:
                        topic = f'/{rs_name}/depth/image_raw'
                else:
                    if depth_spec.startswith('/'):
                        topic = depth_spec
                    else:
                        topic = f'/{rs_name}/{depth_spec}'
                mapping[rs_name] = topic

        # Ensure we have a topic for every camera we are saving
        for rs_name in self.cam_dir_mapping.keys():
            if rs_name in mapping:
                continue
            if rs_name == 'camera_2':
                mapping[rs_name] = f'/{rs_name}/aligned_depth_to_color/image_raw'
            else:
                mapping[rs_name] = f'/{rs_name}/depth/image_raw'

        return mapping

    def obtain_aloha_sensor_data(self):
        """
        Obtain and store sensor data for the ALOHA setup using raw ROS topics.

        This version supports an arbitrary set of active cameras given by
        self.cam_dir_mapping. For each camera name (ROS namespace), we subscribe
        to:
          - /<cam>/color/camera_info
          - <configured depth topic>
          - /<cam>/color/image_raw
        """
        # Initialize per-camera data containers
        self.cam_data = {
            cam_name: self._empty_cam_data()
            for cam_name in self.cam_dir_mapping.keys()
        }

        depth_topic_mapping = self._get_depth_topic_mapping()

        for cam_name, save_dir in self.cam_dir_mapping.items():
            # Camera info subscriber
            rospy.Subscriber(
                f'/{cam_name}/color/camera_info',
                CameraInfo,
                lambda msg, sd=save_dir, cn=cam_name: self.save_camera_info(
                    msg,
                    [
                        os.path.join(sd, 'depth_info.json'),
                        os.path.join(sd, 'color_info.json'),
                    ],
                    cam=cn,
                ),
            )

            # Depth image subscriber
            depth_topic = depth_topic_mapping.get(cam_name)
            rospy.Subscriber(
                depth_topic,
                Image,
                lambda msg, sd=save_dir, cn=cam_name: self.save_depth_image(
                    msg,
                    os.path.join(sd, 'depth_image.png'),
                    cam=cn,
                ),
            )

            # Color image subscriber
            rospy.Subscriber(
                f'/{cam_name}/color/image_raw',
                Image,
                lambda msg, sd=save_dir, cn=cam_name: self.save_image(
                    msg,
                    os.path.join(sd, 'color_image.png'),
                    cam=cn,
                ),
            )

        # Block until we have all required sensor data
        while not self.got_all():
            print("Waiting for sensor data")
            rospy.sleep(1)
        print("All sensor data saved")

    def got_all(self):
        """Return True when we have received all required sensor data."""
        if not self.cam_data:
            return False
        flag = True
        for _, data in self.cam_data.items():
            for v in data.values():
                flag = flag and (v is not None)
        return flag

    def save_image(self, msg, file_name, cam='front'):
        """Save a color image from a ROS message."""
        if self.got_all():
            return

        bridge = CvBridge()
        frame_data = self.cam_data[cam]
        frame_data['cur_color_img'] = bridge.imgmsg_to_cv2(msg, "bgr8")
        cv2.imwrite(file_name, frame_data['cur_color_img'])

    def save_depth_image(self, msg, file_name, cam='front'):
        """Save a depth image from a ROS message."""
        if self.got_all():
            return

        bridge = CvBridge()
        frame_data = self.cam_data[cam]
        frame_data['cur_depth_img'] = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        cv2.imwrite(file_name, frame_data['cur_depth_img'])

    def save_camera_info(self, msg, file_names, cam='front'):
        """Save camera info message to one or more JSON files."""
        if self.got_all():
            return

        camera_info_dict = {
            'height': msg.height,
            'width': msg.width,
            'distortion_model': msg.distortion_model,
            'D': msg.D,
            'K': msg.K,
            'R': msg.R,
            'P': msg.P,
            'binning_x': msg.binning_x,
            'binning_y': msg.binning_y,
            'roi': {
                'x_offset': msg.roi.x_offset,
                'y_offset': msg.roi.y_offset,
                'height': msg.roi.height,
                'width': msg.roi.width,
                'do_rectify': msg.roi.do_rectify,
            },
        }

        frame_data = self.cam_data[cam]
        frame_data['depth_info'] = camera_info_dict
        frame_data['color_info'] = camera_info_dict

        for file_name in file_names:
            with open(file_name, 'w') as f:
                json.dump(camera_info_dict, f, indent=4)

###############################

class openworld_base(aloha_base):
    def __init__(self,   para, init_ros = True, only_perception = False):
        if only_perception:
            rospy.logwarn("Warning: Be sure you are only doing perception!")
            return
        # custom initialization goes here
        self.use_precomputed_path = para['execution']['use_precomputed_path']
        self.task_name = para['task_name']
        self.pre_obj_names = para['text_prompt'].strip('.')
        super().__init__(para, init_ros=init_ros)
        
##############################
    def motion_plan(self, body_path: BodyPath):
        # return body_path
        total_steps = self.para['execution']['refine_num']
        num_steps = np.ceil(total_steps / len(body_path.path)-1).astype(int)
        refined_cmd = body_path.refine(num_steps=num_steps)
        # # debug
        # if len(refined_cmd.path) <30:
        #     print(f"Refine failed, len(refined_cmd.path): {len(refined_cmd.path)}, total_steps: {total_steps}")
        return refined_cmd

    def calc_tamp_cmd(self, **kwargs):
        if not self.use_precomputed_path:
        # # real-time computing

            root_dir = get_file_path(__file__, '../openworld_aloha/estimation/temp_vis')
            save_dir = os.path.join(root_dir, 'realrobot')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            back_save_dir = os.path.join(root_dir, 'back_realrobot')
            if not os.path.exists(back_save_dir):
                os.makedirs(back_save_dir)
            cam_dir_mapping = {'camera_2': save_dir, 'camera_1': back_save_dir}

            data_saver = observation_to_file(cam_dir_mapping)

            # grasp experiment
            # sequence = test_grasp_real(env_type='real', use_gui = True, file_path=save_dir)

            # TAMP
            sequence = compute_TAMP_cmd(self.para, env_type = 'real', task_name= self.task_name, 
                                        pre_obj_names = self.pre_obj_names,  file_path=save_dir, **kwargs)
        else:
            # pre-computed plan
            cmd_path = os.path.join(root_path, 'temp/sequence_pick.pkl')
            sequence = read_pickle(cmd_path)

        path_list = self.process_cmd(sequence)
        return path_list


    
    def process_cmd(self, sequence: Sequence):
        bodypath_list = []
        # convert sequence to body path, assuming initially gripper empty
        gripper_attachments = {'left': [], 'right': []}
        self.gripper_joints = {'left': [], 'right': []}
        self.arm_joints = {'left': [], 'right': []}
        last_joint_cmd = None
        marker_iter = iter(getattr(sequence, 'graphstate_markers', ()))
        next_marker = next(marker_iter, None)

        def append_graphstate(graphstate):
            print(f"Command {graphstate} is imitation learning")
            if graphstate.commands is None:
                graphstate._build_commands()
            bimanual_bodypath = BodyPath(0, [graphstate], group_id=-1)
            bodypath_list.append(bimanual_bodypath)
            for sub_cmd in graphstate.commands:
                if not isinstance(sub_cmd, GroupTrajectory):
                    continue
                arm_side = sub_cmd.group.split('_')[0]
                if len(sub_cmd.joints) == 2:
                    gripper_attachments[arm_side] = sub_cmd.attachments.copy()

        for i, cmd in enumerate(sequence.commands):
            while next_marker is not None and next_marker[0] == i:
                append_graphstate(next_marker[1])
                next_marker = next(marker_iter, None)
            if isinstance(cmd, GroupTrajectory):
                arm_side = cmd.group.split('_')[0]
                group_index = 0 if 'left' in arm_side else 1

                if 'arm' in cmd.group:
                    # if the 1st wp of the next cmd is not the same as the last wp of the current cmd
                    if last_joint_cmd is not None and not np.allclose(last_joint_cmd.path[-1][:6], cmd.path[0][:6]):
                        np.insert(cmd.path, 0, last_joint_cmd.path[-1])
                    bodypath = BodyPath(0, cmd.path, joints=cmd.joints, attachments=gripper_attachments[arm_side], group_id=group_index)
                    bodypath_list.append(bodypath)
                    last_joint_cmd = cmd
                    self.arm_joints[arm_side] = cmd.joints
                elif 'gripper' in cmd.group:
                    self.gripper_joints[arm_side] = cmd.joints
                    gripper_attachments[arm_side] = cmd.attachments.copy()
            elif isinstance(cmd, Graphstate):
                raise RuntimeError("Graphstate must be stored in sequence.graphstate_markers")
            else:
                # raise NotImplementedError
                print(f"Command {cmd} is not included")

        while next_marker is not None:
            append_graphstate(next_marker[1])
            next_marker = next(marker_iter, None)
                    
        bodypaths_cmd = Command(bodypath_list)

        adjusted_path = self.percept_adjust(bodypaths_cmd.body_paths)

        return adjusted_path

    
    def percept_adjust(self, body_paths):
        cur_conf = self.get_current_config()
        l_cur_conf = list(cur_conf[:6])
        r_cur_conf = list(cur_conf[6:])
        left_id, right_id = init_conf_tamp_idx(body_paths)

        parepare_path = []

        for side, id, arm_conf in zip(['left', 'right'], [left_id, right_id], [l_cur_conf, r_cur_conf]):
            group_index = 0 if 'left' in side else 1
            
            if id is not None:
                qpos_tgt = list(body_paths[id].path[0])
                joints_full = list(body_paths[id].joints)
                arm_prepath = BodyPath(0, [arm_conf, qpos_tgt], \
                    joints=joints_full, group_id = group_index, attachments=body_paths[id].attachments)
            else:
                # return to percept pose
                qpos_tgt = list(PERCEPT_ARM_POSE)
                joints_full = list(self.arm_joints[side])
                
                # if the command does not contain this arm
                if len(joints_full) == 0:
                    continue

                attached = []
                if self.holding[side] != -1:
                    attached.append(self.holding[side])

                arm_prepath = BodyPath(0, [arm_conf, qpos_tgt], \
                    joints=joints_full, group_id = group_index, attachments=attached)

            if not self.use_precomputed_path:
                arm_prepath = arm_prepath.refine(num_steps=self.para['execution']['refine_num'])
            parepare_path.append(arm_prepath)


        adjusted_path = parepare_path + body_paths

        # TODO: percept and correct the grasp pose, replan if necessary

        return adjusted_path


        


#######################################        


