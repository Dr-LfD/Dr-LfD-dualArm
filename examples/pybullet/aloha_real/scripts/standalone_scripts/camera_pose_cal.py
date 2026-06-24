import numpy as np

import sys
import json
import os
import re

root_path = next(('/'+ os.path.join(*os.path.dirname(os.path.abspath(__file__)).split(os.sep)[:i+1]) + os.sep
                    for i in range(len(os.path.dirname(os.path.abspath(__file__)).split(os.sep))) 
                    if os.path.dirname(os.path.abspath(__file__)).split(os.sep)[i] == 'pddlstream_aloha'), None)
sys.path.append(root_path) if root_path not in sys.path else None

from examples.pybullet.utils.pybullet_tools.utils import Pose, multiply, invert, Point, Euler, euler_from_quat

from pddlstream.utils import get_file_path

def T_from_launch(file_path):
    with open(file_path, 'r') as file:
        file_content = file.read()

    args_pattern = r'args="([-\d. ]+)'
    args_match = re.search(args_pattern, file_content)

    if args_match:
        args_values = args_match.group(1).split()
        translation = np.array([float(x) for x in args_values[:3]])
        rotation = np.array([float(x) for x in args_values[3:]])
        transform = [translation, rotation]
        print(f'Transform from {file_path}: {transform}')
    else:
        raise NameError("Args values not found in the file.")
    
    return transform
    
# ## back 435
# T_l_c = [np.array([0.474196, 0.324455, 0.762336]), np.array([0.404863, 0.452761, -0.572874, 0.550372])]
# T_r_c = [np.array([0.44092, -0.336009, 0.803077]), np.array([-0.451334, 0.387828, 0.560576, 0.575884])]

# ## front 435
# T_l_c = [np.array([0.438616, -0.178511, 0.806716]), np.array([-0.438192, 0.418452, 0.55245, 0.572437])]
# T_r_c = [np.array([0.454733, 0.204313, 0.845433]), np.array([0.435992, 0.42025, -0.559446, 0.56597])]
def calculate_extparam(T_o_r, T_o_l, out_json_path, right_ext_launch, left_ext_launch = None):
    T_r_c = T_from_launch(right_ext_launch)
    T_o_c_2 = multiply(T_o_r, T_r_c)
    print("calibration from right ", T_o_c_2[0], euler_from_quat(T_o_c_2[1]))


    if left_ext_launch is None:
        T_o_c = [np.array(T_o_c_2[0]), np.array(T_o_c_2[1])]
    else: 
        T_l_c = T_from_launch(left_ext_launch)
        T_o_c_1 = multiply(T_o_l, T_l_c)
        print("calibration from left ", T_o_c_1[0], euler_from_quat(T_o_c_1[1]))

        translation1 = np.array(T_o_c_1[0])
        translation2 = np.array(T_o_c_2[0])
        translation_avg = (translation1 + translation2)/2

        rotation1 = np.array(T_o_c_1[1])
        rotation2 = np.array(T_o_c_2[1])
        rotation_avg = (rotation1 + rotation2)/np.linalg.norm(rotation1 + rotation2)

        T_o_c = [translation_avg, rotation_avg]
        print("calibration from both ", T_o_c[0], euler_from_quat(T_o_c[1]))

    cam_pose = {'xyz': T_o_c[0].tolist(), 'wxyz': T_o_c[1].tolist()}

    with open(os.path.join(out_json_path), 'w') as f:
        json.dump(cam_pose, f)
    return cam_pose


if __name__ == '__main__':
    tempvis_dir = os.path.join(root_path, 'examples/pybullet/aloha_real/openworld_aloha/estimation/temp_vis/')

    # aloha
    T_o_r = Pose(point=Point(x=0.469), euler=Euler(yaw=-np.pi))
    T_o_l = Pose(point=Point(x=-0.469))

    # ## real435 at cybber port
    # out_json_path = os.path.join(tempvis_dir, 'camera_pose.json')
    # right_ext_launch = os.path.join(tempvis_dir, 'cam2right_extparam_optical.launch')
    # left_ext_launch = os.path.join(tempvis_dir, 'cam2left_extparam_optical.launch')
    # cam_pose = calculate_extparam(T_o_r, T_o_l,out_json_path, right_ext_launch, left_ext_launch)

    ## femto bolt at cyber port
    out_json_path = os.path.join(tempvis_dir, 'femto_bolt_camera_pose.json')
    right_ext_launch = os.path.join(tempvis_dir, 'right_femto.launch')
    left_ext_launch = os.path.join(tempvis_dir, 'left_femto.launch')
    cam_pose = calculate_extparam(T_o_r, T_o_l,out_json_path, right_ext_launch, left_ext_launch)


    # ## L515
    # out_json_path = os.path.join(tempvis_dir, 'camera_pose_l515.json')
    # right_ext_launch = os.path.join(tempvis_dir, 'l515_cam2right_extparam.launch')
    # left_ext_launch = os.path.join(tempvis_dir, 'l515_cam2left_extparam.launch')
    # cam_pose = calculate_extparam(T_o_r, T_o_l,out_json_path, right_ext_launch, left_ext_launch)

    # ## franka high
    # out_json_path = get_file_path(__file__, 'temp_vis/franka_cam_high.json')
    # right_ext_launch = get_file_path(__file__, 'temp_vis/franka_high_right_extparam.launch')
    # cam_pose = calculate_extparam(T_o_r, T_o_l,out_json_path, right_ext_launch)

    # # franka
    # T_o_r = Pose(point=Point(x=0.595, z = -0.2), euler=Euler(yaw=-np.pi))
    # T_o_l = Pose(point=Point(x=-0.595, z = -0.2))

    # ## franka front
    # out_json_path = get_file_path(__file__, 'temp_vis/franka_cam_front.json')
    # right_ext_launch = get_file_path(__file__, 'temp_vis/franka_front_right_extparam.launch')
    # cam_pose = calculate_extparam(T_o_r, T_o_l, out_json_path, right_ext_launch)

    # ## franka_high
    # out_json_path = get_file_path(__file__, 'temp_vis/franka_cam_high.json')
    # right_ext_launch = get_file_path(__file__, 'temp_vis/franka_high_right_extparam.launch')
    # cam_pose = calculate_extparam(T_o_r, T_o_l, out_json_path, right_ext_launch)
