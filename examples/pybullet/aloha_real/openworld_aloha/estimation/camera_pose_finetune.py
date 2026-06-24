import rospy
import sys
import tf
import tf2_ros
import geometry_msgs.msg

import termios
import tty
import os
import time
import math
import json

root_path = next(('/'+ os.path.join(*os.path.dirname(os.path.abspath(__file__)).split(os.sep)[:i+1]) + os.sep
                    for i in range(len(os.path.dirname(os.path.abspath(__file__)).split(os.sep))) 
                    if os.path.dirname(os.path.abspath(__file__)).split(os.sep)[i] == 'pddlstream_aloha'), None)
sys.path.append(root_path) if root_path not in sys.path else None

from examples.pybullet.aloha_real.openworld_aloha.policy_simp import get_compatible_campose, optical_to_camlink

def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def main():
    return


def print_status(status):
    sys.stdout.write('%-8s%-8s%-8s%-40s\r' % (status['mode'], status[status['mode']]['value'], status[status['mode']]['step'], status['message']))


def publish_status(broadcaster, status, child_frame):
    static_transformStamped = geometry_msgs.msg.TransformStamped()
    static_transformStamped.header.stamp = rospy.Time.now()
    static_transformStamped.header.frame_id = 'world'

    static_transformStamped.child_frame_id = child_frame
    static_transformStamped.transform.translation.x = status['x']['value']
    static_transformStamped.transform.translation.y = status['y']['value']
    static_transformStamped.transform.translation.z = status['z']['value']

    quat = tf.transformations.quaternion_from_euler(math.radians(status['roll']['value']),
                                                    math.radians(status['pitch']['value']),
                                                    math.radians(status['azimuth']['value']))
    static_transformStamped.transform.rotation.x = quat[0]
    static_transformStamped.transform.rotation.y = quat[1]
    static_transformStamped.transform.rotation.z = quat[2]
    static_transformStamped.transform.rotation.w = quat[3]
    broadcaster.sendTransform(static_transformStamped)

def save_calibrated(status, filename):
    quat = tf.transformations.quaternion_from_euler(math.radians(status['roll']['value']),
                                                    math.radians(status['pitch']['value']),
                                                    math.radians(status['azimuth']['value']))
    new_tf_map_to_optical = {'xyz': [status['x']['value'], status['y']['value'], status['z']['value']], 'wxyz': quat.tolist()}
    new_tf_map_to_camlink_list = optical_to_camlink(new_tf_map_to_optical)
    new_tf_map_to_camlink = {'xyz': new_tf_map_to_camlink_list[0], 'wxyz': new_tf_map_to_camlink_list[1]}
    with open(filename, 'w') as f:
        json.dump(new_tf_map_to_camlink, f, indent=4)

def extract_status(cam_pose):
    euler_angles = tf.transformations.euler_from_quaternion(cam_pose[1])
    x, y, z, yaw, pitch, roll = cam_pose[0][0], cam_pose[0][1], cam_pose[0][2], math.degrees(euler_angles[2]), math.degrees(euler_angles[1]), math.degrees(euler_angles[0])

    status = {'mode': 'pitch',
                  'x': {'value': x, 'step': 0.01},
                  'y': {'value': y, 'step': 0.01},
                  'z': {'value': z, 'step': 0.01},
                  'azimuth': {'value': yaw, 'step': 1},
                  'pitch': {'value': pitch, 'step': 1},
                  'roll': {'value': roll, 'step': 1},
                  'message': ''}
    
    return status

def finetune_status(status):
    kk = getch()
    status['message'] = ''
    try:
        key_idx = status_keys.index(kk)
        status['mode'] = list(status.keys())[key_idx]
    except ValueError as e:
        if kk.upper() == 'Q':
            sys.stdout.write('\n')
            exit(0)
        elif kk == '4':
            status[status['mode']]['value'] -= status[status['mode']]['step']
        elif kk == '6':
            status[status['mode']]['value'] += status[status['mode']]['step']
        elif kk == '-':
            status[status['mode']]['step'] /= 2.0
        elif kk == '+':
            status[status['mode']]['step'] *= 2.0
        else:
            status['message'] = 'Invalid key:' + kk

    return status

if __name__ == '__main__':
    tempvis_path = '/home/robotics/CoMa_code_clean/CoMa_ws/src/CoMa/examples/pybullet/aloha_real/openworld_aloha/estimation/temp_vis/'
    front_ext_json =  os.path.join(tempvis_path, 'camera_pose.json')
    back_ext_json =  os.path.join(tempvis_path, 'back_camera_pose.json')
    front_cam_dict = json.load(open(front_ext_json))
    front_cam_pose = get_compatible_campose(front_cam_dict)
    # front_cam_pose =  [front_cam_dict['xyz'], front_cam_dict['wxyz']]
    front_cam_frame = 'camera_2_color_optical_frame'
    # front_cam_frame = 'camera_2_link'
    front_status = extract_status(front_cam_pose)

    back_cam_dict = json.load(open(back_ext_json))
    back_cam_pose = get_compatible_campose(back_cam_dict)
    # back_cam_pose =  [back_cam_dict['xyz'], back_cam_dict['wxyz']]
    back_cam_frame = 'camera_1_color_optical_frame'
    back_status = extract_status(back_cam_pose)
    
    print('front_status:', front_status)
    print('back_status:', back_status)

    rospy.init_node('my_static_tf2_broadcaster')
    broadcaster = tf2_ros.StaticTransformBroadcaster()

    print
    print ('Press the following keys to change mode: x, y, z, (a)zimuth, (p)itch, (r)oll')
    print ('For each mode, press 6 to increase by step and 4 to decrease')
    print ('Press + to multiply step by 2 or - to divide')
    print
    print ('Press Q to quit')
    print

    status_keys = [key[0] for key in front_status.keys()]
    print ('%-8s%-8s%-8s%s' % ('Mode', 'value', 'step', 'message'))
    while True:
        publish_status(broadcaster, front_status, front_cam_frame)
        publish_status(broadcaster, back_status, back_cam_frame)


        status = finetune_status(back_status)

        print_status(back_status)
        publish_status(broadcaster, back_status, back_cam_frame)
        save_calibrated(back_status, os.path.join(tempvis_path, 'new_back.json'))

    #rospy.spin()