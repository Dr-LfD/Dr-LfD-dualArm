#!/usr/bin/env python

import sys
import os
root_path = next(('/'+ os.path.join(*os.path.dirname(os.path.abspath(__file__)).split(os.sep)[:i+1]) + os.sep
                    for i in range(len(os.path.dirname(os.path.abspath(__file__)).split(os.sep))) 
                    if os.path.dirname(os.path.abspath(__file__)).split(os.sep)[i] == 'pddlstream_aloha'), None)
sys.path.append(root_path) if root_path not in sys.path else None
os.chdir(root_path)

import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import json

from pddlstream.utils import  get_file_path

counter = {'got_color': False, 'got_depth': False, 'got_color_info': False, 'got_depth_info': False}
def save_image(msg, file_name):
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    cv2.imwrite(file_name, cv_image)

    global counter
    counter['got_color'] = True
    values = counter.values()
    if all(values):
        rospy.signal_shutdown('saved')

def save_depth_image(msg, file_name):
    bridge = CvBridge()
    depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    cv2.imwrite(file_name, depth_image)
    global counter
    counter['got_depth'] = True
    values = counter.values()
    if all(values):
        rospy.signal_shutdown('saved')


def save_camera_info(msg, file_name):
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
            'do_rectify': msg.roi.do_rectify
        }
    }
    
    with open(file_name, 'w') as f:
        json.dump(camera_info_dict, f, indent=4)
    
    global counter
    if 'depth' in file_name:
        counter['got_depth_info'] = True
    else:   
        counter['got_color_info'] = True
    values = counter.values()
    if all(values):
        rospy.signal_shutdown('saved')


def main(bag_name = 'realrobot'):
    rospy.init_node('image_and_camera_info_saver', anonymous=True)
    
    import os
    # root_dir = '/mnt/bags'
    root_dir = get_file_path(__file__, 'temp_vis')
    save_dir = os.path.join(root_dir, bag_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # delete all files in the directory
    files = os.listdir(save_dir)
    for file in files:
        os.remove(os.path.join(save_dir, file))

    rospy.Subscriber('/camera/aligned_depth_to_color/camera_info', CameraInfo, lambda msg: save_camera_info(msg, os.path.join(save_dir, 'depth_info.json')))
    rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, lambda msg: save_depth_image(msg, os.path.join(save_dir, 'depth_image.png')))
    rospy.Subscriber('/camera/color/camera_info', CameraInfo, lambda msg: save_camera_info(msg, os.path.join(save_dir, 'color_info.json')))
    rospy.Subscriber('/camera/color/image_raw', Image, lambda msg: save_image(msg, os.path.join(save_dir, 'color_image.png')))

    
    rospy.spin()
    return 0

if __name__ == '__main__':
    main(bag_name='taperight')
