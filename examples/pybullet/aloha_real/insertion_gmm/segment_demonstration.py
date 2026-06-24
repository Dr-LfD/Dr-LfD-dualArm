# from insertion_eepose_reader

import h5py
import numpy as np
from .keypose_extraction import _find_keypose_idx_insertion,_load_traj_from_fileptr, _smooth, DT

def read_qpos_data(input_hdf5_path, from_sim = True):
    # ensure path exist
    if not os.path.exists(input_hdf5_path):
        raise NameError("File not found!")
    with h5py.File(input_hdf5_path, "r") as f:
        print("Keys: {}".format(f.keys()))
        keys = list(f.keys())

        print("\n--- Action --- ")

        action = f['action'][()]
        print("action.shape: {}".format(action.shape))

        obs_grp = f['observations']
        obs_keys = list(obs_grp.keys())


        print(f"\nObservation keys: {obs_keys}")

        print("\n--- Joint --- ")

        # obs_efforts = obs_grp['effort'][()]
        obs_qpos = obs_grp['qpos'][()]
        obs_qvel = obs_grp['qvel'][()]

        # print("joint_efforts.shape: {}".format(obs_efforts.shape))
        print("joint_qpos.shape: {}".format(obs_qpos.shape))
        print("joint_qvel.shape: {}".format(obs_qvel.shape))

        print("\n--- Observations --- ")

        obs_images = obs_grp['images']
        obs_image_keys = list(obs_images.keys())
        print(f"obs_image_keys: {obs_image_keys}")

        if from_sim:
            obs_cam_high = obs_images['top'][()]
        else:
            obs_cam_high = obs_images['cam_high'][()]

        print("obs_cam_high.shape: {}".format(obs_cam_high.shape))
        
        this_traj = _load_traj_from_fileptr(f)
        return obs_qpos, obs_cam_high, this_traj

def switch_detect(smoothed_qpos, gripper_change_rate, x_threshold = 0.1,  dx_threshold = -1, type = 'grasp', clip_start = 0):
    # gripper_data = qpos_data[:, -1]
    # smoothed_qpos = _smooth(gripper_data, smooth_window_size)
    # gripper_change_rate = np.diff(smoothed_qpos) / DT

    if type == 'grasp':
        abrupt_window_size = 10
        for id in range(clip_start, len(gripper_change_rate)-abrupt_window_size+1):
            dx_window = gripper_change_rate[id:id+abrupt_window_size]
            x_window = smoothed_qpos[id:id+abrupt_window_size]
            if  np.min(dx_window)< dx_threshold:
                if x_window[-1] < x_threshold:
                    id_list = np.arange(id+abrupt_window_size, id+len(smoothed_qpos))
                    return id_list
    elif type == 'release':
        abrupt_window_size = 10
        for id in range(clip_start, len(gripper_change_rate)-abrupt_window_size+1):
            dx_window = gripper_change_rate[id:id+abrupt_window_size]
            x_window = smoothed_qpos[id:id+abrupt_window_size]
            if  np.max(dx_window)> dx_threshold:
                if x_window[-1] > x_threshold:
                    id_list = np.arange(id+abrupt_window_size, id+len(smoothed_qpos))
                    return id_list
            
    raise NameError("No grasp detected!")

    




def play_vid(img_seq):
    import cv2

    # input: img_seq: (500, 128, 128, 3)
    # play each img and pause for 0.01s
    for id, img in enumerate(img_seq):
        
        #  display the id on the upper-left corner of the image
        cv2.putText(img, str(id), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        cv2.imshow('image', img)
        cv2.waitKey(1)


import sys
import os
EXE_FOLDER = next(('/'+ os.path.join(*os.path.dirname(os.path.abspath(__file__)).split(os.sep)[:i+1]) + os.sep
                    for i in range(len(os.path.dirname(os.path.abspath(__file__)).split(os.sep))) 
                    if os.path.dirname(os.path.abspath(__file__)).split(os.sep)[i] == 'pddlstream_aloha'), None)
sys.path.append(EXE_FOLDER)
from examples.pybullet.aloha_real.scripts.aloha_tamp_constants import qpos_to_xyzrpy

# start from right gripper close, end at right gripper release
def extract_transfer_ids(left_qpos_data, right_qpos_data, left_dq, right_dq,  seq_len = 500, \
                         clip_start = 0, clip_end = 0):
    # # below for scripted
    # leftarm_insertion_ids = switch_detect(left_qpos_data, x_threshold=0.62, dx_threshold=-0.7)
    # # below for human
    # leftarm_insertion_ids = switch_detect(left_qpos_data, x_threshold=0.95, dx_threshold=-0.4)
    # rightarm_insertion_ids = switch_detect(right_qpos_data, x_threshold=0.62, dx_threshold=-0.7)

    # below for real tape
    # graspl_ids = switch_detect(left_qpos_data, left_dq, x_threshold=0.35, dx_threshold=-0.7, type='grasp')
    grasp_id = switch_detect(right_qpos_data, right_dq, x_threshold=0.35, dx_threshold=-0.7, type='grasp')
    releaser_ids = switch_detect(right_qpos_data, right_dq, x_threshold=0.6, dx_threshold= 0.7, clip_start=clip_start, type='release')

    start_id = grasp_id[0]
    end_id =  max(start_id +clip_end, releaser_ids[0] - clip_end)
    common_ids = np.arange(start_id, end_id)

    if len(common_ids) == 0:
        raise NameError("No common ids found!")
    return common_ids




def extract_insertion_ids(left_qpos_data, right_qpos_data, left_dq, right_dq, seq_len = 500, clip_start = 0):
    leftarm_insertion_ids = switch_detect(left_qpos_data, left_dq, x_threshold=0.8, dx_threshold=-0.7)
    rightarm_insertion_ids = switch_detect(right_qpos_data, right_dq, x_threshold=0.36, dx_threshold=-0.7)
    start_id = max(leftarm_insertion_ids[0], rightarm_insertion_ids[0])
    common_ids = np.arange(start_id, seq_len-clip_start)
    return common_ids

def hdf5_to_txt(input_hdf5_path,  output_txt_path=None, extraction_fn = None, postprocess_fn=None, play_orig_vid = False, play_seg_vid = False, draw_curve = False, from_sim = True):
    if postprocess_fn is None or extraction_fn is None:
        raise NameError("Please specify the required function!")

    obs_qpos, obs_cam_high, trajectory_data = read_qpos_data(input_hdf5_path, from_sim=from_sim)


    if play_orig_vid:
        play_vid(obs_cam_high)

    seq_len = obs_qpos.shape[0]


    clip_start = 100
    clip_end = 100
    left_qpos_data = obs_qpos[clip_start:, :7]
    right_qpos_data = obs_qpos[clip_start:, 7:]
    cam_clip = obs_cam_high[clip_start:]


    # process and visualize the gripper values
    smoothed_left_qpos = _smooth(left_qpos_data[:, -1], 5)
    smoothed_right_qpos = _smooth(right_qpos_data[:, -1], 5)
    left_qvel = np.diff(smoothed_left_qpos, axis=0) / DT
    right_qvel = np.diff(smoothed_right_qpos, axis=0) / DT

    import matplotlib.pyplot as plt

    plt.clf()
    plt.plot(smoothed_left_qpos, label='smoothed_left_qpos')
    plt.plot(smoothed_right_qpos, label='smoothed_right_qpos')

    plt.plot(left_qvel, label= 'left_gripper_change_rate')
    plt.plot(right_qvel, label= 'right_gripper_change_rate')
    plt.legend()
    pic_name = 'gripper_vals.png'
    pic_path = os.path.join('temp', pic_name)
    plt.savefig(pic_path)
    if draw_curve:
        plt.show()

    common_ids = extraction_fn(smoothed_left_qpos, smoothed_right_qpos, left_qvel, right_qvel, \
                               seq_len = seq_len , clip_start = clip_start, clip_end = clip_end)
    
    # another standard: the first time that right gripper below 0.3, and 
    
    if play_seg_vid:
        play_vid(cam_clip[common_ids] )

    left_insertion_cfgs = left_qpos_data[common_ids]
    right_insertion_cfgs = right_qpos_data[common_ids]

    insertion_data = postprocess_fn(left_insertion_cfgs, right_insertion_cfgs)

    # save into txt
    if output_txt_path is None:
        print('debugging')
        return
    with open (output_txt_path, "a") as f:
        np.savetxt(f, insertion_data)



def qpos2eepos_postprocessor(left_insertion_cfgs, right_insertion_cfgs):
    ee_left_flat_ls =  [qpos_to_xyzrpy(left_cfg, 0) for left_cfg in left_insertion_cfgs]
    ee_right_flat_ls = [qpos_to_xyzrpy(right_cfg, 1) for right_cfg in right_insertion_cfgs]

    # data format: idx, ee_left, ee_right
    insertion_data = np.hstack((np.arange(len(left_insertion_cfgs)).reshape(-1, 1), ee_left_flat_ls, ee_right_flat_ls))
    return insertion_data

def rawqpos_postprocessor(left_insertion_cfgs, right_insertion_cfgs):
    # data format: idx, ee_left, ee_right
    insertion_data = np.hstack((np.arange(len(left_insertion_cfgs)).reshape(-1, 1), left_insertion_cfgs[:, :6], right_insertion_cfgs[:, :6]))
    return insertion_data

import os
def get_file_path(file, rel_path):
    directory = os.path.dirname(os.path.abspath(file))
    return os.path.join(directory, rel_path)


def segment_insertion():
####################### insertion ############################
    # input_hdf5_path = "/home/user/yzchen_ws/TAMP-ubuntu22/ALOHA/act/dataset/sim_insertion_human/episode_8.hdf5"
    # input_hdf5_path = "/home/user/yzchen_ws/TAMP-ubuntu22/ALOHA/act/dataset/sim_insertion_scripted/sim_insertion_scripted/episode_8.hdf5"
    # hdf5_to_txt(input_hdf5_path)

    file_dir = "/home/user/yzchen_ws/TAMP-ubuntu22/ALOHA/act/dataset/sim_insertion_human/"
    # file_dir = "/home/user/yzchen_ws/TAMP-ubuntu22/ALOHA/act/dataset/sim_insertion_scripted/sim_insertion_scripted/"

    for i in range(50):
        input_hdf5_path = file_dir + f"episode_{i}.hdf5"
        output_txt_path = get_file_path(__file__ ,"insertion_human_qpos.txt")
        hdf5_to_txt(input_hdf5_path, output_txt_path, extraction_fn=extract_insertion_ids, postprocess_fn= rawqpos_postprocessor, play_seg_vid = False, draw_curve = False)
        print(f"Process file {i} done!")

def segment_transfer():
################### transfer ############################
    # file_dir = "/home/user/yzchen_ws/TAMP-ubuntu22/ALOHA/act/dataset/sim_transfer_cube_human/sim_transfer_cube_human/"

    file_dir = "/home/user/yzchen_ws/TAMP-ubuntu22/ALOHA/act/dataset/aloha_transfer_tape/"

    for i in range(0, 50):
        input_hdf5_path = file_dir + f"episode_{i}.hdf5"
        output_txt_path = get_file_path(__file__ ,"transfer_tape_jointdata.txt")
        hdf5_to_txt(input_hdf5_path, output_txt_path, extraction_fn=extract_transfer_ids, postprocess_fn= rawqpos_postprocessor, play_orig_vid=False, play_seg_vid = True, draw_curve = False, from_sim=False)
        print(f"Process file {i} done!")




if __name__ == "__main__":
    segment_transfer()











