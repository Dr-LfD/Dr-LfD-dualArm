import numpy as np
import os
import cv2
import json

import sys
root_path = next(('/'+ os.path.join(*os.path.dirname(os.path.abspath(__file__)).split(os.sep)[:i+1]) + os.sep
                    for i in range(len(os.path.dirname(os.path.abspath(__file__)).split(os.sep))) 
                    if os.path.dirname(os.path.abspath(__file__)).split(os.sep)[i] == 'pddlstream_aloha'), None)
sys.path.append(root_path) if root_path not in sys.path else None
os.chdir(root_path)

from examples.pybullet.aloha_real.insertion_gmm.segment_demonstration import play_vid,  _smooth
from examples.pybullet.aloha_real.scripts.aloha_tamp_constants import PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN, DT, qpos_to_eetrans
import h5py
from examples.pybullet.aloha_real.openworld_aloha.simple_worlds import  render_pose


MJ2BULLET_OFFSET = np.array([0, -0.5, 0])

def compute_eepath(joint_val_list, robot_id):

    ee_path = [ qpos_to_eetrans(joint_val, robot_id) for joint_val in joint_val_list]
    return ee_path

def get_sparse_eeposes(ee_path, candidate_range, obj_center, max_dist = 0.2, output_len = 5):
    candidate_ids = []
    for idx in candidate_range:
        eetrans_t = ee_path[idx][:3, 3].reshape(3)

        grasp_dist = np.linalg.norm(eetrans_t - obj_center)  
        if grasp_dist < max_dist:
            candidate_ids.append(idx)

    sampled_ids = np.linspace(candidate_ids[0], candidate_ids[-1], num=output_len).astype(int)
    return list(sampled_ids)


def filter_grasp_by_obj(obj_center, ee_path,  grasp_ids,  \
                        threshold = 0.1, extend_num = 60):
    ## first enlarge the list to 2*extend_num + num(grasp_ids), then shrink to 32. 
    filtered_grasp_ids = []

    dist_traj = []
    for idx in grasp_ids:
        eetrans_t = ee_path[idx][:3, 3].reshape(3)

        grasp_dist = np.linalg.norm(eetrans_t - obj_center)  
        dist_traj.append(grasp_dist)
        if grasp_dist < threshold:
            filtered_grasp_ids.append(idx)

    ## TODO: get a sparse re- and post-grasp poses
    lower_bound = max(0, filtered_grasp_ids[0] - extend_num)
    upper_bound = min(len(ee_path), filtered_grasp_ids[-1] + extend_num)
    # return list(range(lower_bound, upper_bound))
    pre_grasp_ids = get_sparse_eeposes(ee_path, range(lower_bound, filtered_grasp_ids[0]), obj_center, max_dist= 2*threshold, output_len= len(filtered_grasp_ids)//2 )
    post_grasp_ids = get_sparse_eeposes(ee_path, range(filtered_grasp_ids[-1], upper_bound), obj_center, max_dist= 2*threshold, output_len= len(filtered_grasp_ids)//2 )
    filtered_grasp_ids = pre_grasp_ids + filtered_grasp_ids + post_grasp_ids
    return filtered_grasp_ids



def actions2grasps(raw_jpose, obj_pcs = None,  plot = False):
    l_joint_vals = raw_jpose[:, :6]
    l_gripper_val = PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(raw_jpose[:, 6])
    r_joint_vals = raw_jpose[:, 7:13]
    r_gripper_val = PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(raw_jpose[:, 13])

    smoothed_gripper_ql = _smooth(l_gripper_val)
    gripper_dql = np.diff(smoothed_gripper_ql, axis=0) / DT

    l_grasp_ids = switch_ids(smoothed_gripper_ql, gripper_dql, x_threshold=1.25, dx_threshold=-0.5, type='grasp')
    # l_release_ids = switch_ids(smoothed_gripper_ql, gripper_dql, x_threshold=0.6, dx_threshold= 0.7, clip_start=100, type='release')

    smoothed_gripper_qr = _smooth(r_gripper_val)
    gripper_dqr = np.diff(smoothed_gripper_qr, axis=0) / DT

    r_grasp_ids = switch_ids(smoothed_gripper_qr, gripper_dqr, x_threshold=0.5, dx_threshold=-0.7, type='grasp')
    # r_release_ids = switch_ids(smoothed_gripper_qr, gripper_dqr, x_threshold=0.6, dx_threshold= 0.7, clip_start=100, type='release')

    left_eepath = compute_eepath(l_joint_vals, 0)
    right_eepath = compute_eepath(r_joint_vals, 1)

    # filter the ids using the distance to the object
    if obj_pcs is not None:
        socket_pc, peg_pc = obj_pcs
        socket_pc = socket_pc.reshape(-1, 3)
        peg_pc = peg_pc.reshape(-1, 3)
        socket_center = np.mean(socket_pc, axis=0)
        peg_center = np.mean(peg_pc, axis=0)
        # socket_center += MJ2BULLET_OFFSET
        # peg_center += MJ2BULLET_OFFSET
        l_grasp_ids = filter_grasp_by_obj(socket_center, left_eepath, l_grasp_ids)
        r_grasp_ids = filter_grasp_by_obj(peg_center, right_eepath, r_grasp_ids)


 
    if plot:
        import matplotlib.pyplot as plt
        plt.clf()
        plt.plot(smoothed_gripper_ql, label='smoothed_left_qpos')
        plt.plot(smoothed_gripper_qr, label='smoothed_right_qpos')

        plt.plot(gripper_dql, label= 'left_gripper_change_rate')
        plt.plot(gripper_dqr, label= 'right_gripper_change_rate')

        # plt.plot(l_dist_traj, label='the distance from left gripper to the object')
        # plt.plot(r_dist_traj, label='the distance from right gripper to the object')


        # draw grasp and release points
        plt.scatter(l_grasp_ids, smoothed_gripper_ql[l_grasp_ids], c='r', label='left_grasp')
        # plt.scatter(l_release_ids, smoothed_gripper_ql[l_release_ids], c='g', label='left_release')
        plt.scatter(r_grasp_ids, smoothed_gripper_qr[r_grasp_ids], c='b', label='right_grasp')
        # plt.scatter(r_release_ids, smoothed_gripper_qr[r_release_ids], c='y', label='right_release')
        plt.legend()
        pic_path =  test_hdf5_path.replace('.hdf5', '_grasp_release.png')
        # pic_path = os.path.join('temp', pic_name)
        plt.savefig(pic_path)        

    if  len(l_grasp_ids) ==0 or len(r_grasp_ids) ==  0:
        return {}
    
    l_grasp_poses = [ee for i, ee in enumerate(left_eepath) if i in l_grasp_ids]
    r_grasp_poses = [ee for i, ee in enumerate(right_eepath) if i in r_grasp_ids]
    l_grasp_actions = [l_gripper_action for i, l_gripper_action in enumerate(l_gripper_val) if i in l_grasp_ids]
    r_grasp_actions = [r_gripper_action for i, r_gripper_action in enumerate(r_gripper_val) if i in r_grasp_ids]


    last_grasp_id = max(l_grasp_ids+ r_grasp_ids)
    # As the data loader in diffgen has already processed the data, we don;t have to filter the ids here
    unfiltered_jpose_ids = list(range(last_grasp_id, len(raw_jpose)))
    # jpose_ids= unfiltered_jpose_ids
    jpose_ids= filter_jpose_ids(unfiltered_jpose_ids, left_eepath, right_eepath, eedist_threshold = 0.2)
    if len(jpose_ids) == 0:
        raise ValueError("No valid joint pose found!")
    filtered_jpose_vals = raw_jpose[jpose_ids]

    # TODO： assign grasp poses using symb mask
    peg_grasps = {'grasp_poses':  r_grasp_poses, 'grasp_actions': r_grasp_actions, 'obj_points': peg_pc, 'arm': 'right'}
    socket_grasps = {'grasp_poses': l_grasp_poses, 'grasp_actions': l_grasp_actions, 'obj_points': socket_pc, 'arm': 'left'}
    # start_grasps = {'left_grasp_ids': l_grasp_ids, 'right_grasp_ids': r_grasp_ids, 'obj_points': start_obj_points}
    # end_releases = {'left_release_ids': l_release_ids, 'right_release_ids': r_release_ids, 'obj_points': end_obj_points}
    contact_info_dict = {'peg_grasps': peg_grasps, 'socket_grasps': socket_grasps, 'pred_joint_vals': filtered_jpose_vals}

    
    return contact_info_dict

def filter_jpose_ids(unfiltered_jpose_ids, left_eepath, right_eepath, eedist_threshold = 0.15, lift_threshold = 0.1):
    filtered_jpose_ids= []
    for idx in unfiltered_jpose_ids:
        l_eetrans = left_eepath[idx][:3, 3].reshape(3)
        r_eetrans = right_eepath[idx][:3, 3].reshape(3)
        l_dist = np.linalg.norm(l_eetrans - r_eetrans)
        if l_dist < eedist_threshold:
            continue

        l_ee_z = l_eetrans[2]
        r_ee_z = r_eetrans[2]
        if l_ee_z < lift_threshold or r_ee_z < lift_threshold:
            continue
        
        filtered_jpose_ids.append(idx)

    return filtered_jpose_ids
    # assert len(filtered_jpose_ids) > 32, "Not enough joint pose ids found!"
    # selected_idices = np.linspace(filtered_jpose_ids[0], filtered_jpose_ids[-1] , num=32).astype(int)

    # return list(selected_idices)

def switch_ids(smoothed_qpos, gripper_change_rate, x_threshold = 0.1,  dx_threshold = -1, type = 'grasp', clip_start = 0):
    # gripper_data = qpos_data[:, -1]
    # smoothed_qpos = _smooth(gripper_data, smooth_window_size)
    # gripper_change_rate = np.diff(smoothed_qpos) / DT
    ret_ids = []
    if type == 'grasp':
        abrupt_window_size = 10
        for id in range(clip_start, len(gripper_change_rate)-abrupt_window_size+1):
            dx_window = gripper_change_rate[id:id+abrupt_window_size]
            x_window = smoothed_qpos[id:id+abrupt_window_size]
            if  np.min(dx_window)< dx_threshold:
                if x_window[-1] < x_threshold:
                    ret_ids.append(id)

    elif type == 'release':
        abrupt_window_size = 10
        for id in range(clip_start, len(gripper_change_rate)-abrupt_window_size+1):
            dx_window = gripper_change_rate[id:id+abrupt_window_size]
            x_window = smoothed_qpos[id:id+abrupt_window_size]
            if  np.max(dx_window)> dx_threshold:
                if x_window[-1] > x_threshold:
                    ret_ids.append(id)

    return ret_ids


def postprocess_jointgrasp(hdf5_path,  play_orig_vid=False):
    data_dict = read_demo_mj(hdf5_path, from_sim=True)
    obs_qpos = data_dict['qpos']
    obs_cam_high = data_dict['cam_high']
    socket_pc = data_dict['socket_pc']
    socket_pc += MJ2BULLET_OFFSET
    peg_pc = data_dict['peg_pc']
    peg_pc += MJ2BULLET_OFFSET

    if play_orig_vid:
        play_vid(obs_cam_high)    


    # identify grasp poses
    contact_info_dict = actions2grasps(obs_qpos, obj_pcs = [socket_pc, peg_pc], plot = True)

    if len(contact_info_dict.keys()) == 0:
        print(f"------{hdf5_path.split('/')[-1]} No valid grasp found!------")
        return

    # save the contact info dict to the hdf5 file
    save_dict_to_hdf5(contact_info_dict, hdf5_path)


    print(f"------Processed  {hdf5_path.split('/')[-1]}----------")



def save_dict_to_hdf5(contact_info_dict, filename):
    filename = filename.replace('.hdf5', '_diffgen.hdf5')
    with h5py.File(filename, 'a') as f:
        def recursively_save_dict_to_group(h5file, path, dictionary):
            for key, item in dictionary.items():
                if isinstance(item, dict):
                    # Check if the group exists, create if it doesn't
                    if key not in h5file[path]:
                        group = h5file.create_group(f'{path}/{key}')
                    recursively_save_dict_to_group(h5file, f'{path}/{key}', item)
                else:
                    # Check if the dataset exists, delete if it does, and create a new one
                    dataset_path = f'{path}/{key}'
                    if dataset_path in h5file:
                        del h5file[dataset_path]
                    h5file.create_dataset(dataset_path, data=item)

        recursively_save_dict_to_group(f, '/', contact_info_dict)

def read_demo_mj(input_hdf5_path, from_sim=False):
    # ensure path exist
    if not os.path.exists(input_hdf5_path):
        raise NameError("File not found!")
    with h5py.File(input_hdf5_path, "r") as f:
        print("Keys: {}".format(f.keys()))
        keys = list(f.keys())

        data_dict = {}

        print("\n--- Action --- ")

        action = f['action'][()]
        print("action.shape: {}".format(action.shape))

        obs_grp = f['observations']
        # obs_efforts = obs_grp['effort'][()]
        obs_qpos = obs_grp['qpos'][()]
        obs_qvel = obs_grp['qvel'][()]

        # print("joint_efforts.shape: {}".format(obs_efforts.shape))
        print("joint_qpos.shape: {}".format(obs_qpos.shape))
        print("joint_qvel.shape: {}".format(obs_qvel.shape))

        # data_dict['qpos'] = obs_qpos
        data_dict['qpos'] = action

        print("\n--- Observations --- ")

        obs_images = obs_grp['images']
        obs_image_keys = list(obs_images.keys())
        print(f"obs_image_keys: {obs_image_keys}")

        if from_sim:
            obs_cam_high = obs_images['top'][()]
        else:
            obs_cam_high = obs_images['cam_high'][()]

        data_dict['cam_high'] = obs_cam_high

        print("obs_cam_high.shape: {}".format(obs_cam_high.shape))

       # get the pc
        data_dict['socket_pc'] = obs_grp['socket_pc']['top'][()]
        data_dict['peg_pc'] = obs_grp['peg_pc']['top'][()]
        
        return data_dict

def visualize_processed(hdf5_path, is_left = True):
    with h5py.File(hdf5_path, "r") as f:
        print("Keys: {}".format(f.keys()))
        if is_left:
            grasps = f['socket_grasps']['grasp_poses'][()]
            tmp_pc = f['socket_grasps']['obj_points'][()] 
        else:
            grasps = f['peg_grasps']['grasp_poses'][()]
            tmp_pc = f['peg_grasps']['obj_points'][()] 
    history_list = []
    for i in range(len(grasps)):
        action_slice = (grasps[i], None)
        history_list.append(action_slice)

    render_pose(history_list, use_gui=True, \
                directory = None, obj_points = tmp_pc)
    
    print(f"------Visualized {hdf5_path.split('/')[-1]}------")

        
if __name__ == '__main__':
    file_dir = '/home/user/yzchen_ws/TAMP-ubuntu22/ALOHA/act/dataset/sim_insertion_tamp/'

    for episode_idx in [0]:
    # for episode_idx in range(50): 
        test_hdf5_path = os.path.join(file_dir, f"episode_{episode_idx}.hdf5")
        if not os.path.exists(test_hdf5_path):
            print(f"------{test_hdf5_path.split('/')[-1]} not found!------")
            continue

        postprocess_jointgrasp(test_hdf5_path,   play_orig_vid=False)
        # visualize_processed(test_hdf5_path, is_left = True)

    # disconnect()