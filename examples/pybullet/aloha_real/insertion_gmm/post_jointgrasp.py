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
from examples.pybullet.aloha_real.openworld_aloha.policy_simp import estimation_policy
from examples.pybullet.aloha_real.openworld_aloha.estimation.pc_utils import filter_pc
from examples.pybullet.aloha_real.openworld_aloha.simple_worlds import load_world_0obj, render_pose
from examples.pybullet.utils.pybullet_tools.utils import CLIENT, connect, disconnect, remove_body, remove_all_debug


def filter_grasp_by_obj(obj_center, robot_id, joint_val_list, grasp_ids,  threshold = 0.1):
    filtered_grasp_ids = []
    for idx in grasp_ids:
        joint_vals = joint_val_list[idx]
        # compute the eepose
        eetrans = qpos_to_eetrans(joint_vals, robot_id)
        eetrans_t = eetrans[:3, 3].reshape(3)
        eepose_R =  eetrans[:3, :3].reshape(9)  

        grasp_dist = np.linalg.norm(eetrans_t - obj_center)  
        if grasp_dist < threshold:
            filtered_grasp_ids.append(idx)

    return filtered_grasp_ids


def actions2grasps(actions, objs = None,  plot = False):
    l_joint_vals = actions[:, :6]
    l_gripper_val = PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(actions[:, 6])
    r_joint_vals = actions[:, 7:13]
    r_gripper_val = PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(actions[:, 13])

    smoothed_gripper_ql = _smooth(l_gripper_val)
    gripper_dql = np.diff(smoothed_gripper_ql, axis=0) / DT

    l_grasp_ids = switch_ids(smoothed_gripper_ql, gripper_dql, x_threshold=0.35, dx_threshold=-0.7, type='grasp')
    l_release_ids = switch_ids(smoothed_gripper_ql, gripper_dql, x_threshold=0.6, dx_threshold= 0.7, clip_start=100, type='release')

    smoothed_gripper_qr = _smooth(r_gripper_val)
    gripper_dqr = np.diff(smoothed_gripper_qr, axis=0) / DT

    r_grasp_ids = switch_ids(smoothed_gripper_qr, gripper_dqr, x_threshold=0.35, dx_threshold=-0.7, type='grasp')
    r_release_ids = switch_ids(smoothed_gripper_qr, gripper_dqr, x_threshold=0.6, dx_threshold= 0.7, clip_start=100, type='release')

    # filter the ids using the distance to the object
    if objs is not None:
        start_pose = objs[0].initial_pose
        l_grasp_ids = filter_grasp_by_obj(start_pose[0], 0, l_joint_vals, l_grasp_ids)
        r_grasp_ids = filter_grasp_by_obj(start_pose[0], 1, r_joint_vals, r_grasp_ids)

        end_pose = objs[-1].initial_pose
        l_release_ids = filter_grasp_by_obj(end_pose[0], 0, l_joint_vals, l_release_ids)
        r_release_ids = filter_grasp_by_obj(end_pose[0], 1, r_joint_vals, r_release_ids)
 
    if plot:
        import matplotlib.pyplot as plt
        plt.clf()
        plt.plot(smoothed_gripper_ql, label='smoothed_left_qpos')
        plt.plot(smoothed_gripper_qr, label='smoothed_right_qpos')

        # plt.plot(gripper_dql, label= 'left_gripper_change_rate')
        # plt.plot(gripper_dqr, label= 'right_gripper_change_rate')

        # draw grasp and release points
        plt.scatter(l_grasp_ids, smoothed_gripper_ql[l_grasp_ids], c='r', label='left_grasp')
        plt.scatter(l_release_ids, smoothed_gripper_ql[l_release_ids], c='g', label='left_release')
        plt.scatter(r_grasp_ids, smoothed_gripper_qr[r_grasp_ids], c='b', label='right_grasp')
        plt.scatter(r_release_ids, smoothed_gripper_qr[r_release_ids], c='y', label='right_release')
        plt.legend()
        pic_name = 'gripper_vals.png'
        pic_path = os.path.join('temp', pic_name)
        plt.savefig(pic_path)        

    # TODO: get the first several grasps and the last several releases
    # associate grasps/releases with the object points
    start_obj_points = [lp.point for lp in objs[0].points]
    # start_obj_points = filter_pc(start_obj_points)
    end_obj_points = [lp.point for lp in objs[-1].points]
    # end_obj_points = filter_pc(end_obj_points)


    l_grasp_poses = []
    for i, grasp_id in enumerate(l_grasp_ids):
        grasp_pose = qpos_to_eetrans(l_joint_vals[grasp_id], 0)
        l_grasp_poses.append(grasp_pose)

    r_grasp_poses = []
    for i, grasp_id in enumerate(r_grasp_ids):
        grasp_pose = qpos_to_eetrans(r_joint_vals[grasp_id], 1)
        r_grasp_poses.append(grasp_pose)
        
    l_release_poses = []
    for i, release_id in enumerate(l_release_ids):
        release_pose = qpos_to_eetrans(l_joint_vals[release_id], 0)
        l_release_poses.append(release_pose)

    r_release_poses = []
    for i, release_id in enumerate(r_release_ids):
        release_pose = qpos_to_eetrans(r_joint_vals[release_id], 1)
        r_release_poses.append(release_pose)

    last_grasp_id = max(l_grasp_ids+ r_grasp_ids)
    first_release_id = min(l_release_ids + r_release_ids)
    assert last_grasp_id < first_release_id
    demo_joint_vals = actions[last_grasp_id:first_release_id]

    # TODO： assign grasp poses using symb mask
    start_grasps = {'grasp_poses':  r_grasp_poses, 'obj_points': start_obj_points, 'arm': 'right'}
    end_grasps = {'grasp_poses': l_release_poses, 'obj_points': end_obj_points, 'arm': 'left'}
    # start_grasps = {'left_grasp_ids': l_grasp_ids, 'right_grasp_ids': r_grasp_ids, 'obj_points': start_obj_points}
    # end_releases = {'left_release_ids': l_release_ids, 'right_release_ids': r_release_ids, 'obj_points': end_obj_points}
    contact_info_dict = {'start_grasps': start_grasps, 'end_grasps': end_grasps, 'pred_joint_vals': demo_joint_vals}

    
    return contact_info_dict

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


def postprocess_jointgrasp(hdf5_path, estimator = None,  play_orig_vid=False,\
                           cam_extparam_mapping = None, debug_save_name = None):
    data_dict = read_demo_realsense(hdf5_path, camera_names = cam_extparam_mapping.keys())
    obs_qpos = data_dict['qpos']
    obs_cam_high = data_dict['cam_high']
    # color_imgs = data_dict['color_img']
    # depth_imgs = data_dict['depth_img']
    # camera_info = json.loads(data_dict['camera_info'])
    color_imgs = {}
    depth_imgs = {}
    camera_infos = {}
    for rs_cam in cam_extparam_mapping.keys():
        color_imgs[rs_cam] = data_dict[f'color_img_{rs_cam}']
        depth_imgs[rs_cam] = data_dict[f'depth_img_{rs_cam}']
        camera_infos[rs_cam] = json.loads(data_dict[f'camera_info_{rs_cam}'])

    if play_orig_vid:
        play_vid(obs_cam_high)    

        

    start_obj, end_obj = get_start_end_objs(color_imgs, depth_imgs, camera_infos, cam_extparam_mapping = cam_extparam_mapping, estimator = estimator, debug_save_name = debug_save_name)

    # identify grasp poses
    contact_info_dict = actions2grasps(obs_qpos, objs = [start_obj, end_obj], plot = True)

    # save the contact info dict to the hdf5 file
    save_dict_to_hdf5(contact_info_dict, hdf5_path)

    ## clear the obj and aabb on GUI
    for obj in [start_obj, end_obj]:
        remove_body(obj)
    remove_all_debug()

    print(f"------Processed  {hdf5_path.split('/')[-1]}----------")


def get_start_end_objs(color_imgs, depth_imgs, camera_infos, estimator = None, debug_save_name = None, cam_extparam_mapping = None):
    assert estimator is not None
    
    objs = []
    start_belief = estimator.estimate_state_multiview(color_imgs, depth_imgs, camera_infos, 0, cam_extparam_mapping = cam_extparam_mapping, debug_save_name = debug_save_name)
    objs.append(start_belief.estimated_objects[0])
    end_belif = estimator.estimate_state_multiview(color_imgs, depth_imgs, camera_infos, 1, cam_extparam_mapping = cam_extparam_mapping)
    objs.append(end_belif.estimated_objects[0])


    return objs

def save_dict_to_hdf5(contact_info_dict, filename):
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

def read_demo_realsense(input_hdf5_path, from_sim=False, camera_names = ['']):
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
        obs_keys = list(obs_grp.keys())


        print(f"\nObservation keys: {obs_keys}")

        print("\n--- Joint --- ")

        # obs_efforts = obs_grp['effort'][()]
        obs_qpos = obs_grp['qpos'][()]
        obs_qvel = obs_grp['qvel'][()]

        # print("joint_efforts.shape: {}".format(obs_efforts.shape))
        print("joint_qpos.shape: {}".format(obs_qpos.shape))
        print("joint_qvel.shape: {}".format(obs_qvel.shape))

        data_dict['qpos'] = obs_qpos

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

        # get color and depth
        for rs_cam in camera_names:
            color_key = f'color_img_{rs_cam}'
            depth_key = f'depth_img_{rs_cam}'
            camera_info_key = f'camera_info_{rs_cam}'
            data_dict[color_key] = f[color_key][()]
            data_dict[depth_key] = f[depth_key][()]
            data_dict[camera_info_key] = f[camera_info_key][()]
        
        return data_dict

def visualize_processed(hdf5_path, is_pred = True):
    with h5py.File(hdf5_path, "r") as f:
        print("Keys: {}".format(f.keys()))
        if is_pred:
            grasps = f['start_grasps']['grasp_poses'][()]
            tmp_pc = f['start_grasps']['obj_points'][()]
        else:
            grasps = f['end_grasps']['grasp_poses'][()]
            tmp_pc = f['end_grasps']['obj_points'][()]

    history_list = []
    for i in range(len(grasps)):
        action_slice = (grasps[i], None)
        history_list.append(action_slice)

    render_pose(history_list, use_gui=True, \
                directory = None, obj_points = tmp_pc)
    
    print(f"------Visualized {hdf5_path.split('/')[-1]}------")

if __name__ == '__main__':
    # file_dir = "/home/xuhang/Desktop/aloha_data/aloha_transfer_tape"
    file_dir = "/ssd1/aloha_data/aloha_transfer_tape/transfer_cup"
    # file_dir = "/home/robotics/CoMa_code_clean/equibot_abstract/data/transfer_tape/raw"
    # for episode_idx in [101]:
    is_visualize = True

    if not is_visualize:
        connect(use_gui=True)
        robot_body, names, movable_bodies, stackable_bodies = load_world_0obj()
        # start phase
        estimator = estimation_policy(robot_body, mode = 'data_process', teleport=False, client=CLIENT, 
                                  seg_branch='sam', text_prompt='cup.')
    temp_vis_dir = os.path.join(root_path, 'examples/pybullet/aloha_real/openworld_aloha/estimation/temp_vis/' )
    cam_extparam_mapping = {'camera_2': os.path.join(temp_vis_dir, 'camera_pose.json'), 'camera_1': os.path.join(temp_vis_dir, 'back_camera_pose.json')}

    for episode_idx in range(0, 30): 
    # for episode_idx in [3,4, 5, 6, 7, 9, 12, 14,17,19]:
        test_hdf5_path = os.path.join(file_dir, f"episode_{episode_idx}.hdf5")
        if not os.path.exists(test_hdf5_path):
            print(f"------{test_hdf5_path} not found!------")
            continue

        if not is_visualize:
            postprocess_jointgrasp(test_hdf5_path, estimator=estimator,  play_orig_vid=False, cam_extparam_mapping = cam_extparam_mapping, debug_save_name = 'hdf5_out/transfer_cup/episode_{}.ply'.format(episode_idx))
        else:
            visualize_processed(test_hdf5_path, is_pred = True)
    if not is_visualize:
        disconnect()