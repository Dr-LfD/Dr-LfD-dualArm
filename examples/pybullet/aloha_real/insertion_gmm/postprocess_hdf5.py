import numpy as np
import os
import json
import cv2
import networkx as nx
import copy
import matplotlib
# Use "Agg" (non-interactive) or "pdf" "svg" for file output
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
plt.clf()
import sys
root_path = next(('/'+ os.path.join(*os.path.dirname(os.path.abspath(__file__)).split(os.sep)[:i+1]) + os.sep
                    for i in range(len(os.path.dirname(os.path.abspath(__file__)).split(os.sep))) 
                    if os.path.dirname(os.path.abspath(__file__)).split(os.sep)[i] == 'pddlstream_aloha'), None)
sys.path.append(root_path) if root_path not in sys.path else None
os.chdir(root_path)


from examples.pybullet.aloha_real.insertion_gmm.segment_demonstration import   _smooth, play_vid


import h5py
from examples.pybullet.aloha_real.openworld_aloha.policy_simp import estimation_policy
from examples.pybullet.aloha_real.openworld_aloha.open_world_utils import load_yaml_params
from examples.pybullet.aloha_real.openworld_aloha.simple_worlds import load_aloha_world_flexible, load_dual_franka_world_flexible, render_pose
from examples.pybullet.utils.pybullet_tools.utils import CLIENT, connect, disconnect, remove_body, remove_all_debug
from examples.pybullet.aloha_real.openworld_aloha.open_world_utils import get_camera_mappings

DT = 0.02


all_colors =  [
    '#FFDDC1',  # Light Peach
    '#C1FFC1',  # Light Green
    '#C1D4FF',  # Light Blue
    '#FFC1E3',  # Light Pink
    '#FFF5C1',  # Soft Yellow
    '#C1FFFF',  # Aqua
    '#F5C1FF',  # Lavender
    '#C1E0FF',  # Sky Blue
    '#FFD4C1',  # Coral
    '#E3FFC1',  # Pale Lime Green
]

# def color_points_from_lp(labeled_points):
#     colored_points = []
#     for lp in labeled_points:
#         color = list(lp.color)
#         point = list(lp.point)
#         colored_points.append(tuple([point, color]))
#     return colored_points


def read_demo_realsense(input_hdf5_path, camera_names = ['']):
    # ensure path exist
    if not os.path.exists(input_hdf5_path):
        raise NameError("File not found!")
    with h5py.File(input_hdf5_path, "r") as f:
        print("Keys: {}".format(f.keys()))

        data_dict = {}

        obs_grp = f['observations']
        data_dict['qpos'] =  obs_grp['qpos'][()]
        data_dict['action'] = f['action'][()]

        # get color and depth
        for rs_cam in camera_names:
            color_key = f'color_img_{rs_cam}'
            depth_key = f'depth_img_{rs_cam}'
            camera_info_key = f'camera_info_{rs_cam}'
            data_dict[color_key] = f[color_key][()]
            data_dict[depth_key] = f[depth_key][()]
            data_dict[camera_info_key] = f[camera_info_key][()]
        
        return data_dict
    
def read_demo_wristcam(input_hdf5_path):
    # ensure path exist
    if not os.path.exists(input_hdf5_path):
        raise NameError("File not found!")
    with h5py.File(input_hdf5_path, "r") as f:
        print("Keys: {}".format(f.keys()))

        image_dict = {}
        obs_grp = f['observations']
        image_dict['left'] = obs_grp['images']['cam_left_wrist'][()]
        image_dict['right'] = obs_grp['images']['cam_right_wrist'][()]
        
        return image_dict
    
## by revising this function, we can adapt to mj sim
def get_jpose_pc_realsense(cfg, hdf5_path, estimator, cam_extparam_mapping = None, **kwargs):

    data_dict = read_demo_realsense(hdf5_path, camera_names = cam_extparam_mapping.keys())
    
    ### if has_per_skill, then use  action to learn traj
    has_per_skill = ['per_skill' in x for x in  cfg['skill_names']]
    has_per_skill = any(has_per_skill)
    if has_per_skill:
        obs_qpos = data_dict['action']
    else:
        obs_qpos = data_dict['qpos']

    color_imgs = {}
    depth_imgs = {}
    camera_infos = {}
    for rs_cam in cam_extparam_mapping.keys():
        color_imgs[rs_cam] = data_dict[f'color_img_{rs_cam}']
        depth_imgs[rs_cam] = data_dict[f'depth_img_{rs_cam}']
        camera_infos[rs_cam] = json.loads(data_dict[f'camera_info_{rs_cam}'])

    detected_objs = get_start_end_objs(cfg, color_imgs, depth_imgs, camera_infos, cam_extparam_mapping = cam_extparam_mapping,estimator = estimator, **kwargs)
    return obs_qpos, detected_objs



def get_start_end_objs(cfg, color_imgs, depth_imgs, camera_infos,  estimator, predict_effect = True, **kwargs):
    assert estimator is not None
    ## NOTE: currently, we detect one object for each type.
    # if num_objs > 1 (e.g., 2 cups), we need to have instance matching algorithm for start and end.
    detected_objs = {}
    start_belief = None
    end_belief = None

    start_belief = estimator.estimate_state_multiview(color_imgs, depth_imgs, camera_infos, 0, filter_surface = cfg['filter_surface'],  **kwargs)
    for obj in start_belief.estimated_objects:
        if obj.category not in detected_objs:
            detected_objs[obj.category] = {}  ## new category
        detected_objs[obj.category]['start'] = obj
        # # Store the belief for later mask creation
        # detected_objs[obj.category]['start_belief'] = start_belief

    if predict_effect:
        end_belief = estimator.estimate_state_multiview(color_imgs, depth_imgs, camera_infos, -1,  filter_surface = cfg['filter_surface'], **kwargs)
        for obj in end_belief.estimated_objects:
            if obj.category not in detected_objs:
                detected_objs[obj.category] = {}  ## new category
            detected_objs[obj.category]['end'] = obj
            # # Store the belief for later mask creation
            # detected_objs[obj.category]['end_belief'] = end_belief

    return detected_objs

####### for grasp processing


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




def vis_sg_seq(scene_graphs, save_name = 'sg_seq.png'):
    import pydot
    # Create a pydot graph to combine all graphs
    dot_graph = pydot.Dot("CombinedGraphs", graph_type="graph", rankdir="LR")  # LR for left-to-right layout

    subgraphs = []
    # Add each NetworkX graph as a subgraph in pydot
    for i, G in enumerate(scene_graphs):
        G.graph['name'] = 'Graph {}'.format(i)
        subgraph = pydot.Cluster(f"cluster_{i}")
        subgraph.set_label(G.graph.get("name", f"Graph {i}"))

        # Add nodes and edges to the subgraph
        for node in G.nodes:
            subgraph.add_node(pydot.Node(str(node)+f'_g{i}'))  # Convert to string for DOT compatibility
        for edge in G.edges:
            subgraph.add_edge(pydot.Edge(str(edge[0])+f'_g{i}', str(edge[1])+f'_g{i}'))

        subgraphs.append(subgraph)

    subgraphs.reverse()
    for subgraph in subgraphs:
        # Add the subgraph to the main graph
        dot_graph.add_subgraph(subgraph)

    # Save to PNG
    dot_graph.write_png(save_name)
    # save to .dot
    dot_string = dot_graph.to_string()
    # with open(save_name.replace('.png', '.dot'), 'w') as f:
    #     f.write(dot_string)
    return dot_string



def get_sg_seq(skill_sg_param,  left_eepath, right_eepath, gripper_actions, detected_objs, out_dot_name='sg_seq.png'):
    def initialize_graph(hand_list, detected_objs):
        """init scene graph"""
        tabletop_sg = nx.Graph()
        tabletop_sg.graph['idx_list'] = []
        tabletop_sg.add_nodes_from(hand_list + ["table"])
        for obj_name in detected_objs.keys():
            tabletop_sg.add_node(obj_name)
            tabletop_sg.add_edge(obj_name, 'table')
        return tabletop_sg

    def update_graph(cur_sg, obj_name, hand, hand_center, obj_center, grasp_ids,  hand_obj_dist_threshold, i):
        """update grasp and release relation"""
        graph_changed = False
        dist = np.linalg.norm(obj_center - hand_center)
        is_hand_obj_close = dist < hand_obj_dist_threshold

        if is_hand_obj_close:
            if not cur_sg.has_edge(obj_name, hand) and i in grasp_ids:
                cur_sg.add_edge(obj_name, hand)
                graph_changed = True
        ## NOTE： drop detection is banned in this version
        # elif cur_sg.has_edge(obj_name, hand) and not cur_sg.has_edge(obj_name, 'table') and i in release_ids:
        #     cur_sg.add_edge(obj_name, 'table')
        #     cur_sg.remove_edge(obj_name, hand)
        #     graph_changed = True

        return graph_changed

    def update_obj_table_relation(cur_sg, hand, obj_name, ee_z, lifted_threshold):
        graph_changed = False
        is_holding = cur_sg.has_edge(hand, obj_name)
        obj_on_table = cur_sg.has_edge(obj_name, 'table')
        if is_holding and obj_on_table and ee_z > lifted_threshold:
            cur_sg.remove_edge(obj_name, 'table')
            graph_changed = True
        return graph_changed

    def update_hand_hand_relation(cur_sg, left_center, right_center, hand_hand_dist_threshold):
        graph_changed = False
        dist = np.linalg.norm(left_center - right_center)
        is_hand_hand_close = dist < hand_hand_dist_threshold
        if is_hand_hand_close and not cur_sg.has_edge('left', 'right'):
            cur_sg.add_edge('left', 'right')
            graph_changed = True
        return graph_changed

    def process_iteration(eepath_range, obj_hand_mapping, obj_key, grasp_key,  reverse=False):
        """support forward and backward iteration"""
        sg_seq = [initialize_graph(['left', 'right'], detected_objs)]
        donebiop = False

        for i in eepath_range:
            graph_changed = False
            cur_sg = copy.deepcopy(sg_seq[-1])

            # if i == 460:
            #     print('debug')

            for obj_name in detected_objs.keys():
                obj = detected_objs[obj_name][obj_key]
                obj_center = obj.initial_pose[0]
                hand = obj_hand_mapping[obj_name]
                hand_center = left_eepath[i][:3, 3] if hand == 'left' else right_eepath[i][:3, 3]

                grasp_updated= update_graph(cur_sg, obj_name, hand, hand_center, obj_center, gripper_actions[hand][grasp_key], skill_sg_param['hand_obj_dist_threshold'], i)
                if grasp_updated:
                    graph_changed = True
                    continue

                ee_z = left_eepath[i][2, 3] if hand == 'left' else right_eepath[i][2, 3]
                graph_changed |= update_obj_table_relation(cur_sg, hand, obj_name, ee_z, skill_sg_param['lifted_threshold'])

            donebiop= update_hand_hand_relation(cur_sg, left_eepath[i][:3, 3], right_eepath[i][:3, 3],                  skill_sg_param['hand_hand_dist_threshold'])
            graph_changed |= donebiop

            if graph_changed:
                cur_sg.graph['idx_list'] = [i]
                sg_seq.append(cur_sg)
            else:
                sg_seq[-1].graph['idx_list'].append(i)

            if donebiop:
                break
        assert donebiop
        if reverse:
            sg_seq.reverse()
        return sg_seq

    # forward iteration
    start_obj_hand = {skill_sg_param['pre_obj_names'][i]: skill_sg_param['pre_arms'][i] for i in range(len(skill_sg_param['pre_obj_names']))}
    start_sg_seq = process_iteration(range(len(left_eepath)), start_obj_hand, 'start', 'grasp_ids')

    ## consider the effect
    if len(skill_sg_param['eff_obj_names']):
        # backward iteration
        end_obj_hand = {skill_sg_param['eff_obj_names'][i]: skill_sg_param['eff_arms'][i] for i in range(len(skill_sg_param['eff_obj_names']))}
        end_sg_seq = process_iteration(range(len(left_eepath) - 1, -1, -1), end_obj_hand, 'end', 'release_ids',  reverse=True)

        # merge the two sequences
        sg_seq = start_sg_seq + end_sg_seq[1:]
    else:
        sg_seq = start_sg_seq

    sg_dot = vis_sg_seq(sg_seq, save_name=out_dot_name)
    return sg_seq

def get_interested_nbrs(sg, pre_obj_names = 'screwdriver'):
    interested_node = [node for node in sg.nodes if pre_obj_names in node][0]
    interested_nbrs = set(sg.neighbors(interested_node))
    return interested_nbrs

def filter_by_sg(skill_sg_param, gripper_actions, sg_seq, pre_obj_names = ['screwdriver']):
    filtered_gripper_actions =      {'left':{'grasp_ids': set(), 'release_ids': set(), 'holding_ids': set(), 'eff_holding_ids': set()},  'right': {'grasp_ids': set(), 'release_ids': set(), 'holding_ids': set(), 'eff_holding_ids': set()}}    

    # hand_set = set(skill_sg_param['pre_arms'])
    hand_set = set(['left', 'right'])
    surface_set = set(['table'])

    hand_status = {'left': 'idle', 'right': 'idle'}

    def update_id():
        # update id 
        for hand in hand_set:
            if hand_status[hand] == 'grasp':
                added_idx = set(gripper_actions[hand]['grasp_ids']) & set(cur_idx_list)
                filtered_gripper_actions[hand]['grasp_ids'] = filtered_gripper_actions[hand]['grasp_ids'].union(added_idx)
            elif hand_status[hand] == 'release':
                added_idx=  set(gripper_actions[hand]['release_ids']) & set(cur_idx_list)
                filtered_gripper_actions[hand]['release_ids'] = filtered_gripper_actions[hand]['release_ids'].union(added_idx)
                filtered_gripper_actions[hand]['eff_holding_ids'] = filtered_gripper_actions[hand]['eff_holding_ids'].union(set(prev_idx_list))
                hand_status[hand] = 'terminated'
            elif hand_status[hand] == 'bimanual':
                filtered_gripper_actions[hand]['holding_ids'] = filtered_gripper_actions[hand]['holding_ids'].union(set(prev_idx_list))
                hand_status[hand] = 'eff_bimanual'
            # elif hand_status[hand] == 'eff_bimanual':
            #     filtered_gripper_actions[hand]['eff_holding_ids'] = filtered_gripper_actions[hand]['eff_holding_ids'].union(set(prev_idx_list))

    done_biop = False
    prev_sg = sg_seq[0]
    prev_intersted_nbrs = {}
    prev_idx_list = prev_sg.graph['idx_list']
    for obj_name in pre_obj_names:
        prev_intersted_nbrs[obj_name] = get_interested_nbrs(prev_sg, obj_name)
    for i in range(1, len(sg_seq)):
        cur_sg = sg_seq[i]
        cur_idx_list = cur_sg.graph['idx_list']
        cur_interested_nbrs = {}

        if cur_sg.has_edge('left', 'right'):
            done_biop = True
            for hand in hand_set:
                hand_status[hand] = 'bimanual'
            update_id()
            continue
            
        for obj_name in pre_obj_names:

            cur_interested_nbrs[obj_name] = get_interested_nbrs(cur_sg, obj_name)
            cur_holding_hands = cur_interested_nbrs[obj_name]  & hand_set
            is_connect_hand = len(cur_holding_hands) > 0
            cur_adj_surfaces = cur_interested_nbrs[obj_name]  & surface_set
            is_connect_table = len(cur_adj_surfaces) > 0


            added_nbrs = cur_interested_nbrs[obj_name]  - prev_intersted_nbrs[obj_name] 
            removed_nbrs = prev_intersted_nbrs[obj_name]  - cur_interested_nbrs[obj_name] 
            added_hand = added_nbrs & hand_set
            removed_hand = removed_nbrs & hand_set
            added_surface = added_nbrs & surface_set

            ## if added nbrs are hand, and the obj connects both hand and the table, then this is grasp
            if len(added_hand) > 0 and is_connect_hand and is_connect_table:
                for hand in added_hand:
                    hand_status[hand] = 'grasp'

            if done_biop:
                if  is_connect_hand and is_connect_table:
                    for hand in cur_holding_hands:
                        hand_status[hand] = 'release'
              
            prev_intersted_nbrs[obj_name]  = cur_interested_nbrs[obj_name] 
            
        update_id()

        prev_sg = cur_sg
        prev_idx_list = cur_idx_list

    return filtered_gripper_actions



def plot_griper_values( filtered_gripper_actions, sg_seq, pic_name = 'gripper_vals.png'):
    plt.clf()
    plt.plot(filtered_gripper_actions['left']['smoothed_gripper_q'], label='smoothed_left_qpos')
    plt.plot(filtered_gripper_actions['right']['smoothed_gripper_q'], label='smoothed_right_qpos')

    # plt.plot(gripper_dql, label= 'left_gripper_change_rate')
    # plt.plot(gripper_dqr, label= 'right_gripper_change_rate')

    colors = ['r', 'g', 'b', 'y', 'm', 'c']
    markers = ['o', 's', 'x', 'D', 'v', '^']
    
    # 遍历左右手
    for hand in ['left', 'right']:
        # 遍历不同的动作类型
        for i, key in enumerate(['grasp', 'release', 'holding', 'eff_holding']):
            # 获取动作的时间点
            ids = filtered_gripper_actions[hand][key + '_ids']
            if len(ids) > 0:
                # 绘制散点图
                plt.scatter(ids, filtered_gripper_actions[hand]['smoothed_gripper_q'][ids],
                            c=colors[i], label=f'{hand}_{key}', marker=markers[i])
    plt.legend()

    ## draw segments using sg
    segments = [(sg.graph['idx_list'][0], sg.graph['idx_list'][-1]) for sg in sg_seq]
    colors = all_colors[:len(segments)]
    for (start, end), color in zip(segments, colors):
        plt.axvspan(start, end, facecolor=color, alpha=0.3)

    plt.savefig(pic_name)


def actions2grasps(cfg,  skill_sg_param,  hdf5_name,  obs_qpos, detected_objs = None,  plot = False, robot_entity = None):
    if cfg['robot_name'] == 'aloha':
        from examples.pybullet.aloha_real.scripts.aloha_tamp_constants import PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN, qpos_to_eetrans

        l_joint_vals = obs_qpos[:, :6]
        l_gripper_val = PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(obs_qpos[:, 6])
        r_joint_vals = obs_qpos[:, 7:13]
        r_gripper_val = PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(obs_qpos[:, 13])

        def compute_eepath(joint_val_list, hand_id):

            ee_path = [ qpos_to_eetrans(joint_val, hand_id) for joint_val in joint_val_list]
            return ee_path
        
    elif cfg['robot_name'] == 'dualfranka':
        l_joint_vals = obs_qpos[:, :7]
        l_gripper_val = obs_qpos[:, 7]
        r_joint_vals = obs_qpos[:, 8:15]
        r_gripper_val = obs_qpos[:, -1]

        def compute_eepath(joint_val_list, hand_id):
            if hand_id == 0: # left
                ee_link =  "panda2_ee_link"
                arm_group = "left_arm"
            elif hand_id == 1: # right
                ee_link =  "panda1_ee_link"
                arm_group = "right_arm"
            else:
                raise ValueError("hand_id should be 0 or 1")
            
            ee_path = []
            for i in range(len(joint_val_list)):
                robot_entity.set_group_positions(arm_group, joint_val_list[i])
                ee_link_pose = robot_entity.get_link_trans(ee_link)
                ee_path.append(ee_link_pose)
            return ee_path
    else:
        raise NotImplementedError("Robot not implemented")

    smoothed_gripper_ql = _smooth(l_gripper_val)
    gripper_dql = np.diff(smoothed_gripper_ql, axis=0) / DT

    l_grasp_ids = switch_ids(smoothed_gripper_ql, gripper_dql, \
                              x_threshold=skill_sg_param['left']['grasp']['x_threshold'], \
                                dx_threshold=skill_sg_param['left']['grasp']['dx_threshold'], type='grasp')
    l_release_ids = switch_ids(smoothed_gripper_ql, gripper_dql,
                                x_threshold=skill_sg_param['left']['release']['x_threshold'], \
                                dx_threshold=skill_sg_param['left']['release']['dx_threshold'], \
                                clip_start=skill_sg_param['left']['release']['clip_start'], type = 'release')

    smoothed_gripper_qr = _smooth(r_gripper_val)
    gripper_dqr = np.diff(smoothed_gripper_qr, axis=0) / DT

    r_grasp_ids = switch_ids(smoothed_gripper_qr, gripper_dqr, \
                                x_threshold=skill_sg_param['right']['grasp']['x_threshold'], \
                                    dx_threshold=skill_sg_param['right']['grasp']['dx_threshold'], type='grasp')
    r_release_ids = switch_ids(smoothed_gripper_qr, gripper_dqr, \
                                x_threshold=skill_sg_param['right']['release']['x_threshold'], \
                                    dx_threshold=skill_sg_param['right']['release']['dx_threshold'], \
                                    clip_start=skill_sg_param['right']['release']['clip_start'], type = 'release')

    gripper_actions = {'left':{'grasp_ids': l_grasp_ids, 'release_ids': l_release_ids}, \
                       'right': {'grasp_ids': r_grasp_ids, 'release_ids': r_release_ids}}    

    left_eepath = compute_eepath(l_joint_vals, 0)
    right_eepath = compute_eepath(r_joint_vals, 1)

    sg_seq = get_sg_seq(skill_sg_param, left_eepath, right_eepath, gripper_actions, detected_objs,
                        out_dot_name = os.path.join(skill_sg_param['output_dir'], 'sg_seq_' + hdf5_name + '.png'))
 
    filtered_gripper_actions = filter_by_sg(skill_sg_param, gripper_actions, sg_seq, pre_obj_names = skill_sg_param['pre_obj_names'])

    for hand in ['left', 'right']:
        ee_trans_list = left_eepath if hand == 'left' else right_eepath
        for id_type in ['grasp_ids', 'release_ids']:
            filtered_gripper_actions[hand][id_type] = sorted(list(filtered_gripper_actions[hand][id_type]))
            val_name = id_type.replace('_ids', '_poses')
            filtered_gripper_actions[hand][val_name] = [ee_trans_list[id] for id in filtered_gripper_actions[hand][id_type]]

        full_holding_idxs = sorted(list(filtered_gripper_actions[hand]['holding_ids']))   
        filtered_gripper_actions[hand]['holding_ids'] = full_holding_idxs[skill_sg_param['holding_truncation']:-skill_sg_param['holding_truncation']:] ## remove the first and last several frames
        filtered_gripper_actions[hand]['holding_jposes'] = [obs_qpos[id] for id in filtered_gripper_actions[hand]['holding_ids']]

        eff_holding_idxs = sorted(list(filtered_gripper_actions[hand]['eff_holding_ids']))
        filtered_gripper_actions[hand]['eff_holding_ids'] = eff_holding_idxs[skill_sg_param['holding_truncation']:-skill_sg_param['holding_truncation']:] ## remove the first and last several frames
        # we do not include eff holding jpose for simplicity

    # dual_arm_holding_ids = filtered_gripper_actions['left']['holding_ids'].intersection(filtered_gripper_actions['right']['holding_ids'])
    # dual_arm_holding_ids_list = sorted(list(dual_arm_holding_ids))
    # filtered_gripper_actions['holding_jposes'] = [obs_qpos[id] for id in dual_arm_holding_ids_list]

    # filtered_gripper_actions['left']['holding_ids'] = filtered_gripper_actions['right']['holding_ids'] = dual_arm_holding_ids_list ## for plot

    if plot:
        filtered_gripper_actions['left']['smoothed_gripper_q'] = smoothed_gripper_ql
        filtered_gripper_actions['right']['smoothed_gripper_q'] = smoothed_gripper_qr
        filtered_gripper_actions['left']['gripper_dq'] = gripper_dql
        filtered_gripper_actions['right']['gripper_dq'] = gripper_dqr
        plot_griper_values(filtered_gripper_actions, sg_seq, pic_name = os.path.join(skill_sg_param['output_dir'],'gripper_vals_' +hdf5_name+ '.png'))

    contact_info_dict = {}
    for obj_name in detected_objs.keys():
        obj_idx = skill_sg_param['pre_obj_names'].index(obj_name)
        ## TODO: this script cannot deal with transfer, as the interacting arm changes
        pre_arm = skill_sg_param['pre_arms'][obj_idx]
        obj_grasp_joint = {'obj': obj_name}
        start_obj = detected_objs[obj_name]['start']
        start_obj_points = [lp.point for lp in start_obj.points]
        start_obj_colors = [lp.color for lp in start_obj.points]

        obj_grasp_joint['start_pc'] = start_obj_points
        obj_grasp_joint['start_colors'] = start_obj_colors
        
        
        # Save grasp poses and IDs
        obj_grasp_joint['grasp_poses'] = filtered_gripper_actions[pre_arm]['grasp_poses']
        obj_grasp_joint['grasp_ids'] = np.array(filtered_gripper_actions[pre_arm]['grasp_ids'], dtype=np.int32)
        
        # Save holding joint poses and IDs
        obj_grasp_joint['joint_poses'] = filtered_gripper_actions[pre_arm]['holding_jposes']
        obj_grasp_joint['holding_ids'] = np.array(filtered_gripper_actions[pre_arm]['holding_ids'], dtype=np.int32)

        ## save the gripper actions
        obj_grasp_joint['grasp_gripper_actions'] = filtered_gripper_actions[pre_arm]['smoothed_gripper_q'][obj_grasp_joint['grasp_ids'] ]


        if 'end' in detected_objs[obj_name]:
            end_obj = detected_objs[obj_name]['end']
            end_obj_points = [lp.point for lp in end_obj.points]
            end_obj_colors = [lp.color for lp in end_obj.points]
            obj_grasp_joint['end_pc'] = end_obj_points
            obj_grasp_joint['end_colors'] = end_obj_colors
            
            
            eff_arm = skill_sg_param['eff_arms'][obj_idx]
            obj_grasp_joint['release_poses'] = filtered_gripper_actions[eff_arm]['release_poses']
            obj_grasp_joint['release_ids'] = np.array(filtered_gripper_actions[eff_arm]['release_ids'], dtype=np.int32)
            if len(obj_grasp_joint['release_poses']) ==0:
                print('no release poses, need to adjust gripper values!')
            
            # Save eff_holding IDs if available
            if len(filtered_gripper_actions[eff_arm]['eff_holding_ids']) > 0:
                obj_grasp_joint['eff_holding_ids'] = np.array(filtered_gripper_actions[eff_arm]['eff_holding_ids'], dtype=np.int32)

            ## save the gripper actions
            obj_grasp_joint['release_gripper_actions'] = filtered_gripper_actions[eff_arm]['smoothed_gripper_q'][obj_grasp_joint['release_ids'] ]


        contact_info_dict[obj_name] = obj_grasp_joint
        # contact_info_dict['dual_jpose'] = filtered_gripper_actions['holding_jposes']
    
    return contact_info_dict, filtered_gripper_actions


def save_abstracted_info(skill_sg_param, hdf5_name, contact_info_dict, original_hdf5_path=None):
    new_hdf5_path = os.path.join(skill_sg_param['output_dir'], hdf5_name + '.hdf5')
    with h5py.File(new_hdf5_path, 'a') as new_hdf5:
        # Read and save cam_high images from original hdf5 if path is provided
        if original_hdf5_path is not None and os.path.exists(original_hdf5_path):
            with h5py.File(original_hdf5_path, "r") as original_hdf5:
                obs_grp = original_hdf5['observations']
                obs_images = obs_grp['images']
                if 'cam_high' in obs_images:
                    cam_high_images = obs_images['cam_high'][()]
                elif 'top' in obs_images:
                    cam_high_images = obs_images['top'][()]
                else:
                    cam_high_images = None
                
                # Save cam_high images (resized to half width and half height)
                if cam_high_images is not None:
                    # Resize each frame to half dimensions
                    h, w = cam_high_images.shape[1], cam_high_images.shape[2]
                    new_h, new_w = h // 2, w // 2
                    resized_images = np.array([
                        cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        for img in cam_high_images
                    ])
                    dataset_path = '/cam_high'
                    if dataset_path in new_hdf5:
                        del new_hdf5[dataset_path]
                    new_hdf5.create_dataset(dataset_path, data=resized_images)

                ## save robot actions (14 dof jposes)
                robot_actions = original_hdf5['action'][()]
                if 'action' in new_hdf5:
                    del new_hdf5['action']
                new_hdf5.create_dataset('action', data=robot_actions)

        
        def recursively_save_dict_to_group(h5file, path, dictionary):
            for key, item in dictionary.items():
                if isinstance(item, dict):
                    # Check if the group exists, create if it doesn't
                    if key not in h5file[path]:
                        group = h5file.create_group(f'{path}/{key}')
                    recursively_save_dict_to_group(h5file, f'{path}/{key}', item)
                elif isinstance(item, list) and len(item) > 0 and isinstance(item[0], np.ndarray):
                    # Handle lists of numpy arrays (e.g., images)
                    dataset_path = f'{path}/{key}'
                    if dataset_path in h5file:
                        del h5file[dataset_path]
                    # Convert list of arrays to a single numpy array
                    array_data = np.array(item)
                    h5file.create_dataset(dataset_path, data=array_data)
                else:
                    # Check if the dataset exists, delete if it does, and create a new one
                    dataset_path = f'{path}/{key}'
                    if dataset_path in h5file:
                        del h5file[dataset_path]
                    h5file.create_dataset(dataset_path, data=item)

        recursively_save_dict_to_group(new_hdf5, '/', contact_info_dict)


def visualize_hdf5(hdf5_path, obj_name, vis_start=True, use_gui=True):
    """
    Visualize processed grasp or release poses for a given object using render_pose.

    Args:
        hdf5_path (str): Path to the processed hdf5 file created by save_abstracted_info.
        obj_name (str): Name of the object group to visualize (must match a key in the file).
        vis_start (bool): If True, visualize grasp poses with start_pc; if False, visualize release poses with end_pc.
        use_gui (bool): Whether to use PyBullet GUI in render_pose.
    """
    with h5py.File(hdf5_path, "r") as f:
        print("Keys: {}".format(list(f.keys())))

        if obj_name not in f:
            print(f"Object {obj_name} not found in {hdf5_path}")
            return

        obj_grp = f[obj_name]

        action_name = "grasp" if vis_start else "release"
        phase_name = "start" if vis_start else "end"
        vis_key = f'{action_name}_poses'
        pc_key = f'{phase_name}_pc'

        if vis_key not in obj_grp or pc_key not in obj_grp:
            print(f"Missing '{vis_key}' or '{pc_key}' for object {obj_name}")
            return
        grasps = obj_grp[vis_key][()]
        tmp_pc = obj_grp[pc_key][()]

    history_list = []
    for i in range(len(grasps)):
        action_slice = (grasps[i], None)
        history_list.append(action_slice)

    render_pose(history_list, use_gui=use_gui, directory=None, obj_points=tmp_pc)


def postprocess_jointgrasp(cfg,skill_sg_param, original_hdf5_path, obs_qpos, detected_objs, robot_entity):
    hdf5_name = os.path.basename(original_hdf5_path).split('.')[0]
    # identify grasp poses
    contact_info_dict, filtered_gripper_actions = actions2grasps(cfg, skill_sg_param, hdf5_name, obs_qpos, detected_objs = detected_objs,  plot = True, robot_entity = robot_entity)

    # save the contact info dict to the hdf5 file
    save_abstracted_info(skill_sg_param, hdf5_name, contact_info_dict, original_hdf5_path=original_hdf5_path)

    ## clear the obj and aabb on GUI
    for obj_key in detected_objs:
        remove_body(detected_objs[obj_key]['start'])
        if 'end' in detected_objs[obj_key]:
            remove_body(detected_objs[obj_key]['end'])
    remove_all_debug()

    print(f"------Processed  {original_hdf5_path.split('/')[-1]}----------")
    return filtered_gripper_actions

def do_visualization(cfg_path, skill_yaml_paths, obj_name="cup", vis_start=True):
    cfg = load_yaml_params(cfg_path, skill_yaml_paths=skill_yaml_paths)
    skill_name = os.path.splitext(os.path.basename(skill_yaml_paths[0]))[0]
    skill_sg_param = cfg[skill_name] 

    for episode_idx in range(30): 
        hdf5_name = f"episode_{episode_idx}"
        new_hdf5_path = os.path.join(skill_sg_param['output_dir'], hdf5_name + '.hdf5')
        visualize_hdf5(new_hdf5_path, obj_name=obj_name, vis_start=vis_start, use_gui=True)



def extract_real_aloha(cfg_path, skill_yaml_paths):
    cfg = load_yaml_params(cfg_path, skill_yaml_paths=skill_yaml_paths)
    skill_name = os.path.splitext(os.path.basename(skill_yaml_paths[0]))[0]
    skill_sg_param = cfg[skill_name] 
    h5_parent_dir = skill_sg_param['h5_parent_dir']
    use_gui = cfg['use_gui']
    text_prompt = cfg['text_prompt']
    if not text_prompt.endswith('.'):
        text_prompt += '.'
    data_type = cfg['data_type']
    if not os.path.exists(skill_sg_param['output_dir']):
        os.makedirs(skill_sg_param['output_dir'])

    ## start pybullet to run the perception model
    connect(use_gui)
    robot_name = cfg['robot_name']
    if robot_name == 'dualfranka':
        robot_entity, names, movable_bodies, stackable_bodies = load_dual_franka_world_flexible()
    elif robot_name == 'aloha':
        robot_entity, names, movable_bodies, stackable_bodies = load_aloha_world_flexible()

    estimator = estimation_policy(robot_entity, mode = 'data_process', teleport=False, client=CLIENT, seg_branch='sam', text_prompt=text_prompt, env_type = 'file',  sam_path = cfg['sam_path'],  use_server = False)

    cam_dir_mapping, cam_extparam_mapping, calibrate_mapping = get_camera_mappings(cfg)

    for episode_idx in range(1,10): 
    # for episode_idx in [0]: # [15, 16, 17]: 
        test_hdf5_path = os.path.join(h5_parent_dir, f"episode_{episode_idx}.hdf5")
        debug_save_name = os.path.join(skill_sg_param['output_dir'], f"episode_{episode_idx}")
        if data_type == 'real':

            # try:
            has_eff = len(skill_sg_param['eff_obj_names']) > 0
            obs_qpos, detected_objs = get_jpose_pc_realsense(cfg,  test_hdf5_path, estimator=estimator,  cam_extparam_mapping = cam_extparam_mapping, cam_dir_mapping = cam_dir_mapping, calibrate_mapping = calibrate_mapping, predict_effect = has_eff, debug_save_name = debug_save_name)

            assumed_objs_pre = skill_sg_param['pre_obj_names']
            if len(detected_objs) != len(assumed_objs_pre):
                print(f"Encounter error in {test_hdf5_path}: detected_objs: {detected_objs.keys()} != len_objs: {assumed_objs_pre}")
                continue

            ## filter bad demo without eff pc
            contain_start = ['start' in detected_objs[obj_name] for obj_name in assumed_objs_pre]
            if not all(contain_start):
                continue

            if has_eff:
                assumed_objs_eff = skill_sg_param['eff_obj_names']
                contain_end = ['end' in detected_objs[obj_name] for obj_name in assumed_objs_eff]
                if not all(contain_end):
                    continue

            filtered_gripper_actions = postprocess_jointgrasp(cfg, skill_sg_param, test_hdf5_path, obs_qpos, detected_objs, robot_entity)
         

        elif data_type == 'mj':
            raise NotImplementedError("Not supported yet")

    disconnect()



if __name__ == '__main__':
    cfg_path = 'examples/pybullet/aloha_real/openworld_aloha/configs/sgBase.yaml'  #sg_cup_random | sg_screwdriver | sg_harrypotter |sg_transfer_cup
    skill_yaml_paths = ['examples/pybullet/aloha_real/openworld_aloha/configs/skill_cfg/screwdriver_per_skill.yaml']

    extract_real_aloha(cfg_path, skill_yaml_paths=skill_yaml_paths)

    # do_visualization(cfg_path, skill_yaml_paths=['examples/pybullet/aloha_real/openworld_aloha/configs/skill_cfg/handoff_cup_per_skill.yaml'], obj_name="cup", vis_start=True)

