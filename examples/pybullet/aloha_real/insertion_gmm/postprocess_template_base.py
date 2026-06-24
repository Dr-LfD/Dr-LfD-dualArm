import numpy as np
import os
import cv2
import json
import copy
from itertools import product, combinations
import networkx as nx
import sys
import ast
import h5py
from collections import namedtuple
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
from tqdm import tqdm

EdgeOpTuple = namedtuple('EdgeOpTuple', ['edge', 'is_add'])
OpNTimeTuple = namedtuple('OpNTimeTuple', ['edge_op', 'time_idx'])

# Custom class for edge_id_tuple that only uses edge_ops for set operations
class EdgeIdTuple:
    def __init__(self, edge_ops, primitive_name, sorted_id_list):
        self.edge_ops = edge_ops
        self.primitive_name = primitive_name
        self.sorted_id_list = sorted_id_list
        self.start_idx = sorted_id_list[0]
        self.end_idx = sorted_id_list[-1]
    
    def __hash__(self):
        # Only hash the edge_ops for set operations
        return hash(self.__repr__())
    
    def __eq__(self, other):
        # Only compare edge_ops for equality in set operations
        if isinstance(other, EdgeIdTuple):
            return self.primitive_name == other.primitive_name
        return False
    
    def __repr__(self):
        return f"EdgeIdTuple(primitive_name={self.primitive_name}, start_idx={self.start_idx}, end_idx={self.end_idx})"

def dilate(binary_list, window_size=3):
    n = len(binary_list)
    output = [False] * n  # Initialize output with F (False)
    radius = window_size // 2  # How far to look on each side (1 for window_size=3)

    for i in range(n):
        # Determine the start and end of the window
        start = max(0, i - radius)
        end = min(n, i + radius + 1)  # +1 because slice end is exclusive
        # Check if any element in the window is True
        if any(binary_list[start:end]):
            output[i] = True

    return output

def get_true_segments(binary_list):
    segments = []
    n = len(binary_list)
    i = 0

    while i < n:
        if binary_list[i]:  # Start of a new True segment
            start = i
            # Find the end of this segment
            while i < n and binary_list[i]:
                i += 1
            end = i - 1
            segments.append((start, end))
        else:
            i += 1

    return segments


def _smooth(data, window_size=5):
    if window_size % 2 == 0:
        print("window size must be odd, add 1 autonomously.")
        window_size += 1
    data = np.pad(data, (window_size // 2, window_size // 2), mode='edge')
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    
def is_containner(obj_name, containner_kw = set(['base', 'bin', 'basket', 'plate', 'caddy', 'stove'])):
    """
    Check if the object is a container based on its name.
    Args:
        obj_name (str): Name of the object.
        containner_kw (set): Keywords indicating a container.
    Returns:
        bool: True if the object is a container, False otherwise.
    """
    return any(kw in obj_name for kw in containner_kw)


def get_switch_segment_ids(action_qpos, gripper_qvel, gripper_qpos=None, 
                          extend_num=5, change_rate_threshold=0.01, 
                          x_threshold=0.02,  type='grasp'):
    """
    Get switch segment IDs that start with action switches and end when gripper change rate is near zero.
    
    Args:
        action_qpos: Gripper position values
        gripper_qvel: Rate of change in gripper position
        gripper_qpos: Smoothed gripper position (optional, uses action_qpos if None)
        extend_num: Number of frames to extend around action switches
        change_rate_threshold: Threshold for considering gripper change rate as "near zero"
        x_threshold: Threshold for gripper position
        dx_threshold: Threshold for gripper change rate
        type: 'grasp' or 'release'
    
    Returns:
        list: List of segment IDs where each segment is a list of consecutive frame indices
    """
    # Get initial action switch IDs
    action_change_rate = np.diff(action_qpos)
    
    # Zero out values in sliding windows where sum is near zero (indicating noise)
    if len(action_change_rate) >= 5:
        for i in range(len(action_change_rate) - 4):
            window = action_change_rate[i:i+5]
            window_sum = np.sum(window)
            
            # If sum is very small and there are non-zero values, zero them all
            if abs(window_sum) < 0.1 and np.any(window != 0):
                action_change_rate[i:i+5] = 0
    if type == 'grasp':
        action_switch_ids = np.where(action_change_rate > 0)[0]  # Closing actions
    else:  # release
        action_switch_ids = np.where(action_change_rate < 0)[0]  # Opening actions
    
    # segment_ids = []
    prehensile_segemnts = []
    
    for switch_id in action_switch_ids:
        # Start segment from the action switch (with extension)
        segment_start = switch_id
        
        # Find where the gripper change rate becomes near zero
        segment_end = None
        
        # Look forward to find where change rate stabilizes near zero
        look_ahead = min(30, len(gripper_qvel) - switch_id - extend_num)  # Limit look ahead
        
        for i in range(switch_id, min(len(gripper_qvel), switch_id + look_ahead)):
            if type == 'grasp' and gripper_qpos[i] < x_threshold:
                if abs(gripper_qvel[i]) < change_rate_threshold:
                    segment_end = i
                    break
            elif type == 'release' and gripper_qpos[i] > x_threshold:
                segment_end = i
                break
        
        if segment_end is None or segment_end < switch_id + extend_num:
            segment_end = switch_id + extend_num
        # Create the segment
        segment = list(range(segment_start, segment_end + 1))
        # ## filter out the failed grasp segments
        # if prehensile_segemnts and prehensile_segemnts[-1][-1] + too_close_threth > segment_start :
        #     prehensile_segemnts.pop()
        prehensile_segemnts.append(segment)
    
    return prehensile_segemnts


def detect_in_hand_status(
    rbt_actions, obj_pcs, exposed_flags, prior_graphs,
    cos_threshold=0.85,
    speed_diff_threshold=0.05,
    min_speed=5e-3,
    consecutive_frames=5,
):
    """
    For each (robot, object) grasp pair declared in prior_graphs, detect the
    first timestep where the object's translational velocity is aligned with
    the gripper's — the predicted in-hand onset (grasp-event boundary).

    All conditions must hold for `consecutive_frames` consecutive frames:
      - cos(obj_vel, gripper_vel) >= cos_threshold
      - ||obj_vel - gripper_vel|| <= speed_diff_threshold  (meters per frame)
      - min(||obj_vel||, ||gripper_vel||) >= min_speed
      - object visible (exposed_flags)

    Returns
    -------
    {rbt_name: {obj_name: {'in_hand_flags': np.ndarray[bool, T],
                            'grasp_event_boundary': int | None}}}
    """
    result = {}
    if prior_graphs is None or 'edge_ops' not in prior_graphs:
        return result

    def _smooth_diff(pos):
        d = np.diff(pos, axis=0)
        return np.stack([_smooth(d[:, k], window_size=5) for k in range(3)], axis=1)

    for rbt_name, edge_op_entries in prior_graphs['edge_ops'].items():
        grasp_obj_names = []
        seen = set()
        for edge_ops, primitive_name in edge_op_entries:
            if 'grasp' not in primitive_name:
                continue
            for edge_op in edge_ops:
                if not edge_op.is_add:
                    continue
                u, v = edge_op.edge
                candidate = v if u == rbt_name else (u if v == rbt_name else None)
                if candidate is None or candidate not in obj_pcs or candidate in seen:
                    continue
                seen.add(candidate)
                grasp_obj_names.append(candidate)

        if not grasp_obj_names:
            continue

        if rbt_name not in rbt_actions:
            raise KeyError(
                f"detect_in_hand_status: robot '{rbt_name}' is in prior_graphs but missing from rbt_actions. "
                f"Ensure sg_params['robots'] matches the robot nodes in the JSON schema. "
                f"rbt_actions keys: {list(rbt_actions.keys())}"
            )

        result[rbt_name] = {}
        eef_pos = np.array(rbt_actions[rbt_name]['eef_pos'])

        for obj_name in grasp_obj_names:
            obj_vis = np.asarray(exposed_flags[obj_name], dtype=bool)
            demo_len = min(len(obj_pcs[obj_name]), len(obj_vis), len(eef_pos))

            in_hand_flags = np.zeros(demo_len, dtype=bool)
            grasp_event_boundary = None

            if demo_len < consecutive_frames + 1:
                result[rbt_name][obj_name] = {
                    'in_hand_flags': in_hand_flags,
                    'grasp_event_boundary': grasp_event_boundary,
                }
                continue

            # Object centroid with carry-forward for occluded frames (vectorized)
            valid = obj_vis[:demo_len]
            raw_centers = obj_pcs[obj_name][:demo_len].mean(axis=1)  # (T, 3)
            fill_idx = np.where(valid, np.arange(demo_len), 0)
            np.maximum.accumulate(fill_idx, out=fill_idx)
            obj_center = raw_centers[fill_idx]
            first_vis = int(np.searchsorted(valid.cumsum(), 1))
            obj_center[:first_vis] = 0.0  # zero frames before any visible centroid

            obj_vel  = _smooth_diff(obj_center)
            grip_vel = _smooth_diff(eef_pos[:demo_len])

            obj_speed  = np.linalg.norm(obj_vel,  axis=1)
            grip_speed = np.linalg.norm(grip_vel, axis=1)
            cos_sim    = np.einsum('ij,ij->i', obj_vel, grip_vel) / (obj_speed * grip_speed + 1e-8)

            aligned = (
                valid[1:]
                & (cos_sim >= cos_threshold)
                & (np.linalg.norm(obj_vel - grip_vel, axis=1) <= speed_diff_threshold)
                & (np.minimum(obj_speed, grip_speed) >= min_speed)
            )
            in_hand_flags[1:] = aligned

            # First run of `consecutive_frames` aligned frames via convolution
            run_sums = np.convolve(aligned.astype(np.int8), np.ones(consecutive_frames, dtype=np.int8), mode='valid')
            hits = np.where(run_sums == consecutive_frames)[0]
            grasp_event_boundary = int(hits[0]) + 1 if len(hits) else None

            result[rbt_name][obj_name] = {
                'in_hand_flags': in_hand_flags,
                'grasp_event_boundary': grasp_event_boundary,
            }

    return result


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

def min_dist_kdtree(kdtree, points):
    """Compute the minimum distance from points to the kdtree."""
    if len(points) == 0:
        return np.array([])
    if isinstance(points, list):
        points = np.array(points)
    distances, _ = kdtree.query(points)

    if isinstance(distances, np.ndarray):
        return np.min(distances)
    return distances
    


def find_first_nonnan(height_list):
    """
    Find the index of the first non-NaN value in a list.
    Args:
        height_list (list): List of heights.
    Returns:
        int: Index of the first non-NaN value, or -1 if all are NaN.
    """
    for i, h in enumerate(height_list):
        if not np.isnan(h):
            return h
    return np.nan

## TODO: instead of using hand center, we can use gripper point cloud for distance calculation
def compute_distance_dict(obj_pcs, exposed_flags,  rbt_actions,  dot_folder = None,  ep_name = None):

    demo_len = len(rbt_actions['robot0']['eef_pos'])
    robots = list(rbt_actions.keys())
    objects = list(obj_pcs.keys())
    ## pre-compute distance
    distance_dict = {}
    # bbx_intersect_dict = {}
    robot_hand_center_dict = {rbt_name: np.array(rbt_actions[rbt_name]['eef_pos']) for rbt_name in rbt_actions.keys()}

    # ## check if the object is visible
    # exposed_flags = {}
    # for obj_name in obj_pcs.keys():
    #     exposed_flags[obj_name] = np.array([True if obj_pcs[obj_name][i].shape[0] > 10 else False for i in range(demo_len)])

    #     if np.sum(exposed_flags[obj_name]) < demo_len:
    #         print(f"Warning: Object {obj_name} is only visible in {np.sum(exposed_flags[obj_name])} / {demo_len} frames.")

    obj_center_dict = {}
    for obj_name in obj_pcs.keys():

        obj_center_dict[obj_name] = np.array([
            obj_pcs[obj_name][i].mean(axis=0) if exposed_flags[obj_name][i] else np.ones((3)) * np.inf
            for i in range(demo_len)
        ])
    from scipy.spatial import KDTree
    # obj_bbx_dict = {}
    obj_kdtree_dict = {}
    # obj_center_dict = {}
    for obj_name in obj_pcs.keys():
        # obj_bbx_dict[obj_name] = [aabb_from_points(obj_pcs[obj_name][i]) for i in range(demo_len)]
        # obj_center_dict[obj_name] = [get_aabb_center(obj_bbx_dict[obj_name][i]) for i in range(demo_len)]
        # obj_center_dict[obj_name] = np.array(obj_center_dict[obj_name])
        obj_kdtree_dict[obj_name] = [KDTree(obj_pcs[obj_name][i]) if exposed_flags[obj_name][i] else None for i in range(demo_len)]
    
    if len(robots) == 2:
        robot_hand_dist = np.linalg.norm(robot_hand_center_dict[robots[0]] - robot_hand_center_dict[robots[1]], axis=1)
        distance_dict['hand_hand'] = robot_hand_dist

    draw_key = []
    for rbt_name, obj_name in product(robots, objects):

        hand_obj_dist = [min_dist_kdtree(obj_kdtree_dict[obj_name][i], robot_hand_center_dict[rbt_name][i]) if exposed_flags[obj_name][i] else np.inf for i in range(demo_len)]
        ## to prevent non-differentiable point
        if np.any(exposed_flags[obj_name]==False):
            distance_dict[f'{rbt_name}_{obj_name}'] = hand_obj_dist
        else:
            distance_dict[f'{rbt_name}_{obj_name}']  = _smooth(hand_obj_dist, window_size=5)
        distance_dict[f'{obj_name}_{rbt_name}'] = distance_dict[f'{rbt_name}_{obj_name}']
        draw_key.append(f'{rbt_name}_{obj_name}')

    for obj_name in objects:

        # obj_height  =obj_center_dict[obj_name][:, 2].copy()
        min_pc_z = [np.min(obj_pcs[obj_name][i][:, 2]) for i in range(demo_len)]
        obj_height= [min_pc_z[i] if exposed_flags[obj_name][i] else min_pc_z[i-1] for i in range(demo_len)]
        ## assume all objects are initially placed on the table
        obj_height -= find_first_nonnan(obj_height) # normalize


    ## to prevent non-differentiable point
        if np.any(exposed_flags[obj_name]==False):
            distance_dict[f'{obj_name}_height'] = obj_height
        else:
            distance_dict[f'{obj_name}_height'] = _smooth(obj_height, window_size=5)  

        draw_key.append(f'{obj_name}_height')


    ## TODO; for assembly, piece_1 can be initially near the base. we have to identify the insertion motion. 
    for obj_name1, obj_name2 in combinations(objects, 2):
        dist_list = []
        both_exposed = []
        for i in range(demo_len):
            if exposed_flags[obj_name1][i] and exposed_flags[obj_name2][i]:
                both_exposed.append(i)
                if is_containner(obj_name1) or is_containner(obj_name2):
                    dist = np.linalg.norm(obj_center_dict[obj_name1][i] - obj_center_dict[obj_name2][i])
                else:
                    dist = min_dist_kdtree(obj_kdtree_dict[obj_name1][i], obj_pcs[obj_name2][i])
                dist_list.append(dist)
            else:
                dist_list.append(np.inf)
        # Smooth only the segment where both are exposed
        if both_exposed:
            exposed_segment = [dist_list[i] for i in both_exposed]
            smoothed = _smooth(exposed_segment, window_size=5)
            for idx, i in enumerate(both_exposed):
                dist_list[i] = smoothed[idx]
        distance_dict[f'{obj_name1}_{obj_name2}'] = dist_list
        distance_dict[f'{obj_name2}_{obj_name1}'] = dist_list
        draw_key.append(f'{obj_name1}_{obj_name2}')

    ## plot distance dict
    if ep_name is not None:
        plot_save_name = os.path.join(dot_folder, f'distance_plot_ep_{ep_name}.png')
        from matplotlib import pyplot as plt
        plt.clf()        

        for key in draw_key:
            plt.plot(distance_dict[key], label=key)
        plt.xlabel('Time Step')
        plt.ylabel('Distance')
        plt.legend()
        plt.savefig(plot_save_name)
        
    return distance_dict


    

def edge_changes_here(edge_op, distance_dict, sg_params, i, rbt_actions):
    """Determine if an edge should be added or removed based on distance and intersection constraints.
    
    Args:
        u, v: Edge vertices
        distance_dict: Dictionary of distances between entities
        bbx_intersect_dict: Dictionary of bounding box intersections
        sg_params: Dictionary containing all scene graph parameters including thresholds and entity lists
        is_to_add: Whether we're checking for adding or removing the edge
        i: Current time index
    
    Returns:
        bool: Whether the edge change should occur now
    """
    robots = sg_params['robots']
    objects = sg_params['interested_objs']
    surface = 'table'

    edge, is_to_add = edge_op
    u, v = edge

    demo_len = len(rbt_actions['robot0']['eef_pos'])

    involved_robots = set([u, v]).intersection(set(robots))
    if len(involved_robots) == 1:
        # if i in rbt_actions[list(involved_robots)[0]]['grasp_ids'] and is_to_add:
        #     return i

        ## delay to allow release movement
        if i in rbt_actions[list(involved_robots)[0]]['release_ids'] and not is_to_add:
            return min(i+10, demo_len-1) 
        
    # Get distance based on edge type
    if u in robots:  # v must be object
        thresh = sg_params['hand_obj_dist_threshold']
        related_distance = distance_dict[f'{u}_{v}'][i]
        # related_intersect = bbx_intersect_dict[f'{u}_{v}'][i]
    elif v in robots:  # u must be object
        thresh = sg_params['hand_obj_dist_threshold']
        related_distance = distance_dict[f'{v}_{u}'][i]
        # related_intersect = bbx_intersect_dict[f'{u}_{v}'][i]
    elif u in objects and v in objects:  # both are objects
        thresh = sg_params['obj_obj_dist_threshold']
        related_distance = distance_dict[f'{u}_{v}'][i]
        # related_intersect = bbx_intersect_dict[f'{u}_{v}'][i]
    elif u == surface or v == surface:
        obj_name = list(set([u, v]) - set([surface]))[0]
        thresh = sg_params['lifted_threshold']
        related_distance = distance_dict[f'{obj_name}_height'][i]
        # For surface edges, we don't use intersection constraints
        # related_intersect = True
    else:
        raise ValueError(f"Invalid edge: {u} {v}")
    
    # For adding edges: must be close OR intersecting
    # For removing edges: must be far OR not intersecting
    add_edges_now = (related_distance < thresh ) and is_to_add 
    remove_edges_now = (related_distance > thresh ) and not is_to_add

    ## output true if all edge_ops are detected
    if add_edges_now or remove_edges_now:
        return i
    else:
        return -1
    

def get_continuous_segments(valid_ids):
    diffs = np.diff(valid_ids)
    breaks = np.where(diffs > 1)[0] + 1
    segments = np.split(valid_ids, breaks)
    return segments



def get_per_rbt_changes(rbt_actions, sg_params, distance_dict, prior_graphs):
    def update_op_to_detect(edge_op_dict, rbt_name, edge_op_id):
        edge_ops, primitive_name = edge_op_dict[rbt_name][edge_op_id]
        change_idx_dict = {edge_op:-1 for edge_op in edge_ops}
        return edge_ops, primitive_name, change_idx_dict

    demo_len = len(rbt_actions['robot0']['eef_pos'])
    robots = sg_params['robots']

    edge_op_dict = prior_graphs['edge_ops']
    edge_op_time_dict = {}
    opNtime_dict = {}
    for rbt_name in robots:
        edge_op_id = 0
        edge_op_time_dict[rbt_name] = []
        opNtime_dict[rbt_name] = []
        edge_ops, primitive_name , change_idx_dict = update_op_to_detect(edge_op_dict, rbt_name, edge_op_id)

        for i in range(1,demo_len):

            for edge_op in edge_ops:
                # update only when the edge_op is not detected
                if change_idx_dict[edge_op] == -1:
                    change_idx = edge_changes_here(edge_op, distance_dict, sg_params,  i, rbt_actions)
                    change_idx_dict[edge_op] = change_idx

            sorted_idx_list = sorted(list(change_idx_dict.values()))
            # if len(sorted_idx_list) == 1:
            #     sorted_idx_list.append(demo_len-1)
            if -1 not in sorted_idx_list:
                for edge_op, idx in change_idx_dict.items():
                    opNtime_dict[rbt_name].append(OpNTimeTuple(edge_op, idx))

                id_change_dict = {v:k for k,v in change_idx_dict.items()}

                sorted_edge_op_list = [id_change_dict[idx] for idx in sorted_idx_list]

                edge_op_time_dict[rbt_name].append(EdgeIdTuple(sorted_edge_op_list, primitive_name, sorted_idx_list))
                
                if edge_op_id < len(edge_op_dict[rbt_name])-1:
                    edge_op_id += 1
                    edge_ops, primitive_name , change_idx_dict = update_op_to_detect(edge_op_dict, rbt_name, edge_op_id)
                else:
                    break

    return edge_op_time_dict, opNtime_dict

def recover_graph(robots, opNtime):
    init_graph = nx.Graph()
    name_prefix = ''.join(robots)
    init_graph.name = f'{name_prefix}_sg_0'

    edge_1st_op = {}
    for edge_op_t in opNtime:
        edge_op, _ = edge_op_t
        changed_edge, is_add = edge_op
        if changed_edge not in edge_1st_op:
            edge_1st_op[changed_edge] = is_add

        init_graph.add_nodes_from(changed_edge)

    lacked_edge = [edge for edge, is_add in edge_1st_op.items() if is_add == False]
    init_graph.add_edges_from(lacked_edge)

    sg_seq = [init_graph.copy()]
    for i, edge_op_t in enumerate(opNtime):
        edge_op, _ = edge_op_t
        cur_sg = sg_seq[-1].copy()
        changed_edge, is_add = edge_op

        if is_add == True:
            cur_sg.add_edge(*changed_edge)
        elif is_add == False:
            cur_sg.remove_edge(*changed_edge)
        else:
            raise ValueError(f"Invalid operation: {is_add}")
        cur_sg.name = f'{name_prefix}_sg_{i+1}'
        sg_seq.append(cur_sg)
    return sg_seq

def find_sg_from_edge(edge_op, detected_all_sg, sorted_opNtime):
    for i, opNtime in enumerate(sorted_opNtime):
        if opNtime.edge_op == edge_op:
            return detected_all_sg[i+1]
    raise ValueError(f"Cannot find sg for edge_op: {edge_op}")

## idea: use VL to extract the sg seq for each robot, and match each sg with index.
def get_actions_from_prior_graphs(sg_params, obj_pcs, rbt_actions, distance_dict, prior_graphs, dot_folder = None, ep_name = None):
    objects = list(obj_pcs.keys())
    surface = 'table'
    robots = list(rbt_actions.keys())
    demo_len = len(rbt_actions['robot0']['eef_pos'])

    matched_actions = {}

    skill_edge_time_dict, opNtime_dict = get_per_rbt_changes(rbt_actions, sg_params,  distance_dict, prior_graphs)

    # Collect sets of edges for each robot
    robot_skill_edges = {r: set(skill_edge_time_dict[r]) for r in robots}

    # Find intersection (edges shared by all robots)
    if len(robot_skill_edges) > 1:
        intersected_edges = set.intersection(*robot_skill_edges.values())
    else:
        intersected_edges = set()

    # Remove shared edges from each robot's list and sort
    for r in robots:
        skill_edge_time_dict[r] = sorted(list(robot_skill_edges[r] - intersected_edges), key=lambda x: x.start_idx)

    # Union of all edges for visualization, sorted by time
    all_edge_skills = sorted(set.union(*robot_skill_edges.values()), key=lambda x: x.start_idx)

    ## recover sg for 2 rbts
    opNtime_set = set([opNtime for rbt in robots for opNtime in opNtime_dict[rbt]])
    sorted_opNtime = sorted(opNtime_set, key=lambda x: x.time_idx)
    detected_all_sg = recover_graph(robots, sorted_opNtime)

    ### decide the contact-rich segments
    detected_all_sg[0].graph['is_biop'] = False
    for i in range(0, len(all_edge_skills)):

        primitive = all_edge_skills[i].primitive_name
        contact_rich = "attach" in primitive or "insert" in primitive or 'bimanual' in primitive

        ## NOTE that the sg seq for all robots changes at most 1 edge
        for edge_op, op_idx in zip(all_edge_skills[i].edge_ops, all_edge_skills[i].sorted_id_list):

            ## find cur sg
            cur_sg = find_sg_from_edge(edge_op, detected_all_sg, sorted_opNtime)

            if len(robots) ==2:
                ## delete the surface nodes
                sg_no_surface = cur_sg.copy()
                sg_no_surface.remove_nodes_from([surface])

                ## Criterion of contact-rich segments: 1) if it is connected by both robots (hammer); 2) Object-object contact. 
                if nx.has_path(sg_no_surface, robots[0], robots[1])\
                    or contact_rich:
                    cur_sg.graph['is_biop'] = True
                else:
                    cur_sg.graph['is_biop'] = False
            else:
                cur_sg.graph['is_biop'] = False

    out_dot_name = os.path.join(dot_folder, f'sg_seq_{ep_name}.png')
    sg_dot = vis_sg_seq(detected_all_sg, save_name=out_dot_name)

    bimanual_flag_list = [sg.graph['is_biop'] for sg in detected_all_sg]
    bimanual_flag_list = dilate(bimanual_flag_list, window_size=3)
    
    ## we associate biop flag with changed edges
    edge_time_biop_pair = dict(zip(sorted_opNtime, bimanual_flag_list[1:]))

    edge_biop_pair = {op_n_time.edge_op: edge_time_biop_pair[op_n_time] for op_n_time in sorted_opNtime}

    biop_segments = get_true_segments(bimanual_flag_list)

    if len(biop_segments) > 0:
        # raise ValueError(f"No bimanual segments detected in episode {ep_name}. Please check the edge changes and scene graphs.")

        for biop_i, biop_seg in enumerate(biop_segments):
            start_id, end_id = biop_seg
            pre_sg = detected_all_sg[start_id]
            pre_start_id = sorted_opNtime[start_id-1].time_idx
            pre_next_start_id = sorted_opNtime[start_id].time_idx
            pre_stage_span = list(range(pre_start_id, pre_next_start_id))
            if len(pre_stage_span) <= 4:
                print(f"Pre-stage of ep {ep_name} is too short: {len(pre_stage_span)}")
            pre_sg.graph['idx_list'] = pre_stage_span

            if end_id == len(detected_all_sg) - 1:
                eff_sg = None
            else:
                eff_sg = detected_all_sg[end_id]
                eff_start_id = sorted_opNtime[end_id].time_idx
                eff_next_start_id = sorted_opNtime[end_id].time_idx
                eff_sg.graph['idx_list'] = list(range(eff_start_id, eff_next_start_id))

            related_edges = sorted_opNtime[start_id:end_id+1]
            related_objs = set()
            for change_edge_id in related_edges:
                changed_edge,  _ = change_edge_id.edge_op
                related_objs = related_objs.union(set(changed_edge).intersection(set(objects)))

                
            matched_actions[f'bimanual_{biop_i}'] = {
                'pre_sg': pre_sg,
                'cur_sg': None,
                'eff_sg': eff_sg,
                'related_rbts': robots,
                'related_objs': list(related_objs),
                'extended_ids': pre_sg.graph['idx_list']
            }


    ## extract unimanual actions
    interested_primitives = sg_params['interested_primitives']
    # Build a mapping from each atomic edge_op to its corresponding scene graph index in detected_all_sg
    # Note: detected_all_sg[0] is the init graph; after each edge_op we append, so index is +1 offset
    # edge_op_to_seq_idx = {op_time.edge_op: (idx + 1) for idx, op_time in enumerate(sorted_opNtime)}

    for rbt_name in robots:

        ## debug 
        if len(skill_edge_time_dict[rbt_name]) == 0:
            raise ValueError(f"No edge changes detected for robot {rbt_name} in episode {ep_name}.")

        # For each robot, build its own sequence of scene graphs and label with idx_list
        # all_sg_cpy = [sg.copy() for sg in detected_all_sg]
        # single_sg_seq = [all_sg_cpy[0]]

        this_opNtime_list = [opNtime for opNtime in opNtime_dict[rbt_name]]
        this_sorted_opNtime = sorted(this_opNtime_list, key=lambda x: x.time_idx)
        single_sg_seq = recover_graph([rbt_name], this_sorted_opNtime)
        prev_change_id = 0
        for i, opNtime in enumerate(this_sorted_opNtime):
            single_sg_seq[i].graph['idx_list'] = list(range(prev_change_id, opNtime.time_idx))
            prev_change_id = opNtime.time_idx

        single_sg_seq[-1].graph['idx_list'] = list(range(prev_change_id, demo_len-1))

        ## match actions fo unimanual
        # rbt_edge_changes = [first_edge_change] + skill_edge_time_dict[rbt_name]
        rbt_edge_changes = skill_edge_time_dict[rbt_name]
        skillseg_nums = len(rbt_edge_changes)
        ## starting the first sg
        cur_sg_idx = 1
        for i in range(skillseg_nums):
            edge_ops = rbt_edge_changes[i].edge_ops
            num_ops  = len(edge_ops)
            primitive = rbt_edge_changes[i].primitive_name
            if primitive not in interested_primitives:
                cur_sg_idx += num_ops
                continue

            ## get pre_sg, cur_sg, eff_sg
            pre_sg_idx = cur_sg_idx-1
            eff_sg_idx = cur_sg_idx+num_ops-1
            if pre_sg_idx >= 0:
                pre_sg = single_sg_seq[pre_sg_idx]
            else:
                pre_sg = None
            cur_sg = single_sg_seq[cur_sg_idx]
            uncompleted_ids = cur_sg.graph['idx_list']
            if eff_sg_idx < len(single_sg_seq)-1:
                eff_sg = single_sg_seq[eff_sg_idx]
                completed_ids = eff_sg.graph['idx_list']
            else:
                eff_sg = None
                completed_ids = uncompleted_ids[-8:]
                uncompleted_ids = uncompleted_ids[:-8]

            ## filter out changes in biop
            if  len(robots) ==2:
                edges_in_biop = [edge_biop_pair[edge_op] for edge_op in edge_ops]
                if all(edges_in_biop):
                    continue

            ## get obj related to unimanual action
            related_objs = set()
            for edge_op in edge_ops:
                focused_edge, is_to_add = edge_op
                related_objs = related_objs.union(set(focused_edge).intersection(set(objects)))

            ## conditioned object in obj-obj contact
            if len(related_objs) >=2:
                ## in pre_sg, surface is not connected to robots
                conditioned_objs = [obj for obj in related_objs if not nx.has_path(pre_sg, obj, rbt_name)]
                related_objs = conditioned_objs + list(related_objs-set(conditioned_objs))
            else:
                related_objs = list(related_objs)

            ## put_ball_basket. basket is the conditioned obj
            objs_str = '_'.join(reversed(list(related_objs)))
            sb_do_sth = f'{rbt_name}_{primitive}_{objs_str}'

            ## set default exteded_ids and essential_ids
            extended_ids = cur_sg.graph['idx_list']
            if  len(extended_ids) < 8:
                if 'place' in sb_do_sth:
                    extended_ids = list(range(max(0, extended_ids[0]), min(demo_len, extended_ids[-1]+8)))
                else:
                    print(f'{sb_do_sth} has less than 8 ids: {extended_ids}')
                    continue
            essential_ids = extended_ids[:len(cur_sg.graph['idx_list'])//2]
                
            matched_actions[sb_do_sth] = {
                'pre_sg': pre_sg,
                'cur_sg': cur_sg,
                'eff_sg': eff_sg,
                'related_rbts': [rbt_name],
                'related_objs': related_objs,
                'essential_ids': essential_ids,
                'extended_ids': extended_ids,
                'completed_ids': completed_ids,
                'uncompleted_ids': uncompleted_ids,
            }

            cur_sg_idx += num_ops

    visualize_info = {
        'sg_dot': sg_dot,
        # 'sorted_opNtime': sorted_opNtime,
    }
    if len(matched_actions) < 2:
        print(f"Too few actions detected. dot_file: {dot_folder}/sg_seq_{ep_name}")
    return matched_actions, visualize_info



def parse_prior_graphs(data):
    # Parse the string representation of initial graph into a list of edges
    if isinstance(data["initial_graph"], str):
        init_edges = ast.literal_eval(data["initial_graph"])
    else:
        init_edges = data["initial_graph"]
    
    init_graph = nx.Graph()
    init_graph.add_edges_from(init_edges)

    robot_nodes = [nd for nd in init_graph.nodes if nd.startswith('robot')]

    for rbt_name in robot_nodes:
        if init_graph.has_edge(rbt_name, 'table'):
            init_graph.remove_edge(rbt_name, 'table')

    # Collect per-robot edge operation specifications
    per_rbt_edge_changes = {}
    for robot_name in robot_nodes:
        cur_graph_cpy = init_graph.copy()
        per_rbt_edge_changes[robot_name] = []
        for mode_change_info in data["ModeChangeDetection"]:
            contact_changes = mode_change_info["contact_changes"]
            edge_ops = []
            for contact_change in contact_changes:
                edge_u, edge_v = contact_change[0]
                edge_tuple = (edge_u, edge_v)
                edge_desc = contact_change[1]

                cur_graph_cpy.add_edge(*edge_tuple) if edge_desc == "add" else cur_graph_cpy.remove_edge(*edge_tuple)

                # Filter out changes not involving this robot 
                if len(robot_nodes) > 1 and not nx.has_path(cur_graph_cpy, edge_u, robot_name) and \
                   not nx.has_path(cur_graph_cpy, edge_v, robot_name):
                    continue

                edge_op = EdgeOpTuple(edge_tuple, edge_desc == "add")
                edge_ops.append(edge_op)

            if len(edge_ops) == 0:
                continue
            per_rbt_edge_changes[robot_name].append((edge_ops, mode_change_info["description"]))

    prior_graphs = {
        'init': init_graph,
        'edge_ops': per_rbt_edge_changes,
    }

    return prior_graphs

def filter_grasp_release_segs(grasp_segs, release_segs):
    filtered_grasp_segs = []
    filtered_release_segs = []

    if len(grasp_segs) == 0 and len(release_segs) == 0:
        return [], []

    if len(release_segs) == 0:
        grasp_ids = np.concatenate(grasp_segs).tolist()
        return grasp_ids, []
    elif len(grasp_segs) == 0:
        release_ids = np.concatenate(release_segs).tolist()
        return [], release_ids

    #         ## debug
    # if len(grasp_segs) != len(release_segs):
    #     print('check me in libero')

    for i_seg in range(len(grasp_segs)):

        ## if two grasps are too close, discard this demo, as this is a failed grasp
        if i_seg+1 < len(grasp_segs) and \
              grasp_segs[i_seg][-1] + 60 > grasp_segs[i_seg+1][0]:
            return None, None

        filtered_grasp_segs.append(grasp_segs[i_seg])

    for i_seg in range(len(release_segs)):
        filtered_release_segs.append(release_segs[i_seg])
    

    grasp_ids = np.concatenate(filtered_grasp_segs).tolist()
    release_ids = np.concatenate(filtered_release_segs).tolist()
    return grasp_ids, release_ids

def sg_monitoring(sg_params, obs_action_data, prior_graphs = None, visualize = False, ep_name = None):
    robots = sg_params['robots']
    # input_hdf5_path = sg_params['input_hdf5_path']
    interested_objs = sg_params['interested_objs']
    # hdf5_name = os.path.basename(input_hdf5_path).split('.')[0]

    obj_pcs = {}
    exposed_flags = {}
    for obj in interested_objs:
        obj_pcs[obj] = obs_action_data[f'{obj}_point_cloud']
        exposed_flags[obj] = obs_action_data[f'{obj}_visible']

    rbt_actions = {}
    for robot in robots:
        joint_vals = obs_action_data[f'{robot}_joint_pos']
        eef_pos = obs_action_data[f'{robot}_eef_pos']
        eef_quat = obs_action_data[f'{robot}_eef_quat']
        gripper_qpos = obs_action_data[f'{robot}_gripper_qpos']
        gripper_qvel = obs_action_data[f'{robot}_gripper_qvel']
      
        rbt_actions[robot] = {
                'eef_pos': eef_pos, 'eef_quat': eef_quat, \
                    'joint_pos': joint_vals, \
                    'gripper_qpos': gripper_qpos, 'gripper_qvel': gripper_qvel}
        
        ## only for parallel gripper
        gripper_actions = obs_action_data[f'{robot}_gripper_actions']

        # grasp_ids, release_ids = action_switch_ids(gripper_actions)
        grasp_segs = get_switch_segment_ids(gripper_actions, gripper_qvel, gripper_qpos, type='grasp')
        release_segs = get_switch_segment_ids(gripper_actions, gripper_qvel, gripper_qpos, type='release')

        grasp_ids, release_ids = filter_grasp_release_segs(grasp_segs, release_segs)

        if grasp_ids is None or release_ids is None:
            print(f"There are failed demonstrations in epsisode {ep_name}, please check the gripper actions!")
            return None
        rbt_actions[robot]['grasp_ids']  = grasp_ids
        rbt_actions[robot]['release_ids']  = release_ids

    in_hand_status = detect_in_hand_status(rbt_actions, obj_pcs, exposed_flags, prior_graphs)
    for rbt_name_ih, per_obj_ih in in_hand_status.items():
        rbt_actions[rbt_name_ih]['in_hand_status'] = per_obj_ih

    dot_folder = sg_params['output_dir']
    if not os.path.exists(dot_folder):
        os.makedirs(dot_folder)

    distance_dict = compute_distance_dict(obj_pcs, exposed_flags, rbt_actions, dot_folder = dot_folder, ep_name = ep_name)

    matched_actions, visualize_info = get_actions_from_prior_graphs(sg_params, obj_pcs, rbt_actions, distance_dict,
    dot_folder = dot_folder, ep_name = ep_name, prior_graphs = prior_graphs) 

    if visualize:
        vid_path = f"/home/user/yzchen_ws/imitation_learning/dexmimicgen/datasets/pc_demos/two_arm_threading/raw/two_arm_threading_{ep_name}.mp4"
        # vid_path = input_hdf5_path.replace('.hdf5', '.mp4')
        save_img_with_sg_seq(vid_path, visualize_info, matched_actions, ep_name=ep_name)

    ## if there is grasp/release ids detected from the gripper, adjust essential ids
    for action_name, action_info in matched_actions.items():
        if 'bimanual' in action_name:
            continue
        rbt_name = action_info['related_rbts'][0]
        # obj_name = action_info['related_objs'][0]

        is_action_interested = False
        for primitive_kw  in sg_params['interested_primitives']:
            if primitive_kw in action_name:
                action_type = primitive_kw
                is_action_interested = True
                break

        extended_ids = action_info['extended_ids']

        if action_type + '_ids' not in rbt_actions[rbt_name] or not is_action_interested:
            # print("Either the action is not interested or the gripper is not dexterous hand!")
            continue
        
        detected_ids = rbt_actions[rbt_name][action_type + '_ids']
        essential_ids = set()
        ## note: as there is failed grasp, we do not use pre_sg
        for sg in [action_info['cur_sg']]:
        # for sg in [action_info['pre_sg'], action_info['cur_sg']]:
            if sg is None:
                continue
            intersect_ids = set(detected_ids).intersection(set(sg.graph['idx_list']))
            essential_ids = essential_ids.union(intersect_ids)
            
        ## order the output
        essential_ids = sorted(list(essential_ids))
        action_info['essential_ids'] = essential_ids

        # assert len(essential_ids) > 0, f"Essential ids for {action_name} is empty, please check the action type and gripper actions!"
        if len(essential_ids) == 0:
            if action_type == 'grasp':
                print(f"Essential ids for {action_name} is empty, ep id is {ep_name}, detected_ids: {detected_ids}, extended_ids: {extended_ids}")
            ## TODO: still cannot cope with failed demo
            continue

        if action_type == 'grasp' and 'in_hand_status' in rbt_actions[rbt_name]:
            related_set = set(action_info['related_objs'])
            for obj_name_ih, ih_data in rbt_actions[rbt_name]['in_hand_status'].items():
                if obj_name_ih in related_set:
                    action_info['grasp_event_boundary'] = ih_data.get('grasp_event_boundary')
                    break

    return matched_actions



def postprocess_for_sgs(sg_params, n_playback = None, num_workers = 1, \
                    prior_graphs = None,  visualize = False, debug_ep_id = None):
    input_hdf5_path = sg_params['input_hdf5_path'] 
    f_in = h5py.File(input_hdf5_path, 'a')

    # env_meta = json.loads(f_in["data"].attrs["env_args"])

    demos = list(f_in["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    assert n_playback is not None or len(demos) <= 100, "n_playback must be provided, or the number of demonstrations is less than 100"
    # if n_playback is not None:
    demos = demos[:n_playback]
    # output_hdf5_path = input_hdf5_path.replace('.hdf5', f'_sg_{n_playback}.hdf5')
    # if os.path.exists(output_hdf5_path):
    #     f_in.close()
    #     print(f"Output file {output_hdf5_path} already exists. Please remove it or change the output path.")
    #     return
    output_hdf5_path = sg_params['output_hdf5_path']
    f_out = h5py.File(output_hdf5_path, 'w')
    # create the data group
    if "data/" in f_out:
        del f_out["data/"]
    f_out.create_group("data")

    ## copy the mapping from instance name to idx if exist
    if 'instance_name2id' in f_in.keys():
        instance_name2id = json.loads(f_in['instance_name2id'][()])
        sg_params['instance_name2id'] = instance_name2id

    # Create progress bar
    pbar = tqdm(total=len(demos), desc="Processing episodes", unit="ep")

    for i in range(0, len(demos), num_workers):
        end = min(i + num_workers, len(demos))
        data_list = []
        ep_list = []
        for j in range(i, end):
            ep = demos[j]
            ep_list.append(ep)

            # debug
            if debug_ep_id is not None and debug_ep_id != j:
                continue
            # else:
            #     print(f'processing {ep}')

            # prepare initial state to reload from
            data_dict = {}
            for rbt in sg_params['robots']:
                data_dict[f'{rbt}_eef_pos'] = f_in["data/{}/obs/{}_eef_pos".format(ep, rbt)][()]
                data_dict[f'{rbt}_eef_quat'] = f_in["data/{}/obs/{}_eef_quat".format(ep, rbt)][()]
                data_dict[f'{rbt}_joint_pos'] = f_in["data/{}/obs/{}_joint_pos".format(ep, rbt)][()]
                data_dict[f'{rbt}_gripper_qvel'] = f_in["data/{}/obs/{}_gripper_qvel".format(ep, rbt)][()][:, 0]
                data_dict[f'{rbt}_gripper_qpos'] = f_in["data/{}/obs/{}_gripper_qpos".format(ep, rbt)][()][:, 0]
            
            actions = f_in["data/{}/actions".format(ep)][()]
            for k, rbt in enumerate(sg_params['robots']):
                data_dict[f'{rbt}_gripper_actions'] = actions[:, (k+1)*7-1]
            # data_dict['robot1_gripper_actions'] = actions[:, 13]

            for obj_name in sg_params['interested_objs']:
                obj_pcs_xyzrgb = f_in["data/{}/obs/{}_point_cloud".format(ep, obj_name)][()]
                obj_pcs = obj_pcs_xyzrgb[:, :, :3]  ## (demo_len, num_points, 3)
                data_dict[obj_name + '_point_cloud'] = obj_pcs

                data_dict[obj_name + '_visible'] = f_in[f"data/{ep}/obs/{obj_name}_visible"][()]

            data_list.append(data_dict)

        if debug_ep_id is not None:

            if len(data_list) > 0:
                matched_actions_list =[ sg_monitoring(sg_params, data_list[0], prior_graphs, visualize, f'demo_{debug_ep_id}')]
            else:
                continue
        else:
            with multiprocessing.Pool(num_workers) as pool:
                tasks = [
                    [sg_params, data_list[j], prior_graphs, visualize, ep_list[j]]
                    for j in range(len(data_list))
                ]
                matched_actions_list = pool.starmap(sg_monitoring, tasks)


        for j, ind in enumerate(range(i, end)):
            ep = demos[ind]
            matched_actions = matched_actions_list[j]
            if matched_actions is None:
                print(f"Skipping demo {ep} due to failed sg monitoring.")
                continue
            if len(matched_actions.keys()) == 0:
                raise ValueError(f"No valid grasp found for demo {ep}")
            
            # Copy the entire episode data from f_in to f_out
            if f"data/{ep}" in f_out:
                del f_out[f"data/{ep}"]
            f_in.copy(f"data/{ep}", f_out["data"])
            
            ## save the matched actions to the hdf5 file
            demo_grp = f_out["data/{}".format(ep)]
            if 'sg_info' in demo_grp.keys():
                del demo_grp['sg_info']
            demo_grp.create_group('sg_info')

            # Save entire matched_actions as a JSON blob for convenience
            def _to_serializable(obj):
                if isinstance(obj, nx.Graph):
                    return nx.node_link_data(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                if isinstance(obj, (list, tuple)):
                    return [_to_serializable(x) for x in obj]
                if isinstance(obj, dict):
                    return {k: _to_serializable(v) for k, v in obj.items()}
                return obj

            matched_actions_json = json.dumps(_to_serializable(matched_actions))
            dt = h5py.string_dtype(encoding='utf-8')
            demo_grp.create_dataset('matched_actions_json', data=matched_actions_json, dtype=dt)

            for action_name, data_dict in matched_actions.items():
                for dict_key, dict_val in data_dict.items():
                    if dict_val is None:
                        continue

                    if isinstance(dict_val, nx.Graph):
                        sg = dict_val
                        graph_data = nx.node_link_data(sg)
                        json_str = json.dumps(graph_data)
                        dt = h5py.string_dtype(encoding='utf-8')
                        demo_grp.create_dataset(f'sg_info/{action_name}/{dict_key}', data=json_str, dtype=dt)
                    else:
                        demo_grp.create_dataset(f'sg_info/{action_name}/{dict_key}', data=dict_val)

        # Update progress bar
        pbar.update(end - i)

    # Close progress bar
    pbar.close()

    param_json = json.dumps(sg_params)
    dt = h5py.string_dtype(encoding='utf-8')
    if 'sg_params' in f_out.keys():
        del f_out['sg_params']
    f_out.create_dataset('sg_params', data=param_json, dtype=dt)
        
    ## important: save the env_meta
    f_out["data"].attrs["env_args"] = f_in["data"].attrs["env_args"]
    f_out["data"].attrs['sg_params'] = param_json
    print(f"------Saved to  {output_hdf5_path.split('/')[-1]}----------")
    f_in.close()
    f_out.close()

def load_yaml_params(cfg_path, skills = None):
    import yaml
    with open(cfg_path, 'r') as file:
        sg_params = yaml.load(file, Loader=yaml.FullLoader)
        if skills is not None:
            sg_params = {k: sg_params[k] for k in skills}
        return sg_params
    

def save_img_with_sg_seq(vid_path, visualize_info, matched_actions, ep_name = None):
    """Visualize video frames with scene graphs and highlight current graph state.
    Places scene graph on the right side of the frame and highlights current state using pydot attributes.
    Also displays action names when current frame is within their time spans.
    
    Args:
        vid_path (str): Path to input video file
        visualize_info (dict): Contains 'sg_dot' and 'all_edge_skills'
        matched_actions (dict): Dictionary of matched actions with their time spans
    """
    import cv2
    import pydot
    from PIL import Image
    import io
    import numpy as np
    import os

    # Load video
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {vid_path}")
        return

    # Create output directory for frames
    output_dir = os.path.join(os.path.dirname(vid_path), 'frames_with_sg', f'{ep_name}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving frames to {output_dir}")

    # Parse the DOT string to get graph structure
    dot_graph = pydot.graph_from_dot_data(visualize_info['sg_dot'])[0]
    all_edge_skills = visualize_info['all_edge_skills']
    
    # Track current graph state
    current_sg_idx = 0
    frame_idx = 0

    # Pre-compute action spans for faster lookup
    action_spans = {}
    for action_name, action_info in matched_actions.items():
        span = []
        if 'bimanual' in action_name:
            span = list(range(action_info['pre_sg'].graph['idx_list'][0], all_edge_skills[-1].time_idx + 1))
        # if action_info['pre_sg'] is not None:
        #     span.extend(action_info['pre_sg'].graph['idx_list'])
        if action_info['cur_sg'] is not None:
            span.extend(action_info['cur_sg'].graph['idx_list'])
        if len(span):  # Only add if there are any indices
            action_spans[action_name] = sorted(list(set(span)))  # Remove duplicates and sort

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Update current graph state based on edge changes
        while (current_sg_idx + 1 < len(all_edge_skills) and 
               all_edge_skills[current_sg_idx + 1].time_idx <= frame_idx):
            current_sg_idx += 1

        # Create a copy of the DOT graph for this frame
        frame_dot = pydot.graph_from_dot_data(visualize_info['sg_dot'])[0]
        
        # Set colors for each subgraph
        for i, subgraph in enumerate(frame_dot.get_subgraphs()):
            if i == (len(all_edge_skills) - current_sg_idx - 1):
                # Highlight current subgraph with red border
                subgraph.set('color', 'red')
                subgraph.set('penwidth', '3.0')
            else:
                # Set other subgraphs to black
                subgraph.set('color', 'black')
                subgraph.set('penwidth', '1.0')

        # Convert DOT to image
        png_data = frame_dot.create_png()
        sg_img = Image.open(io.BytesIO(png_data))
        sg_img = np.array(sg_img)
        
        # Convert RGBA to BGR for OpenCV
        if sg_img.shape[2] == 4:
            sg_img = cv2.cvtColor(sg_img, cv2.COLOR_RGBA2BGR)
            
        # resize frame to match scene graph height
        target_height = sg_img.shape[0]
        frame_height, frame_width = frame.shape[:2]
        frame_wh_ratio = frame_width / frame_height
        if frame_height != target_height:
            frame = cv2.resize(frame, (int(frame_wh_ratio*target_height), target_height))

        # Create a combined frame with video on left and scene graph on right
        combined_width = frame.shape[1] + sg_img.shape[1]
        combined_height = max(frame.shape[0], target_height)
        combined_frame = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
        
        # Place video frame on the left
        combined_frame[:frame.shape[0], :frame.shape[1]] = frame
        
        # Place scene graph on the right
        combined_frame[:target_height, frame.shape[1]:] = sg_img

        # Add frame number
        # Dynamically set font scale and thickness based on frame height
        base_height = 720  # Reference height for scaling
        font_scale = max(0.5, frame.shape[0] / base_height)
        font_thickness = max(1, int(frame.shape[0] / 360))
        x_indent = max(10, int(0.015 * frame.shape[1]))
        y_base = int(0.04 * frame.shape[0])
        y_step = int(0.045 * frame.shape[0])

        cv2.putText(combined_frame, f"Frame: {frame_idx}", (x_indent, y_base),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)

        # Display active actions
        y_offset = y_base + y_step
        for action_name, span in action_spans.items():
            if frame_idx in span:
                cv2.putText(combined_frame, action_name, (x_indent, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, (0, 255, 255), font_thickness)  # Yellow color
                y_offset += y_step  # Move down for next action

        # Save frame as image
        frame_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.png")
        cv2.imwrite(frame_path, combined_frame)
        
        frame_idx += 1

    # Clean up
    cap.release()
    print(f"Saved {frame_idx} frames to {output_dir}")
    print("You can use ffmpeg to convert these frames to a video if needed:")
    print(f"ffmpeg -framerate 30 -i {output_dir}/frame_%06d.png -c:v libx264 -pix_fmt yuv420p {output_dir}/output.mp4")


