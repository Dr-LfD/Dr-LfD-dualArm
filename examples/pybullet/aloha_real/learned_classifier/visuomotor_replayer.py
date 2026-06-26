
import numpy as np
import yaml
import time
import os
import sys
import os
EXE_FOLDER = next(p for p in (os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *([os.pardir] * k))) for k in range(16)) if os.path.isfile(os.path.join(p, '.repo_root')))
sys.path.append(EXE_FOLDER) if EXE_FOLDER not in sys.path else None

import pybullet as p

# from examples.pybullet.aloha_real.openworld_aloha.simple_worlds import load_world_3obj
from collections import defaultdict
from pddlstream.utils import get_file_path
from examples.pybullet.utils.pybullet_tools.aloha_primitives import  get_grasp_gen, get_stable_gen, BodyConf, BodyPose
from examples.pybullet.utils.pybullet_tools.utils import Pose, \
    Point, set_default_camera, CUBOID_URDF, CUBOID_DOWN_URDF, CUBOID_TALL_URDF,\
    load_model, HideOutput, draw_global_system, VX300_URDF, Euler, connect, pairwise_collision, DEFAULT_CLIENT, stable_z

from examples.pybullet.aloha_real.scripts.aloha_tamp_constants import qpos_to_eepose


import torch

SEQ_OFFSET = 0

FLOOR_OFFSET = -0.01
OBJ_HIGHT = 0.04

DEFAULT_DIST = np.inf

def load_preinsertion_world(pybullet_use_gui):
    connect(use_gui=pybullet_use_gui)

    # TODO: store internal world info here to be reloaded
    set_default_camera(yaw=30)
    draw_global_system()
    with HideOutput():
        # static
        robot_l = load_model(VX300_URDF, fixed_base=True, pose=Pose(Point(x=-0.469, y = 0), euler= Euler(0, 0, 0))) # DRAKE_IIWA_URDF | KUKA_IIWA_URDF
        robot_r = load_model(VX300_URDF, fixed_base=True, pose=Pose(Point(x=0.469, y = 0), euler= Euler(0, 0, -np.pi)))
        floor = load_model('models/aloha_desk.urdf',pose=Pose(Point(x=-0, y = -0, z = FLOOR_OFFSET)), fixed_base=True)
        # sink = load_model('models/virtual_sink.urdf', pose=Pose(Point(x=0.0, y = 0.0, z = 0.01)), fixed_base=True)
        
        wall_behind = load_model('models/wall.urdf', pose=Pose(Point(x=0.0, y = 0.43, z = 0.5)), fixed_base=True)
        front_pillar = load_model('models/pillar.urdf', pose=Pose(Point(x=0.0, y = -0.3, z = 0.2)), fixed_base=True)
        
        # holding
        socket = load_model(CUBOID_URDF, pose=Pose(Point(x=0.2, y=-0.2, z=0.0)), fixed_base=False)
        peg = load_model(CUBOID_DOWN_URDF, pose= Pose(Point(x=-0.25, y=0.2, z=0.0)), fixed_base=False)

        # dynamic
        cup = load_model(CUBOID_TALL_URDF,fixed_base=False, pose=Pose(Point(x=0, y=0, z=0.1)))
    
    
    body_names = {
        # sink: 'sink',
        socket: 'socket',
        peg: 'peg',
        cup: 'cup',
        floor: 'floor',
        wall_behind: 'wall_behind',
        front_pillar: 'front_pillar',
    }
    # movable_bodies = [ socket, peg]
    movable_bodies = [socket, peg, cup]
    stackable_bodies = [floor]

    return robot_l, robot_r,  body_names, movable_bodies, stackable_bodies

# get the data of (t, q_left, q_right)
def load_demonstration(file_name):

    qpos_mat = np.loadtxt(file_name)

    qpos_demos_left = []
    qpos_demos_right = []

    start_clips = 30
    end_clips = 30

    idx = SEQ_OFFSET
    for i in range(1, len(qpos_mat)):
        if qpos_mat[i][0] < qpos_mat[i-1][0]:
            cliped_start = idx+ start_clips
            cliped_end = i- end_clips
            if cliped_end <= cliped_start:
                continue 
            # demos.append(ee_left[idx:i])
            qpos_demos_left.append(qpos_mat[cliped_start:cliped_end, 1:7])
            qpos_demos_right.append(qpos_mat[cliped_start:cliped_end, 8:14])
            idx = i


    return qpos_demos_left, qpos_demos_right



def customized_collision_fn(body1, body2, max_distance = 0.5, draw_debugline = True):
    client = DEFAULT_CLIENT
    pts = client.getClosestPoints(
        bodyA=int(body1), bodyB=int(body2), distance=max_distance
    )
    
    if len(pts) == 0:
        return DEFAULT_DIST
    else:
        closest_id = np.argmin([pt[8] for pt in pts])
        distance = pts[closest_id][8]
        #print("distance=",distance)
        ptA = pts[closest_id][5]
        ptB = pts[closest_id][6]

        if draw_debugline:
            lineWidth = 3
            colorRGB = [1, 0, 0]
            lineId = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                        lineToXYZ=[0, 0, 0],
                                        lineColorRGB=colorRGB,
                                        lineWidth=lineWidth,
                                        lifeTime=0)

            p.addUserDebugLine(lineFromXYZ=ptA,
                            lineToXYZ=ptB,
                            lineColorRGB=colorRGB,
                            lineWidth=lineWidth,
                            lifeTime=0,
                            replaceItemUniqueId=lineId)
    return distance

def check_collision(movable_body, moving_bodies, **kwargs):
    min_dist = DEFAULT_DIST
    for body in moving_bodies:
        distance = customized_collision_fn(body, movable_body, **kwargs)
        min_dist = min(min_dist, distance)
        is_collide = pairwise_collision(body, movable_body)
        if is_collide:
            return True, min_dist
    # if any(pairwise_collision(body, movable_body) for body in moving_bodies):
    #     return True
    return False, min_dist

def update_sdf(min_dist_array, time_step, left_grasp= None, right_grasp=None, left_conf=None, right_conf=None, colpos=None, draw_debugline = False):
    left_bodies = [left_grasp.body, left_conf.body]
    right_bodies = [right_grasp.body, right_conf.body]
    moving_bodies = left_bodies + right_bodies
    movable_body = colpos.body
    collide_left, left_sdf = check_collision(movable_body, left_bodies, draw_debugline=draw_debugline)
    collide_right, right_sdf = check_collision(movable_body, right_bodies, draw_debugline=draw_debugline)

    # if collide_left:
    #     min_dist_array[0][time_step] = (0, left_grasp, left_conf, colpos)
    # if collide_right: 
    #     min_dist_array[1][time_step] = (0, right_grasp, right_conf, colpos)
    if collide_left:
        min_dist_array[0][time_step] = 0
    if collide_right:
        min_dist_array[1][time_step] = 0

    if draw_debugline:
        p.removeAllUserDebugItems()
    return left_sdf, right_sdf


def syn_data(raw_data, save_path):
    # save two list of tuples to pytorch datasets
    sample_data = [tp for data in raw_data for tp in data]
    sdf_ls, eepose_ls, colpos_2d_ls = zip(*sample_data)
    # sdf_ls, grasp_ls, conf_ls, colpos_ls = zip(*sample_data)
    # eepose_ls = [tuple(qpos_to_eepose(conf.configuration, conf.body)[0]) for conf in conf_ls]
    # colpos_2d_ls = [tuple(colpos.pose[0][:2]) for colpos in colpos_ls]

    sdf_tensor = torch.tensor(sdf_ls, dtype=torch.float32)
    eepose_tensor = torch.tensor(eepose_ls, dtype=torch.float32)
    colpos_tensor = torch.tensor(colpos_2d_ls, dtype=torch.float32)
    data_dict = {'sdf': sdf_tensor, 'eepose': eepose_tensor, 'colpos': colpos_tensor}
    torch.save(data_dict, save_path)
    print('Saved data to:', save_path)

    return data_dict

def extract_useful(min_dist, grasp, conf, colpos):
    eepose = tuple(qpos_to_eepose(conf.configuration, conf.body)[0])
    colpos_2d = colpos.pose[0][:2]
    return (min_dist, eepose, colpos_2d,)
    # return min_dist, grasp, conf, colpos
     
def main():

    real_time_render = True
    algorithm = "adaptive"
    unit = False
    pybullet_use_gui = True

    robot_l, robot_r, body_names, movable_bodies, stackable_bodies = load_preinsertion_world(pybullet_use_gui)

    reversed_dict = {v: k for k, v in body_names.items()}

    left_body = reversed_dict['socket']
    right_body = reversed_dict['peg']

    grasp_dict = defaultdict(lambda: 'top')
    grasp_dict[left_body] = 'sidePitched'
    grasp_dict[right_body] = 'topPitched'
    grasp_gen_fn = get_grasp_gen(grasp_dict)
    # get the eepose of left gripper


    pose_gen_fn = get_stable_gen(fixed=stackable_bodies)
    movable_body = reversed_dict['cup']

    file_name = get_file_path(__file__, "../insertion_gmm/insertion_human_qpos.txt")
    qpos_demos_left, qpos_demos_right = load_demonstration(file_name)

    # set the sample variations which is irrelevant to the demosntration
    

    # construct a meshgrid for the colpos
    col_min = [-0.35, -0.2]
    col_max = [0.35, 0.2]
    col_coords_xy = np.meshgrid(np.linspace(col_min[0], col_max[0], 20), np.linspace(col_min[1], col_max[1], 20))
    col_coords_z = stable_z(reversed_dict['cup'], reversed_dict['floor'])
    col_coords = np.array([col_coords_xy[0].flatten(), col_coords_xy[1].flatten(), col_coords_z*np.ones_like(col_coords_xy[0].flatten())]).T

    max_rand_iter = 400

    left_path = get_file_path(__file__, 'left_sdf_data_%d.pt'%max_rand_iter)
    right_path = get_file_path(__file__, 'right_sdf_data_%d.pt'%max_rand_iter)


    min_dsit_arrs = []
    left_data_list2 = []
    right_data_list2 = []

    for demo_id in range(len(qpos_demos_left)):
        print('Demo:', demo_id)
        qpos_demo_left = qpos_demos_left[demo_id]
        qpos_demo_right = qpos_demos_right[demo_id]
        demo_length = len(qpos_demo_left)

        # # set robot iintial pose, gripper status, and the object pose
        # colpos_gen = pose_gen_fn(movable_body, reversed_dict['floor'])
        # for rand_id in range(max_rand_iter):
        #     colpos = next(colpos_gen)[0]

        for col_crd in col_coords:
            colpos = BodyPose(body=reversed_dict['cup'], pose=Pose(point = col_crd, euler = Euler(0, 0, 0)))
            colpos.assign()
            left_grasp_gen = grasp_gen_fn(robot_l, left_body)
            right_grasp_gen = grasp_gen_fn(robot_r, right_body)
            left_grasp = next(left_grasp_gen)[0]
            right_grasp = next(right_grasp_gen)[0]

            # two rows, left and right
            min_dist_array = np.ones(shape=(2, demo_length))*DEFAULT_DIST

            left_data_list = []
            right_data_list = []

            min_dist_left, min_dist_right = DEFAULT_DIST, DEFAULT_DIST
            min_dist = DEFAULT_DIST
            for stmp in range(demo_length-1, -1, -1):

                left_conf = BodyConf(robot_l, qpos_demo_left[stmp])
                right_conf = BodyConf(robot_r, qpos_demo_right[stmp])
                # NOTE: must first assign conf, then assign the grasp
                left_conf.assign()
                right_conf.assign()
                left_grasp.assign()
                right_grasp.assign()


                results = update_sdf(min_dist_array, stmp, left_grasp=left_grasp, right_grasp=right_grasp, left_conf=left_conf, right_conf=right_conf, colpos=colpos)

                min_dist_left = min(min_dist_left, results[0])
                min_dist_right = min(min_dist_right, results[1])
                min_dist = min(min_dist, min(results))

                left_data_list.append(extract_useful(min_dist_left, left_grasp, left_conf, colpos))
                right_data_list.append(extract_useful(min_dist_right, right_grasp, right_conf, colpos))

            min_dsit_arrs.append(min_dist_array)
            left_data_list2.append(left_data_list[::-1])
            right_data_list2.append(right_data_list[::-1])
            


    left_dataset = syn_data(left_data_list2, left_path)
    right_dataset = syn_data(right_data_list2, right_path)

    print('Data created.')



        

    



if __name__ == '__main__':
    main()