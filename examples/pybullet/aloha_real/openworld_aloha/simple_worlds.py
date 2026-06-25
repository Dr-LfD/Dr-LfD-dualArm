
import sys, os
root_path = next(p for p in (os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *([os.pardir] * k))) for k in range(16)) if os.path.isfile(os.path.join(p, '.repo_root')))
sys.path.append(root_path) if root_path not in sys.path else None
from examples.pybullet.utils.pybullet_tools.utils import connect,  set_pose, Pose, \
    Point, set_default_camera, HideOutput, CUBOID_URDF, CUBOID_DOWN_URDF, CUBOID_TALL_URDF, SINK_URDF, BASKET_URDF, \
     load_model,  disconnect,  mesh_from_points, draw_global_system, ALOHA_URDF, DUAL_FRANKA_URDF, PANDA_DUAL_URDF, PANDA_ARM_URDF, \
       ensure_dir, save_image, create_mesh, quat_from_matrix, multiply, invert, set_joint_positions,\
       Euler, create_box, RGBA, remove_body, draw_pose

from examples.pybullet.aloha_real.openworld_aloha.entities import Object, LabeledPoint
from examples.pybullet.aloha_real.openworld_aloha.primitives import GroupConf
from examples.pybullet.aloha_real.openworld_aloha.robot_entities import ALOHARobot, DUALfrankaRobot, PandaDualRobot, PandaSingleRobot
from examples.pybullet.aloha_real.openworld_aloha.estimation.geometry import refine_shape
from examples.pybullet.aloha_real.openworld_aloha.estimation.belief import EstimatedObject
from examples.pybullet.aloha_real.openworld_aloha.estimation.bounding import estimate_oobb
from examples.pybullet.aloha_real.openworld_aloha.estimation.concave import concave_mesh, create_concave_mesh
from examples.pybullet.aloha_real.openworld_aloha.estimation.pc_utils import filter_pc, add_projected_point

import numpy as np

FLOOR_OFFSET = -0.01
OBJ_HIGHT = 0.04


def load_world_0obj():
    # for primitives_test.py
    set_default_camera(yaw=30)
    draw_global_system()
    with HideOutput():
        #add_data_path()
        robot_body = load_model(ALOHA_URDF, fixed_base=True, pose=Pose(Point(x=0, y = 0, z = 0.0)))
     
        wall_behind = load_model('models/wall.urdf', pose=Pose(Point(x=0.0, y = 0.40, z = 0.5)), fixed_base=True)
        # front_pillar = load_model('models/pillar.urdf', pose=Pose(Point(x=0.0, y = -0.3, z = 0.0)), fixed_base=True)

    body_names = {
        wall_behind: 'wall_behind',
        # front_pillar: 'front_pillar',
    }
    movable_bodies = []# socket]#, peg]
    stackable_bodies = [] #[floor]
    return robot_body,  body_names, movable_bodies, stackable_bodies

def load_scene(body_names, movable_bodies, stackable_bodies, obj_info_list = [], additional_list = [], table_center = None):
    if len(obj_info_list):
        socket_info, peg_info, colObs_info = obj_info_list

    if table_center is not None:
        low_table_center = np.array(table_center)-np.array([0, 0 , 0.03])
        obj = load_model('models/aloha_desk.urdf',pose=Pose(low_table_center), fixed_base=True, scale = 2)
        stackable_bodies.append(obj)
        body_names[obj] = "table"

    for obj_name in additional_list:
        if obj_name == 'floor':
            obj = load_model('models/aloha_desk.urdf',pose=Pose(Point(x=-0, y = -0, z = FLOOR_OFFSET)), fixed_base=True)
            stackable_bodies.append(obj)
        elif obj_name =='socket':
            socket_height = 0.1
            obj = Object(
                create_box(
                    w=0.04,
                    l=0.04,
                    h=socket_height,
                    color=RGBA(14 / 256.0, 14 / 256.0, 213 / 256.0, 1.0),  # blue
                    mass=0.1,
                    # client=client,
                ),
                category='socket',
                manual_compute_points=True,
                # client=client,
            )
            # set_pose(socket, Pose(point=Point(x=-0.0, y=-0.0, z=0.1), euler = Euler(0, 0, 0.25* math.pi)))
            set_pose(obj, Pose(point=Point(x=socket_info.x, y=socket_info.y, z=socket_height *0.5 -FLOOR_OFFSET), \
                                  euler = Euler(0, 0, socket_info.yaw)))
            movable_bodies.append(obj)
        elif obj_name =='peg':
            peg_height = 0.02
            obj = Object(
                create_box(
                    w=0.1,
                    l=0.02,
                    h=peg_height,
                    color=RGBA(244 / 256.0, 14 / 256.0, 13 / 256.0, 1.0), # red
                    mass=0.1,
                    # client=client,
                ),
                category='peg',
                manual_compute_points=True,
                # client=client,
            )
            # set_pose(peg, Pose(point=Point(x=-0.0, y=-0.0, z=0.1), euler = Euler(0, 0, 0.25* math.pi)))
            set_pose(obj, Pose(point=Point(x=peg_info.x, y=peg_info.y, z=peg_height *0.5 -FLOOR_OFFSET), \
                                  euler = Euler(0, 0, peg_info.yaw)))
            movable_bodies.append(obj)
        elif obj_name =='cup':
            cup_height = 0.2
            obj = Object(
                create_box(
                    w=0.05,
                    l=0.05,
                    h=cup_height,
                    color=RGBA(244 / 256.0, 194 / 256.0, 13 / 256.0, 1.0), # yellow
                    mass=0.1,
                    # client=client,
                ),
                category='cup',
                manual_compute_points=True,
                # client=client,
            )
            # set_pose(peg, Pose(point=Point(x=-0.0, y=-0.0, z=0.1), euler = Euler(0, 0, 0.25* math.pi)))
            set_pose(obj, Pose(point=Point(x=colObs_info.x, y=colObs_info.y, z=cup_height *0.5 -FLOOR_OFFSET), \
                                  euler = Euler(0, 0, colObs_info.yaw)))
            movable_bodies.append(obj)
        elif obj_name =='sink':
            obj = load_model(SINK_URDF, pose=Pose(Point(x=-0.25, y = -0.0, z = -0.02)), fixed_base=True) 
            stackable_bodies.append(obj)
        elif obj_name =='gearbox':
            obj = load_model(SINK_URDF, pose=Pose(Point(x=0, y = 0.00, z = 0.01)), fixed_base=True, scale=0.5) 
            # stackable_bodies.append(obj)
        elif obj_name == 'lowerright_pad':
            obj = load_model(SINK_URDF, pose=Pose(Point(x=0.3, y = -0.2, z = -0.02)), fixed_base=True, scale=0.6) 
            stackable_bodies.append(obj)
        elif obj_name == 'upperright_pad':
            obj = load_model(SINK_URDF, pose=Pose(Point(x=0.3, y = 0.2, z = -0.02)), fixed_base=True, scale=0.6) 
            stackable_bodies.append(obj)
        elif obj_name == 'left_pad':
            obj = load_model(SINK_URDF, pose=Pose(Point(x=-0.15, y = 0.12, z = -0.02)), fixed_base=True, scale=0.3)
            stackable_bodies.append(obj)
        elif obj_name == 'right_pad':
            obj = load_model(SINK_URDF, pose=Pose(Point(x=0.18, y = -0.1, z = -0.02)), fixed_base=True, scale=0.3)
            stackable_bodies.append(obj)
        elif obj_name == 'left_basket': # 32*20*15
            obj = load_model(BASKET_URDF, pose=Pose(Point(x=-0.25, y = 0.25, z = 0.00)), fixed_base=True, scale=1)
            stackable_bodies.append(obj)
        else:
            raise ValueError(f"Unknown object name: {obj_name}")

        body_names[obj] = obj_name


def load_aloha_world_flexible(additional_list = [], obj_info_list = [], **kwargs):
    robot_body,  body_names, movable_bodies, stackable_bodies = load_world_0obj()

    with HideOutput():
        load_scene(body_names, movable_bodies, stackable_bodies, obj_info_list = obj_info_list, additional_list = additional_list)
    
    robot_entity = ALOHARobot(robot_body=robot_body, **kwargs)
    return robot_entity,  body_names, movable_bodies, stackable_bodies

## for huawei franka
def load_dual_franka_world_flexible(additional_list = [], obj_info_list = [], **kwargs):
    set_default_camera(yaw=45)
    draw_global_system()
    body_names = {}
    movable_bodies = []
    stackable_bodies = [] 

    with HideOutput():
        robot_body = load_model(DUAL_FRANKA_URDF, fixed_base=True, pose=Pose(Point(x=0, y = 0, z = 0.0)))

        load_scene(body_names, movable_bodies, stackable_bodies, obj_info_list = obj_info_list, additional_list = additional_list)

    robot_entity = DUALfrankaRobot(robot_body=robot_body, **kwargs)

    return robot_entity,  body_names, movable_bodies, stackable_bodies




def load_panda_dual_world_flexible(additional_list = [], obj_info_list = [], table_center = None, **kwargs):
    set_default_camera(yaw=45, distance = 2)
    draw_global_system()
    body_names = {}
    movable_bodies = []
    stackable_bodies = [] 

    with HideOutput():
        robot_body = load_model(PANDA_DUAL_URDF, fixed_base=True, pose=Pose(Point(x=0, y = 0, z = 0.0)))

        load_scene(body_names, movable_bodies, stackable_bodies, obj_info_list = obj_info_list, additional_list = additional_list, table_center=table_center)

    robot_entity = PandaDualRobot(robot_body=robot_body, **kwargs)

    return robot_entity,  body_names, movable_bodies, stackable_bodies

def load_panda_world_flexible(additional_list = [], obj_info_list = [], franka_base_pos = [-0.66, 0, 0.912], table_center = None,   **kwargs):
    set_default_camera(yaw=45)
    draw_global_system()
    body_names = {}
    movable_bodies = []
    stackable_bodies = [] 

    with HideOutput():
        # robot_body = load_model(PANDA_ARM_URDF, fixed_base=True, pose=Pose(Point(x=-0.66, y = 0, z = 0.912)))
        robot_body = load_model(PANDA_ARM_URDF, fixed_base=True, pose=Pose(np.array(franka_base_pos)))

        load_scene(body_names, movable_bodies, stackable_bodies, obj_info_list = obj_info_list, additional_list = additional_list, table_center=table_center)

    robot_entity = PandaSingleRobot(robot_body=robot_body, **kwargs)

    return robot_entity,  body_names, movable_bodies, stackable_bodies

####################################################### for ddpm world
SIDE_ALOHA_URDF = 'models/vx300/gazebo_sidecam_aloha.urdf'
def load_world_sidecam():
    # for primitives_test.py
    set_default_camera(yaw=30)
    draw_global_system()
    with HideOutput():
        #add_data_path()
        robot_body = load_model(SIDE_ALOHA_URDF, fixed_base=True, pose=Pose(Point(x=0, y = 0, z = 0.0)))
        # floor = load_model('models/aloha_desk.urdf',pose=Pose(Point(x=-0, y = -0, z = FLOOR_OFFSET)), fixed_base=True)
        # socket = load_model(CUBOID_URDF,fixed_base=False, pose=Pose(Point(x=-0.1, y = 0.0, z=0.1 *0.5 -FLOOR_OFFSET)))
        # peg = load_model(CUBOID_DOWN_URDF,fixed_base=False)
        
        wall_behind = load_model('models/wall.urdf', pose=Pose(Point(x=0.0, y = 0.40, z = 0.5)), fixed_base=True)
        front_pillar = load_model('models/pillar.urdf', pose=Pose(Point(x=0.0, y = -0.3, z = 0.2)), fixed_base=True)
        


    body_names = {
        # socket: 'socket',
        # peg: 'peg',
        # floor: 'floor',
        wall_behind: 'wall_behind',
        front_pillar: 'front_pillar',
    }
    movable_bodies = []# socket]#, peg]
    stackable_bodies = [] #[floor]

    robot_entity = ALOHARobot(robot_body=robot_body)
    return robot_entity,  body_names, movable_bodies, stackable_bodies


def set_arm_pose(robot, arm_name, pose6d):
    conf = GroupConf(robot, arm_name, pose6d)
    conf.assign()

def set_arm_poses(robot,  dual_jpose):
    single_arm_dof = len(dual_jpose) // 2
    # set_arm_pose(robot, 'left_arm', dual_jpose[:6])
    # set_arm_pose(robot, 'right_arm', dual_jpose[single_arm_dof:single_arm_dof+6])

    set_arm_pose(robot, 'left_arm', dual_jpose[:single_arm_dof])
    set_arm_pose(robot, 'right_arm', dual_jpose[single_arm_dof:])
    

def save_rgb_image(camera, directory, file_name):
    # camera.draw(draw_cone=True)
    camera_image = camera.get_image(segment=False)

    # save_camera_images_at(camera_image, dir) 

    ensure_dir(directory)
    rgb_image, depth_image, seg_image = camera_image[:3]
    # depth_image = simulate_depth(depth_image)
    save_image(
        os.path.join(directory, file_name), rgb_image
    )  # [0, 255]
    

    
import time

def obj_points_to_body(obj_points, sc_network = None, category = 'other', concave = True, filter = False, add_bottom = False, previous_pose = None):

    obj_points = np.asarray(obj_points)
    # Filter on the full array so the kept rows carry their rgb along with xyz.
    if filter:
        obj_points = filter_pc(obj_points, use_DBSCAN = False)

    # Separate xyz from rgb so geometric ops never see the colour channels.
    obj_xyz = obj_points[:, :3]
    obj_colors = obj_points[:, 3:6] if obj_points.shape[1] >= 6 else None

    oobb = estimate_oobb(obj_xyz)
    center = np.array(oobb.pose[0])

    if previous_pose is not None:
        pose_difference = np.array(previous_pose[0]) - center
    else:
        pose_difference = np.array([0, 0, 0])


    centered_obj_points = obj_xyz - center + pose_difference
    origin_pose = Pose(Point(x=center[0], y=center[1], z=center[2]))

    ## assume flat surface, adding bottom points for safe motion planning
    if add_bottom:
        bottom_pc = add_projected_point(centered_obj_points, num_ratio=0.5)
        mesh_pc = np.concatenate([centered_obj_points, bottom_pc], axis=0)
    else:
        mesh_pc = centered_obj_points


    ## complete the shape if possible
    if sc_network is not None:
        refined_mesh = refine_shape(sc_network, mesh_pc, use_points=True, min_z = 0)
        refined_points = refined_mesh.vertices
        obj_mesh = concave_mesh(refined_points) if concave else mesh_from_points(refined_points)

    else:
        obj_mesh = concave_mesh(mesh_pc) if concave else mesh_from_points(mesh_pc)

    if concave:
        obj_id = create_concave_mesh(
            obj_mesh, under=True, color=(1, 0, 0), mass = 0.2
        )
    else:
        obj_id = create_mesh(
            obj_mesh, under=True, color=(1, 0, 0), mass = 0.2
        )

    _neutral = (0.7, 0.7, 0.7)
    if obj_colors is not None:
        labeled_points = [
            LabeledPoint(Point(x=p[0], y=p[1], z=p[2]), tuple(float(c) for c in col), category)
            for p, col in zip(centered_obj_points, obj_colors)
        ]
    else:
        labeled_points = [LabeledPoint(Point(x=p[0], y=p[1], z=p[2]), _neutral, category) for p in centered_obj_points]

    return obj_id, labeled_points, origin_pose


def load_mesh(obj_points, sc_network = None, category = 'other',  concave = True, filter = False, add_bottom = False):
    obj_id, labeled_points, origin_pose = obj_points_to_body(obj_points, sc_network, category, concave, filter, add_bottom)

    # Place the body at the final world pose BEFORE creating EstimatedObject.
    # EstimatedObject captures `initial_pose = get_pose(body)` at construction time.
    set_pose(obj_id, origin_pose)
    obj_estimate = EstimatedObject(
        obj_id,
        category=category,
        labeled_points=labeled_points,
        is_fragile=False,
        pc_normalized=True,
    )
    return obj_estimate, origin_pose

def update_mesh(existed_obj, obj_points, sc_network = None, category = 'other', concave = True, filter = False, add_bottom = False):

    ## TODO: handle disappear and appear
    if obj_points is None:
        existed_obj.remove()
        return False
    else:
        new_body, new_labeled_points, new_pose = obj_points_to_body(obj_points, sc_network, category, concave, filter, add_bottom, 
        previous_pose=None)  

        # Place the new body at the final pose BEFORE swapping it into existed_obj.
        set_pose(new_body, new_pose)
        existed_obj.only_update_pc(new_body, new_labeled_points)
        # Refresh initial_pose, which is captured at object construction time.
        existed_obj.initial_pose = existed_obj.observed_pose
        # existed_obj.body is now at new_pose already, so no extra set_pose(existed_obj, ...) needed.

        return True

def render_pose(denoising_history, use_gui, \
                directory = None, save_pic_every = 10,\
                    obj_points = None, has_eff = False, \
                    moving_pc_list = None, robot_name = 'aloha'):
    # denoising_history = denoising_history.detach().cpu().numpy()

    connect(use_gui = use_gui)

    if robot_name == 'aloha':
        robot, names, movable_bodies, stackable_bodies = load_world_sidecam()
    elif robot_name == 'panda_dual':
        robot, names, movable_bodies, stackable_bodies = load_panda_dual_world_flexible()
    elif robot_name == 'panda':
        robot, names, movable_bodies, stackable_bodies =load_panda_world_flexible()

    # load mesh from points
    obj_estimate = None
    if obj_points is not None:
        obj_estimate, _ = load_mesh(obj_points, concave=True)

    
    # clone the gripper
    side = robot.side_from_arm("left_arm")
    arm_group, gripper_group, tool_name = robot.manipulators[side]
    gripper = robot.get_component(gripper_group)
    ## the pose of ee in the gripper center frame
    parent_from_tool = robot.get_parent_from_tool(side)  

    closed_conf, open_conf = robot.close_open_conf()
    set_joint_positions(
        gripper, robot.get_component_joints(gripper_group), open_conf, 
    )

    # visualize grasp pose
    for m, feature in enumerate(denoising_history):

        # if the pc is moving, re-load the object
        if moving_pc_list is not None:
            if obj_estimate is not None:
                remove_body(obj_estimate)

            obj_estimate, _ = load_mesh(moving_pc_list[m])

        glb_grasp_pose, qpos_dual = feature
        if glb_grasp_pose is None:
            print("No grasp is predicted")
            break
        # set the gripper
        grasp_quat = quat_from_matrix(glb_grasp_pose[:3, :3])
        grasp_xyz = glb_grasp_pose[:3, 3]
        grasp_pose_7d = (grasp_xyz, grasp_quat)
        draw_pose(grasp_pose_7d, length=0.1)
        ## if ee frame is not the origin of gripper, we should transform to the gripper origin
        set_pose(
            gripper,
            multiply(grasp_pose_7d, invert(parent_from_tool)),
        )

        if  directory is not None and save_pic_every>0:
            if m % save_pic_every == 0:
                file_name = f"grasp_{m}.png"
                save_rgb_image(robot.cameras[0], directory, file_name)

        time.sleep(0.1)

    # visualize qpose
    for m, feature in enumerate(denoising_history):

        glb_grasp_pose, qpos_dual = feature
        if qpos_dual is None:
            print("No qpos is predicted")
            break
        # elif len(qpos_dual)==12:
        #     qpos_12d = qpos_dual
        # elif len(qpos_dual)==14:
        #     qpos_12d = np.concatenate((qpos_dual[:6], qpos_dual[7:13]))
        # else:
        #     raise ValueError("The qpos_dual should be 12 or 14 dim")

        # set_arm_poses(robot, qpos_12d)

        set_arm_poses(robot, qpos_dual)

        # TODO: load the object and gripper pose

        if  directory is not None and save_pic_every>0:
            if m % save_pic_every == 0:
                file_name = f"jpose_{m}.png"
                save_rgb_image(robot.cameras[0], directory, file_name)

        time.sleep(0.01)

    # NOTE: the eff grasp pose is relative to object seen at begining.
    # assume that objects at the end is only different in translation. 
    if has_eff:

        # visualize grasp pose in effect
        for m, feature in enumerate(denoising_history):

            glb_grasp_pose, qpos_12d = feature
            if glb_grasp_pose is None:
                print("No grasp is predicted")
                break
            # set the gripper
            grasp_quat = quat_from_matrix(glb_grasp_pose[4:7, :3])
            grasp_xyz = glb_grasp_pose[4:7, 3]
            grasp_pose_7d = (grasp_xyz, grasp_quat)
            set_pose(
                gripper,
                multiply(grasp_pose_7d, invert(parent_from_tool)),
            )

            if  directory is not None and save_pic_every>0:
                if m % save_pic_every == 0:
                    file_name = f"grasp_eff_{m}.png"
                    save_rgb_image(robot.cameras[0], directory, file_name)

            time.sleep(0.1)


    # save the final pose
    if directory is not None:
        file_name = f"pose_final.png"
        save_rgb_image(robot.cameras[0], directory, file_name)
    disconnect()

def get_cloned_gripper(robot, side = 'right'):
    side = robot.side_from_arm(side + "_arm")
    arm_group, gripper_group, tool_name = robot.manipulators[side]
    gripper = robot.get_component(gripper_group)
    parent_from_tool = robot.get_parent_from_tool(side)  

    closed_conf, open_conf = robot.close_open_conf()
    set_joint_positions(
        gripper, robot.get_component_joints(gripper_group), open_conf, 
    )
    return gripper, parent_from_tool



def render_history(denoising_history, use_gui, \
                directory = None, save_pic_every = 10,\
                    agent_obs = None, vis_eff = False, sleep_time = 0.1):
    # denoising_history = denoising_history.detach().cpu().numpy()

    connect(use_gui = use_gui)
    robot_body, names, movable_bodies, stackable_bodies = load_world_sidecam()
          
    robot = ALOHARobot(robot_body)
    gripper, parent_from_tool = get_cloned_gripper(robot, side='right')

    def vis_grasp(grasp_key):
        ## visualize grasp
        for m, action_slice in enumerate(denoising_history):

            glb_grasp_pose = action_slice.get(grasp_key)
            if glb_grasp_pose is None:
                print("No grasp is predicted")
                break
            # set the gripper
            if vis_eff:
                grasp_quat = quat_from_matrix(glb_grasp_pose[4:7, :3])
                grasp_xyz = glb_grasp_pose[4:7, 3]
            else:
                grasp_quat = quat_from_matrix(glb_grasp_pose[:3, :3])
                grasp_xyz = glb_grasp_pose[:3, 3]
            grasp_pose_7d = (grasp_xyz, grasp_quat)
            set_pose(
                gripper,
                multiply(grasp_pose_7d, invert(parent_from_tool)),
            )

            if  directory is not None and save_pic_every>0:
                if m % save_pic_every == 0:
                    file_name = f"grasp_{m}.png"
                    save_rgb_image(robot.cameras[0], directory, file_name)

            time.sleep(sleep_time)
        else:
            print('grasp vis iteration done')

    def vis_jpose(joint_key, side = ''):
        ## visualize qpose
            
        for m, action_slice in enumerate(denoising_history):
            qpos = action_slice.get(joint_key)
            if qpos is None:
                print("No qpos is predicted")
                break

            if side =='':
                set_arm_poses(robot, qpos)
            else:
                set_arm_pose(robot, side + '_arm', qpos[:6])

            # TODO: load the object and gripper pose

            if  directory is not None and save_pic_every>0:
                if m % save_pic_every == 0:
                    file_name = f"jpose_{m}.png"
                    save_rgb_image(robot.cameras[0], directory, file_name)

            time.sleep(sleep_time)
        
    # load mesh from points
    objs_estimate = {}
    pc_keys = [k for k in agent_obs.keys() if 'pc' in k]
    for pc_key in pc_keys:
        pc_prefix = pc_key.strip('pc')
        obj_points = agent_obs[pc_key].cpu().numpy().reshape(-1, 3)
        obj_estimate, _ = load_mesh(obj_points)

        grasp_key = pc_prefix + 'grasp'
        vis_grasp(grasp_key)
        joint_key = pc_prefix + 'jpose'
        vis_jpose(joint_key, side = pc_prefix.split('_')[0])

        ## remove the mesh
        remove_body(obj_estimate)
      
    # save the final pose
    if directory is not None:
        file_name = f"pose_final.png"
        save_rgb_image(robot.cameras[0], directory, file_name)
    disconnect()
