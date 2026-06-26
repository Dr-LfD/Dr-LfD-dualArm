
import sys, os
root_path = next(p for p in (os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *([os.pardir] * k))) for k in range(16)) if os.path.isfile(os.path.join(p, '.repo_root')))
sys.path.append(root_path) if root_path not in sys.path else None
from examples.pybullet.utils.pybullet_tools.utils import set_pose, Pose, \
    Point, set_default_camera, HideOutput, SINK_URDF, BASKET_URDF, \
     load_model, mesh_from_points, draw_global_system, ALOHA_URDF, DUAL_FRANKA_URDF, PANDA_DUAL_URDF, PANDA_ARM_URDF, \
       create_mesh, Euler, create_box, RGBA

from examples.pybullet.aloha_real.openworld_aloha.entities import Object, LabeledPoint
from examples.pybullet.aloha_real.openworld_aloha.robot_entities import ALOHARobot, DUALfrankaRobot, PandaDualRobot, PandaSingleRobot
from examples.pybullet.aloha_real.openworld_aloha.estimation.geometry import refine_shape
from examples.pybullet.aloha_real.openworld_aloha.estimation.belief import EstimatedObject
from examples.pybullet.aloha_real.openworld_aloha.estimation.bounding import estimate_oobb
from examples.pybullet.aloha_real.openworld_aloha.estimation.concave import concave_mesh, create_concave_mesh
from examples.pybullet.aloha_real.openworld_aloha.estimation.pc_utils import filter_pc, add_projected_point

import numpy as np

FLOOR_OFFSET = -0.01


def load_world_0obj():
    # for primitives_test.py
    set_default_camera(yaw=30)
    draw_global_system()
    with HideOutput():
        robot_body = load_model(ALOHA_URDF, fixed_base=True, pose=Pose(Point(x=0, y = 0, z = 0.0)))
        wall_behind = load_model('models/wall.urdf', pose=Pose(Point(x=0.0, y = 0.40, z = 0.5)), fixed_base=True)

    body_names = {
        wall_behind: 'wall_behind',
    }
    movable_bodies = []
    stackable_bodies = []
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
                ),
                category='socket',
                manual_compute_points=True,
            )
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
                ),
                category='peg',
                manual_compute_points=True,
            )
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
                ),
                category='cup',
                manual_compute_points=True,
            )
            set_pose(obj, Pose(point=Point(x=colObs_info.x, y=colObs_info.y, z=cup_height *0.5 -FLOOR_OFFSET), \
                                  euler = Euler(0, 0, colObs_info.yaw)))
            movable_bodies.append(obj)
        elif obj_name =='sink':
            obj = load_model(SINK_URDF, pose=Pose(Point(x=-0.25, y = -0.0, z = -0.02)), fixed_base=True)
            stackable_bodies.append(obj)
        elif obj_name =='gearbox':
            obj = load_model(SINK_URDF, pose=Pose(Point(x=0, y = 0.00, z = 0.01)), fixed_base=True, scale=0.5)
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
        robot_body = load_model(PANDA_ARM_URDF, fixed_base=True, pose=Pose(np.array(franka_base_pos)))

        load_scene(body_names, movable_bodies, stackable_bodies, obj_info_list = obj_info_list, additional_list = additional_list, table_center=table_center)

    robot_entity = PandaSingleRobot(robot_body=robot_body, **kwargs)

    return robot_entity,  body_names, movable_bodies, stackable_bodies


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
