import os
import time

from examples.pybullet.utils.pybullet_tools.utils import (
    INF,
    Point,
    Pose,
    elapsed_time,
    invert,
    multiply,
    point_from_pose,
    pose_from_tform,
    tform_points,
    quaternion_from_matrix,
)

# from examples.pybullet.aloha_real.openworld_aloha.graspnet import GRASPNET_POSE, filter_identical_grasps
# from examples.pybullet.aloha_real.openworld_aloha.simulation.lis import USING_ROS
import numpy as np



GRASP_MODES = ["graspnet", "gpd"]

#######################################################

GPD_GRIPPER_ADJUSTMENT = Pose(
    point=Point(x=-0.08)
)  # for l_gripper_palm_joint in pr2_l_gripper.urdf
GPD_TOOL_ADJUSTMENT = Pose(
    point=Point(x=0.035)
)  # (0.03376995027065277, -0.0005300119519233704, 0.0011900067329406738)


def local_gpd(points):
    from examples.pybullet.aloha_real.openworld_aloha.grasp.gpd_interface  import generate_grasps

    entries = generate_grasps(points)
    grasps = [(grasp[:3], grasp[3:7]) for grasp in entries]
    scores = [grasp[7] for grasp in entries]
    return grasps, scores

def gpd_predict_grasps(points_world, camera_pose, use_tool=True):
    # Assumes camera_position = 0 0 0
    start_time = time.time()
    assert len(points_world) >= 1

    # reference_pose = Pose()
    reference_pose = Pose(point_from_pose(camera_pose))
    # reference_pose = camera_pose

    points_reference = np.array(tform_points(invert(reference_pose), points_world))
    grasps, scores = local_gpd(points_reference)

    if len(grasps) == 0:
        return [], []
    
    grasps, scores = zip(
        *sorted(zip(grasps, scores), key=lambda pair: pair[-1], reverse=True)
    )

    print(
        "Grasps: {} | Min likelihood: {:.3f} | Max likelihood: {:.3f} | Time: {:.3f} sec".format(
            len(grasps),
            min(scores, default=-INF),
            max(scores, default=-INF),
            elapsed_time(start_time),
        )
    )

    adjustment = GPD_TOOL_ADJUSTMENT if use_tool else GPD_GRIPPER_ADJUSTMENT
    grasps = [
        multiply(reference_pose, grasp, adjustment) for grasp in grasps
    ]  # world_from_tool

    return grasps, scores


#######################################################

ADIAN_GRASPNET_ADJUSTMENT = ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5, 0.5))


def local_graspnet(points):
    from grasp.graspnet_interface import generate_grasps

    tforms, scores = generate_grasps(points, pc_colors=None)
    grasps = list(map(pose_from_tform, tforms))
    return grasps, scores


def graspnet_predict_grasps(points_world, camera_pose):
    start_time = time.time()
    assert len(points_world) >= 1

    # reference_pose = Pose()
    # reference_pose = Pose(point_from_pose(camera_pose))
    reference_pose = camera_pose

    points_reference = tform_points(invert(reference_pose), points_world)
    if USING_ROS:
        grasps, scores = query_grasp_server(points_reference, grasp_mode="graspnet")
    else:
        grasps, scores = local_graspnet(points_reference)
    grasps, scores = zip(
        *sorted(zip(grasps, scores), key=lambda pair: pair[-1], reverse=True)
    )
    grasps, scores = zip(*filter_identical_grasps(zip(grasps, scores)))

    print(
        "Grasps: {} | Min likelihood: {:.3f} | Max likelihood: {:.3f} | Time: {:.3f} sec".format(
            len(grasps),
            min(scores, default=-INF),
            max(scores, default=-INF),
            elapsed_time(start_time),
        )
    )

    adjustment = multiply(invert(GRASPNET_POSE))  # Pose(point=Point(x=-0.08)))
    # adjustment = ADIAN_GRASPNET_ADJUSTMENT
    grasps = [
        multiply(reference_pose, grasp, adjustment) for grasp in grasps
    ]  # world_from_tool

    return grasps, scores


def riemann_predict_grasps(obj, camera_pose):
    # Assumes camera_position = 0 0 0
    start_time = time.time()
    if obj.pc_normalized:
        points_world = obj.get_world_labeled_points()
        pc_center = obj.observed_pose[0]
    else:
        points_world = obj.labeled_points
        pc_center = obj.observed_pose[0]
    assert len(points_world) >= 1

    grasps, scores = world_riemann(points_world, pc_center)
    return grasps, scores

def world_riemann(points, pc_center, try_number = 3):
    print("Riemann grasp prediction with try number = ", try_number)
    # save points and pc_center as npz
    save_path = os.path.expandvars("${WS_ROOT}/RiEMann_seg/experiments/real_world/input.npz")
    load_path = os.path.expandvars("${WS_ROOT}/RiEMann_seg/experiments/real_world/output.npz")
    xyz = np.array([p.point for p in points])
    rgb = np.array([p.color for p in points])
    np.savez(save_path, xyz = xyz, rgb = rgb,  pc_center = pc_center, try_number = try_number)
    # call riemann grasp prediction
    import subprocess
    conda_env = os.path.expandvars("${HOME}/miniforge3/envs/equi/bin/python")
    python_path = os.path.expandvars("${WS_ROOT}/RiEMann_seg/scripts/testing/riemann_interface.py")

    # subprocess.run([conda_env, python_path, save_path, load_path],  shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    # Construct the command
    python_cmd = [conda_env, python_path, "--input_npz_path", save_path, "--output_npz_path", load_path]

    # Run the command
    process = subprocess.Popen(python_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait for the process to complete and capture output
    stdout, stderr = process.communicate()

    # load grasp prediction
    data = np.load(load_path)
    grasps = []
    scores = []
    for i in range(try_number):
        pred_t = data['pred_pos'][i]
        pred_r = data['pred_rot'][i]
        transform_mat = np.eye(4)
        transform_mat[:3, :3] = pred_r
        transform_mat[:3, 3] = pred_t
        pred_quat = quaternion_from_matrix(transform_mat)
        pred_pos = (pred_t, pred_quat)
        grasps.append(pred_pos)
        scores.append(1.0)
    return grasps, scores

