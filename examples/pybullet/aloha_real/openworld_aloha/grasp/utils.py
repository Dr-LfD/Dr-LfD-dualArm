import time

from examples.pybullet.utils.pybullet_tools.utils import (
    INF,
    Point,
    Pose,
    elapsed_time,
    invert,
    multiply,
    point_from_pose,
    tform_points,
)

import numpy as np

#######################################################

GPD_GRIPPER_ADJUSTMENT = Pose(
    point=Point(x=-0.08)
)  # for l_gripper_palm_joint in pr2_l_gripper.urdf
GPD_TOOL_ADJUSTMENT = Pose(
    point=Point(x=0.035)
)  # (0.03376995027065277, -0.0005300119519233704, 0.0011900067329406738)


def local_gpd(points):
    from examples.pybullet.aloha_real.openworld_aloha.grasp.gpd_interface import generate_grasps

    entries = generate_grasps(points)
    grasps = [(grasp[:3], grasp[3:7]) for grasp in entries]
    scores = [grasp[7] for grasp in entries]
    return grasps, scores


def gpd_predict_grasps(points_world, camera_pose, use_tool=True):
    # Assumes camera_position = 0 0 0
    start_time = time.time()
    assert len(points_world) >= 1

    reference_pose = Pose(point_from_pose(camera_pose))

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
