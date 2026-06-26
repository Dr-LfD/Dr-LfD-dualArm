from __future__ import print_function
import math

from examples.pybullet.utils.pybullet_tools.ikfast.ikfast import (
    closest_inverse_kinematics,
    get_ik_joints,
    get_ik_fn,
)
from examples.pybullet.utils.pybullet_tools.utils import (
    INF,
    get_collision_fn,
    interpolate_joint_waypoints,
    interpolate_poses,
    invert,
    link_from_name,
    multiply,
    pairwise_collisions,
    safe_zip,
    set_joint_positions,
    set_pose,
)

# TODO: could update MAX_DISTANCE globally
COLLISION_DISTANCE = 5e-3  # Distance from fixed obstacles
# MOVABLE_DISTANCE = 1e-2 # Distance from movable objects
MOVABLE_DISTANCE = COLLISION_DISTANCE
EPSILON = 1e-3
SELF_COLLISIONS = True  # TODO: check self collisions
# URDF_USE_SELF_COLLISION: by default, Bullet disables self-collision. This flag let's you enable it.

MAX_IK_DISTANCE = INF
DISABLE_ALL_COLLISIONS = True


def compute_gripper_path(pose, grasp, pos_step_size=0.02):
    # TODO: move linearly out of contact and then interpolate (ensure no collisions with the table)
    # grasp -> pregrasp
    grasp_pose = multiply(pose.get_pose(), invert(grasp.grasp))
    pregrasp_pose = multiply(pose.get_pose(), invert(grasp.pregrasp))
    gripper_path = list(
        interpolate_poses(grasp_pose, pregrasp_pose, pos_step_size=pos_step_size)
    )
    # handles = list(flatten(draw_pose(waypoint_pose, length=0.02) for waypoint_pose in gripper_path))
    return gripper_path


def create_grasp_attachment(robot, side, grasp, **kwargs):
    # TODO: robot.get_tool_link(side)
    arm_group, gripper_group, tool_name = robot.manipulators[side]
    return grasp.create_attachment(
        robot, link=link_from_name(robot, tool_name, **kwargs)
    )


def plan_workspace_motion(
    robot, side, tool_waypoints, attachment=None, obstacles=[], max_attempts=2,
    return_ids=False, waypoint_resolution_deg=1.2, **kwargs
):  # , randomize=True, teleport=False):

    assert tool_waypoints
    waypoint_resolution_deg = float(waypoint_resolution_deg)
    if waypoint_resolution_deg <= 0:
        raise ValueError(f"waypoint_resolution_deg must be positive, got {waypoint_resolution_deg}")
    # TODO: omit collisions between the attachment and surface
    # TODO: check attachment collisions after a certain tool distance

    # world.carry_conf.assign()
    ik_info = robot.ik_info[side]
    arm_group, _, tool_name = robot.manipulators[side]
    tool_link = link_from_name(robot, tool_name, **kwargs)
    ik_joints = get_ik_joints(robot, ik_info, tool_link, **kwargs)  # Arm + torso
    fixed_joints = set(ik_joints) - set(robot.get_group_joints(arm_group))  # Torso only
    arm_joints = [j for j in ik_joints if j not in fixed_joints]  # Arm only
    extract_arm_conf = lambda q: [
        p for j, p in safe_zip(ik_joints, q) if j not in fixed_joints
    ]
    # tool_path = interpolate_poses(tool_waypoints[0], tool_waypoints[-1])

    ik_fn = get_ik_fn(ik_info, method=robot.ik_method, fixed_joints=fixed_joints)

    parts = [robot] + ([] if attachment is None else [attachment.child])
    attachments = [attachment] if attachment is not None else []
    collision_fn = get_collision_fn(
        robot,
        arm_joints,
        obstacles=obstacles,
        attachments=attachments,
        self_collisions=SELF_COLLISIONS,
        disabled_collisions=robot.disabled_collisions,
        disable_collisions=DISABLE_ALL_COLLISIONS,
        custom_limits=robot.custom_limits,
        **kwargs
    )

    # aq = next(get_arm_ik_generator(arm, pos, quat, torso, upper_limits=None, max_attempts=10))
    # aq = solve_inverse_kinematics(robot, side, grasp_pose, obstacles=[], collision_buffer=0.)
    got_ik = False
    for attempts in range(max_attempts):
        ## initially, set robot to default conf
        robot.reset()
        for arm_conf in closest_inverse_kinematics(
            ik_fn,
            robot,
            ik_info,
            tool_link,
            tool_waypoints[0],
            max_candidates=INF,
            max_attempts=100,
            max_time=INF,
            max_distance=MAX_IK_DISTANCE,
            verbose=False,
            **kwargs
        ):
            # TODO: can also explore multiple ways to proceed
            got_ik = True

            arm_conf = extract_arm_conf(arm_conf)

            if collision_fn(arm_conf, verbose=False):
                continue
            arm_waypoints = [arm_conf]
            set_joint_positions(robot, arm_joints, arm_conf, **kwargs)
            for i_wp, tool_pose in enumerate(tool_waypoints[1:]):
                # TODO: joint weights
                arm_conf = next(
                    closest_inverse_kinematics(
                        ik_fn,
                        robot,
                        ik_info,
                        tool_link,
                        tool_pose,
                        max_candidates=INF,
                        max_attempts=20,
                        max_time=INF,
                        max_distance=MAX_IK_DISTANCE,
                        verbose=False,
                        **kwargs
                    ),
                    None,
                )
                if arm_conf is None:
                    break

                arm_conf = extract_arm_conf(arm_conf)
                if collision_fn(arm_conf, verbose=False):
                    break ## as tool_waypoints only has 2 waypoints, each waypoint should be collision free
                arm_waypoints.append(arm_conf)
                set_joint_positions(robot, arm_joints, arm_conf, **kwargs)
            else:
                set_joint_positions(robot, arm_joints, arm_waypoints[-1], **kwargs)
                if attachment is not None:
                    attachment.assign()
                if any(
                    pairwise_collisions(
                        part,
                        obstacles,
                        max_distance=(COLLISION_DISTANCE + EPSILON),
                        **kwargs
                    )
                    for part in parts
                ) and not DISABLE_ALL_COLLISIONS:
                    continue
                arm_path_and_out_idx = interpolate_joint_waypoints(
                    robot,
                    arm_joints,
                    arm_waypoints,
                    resolutions=math.radians(waypoint_resolution_deg),
                    return_ids = return_ids,
                    **kwargs
                ) 
                if return_ids:
                    arm_path,  out_idx = arm_path_and_out_idx
                else:
                    arm_path = arm_path_and_out_idx

                if any(collision_fn(q) for q in arm_path):
                    continue

                print(
                    "Found path with {} waypoints and {} configurations after {} attempts".format(
                        len(arm_waypoints), len(arm_path), attempts + 1
                    )
                )
                if return_ids:
                    return arm_path, out_idx
                return arm_path
            
    print("IK reults: {}".format(got_ik))
    return None


#######################################################


def workspace_collision(
    robot,
    arm,
    gripper_path,
    grasp=None,
    open_gripper=True,
    obstacles=[],
    max_distance=0.0,
    **kwargs
):
    if(DISABLE_ALL_COLLISIONS):
        return False
    side = robot.side_from_arm(arm)
    _, gripper_group, tool_name = robot.manipulators[side]
    gripper = robot.get_component(gripper_group)

    if open_gripper:
        # TODO: make a separate method?
        robot.set_open_gripper(arm)

    parent_from_tool = robot.get_parent_from_tool(side)
    parts = [gripper]  # , obj]
    if grasp is not None:
        parts.append(grasp.body)
    for i, gripper_pose in enumerate(
        gripper_path
    ):  # TODO: be careful about the initial pose
        set_pose(gripper, multiply(gripper_pose, invert(parent_from_tool)), **kwargs)
        if grasp is not None:
            set_pose(grasp.body, multiply(gripper_pose, grasp.value), **kwargs)
        # attachment.assign()
        distance = (
            (COLLISION_DISTANCE + EPSILON)
            if (i == len(gripper_path) - 1)
            else max_distance
        )
        if any(
            pairwise_collisions(part, obstacles, max_distance=distance, **kwargs)
            for part in parts
        ):
            # TODO: some gripper object collisions are still slipping through
            return True
    return False


def plan_prehensile(
    robot, arm, obj, pose, grasp, environment=[], is_placing = False, **kwargs
):  # , teleport=False):
    obstacles = list(environment)# + [obj]
    # obstacles = [obst for obst in obstacles if obst not in pose.ancestors()]
    pose.assign()
    gripper_path = compute_gripper_path(pose, grasp)  # grasp -> pregrasp
    gripper_waypoints = gripper_path[:1] + gripper_path[-1:]
    if workspace_collision(
        robot, arm, gripper_path, grasp=None, obstacles=obstacles, **kwargs
    ):
        return None
    side = robot.side_from_arm(arm)

    attachment = None
    if is_placing:
        attachment = create_grasp_attachment(robot, side, grasp, **kwargs)
        attachment.assign()

    arm_path = plan_workspace_motion(
        robot, side, gripper_waypoints, attachment=attachment, obstacles=obstacles, **kwargs
    )
    return arm_path
