"""Minimum Distance Field (MDF) construction for visuomotor-policy safety certification.

Builds a per-skill 3D distance field from replayed demonstrations.
"""

import numpy as np
import scipy.ndimage
import os
import sys
import pickle
import h5py
import argparse

EXE_FOLDER = next(p for p in (os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *([os.pardir] * k))) for k in range(16)) if os.path.isfile(os.path.join(p, '.repo_root')))
sys.path.append(EXE_FOLDER) if EXE_FOLDER not in sys.path else None

import pybullet as p
from collections import defaultdict

from pddlstream.utils import get_file_path
from examples.pybullet.utils.pybullet_tools.aloha_primitives import BodyConf
from examples.pybullet.utils.pybullet_tools.utils import (
    Pose, Point, set_default_camera, load_model, HideOutput,
    draw_global_system, VX300_URDF, Euler, connect, DEFAULT_CLIENT,
    stable_z, get_all_links, get_aabb, AABB,
    joints_from_names, set_joint_positions, link_from_name, get_link_pose
)
from examples.pybullet.aloha_real.openworld_aloha.simple_worlds import load_aloha_world_flexible
# customized_collision_fn (and its module visuomotor_replayer) is only needed by the offline
# MDF construction path; it is imported lazily inside get_occupied_voxels_at_timestep so the
# runtime query API (load_mdf_dict / query_mdf_safe) imports without that build-time dependency.


def create_voxel_grid(workspace_bounds, voxel_size):
    """
    Create a 3D voxel grid over the workspace.
    
    Args:
        workspace_bounds: Dict with 'min' and 'max' keys, each containing [x, y, z]
        voxel_size: Size of each voxel (uniform in all dimensions)
    
    Returns:
        grid_shape: Tuple (nx, ny, nz) - number of voxels in each dimension
        grid_origin: Array [x, y, z] - origin of the grid in world coordinates
        voxel_coords: Array of shape (nx*ny*nz, 3) - world coordinates of each voxel center
    """
    min_bounds = np.array(workspace_bounds['min'])
    max_bounds = np.array(workspace_bounds['max'])
    
    # Calculate grid dimensions
    grid_size = max_bounds - min_bounds
    grid_shape_tuple = tuple(np.ceil(grid_size / voxel_size).astype(int))
    grid_shape = np.array(grid_shape_tuple)  # Convert to array for scalar multiplication
    
    # Adjust bounds to be integer multiples of voxel_size
    adjusted_max = min_bounds + grid_shape * voxel_size
    grid_origin = min_bounds
    
    # Create meshgrid of voxel centers
    ranges = [np.linspace(min_bounds[i] + voxel_size/2, adjusted_max[i] - voxel_size/2, grid_shape_tuple[i]) 
              for i in range(3)]
    xx, yy, zz = np.meshgrid(ranges[0], ranges[1], ranges[2], indexing='ij')
    
    voxel_coords = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
    
    return grid_shape_tuple, grid_origin, voxel_coords


def world_to_voxel_coords(point_world, grid_origin, voxel_size):
    """Convert world coordinates to voxel indices."""
    point_world = np.array(point_world)
    voxel_idx = np.floor((point_world - grid_origin) / voxel_size).astype(int)
    return voxel_idx


def get_occupied_voxels_at_timestep(robot_body, grid_origin, voxel_size, grid_shape):
    """
    Get the set of occupied voxels at a given timestep (only arms, no objects).
    
    Args:
        robot_body: Robot body (ALOHA has both arms in one body)
        grid_origin: Origin of voxel grid in world coordinates
        voxel_size: Size of each voxel
        grid_shape: Shape of the grid (nx, ny, nz)
    
    Returns:
        occupied_mask: Boolean array of shape grid_shape indicating occupied voxels
    """
    occupied_mask = np.zeros(grid_shape, dtype=bool)

    # Build voxel centers grid
    ranges = [
        np.linspace(
            grid_origin[i] + voxel_size/2,
            grid_origin[i] + grid_shape[i] * voxel_size - voxel_size/2,
            grid_shape[i]
        ) for i in range(3)
    ]
    xx, yy, zz = np.meshgrid(ranges[0], ranges[1], ranges[2], indexing='ij')

    # Create a small sphere to probe distance at voxel centers
    sphere_radius = voxel_size / 2.0
    col_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=sphere_radius)
    sphere_body = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=col_shape)

    # build-time only dependency (see module-top note); imported lazily here
    from examples.pybullet.aloha_real.learned_classifier.visuomotor_replayer import customized_collision_fn

    try:
        # Iterate through all voxel centers; mark occupied if within sphere radius of robot
        for ix in range(grid_shape[0]):
            for iy in range(grid_shape[1]):
                for iz in range(grid_shape[2]):
                    center = [xx[ix, iy, iz], yy[ix, iy, iz], zz[ix, iy, iz]]
                    p.resetBasePositionAndOrientation(sphere_body, center, [0, 0, 0, 1])
                    dist = customized_collision_fn(robot_body, sphere_body, max_distance=voxel_size * 1.5, draw_debugline=False)
                    if dist <= sphere_radius:
                        occupied_mask[ix, iy, iz] = True
    finally:
        # Clean up the temporary probe body
        try:
            p.removeBody(sphere_body)
        except Exception:
            pass

    return occupied_mask


EE_TOOL_LINKS = ['puppet_left/ee_gripper_link', 'puppet_right/ee_gripper_link']


def voxel_centers_grid(grid_origin, voxel_size, grid_shape):
    """World-coordinate centers of every voxel as an (nx, ny, nz, 3) array."""
    ranges = [
        np.linspace(grid_origin[i] + voxel_size/2,
                    grid_origin[i] + grid_shape[i] * voxel_size - voxel_size/2,
                    grid_shape[i]) for i in range(3)
    ]
    xx, yy, zz = np.meshgrid(ranges[0], ranges[1], ranges[2], indexing='ij')
    return np.stack([xx, yy, zz], axis=-1)


def get_occupied_voxels_ee(tool_positions, grid_shape, ee_radius, centers):
    """
    Occupancy from the end-effector (gripper) + held-object swept tube only: mark voxels
    within ee_radius of either tool-link position. Unlike the whole-arm collision probe,
    this yields a COMPACT danger zone at the insertion site (the arms pivot from the table
    edge and otherwise blanket the whole reachable tabletop, leaving no on-table spot to
    set a blocking obstacle aside). Vectorized -- no per-voxel physics query. `centers` is
    the fixed (nx,ny,nz,3) voxel-center grid (voxel_centers_grid), built once by the caller.
    """
    occupied_mask = np.zeros(grid_shape, dtype=bool)
    for tp in tool_positions:
        dist = np.linalg.norm(centers - np.asarray(tp, dtype=float), axis=-1)
        occupied_mask |= (dist <= ee_radius)
    return occupied_mask


def compute_edt_map(occupied_mask, voxel_size):
    """
    Compute Euclidean Distance Transform map for occupied voxels.
    
    Args:
        occupied_mask: Boolean array indicating occupied voxels (True = occupied, False = free)
        voxel_size: Size of each voxel
    
    Returns:
        edt_map: Array of same shape as occupied_mask, containing distance to nearest occupied voxel
                 Distance is in world units (meters)
    """
    # scipy.ndimage.distance_transform_edt computes distance from each True pixel 
    # to the nearest False pixel. We want distance from each voxel to the nearest occupied voxel.
    
    # Invert mask: True for free space, False for occupied
    free_mask = ~occupied_mask
    
    # Compute distance transform: distance from each True (free) voxel to nearest False (occupied) voxel
    # The result is in voxel units, so we multiply by voxel_size to get meters
    # Note: distance_transform_edt returns float64, and distances are in voxel coordinates
    edt_map_voxels = scipy.ndimage.distance_transform_edt(free_mask)
    edt_map = edt_map_voxels.astype(np.float64) * voxel_size
    
    # For occupied voxels themselves, distance should be 0
    edt_map[occupied_mask] = 0.0
    
    return edt_map


def load_demonstrations_hdf5(hdf5_paths):
    """
    Load demonstrations from HDF5 files (referring to postprocess_hdf5.py).
    
    Args:
        hdf5_paths: List of paths to HDF5 files or a single path
    
    Returns:
        qpos_demos_left: List of arrays, each containing left arm joint positions for one demo
        qpos_demos_right: List of arrays, each containing right arm joint positions for one demo
    """
    if isinstance(hdf5_paths, str):
        hdf5_paths = [hdf5_paths]
    
    qpos_demos_left = []
    qpos_demos_right = []
    
    for hdf5_path in hdf5_paths:
        if not os.path.exists(hdf5_path):
            print(f"Warning: File not found: {hdf5_path}")
            continue
            
        with h5py.File(hdf5_path, "r") as f:
            # Read qpos from observations (same format as postprocess_hdf5.py)
            obs_grp = f['observations']
            obs_qpos = obs_grp['qpos'][()]  # Shape: (T, 14) for ALOHA
            
            # Split into left and right arm joints
            # ALOHA: left arm joints 0:7 (6 arm + 1 gripper), right arm joints 7:14
            l_joint_vals = obs_qpos[:, :7]  # Left arm joints
            r_joint_vals = obs_qpos[:, 7:14]  # Right arm joints
            
            qpos_demos_left.append(l_joint_vals)
            qpos_demos_right.append(r_joint_vals)
    
    return qpos_demos_left, qpos_demos_right


def construct_mdf(hdf5_paths, workspace_bounds, voxel_size=0.01,
                  task='clean_cup', pybullet_use_gui=False, start_frame_idx=0,
                  num_demos=None, frame_stride=1, ee_only=False, ee_radius=0.08,
                  qpos_demos=None, ee_dist_threshold=None, z_min=None):
    """
    Construct Minimum Distance Field from demonstrations (only arms, no objects).
    
    Args:
        hdf5_paths: Path or list of paths to HDF5 demonstration files
        workspace_bounds: Dict with 'min' and 'max' keys for workspace bounds
        voxel_size: Size of each voxel in meters (default 1cm)
        task: Task name (default 'clean_cup')
        pybullet_use_gui: Whether to show PyBullet GUI
        start_frame_idx: Index of the first frame to process (default 0)
    
    Returns:
        mdf: Array of shape grid_shape containing minimum distance values
        grid_shape: Shape of the grid
        grid_origin: Origin of the grid in world coordinates
        voxel_size: Size of each voxel
    """
    # Load world (using load_aloha_world_flexible for clean_cup task)
    connect(use_gui=pybullet_use_gui)
    robot_entity, body_names, movable_bodies, stackable_bodies = load_aloha_world_flexible()
    
    # Get robot body (ALOHA has both arms in one URDF body)
    robot_body = robot_entity.body
    # Tool links needed for EE-tube occupancy and/or the insertion (EE-proximity) frame filter.
    need_tool_links = ee_only or (ee_dist_threshold is not None)
    ee_tool_links = [link_from_name(robot_body, n) for n in EE_TOOL_LINKS] if need_tool_links else None
    n_skipped_frames = 0
    
    # Load demonstrations (pre-loaded txt demos take precedence over HDF5).
    if qpos_demos is not None:
        qpos_demos_left, qpos_demos_right = qpos_demos
    else:
        qpos_demos_left, qpos_demos_right = load_demonstrations_hdf5(hdf5_paths)

    if len(qpos_demos_left) == 0:
        raise ValueError("No valid demonstrations found in provided HDF5 files")
    
    # Create voxel grid
    grid_shape, grid_origin, voxel_coords = create_voxel_grid(workspace_bounds, voxel_size)
    print(f"Created voxel grid with shape {grid_shape}, origin {grid_origin}, voxel size {voxel_size}")
    # The voxel-center grid is fixed for the whole run; build it once for the EE-tube path.
    ee_centers = voxel_centers_grid(grid_origin, voxel_size, grid_shape) if ee_only else None
    # Drop table/base URDF links below z_min from occupancy: the ALOHA URDF includes the
    # table, whose z~0 links otherwise paint a full-width swept floor band that makes every
    # on-table xy look occupied. Precompute the below-floor z-layers once.
    z_below_floor = None
    if z_min is not None:
        zs_grid = grid_origin[2] + (np.arange(grid_shape[2]) + 0.5) * voxel_size
        z_below_floor = zs_grid < z_min
    
    # Initialize MDF with large values
    mdf = np.ones(grid_shape) * np.inf
    
    # Track occupied voxels (union across all timesteps)
    occupied_voxels_mask = np.zeros(grid_shape, dtype=bool)
    
    # Determine how many demonstrations to process
    total_demos = len(qpos_demos_left)
    demos_to_process = total_demos if num_demos is None else min(num_demos, total_demos)

    # Process each demonstration
    for demo_id in range(demos_to_process):
        print(f'Processing demo {demo_id+1}/{demos_to_process}')
        qpos_demo_left = qpos_demos_left[demo_id]
        qpos_demo_right = qpos_demos_right[demo_id]
        demo_length = len(qpos_demo_left)
        
        # Ensure start_frame_idx is within bounds
        actual_start_idx = min(start_frame_idx, demo_length - 1) if demo_length > 0 else 0
        
        # Process each timestep starting from start_frame_idx (subsampled by frame_stride)
        for t in range(actual_start_idx, demo_length, frame_stride):
            # Set robot configurations for both arms directly using robot_entity
            # ALOHA qpos format: [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
            left_arm_qpos = qpos_demo_left[t][:6]  # 6 arm joints
            left_gripper_qpos = qpos_demo_left[t][6:7] if len(qpos_demo_left[t]) > 6 else []
            right_arm_qpos = qpos_demo_right[t][:6]  # 6 arm joints
            right_gripper_qpos = qpos_demo_right[t][6:7] if len(qpos_demo_right[t]) > 6 else []
            
            # Set joint positions directly using robot_entity.joint_groups
            # Left arm
            left_arm_joints = joints_from_names(robot_body, robot_entity.joint_groups['left_arm'])
            set_joint_positions(robot_body, left_arm_joints, left_arm_qpos)
            
            ## TODO: joint to position
            # # Left gripper (if present)
            # if len(left_gripper_qpos) > 0:
            #     left_gripper_joints = joints_from_names(robot_body, robot_entity.joint_groups['left_gripper'])
            #     # Convert gripper position to joint positions (two finger joints move in opposite directions)
            #     left_gripper_positions = [left_gripper_qpos[0], -left_gripper_qpos[0]]
            #     set_joint_positions(robot_body, left_gripper_joints, left_gripper_positions)
            
            # Right arm
            right_arm_joints = joints_from_names(robot_body, robot_entity.joint_groups['right_arm'])
            set_joint_positions(robot_body, right_arm_joints, right_arm_qpos)
            
            # # Right gripper (if present)
            # if len(right_gripper_qpos) > 0:
            #     right_gripper_joints = joints_from_names(robot_body, robot_entity.joint_groups['right_gripper'])
            #     right_gripper_positions = [right_gripper_qpos[0], -right_gripper_qpos[0]]
            #     set_joint_positions(robot_body, right_gripper_joints, right_gripper_positions)
            
            # Restrict the swept volume to the ACTUAL insertion (post_mj_jointgrasp):
            # pred_joint_vals is the transport phase where the two EEs are far apart; insertion
            # begins when their distance drops below the threshold. Skip non-insertion frames so
            # the danger zone is tight at the insertion point (not the convergence corridor).
            if ee_tool_links is not None:
                tool_positions = [get_link_pose(robot_body, tl)[0] for tl in ee_tool_links]
            if ee_dist_threshold is not None:
                ee_gap = float(np.linalg.norm(np.array(tool_positions[0]) - np.array(tool_positions[1])))
                if ee_gap >= ee_dist_threshold:
                    n_skipped_frames += 1
                    continue

            # Get occupied voxels at this timestep
            if ee_only:
                # Compact end-effector + held-object swept tube (gripper tool links only).
                occupied_mask = get_occupied_voxels_ee(
                    tool_positions, grid_shape, ee_radius, ee_centers
                )
            else:
                # Whole-arm swept volume (table/base links removed by the z_min filter below).
                occupied_mask = get_occupied_voxels_at_timestep(
                    robot_body, grid_origin, voxel_size, grid_shape
                )

            # Remove the table/base URDF floor band so on-table placements stay free.
            if z_below_floor is not None:
                occupied_mask[:, :, z_below_floor] = False

            # Compute EDT map for this timestep
            edt_map = compute_edt_map(occupied_mask, voxel_size)
            
            # Debug: print EDT statistics for first timestep of first demo and periodically
            if (demo_id == 0 and t == actual_start_idx) or (t % 100 == 0 and demo_id == 0):
                num_occupied = np.sum(occupied_mask)
                total_voxels = occupied_mask.size
                edt_max = np.max(edt_map)
                edt_mean = np.mean(edt_map)
                print(f"  Timestep {t}: Occupied {num_occupied}/{total_voxels} ({100*num_occupied/total_voxels:.1f}%), "
                      f"EDT range: [{np.min(edt_map):.4f}, {edt_max:.4f}]m, mean: {edt_mean:.4f}m")
            
            # Update MDF: take minimum across timesteps
            # This means we're tracking the CLOSEST the robot ever got to each voxel
            mdf = np.minimum(mdf, edt_map)
            
            # Update occupied voxels mask: union across all timesteps
            occupied_voxels_mask = np.logical_or(occupied_voxels_mask, occupied_mask)
            
            # Track how MDF changes over time (for debugging)
            if demo_id == 0 and t % 100 == 0:
                mdf_max = np.max(mdf[~np.isinf(mdf)]) if np.any(~np.isinf(mdf)) else np.nan
                print(f"    MDF after timestep {t}: max = {mdf_max:.4f}m")
        
        processed_count = demo_length - actual_start_idx
        print(f'  Processed {processed_count} timesteps (from frame {actual_start_idx} to {demo_length-1})')
    
    if ee_dist_threshold is not None:
        print(f"Skipped {n_skipped_frames} non-insertion frames (EE gap >= {ee_dist_threshold}m)")

    # Replace inf with a large value for unvisited regions
    mdf[np.isinf(mdf)] = np.max(mdf[~np.isinf(mdf)]) if np.any(~np.isinf(mdf)) else 1.0
    
    # Debug: Print final MDF statistics
    print(f"\nFinal MDF statistics:")
    print(f"  MDF range: [{np.min(mdf):.4f}, {np.max(mdf):.4f}] meters")
    print(f"  MDF mean: {np.mean(mdf):.4f} meters")
    print(f"  MDF median: {np.median(mdf):.4f} meters")
    print(f"  Voxels with distance > 0.1m: {np.sum(mdf > 0.1)} ({100*np.sum(mdf > 0.1)/mdf.size:.1f}%)")
    print(f"  Voxels with distance > 0.2m: {np.sum(mdf > 0.2)} ({100*np.sum(mdf > 0.2)/mdf.size:.1f}%)")
    
    # Note: MDF represents the MINIMUM distance the robot ever got to each voxel
    # If the range is small, it means the robot moved through most of the workspace
    
    # Print occupied voxels statistics
    num_occupied = np.sum(occupied_voxels_mask)
    total_voxels = occupied_voxels_mask.size
    print(f"\nOccupied voxels statistics:")
    print(f"  Total occupied voxels: {num_occupied}/{total_voxels} ({100*num_occupied/total_voxels:.1f}%)")
    
    return mdf, grid_shape, grid_origin, voxel_size, occupied_voxels_mask


def save_mdf(mdf, grid_shape, grid_origin, voxel_size, save_path, occupied_voxels_mask=None):
    """Save MDF to file."""
    mdf_data = {
        'mdf': mdf,
        'grid_shape': grid_shape,
        'grid_origin': grid_origin,
        'voxel_size': voxel_size
    }
    
    if occupied_voxels_mask is not None:
        mdf_data['occupied_voxels_mask'] = occupied_voxels_mask
    
    with open(save_path, 'wb') as f:
        pickle.dump(mdf_data, f)
    
    print(f'Saved MDF to: {save_path}')
    print(f'  Grid shape: {grid_shape}')
    print(f'  Grid origin: {grid_origin}')
    print(f'  Voxel size: {voxel_size}')
    print(f'  Distance range: [{np.min(mdf):.4f}, {np.max(mdf):.4f}]')


def load_mdf(load_path):
    """Load MDF from file."""
    with open(load_path, 'rb') as f:
        mdf_data = pickle.load(f)
    
    occupied_voxels_mask = mdf_data.get('occupied_voxels_mask', None)

    return mdf_data['mdf'], mdf_data['grid_shape'], mdf_data['grid_origin'], mdf_data['voxel_size'], occupied_voxels_mask


def load_mdf_dict(load_path):
    """Load MDF as a dict (mdf, grid_shape, grid_origin, voxel_size), for query use."""
    with open(load_path, 'rb') as f:
        return pickle.load(f)


def query_mdf_safe(mdf_data, points_world, safety_margin):
    """
    Certify whether an object (given by its world points) clears the swept-volume MDF.

    The MDF stores, for each workspace voxel, the minimum distance ever reached to the
    manipulator's swept volume over the demonstration. An object point falling in a voxel
    whose MDF value is below safety_margin lies inside (or too close to) the risky region.

    Args:
        mdf_data: dict with keys 'mdf', 'grid_shape', 'grid_origin', 'voxel_size'.
        points_world: (N, 3) array of object points in the MDF world frame.
        safety_margin: minimum allowed clearance (meters) between object and swept volume.

    Returns:
        True  -> object clears the swept volume (CFreeMDF holds).
        False -> object intrudes the risky region (blocking).
    """
    mdf = mdf_data['mdf']
    grid_origin = np.asarray(mdf_data['grid_origin'], dtype=float)
    voxel_size = float(mdf_data['voxel_size'])
    nx, ny, nz = mdf.shape

    pts = np.asarray(points_world, dtype=float).reshape(-1, 3)
    if len(pts) == 0:
        return True

    idx = world_to_voxel_coords(pts, grid_origin, voxel_size)
    inside = (
        (idx[:, 0] >= 0) & (idx[:, 0] < nx) &
        (idx[:, 1] >= 0) & (idx[:, 1] < ny) &
        (idx[:, 2] >= 0) & (idx[:, 2] < nz)
    )
    # Points outside the grid are far from the swept volume -> not blocking by themselves.
    if not np.any(inside):
        return True

    vi = idx[inside]
    dists = mdf[vi[:, 0], vi[:, 1], vi[:, 2]]
    return bool(np.min(dists) >= safety_margin)


def occupied_voxels_to_point_cloud(occupied_voxels_mask, grid_shape, grid_origin, voxel_size):
    """
    Convert occupied voxels mask to point cloud (XYZ only, no intensity).
    
    Args:
        occupied_voxels_mask: Boolean array of shape grid_shape indicating occupied voxels
        grid_shape: Shape of the grid (nx, ny, nz)
        grid_origin: Origin of the grid in world coordinates
        voxel_size: Size of each voxel
    
    Returns:
        point_cloud: Array of shape (N, 3) where columns are [x, y, z]
    """
    # Create meshgrid of voxel centers
    ranges = [np.linspace(grid_origin[i] + voxel_size/2, 
                          grid_origin[i] + grid_shape[i] * voxel_size - voxel_size/2, 
                          grid_shape[i]) 
              for i in range(3)]
    xx, yy, zz = np.meshgrid(ranges[0], ranges[1], ranges[2], indexing='ij')
    
    # Flatten coordinates
    x_flat = xx.flatten()
    y_flat = yy.flatten()
    z_flat = zz.flatten()
    occupied_flat = occupied_voxels_mask.flatten()
    
    # Only include occupied voxels
    x_occupied = x_flat[occupied_flat]
    y_occupied = y_flat[occupied_flat]
    z_occupied = z_flat[occupied_flat]
    
    # Stack into point cloud format [x, y, z] (XYZ only, no intensity)
    point_cloud = np.column_stack([x_occupied, y_occupied, z_occupied])
    
    return point_cloud


def mdf_to_point_cloud(mdf, grid_shape, grid_origin, voxel_size, min_distance_threshold=None):
    """
    Convert MDF grid to point cloud with distance values as intensity.
    
    Args:
        mdf: MDF array of shape grid_shape
        grid_shape: Shape of the grid (nx, ny, nz)
        grid_origin: Origin of the grid in world coordinates
        voxel_size: Size of each voxel
        min_distance_threshold: Optional threshold - only include points with distance >= threshold
    
    Returns:
        point_cloud: Array of shape (N, 4) where columns are [x, y, z, intensity]
        intensity values are the distance from the MDF
    """
    # Create meshgrid of voxel centers
    ranges = [np.linspace(grid_origin[i] + voxel_size/2, 
                          grid_origin[i] + grid_shape[i] * voxel_size - voxel_size/2, 
                          grid_shape[i]) 
              for i in range(3)]
    xx, yy, zz = np.meshgrid(ranges[0], ranges[1], ranges[2], indexing='ij')
    
    # Flatten coordinates
    x_flat = xx.flatten()
    y_flat = yy.flatten()
    z_flat = zz.flatten()
    distance_flat = mdf.flatten()
    
    # Apply threshold if specified
    if min_distance_threshold is not None:
        mask = distance_flat >= min_distance_threshold
        x_flat = x_flat[mask]
        y_flat = y_flat[mask]
        z_flat = z_flat[mask]
        distance_flat = distance_flat[mask]
    
    # Stack into point cloud format [x, y, z, intensity]
    point_cloud = np.column_stack([x_flat, y_flat, z_flat, distance_flat])
    
    return point_cloud


def save_point_cloud_ply(point_cloud, save_path):
    """
    Save point cloud to PLY format.
    
    Args:
        point_cloud: Array of shape (N, 4) with [x, y, z, intensity]
        save_path: Path to save PLY file
    """
    header = f"""ply
format ascii 1.0
comment MDF point cloud with distance as intensity
element vertex {len(point_cloud)}
property float x
property float y
property float z
property float intensity
end_header
"""
    
    with open(save_path, 'w') as f:
        f.write(header)
        for point in point_cloud:
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {point[3]:.6f}\n")
    
    print(f'Saved point cloud to: {save_path} ({len(point_cloud)} points)')


def save_point_cloud_numpy(point_cloud, save_path):
    """Save point cloud as numpy array."""
    np.save(save_path, point_cloud)
    print(f'Saved point cloud to: {save_path} ({len(point_cloud)} points)')


def numpy_to_pointcloud2(point_cloud, frame_id='world', timestamp=None, has_intensity=True):
    """
    Convert numpy point cloud to ROS PointCloud2 message.
    
    Args:
        point_cloud: Array of shape (N, 4) with [x, y, z, intensity] or (N, 3) with [x, y, z]
        frame_id: Frame ID for the point cloud message
        timestamp: ROS time stamp (if None, uses current time - requires initialized ROS node)
        has_intensity: Whether point cloud has intensity field (default True for MDF)
    
    Returns:
        sensor_msgs.msg.PointCloud2 message
    """
    try:
        import rospy
        from sensor_msgs.msg import PointCloud2, PointField
        from std_msgs.msg import Header
    except ImportError:
        raise ImportError("ROS libraries not available. Install ROS and source setup.bash")
    
    # Check if node is initialized

    rospy.init_node('mdf_pointcloud_publisher', anonymous=True)
    
    # Create header
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id
    
    # Create point cloud message
    cloud_msg = PointCloud2()
    cloud_msg.header = header
    cloud_msg.height = 1
    cloud_msg.width = len(point_cloud)
    cloud_msg.is_dense = False
    
    if has_intensity and point_cloud.shape[1] == 4:
        # Define fields: x, y, z, intensity
        cloud_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        cloud_msg.point_step = 16  # 4 bytes per field * 4 fields
        cloud_msg.data = point_cloud.astype(np.float32).tobytes()
    else:
        # Define fields: x, y, z only (XYZ format)
        cloud_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        cloud_msg.point_step = 12  # 4 bytes per field * 3 fields
        cloud_msg.data = point_cloud.astype(np.float32).tobytes()
    
    cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
    
    return cloud_msg


def publish_point_cloud_ros(point_cloud, topic_name='/mdf_pointcloud', frame_id='world', 
                           rate=10.0):
    """
    Publish point cloud as ROS PointCloud2 message with intensity (distance) field.
    
    Args:
        point_cloud: Array of shape (N, 4) with [x, y, z, intensity] where intensity is distance
        topic_name: ROS topic name to publish to
        frame_id: Frame ID for the point cloud message
        rate: Publishing rate in Hz
    """
    try:
        import rospy
        from sensor_msgs.msg import PointCloud2
    except ImportError:
        raise ImportError("ROS libraries not available. Install ROS and source setup.bash")
    
    # Initialize ROS node if not already initialized
    try:
        rospy.get_node_uri()
    except rospy.exceptions.ROSInitException:
        rospy.init_node('mdf_pointcloud_publisher', anonymous=True)
    
    # Create publisher
    pub = rospy.Publisher(topic_name, PointCloud2, queue_size=1)
    
    # Convert to PointCloud2 message (this will use rospy.Time.now() which requires initialized node)
    cloud_msg = numpy_to_pointcloud2(point_cloud, frame_id=frame_id)
    
    # Publish
    rate_obj = rospy.Rate(rate)
    rospy.loginfo(f"Publishing point cloud to {topic_name} at {rate} Hz ({len(point_cloud)} points)")
    
    while not rospy.is_shutdown():
        cloud_msg.header.stamp = rospy.Time.now()
        pub.publish(cloud_msg)
        rate_obj.sleep()


def publish_point_cloud_xyz_ros(point_cloud, topic_name='/occupied_voxels', frame_id='world', rate=10.0):
    """
    Publish XYZ point cloud (no intensity) as ROS PointCloud2 message.
    
    Args:
        point_cloud: Array of shape (N, 3) with [x, y, z]
        topic_name: ROS topic name to publish to
        frame_id: Frame ID for the point cloud message
        rate: Publishing rate in Hz
    """
    try:
        import rospy
        from sensor_msgs.msg import PointCloud2
    except ImportError:
        raise ImportError("ROS libraries not available. Install ROS and source setup.bash")
    
    # Initialize ROS node if not already initialized
    try:
        rospy.get_node_uri()
    except rospy.exceptions.ROSInitException:
        rospy.init_node('mdf_pointcloud_publisher', anonymous=True)
    
    # Create publisher
    pub = rospy.Publisher(topic_name, PointCloud2, queue_size=1)
    
    # Convert to PointCloud2 message (XYZ only, no intensity)
    cloud_msg = numpy_to_pointcloud2(point_cloud, frame_id=frame_id, has_intensity=False)
    
    # Publish
    rate_obj = rospy.Rate(rate)
    rospy.loginfo(f"Publishing XYZ point cloud to {topic_name} at {rate} Hz ({len(point_cloud)} points)")
    
    while not rospy.is_shutdown():
        cloud_msg.header.stamp = rospy.Time.now()
        pub.publish(cloud_msg)
        rate_obj.sleep()


def publish_mdf_and_occupied_voxels_ros(mdf_path, mdf_topic='/mdf_pointcloud', 
                                         occupied_topic='/occupied_voxels', 
                                         frame_id='world', min_distance_threshold=0.0, rate=10.0):
    """
    Load MDF and occupied voxels, then publish both as ROS PointCloud2 messages simultaneously.
    
    Args:
        mdf_path: Path to saved MDF file
        mdf_topic: ROS topic name for MDF point cloud (with distance as intensity)
        occupied_topic: ROS topic name for occupied voxels (XYZ only)
        frame_id: Frame ID for the point cloud messages
        min_distance_threshold: Only include MDF points with distance >= threshold
        rate: Publishing rate in Hz
    """
    try:
        import rospy
        from sensor_msgs.msg import PointCloud2
        import threading
    except ImportError:
        raise ImportError("ROS libraries not available. Install ROS and source setup.bash")
    
    # Initialize ROS node if not already initialized
    try:
        rospy.get_node_uri()
    except rospy.exceptions.ROSInitException:
        rospy.init_node('mdf_pointcloud_publisher', anonymous=True)
    
    # Load MDF data (includes occupied_voxels_mask)
    mdf, grid_shape, grid_origin, voxel_size, occupied_voxels_mask = load_mdf(mdf_path)
    print(f"Loaded MDF from: {mdf_path}")
    
    # Convert MDF to point cloud (with intensity)
    mdf_point_cloud = mdf_to_point_cloud(mdf, grid_shape, grid_origin, voxel_size, 
                                         min_distance_threshold=min_distance_threshold)
    
    print(f"MDF point cloud statistics:")
    print(f"  Number of points: {len(mdf_point_cloud)}")
    print(f"  Distance range: [{np.min(mdf_point_cloud[:, 3]):.4f}, {np.max(mdf_point_cloud[:, 3]):.4f}]")
    
    # Convert occupied voxels to point cloud (XYZ only)
    occupied_point_cloud = None
    if occupied_voxels_mask is not None:
        occupied_point_cloud = occupied_voxels_to_point_cloud(occupied_voxels_mask, grid_shape, grid_origin, voxel_size)
        print(f"Occupied voxels point cloud statistics:")
        print(f"  Number of points: {len(occupied_point_cloud)}")
    
    # Create publishers
    mdf_pub = rospy.Publisher(mdf_topic, PointCloud2, queue_size=1)
    occupied_pub = rospy.Publisher(occupied_topic, PointCloud2, queue_size=1) if occupied_point_cloud is not None else None
    
    # Convert to PointCloud2 messages
    mdf_cloud_msg = numpy_to_pointcloud2(mdf_point_cloud, frame_id=frame_id, has_intensity=True)
    occupied_cloud_msg = numpy_to_pointcloud2(occupied_point_cloud, frame_id=frame_id, has_intensity=False) if occupied_point_cloud is not None else None
    
    # Publish both topics
    rate_obj = rospy.Rate(rate)
    rospy.loginfo(f"Publishing MDF to {mdf_topic} and occupied voxels to {occupied_topic} at {rate} Hz")
    
    while not rospy.is_shutdown():
        timestamp = rospy.Time.now()
        
        # Update and publish MDF
        mdf_cloud_msg.header.stamp = timestamp
        mdf_pub.publish(mdf_cloud_msg)
        
        # Update and publish occupied voxels if available
        if occupied_cloud_msg is not None:
            occupied_cloud_msg.header.stamp = timestamp
            occupied_pub.publish(occupied_cloud_msg)
        
        rate_obj.sleep()


def publish_mdf_point_cloud_ros(mdf_path, topic_name='/mdf_pointcloud', frame_id='world',
                                min_distance_threshold=0.0, rate=10.0):
    """
    Load MDF, convert to point cloud, and publish as ROS PointCloud2 with intensity (distance).
    
    Args:
        mdf_path: Path to saved MDF file
        topic_name: ROS topic name to publish to
        frame_id: Frame ID for the point cloud message
        min_distance_threshold: Only include points with distance >= threshold
        rate: Publishing rate in Hz
    """
    # Load MDF
    mdf, grid_shape, grid_origin, voxel_size, _ = load_mdf(mdf_path)
    print(f"Loaded MDF from: {mdf_path}")
    
    # Convert to point cloud
    point_cloud = mdf_to_point_cloud(mdf, grid_shape, grid_origin, voxel_size, 
                                     min_distance_threshold=min_distance_threshold)
    
    print(f"Point cloud statistics:")
    print(f"  Number of points: {len(point_cloud)}")
    print(f"  Distance range: [{np.min(point_cloud[:, 3]):.4f}, {np.max(point_cloud[:, 3]):.4f}]")
    
    # Publish as ROS message
    publish_point_cloud_ros(point_cloud, topic_name=topic_name, frame_id=frame_id, rate=rate)


def publish_mdf_as_point_cloud(mdf_path, output_path=None, output_format='ply', 
                               min_distance_threshold=0.0):
    """
    Read MDF from file and publish/save as point cloud with distance as intensity.
    
    Args:
        mdf_path: Path to saved MDF file
        output_path: Path to save point cloud (if None, derives from mdf_path)
        output_format: Format to save ('ply' or 'npy')
        min_distance_threshold: Only include points with distance >= threshold
    """
    # Load MDF
    mdf, grid_shape, grid_origin, voxel_size, _ = load_mdf(mdf_path)
    print(f"Loaded MDF from: {mdf_path}")
    
    # Convert to point cloud
    point_cloud = mdf_to_point_cloud(mdf, grid_shape, grid_origin, voxel_size, 
                                     min_distance_threshold=min_distance_threshold)
    
    # Determine output path
    if output_path is None:
        base_name = os.path.splitext(mdf_path)[0]
        if output_format == 'ply':
            output_path = f"{base_name}_pointcloud.ply"
        else:
            output_path = f"{base_name}_pointcloud.npy"
    
    # Save point cloud
    if output_format == 'ply':
        save_point_cloud_ply(point_cloud, output_path)
    else:
        save_point_cloud_numpy(point_cloud, output_path)
    
    print(f"Point cloud statistics:")
    print(f"  Number of points: {len(point_cloud)}")
    print(f"  Distance range: [{np.min(point_cloud[:, 3]):.4f}, {np.max(point_cloud[:, 3]):.4f}]")
    
    return point_cloud


def construct_mdf_and_save_point_cloud(num_demos=None):
    """Main function to construct MDF and save point cloud."""
    # Configuration
    pybullet_use_gui = True  # Set to True for visualization
    voxel_size = 0.02
    task = 'clean_cup'  # Default task
    start_frame_idx = 180
    
    # Workspace bounds (adjust based on your workspace)
    workspace_bounds = {
        'min': [-0.5, -0.5, -0.05],  # [x_min, y_min, z_min]
        'max': [0.5, 0.5, 0.5]       # [x_max, y_max, z_max]
    }
    
    # Load demonstration files (HDF5 format)
    # Update this path to point to your clean_cup demonstration HDF5 files
    hdf5_dir = "/ssd1/aloha_data/clean_cup"  # Default path for clean_cup data
    hdf5_files = []
    
    # You can either provide a directory or specific file paths
    if os.path.isdir(hdf5_dir):
        # Load all episode files in the directory
        for fname in sorted(os.listdir(hdf5_dir)):
            if fname.startswith('episode_') and fname.endswith('.hdf5'):
                hdf5_files.append(os.path.join(hdf5_dir, fname))
    else:
        # Or specify individual files
        hdf5_files = [hdf5_dir]  # If it's a single file path
    
    if len(hdf5_files) == 0:
        raise ValueError(f"No HDF5 files found at {hdf5_dir}. Please update the path.")
    
    print(f"Found {len(hdf5_files)} HDF5 file(s) to process")
    
    # Construct MDF
    print("Constructing Minimum Distance Field...")
    mdf, grid_shape, grid_origin, voxel_size, occupied_voxels_mask = construct_mdf(
        hdf5_files, 
        workspace_bounds, 
        voxel_size=voxel_size,
        task=task,
        pybullet_use_gui=pybullet_use_gui,
        start_frame_idx=start_frame_idx,
        num_demos=num_demos
    )
    
    # Save MDF
    mdf_path = get_file_path(__file__, f'mdf_data_{task}.pkl')
    save_mdf(mdf, grid_shape, grid_origin, voxel_size, mdf_path, occupied_voxels_mask=occupied_voxels_mask)
    
    # Convert to point cloud and save
    print("\nConverting MDF to point cloud...")
    point_cloud = publish_mdf_as_point_cloud(
        mdf_path, 
        output_format='ply',
        min_distance_threshold=0.0  # Include all points
    )
    
    print("\nMDF construction complete!")

def load_insertion_txt_demos(txt_path):
    """
    Load insertion-only joint trajectories from the recorded human-qpos txt (same format/logic
    as visuomotor_replayer.load_demonstration): each demo is delimited by a reset in the time
    column; 30 frames are clipped from each end. Columns: t, L-arm(1:7), L-grip(7), R-arm(8:14),
    R-grip(14). These frames are the INSERTION segment only (no pickup/approach).
    """
    qpos_mat = np.loadtxt(txt_path)
    qpos_demos_left, qpos_demos_right = [], []
    start_clips = end_clips = 30
    idx = 0
    for i in range(1, len(qpos_mat)):
        if qpos_mat[i][0] < qpos_mat[i - 1][0]:
            cs, ce = idx + start_clips, i - end_clips
            if ce > cs:
                qpos_demos_left.append(qpos_mat[cs:ce, 1:7])
                qpos_demos_right.append(qpos_mat[cs:ce, 8:14])
            idx = i
    return qpos_demos_left, qpos_demos_right


def construct_mdf_insertion(num_demos=3, voxel_size=0.02, frame_stride=5,
                            ee_only=False, ee_radius=0.08, ee_dist_threshold=0.20,
                            z_min=0.04, pybullet_use_gui=False):
    """
    Build the swept-volume MDF for the mj bimanual peg-in-hole insertion skill from the
    recorded INSERTION-segment human qpos (insertion_gmm/insertion_human_qpos.txt) and save it
    next to this module as 'mdf_data_mj_insertion.pkl' (loaded at plan time to certify CFreeMDF).

    Default = WHOLE-ARM occupancy with a z_min table-floor filter: insertion-only frames keep
    the arms near the converged insertion config, and the z_min filter removes the ALOHA URDF's
    table/base links (the only thing that previously made the whole-arm volume blanket the
    tabletop). This captures the true arm sweep above the table while leaving on-table room to
    set a blocking obstacle aside. ee_only=True is the alternative (gripper-tube occupancy only).
    """
    txt_path = get_file_path(__file__, '../insertion_gmm/insertion_human_qpos.txt')
    qpos_demos_left, qpos_demos_right = load_insertion_txt_demos(txt_path)
    if len(qpos_demos_left) == 0:
        raise ValueError(f"No insertion demos parsed from {txt_path}")

    # Workspace covering the ALOHA tabletop; insertion happens near the center.
    workspace_bounds = {'min': [-0.5, -0.5, -0.05], 'max': [0.5, 0.5, 0.5]}

    print(f"Constructing insertion MDF from {len(qpos_demos_left)} insertion-segment demos "
          f"(use {num_demos}, stride {frame_stride}, ee_only={ee_only}, ee_dist<{ee_dist_threshold}, "
          f"z_min={z_min})...")
    mdf, grid_shape, grid_origin, voxel_size, occupied_voxels_mask = construct_mdf(
        None, workspace_bounds, voxel_size=voxel_size, task='mj_insertion',
        pybullet_use_gui=pybullet_use_gui, start_frame_idx=0,
        num_demos=num_demos, frame_stride=frame_stride, ee_only=ee_only, ee_radius=ee_radius,
        qpos_demos=(qpos_demos_left, qpos_demos_right), ee_dist_threshold=ee_dist_threshold,
        z_min=z_min,
    )

    mdf_path = get_file_path(__file__, 'mdf_data_mj_insertion.pkl')
    save_mdf(mdf, grid_shape, grid_origin, voxel_size, mdf_path,
             occupied_voxels_mask=occupied_voxels_mask)
    return mdf_path


def visualize_mdf(mdf_path, out_path=None, z_min=0.04,
                  colobs_init=(-0.125, 0.0),
                  patch_x=(-0.28, 0.27), patch_y=(-0.18, 0.18)):
    """
    3-panel MDF distance-field PNG: min distance projected along each axis.
    Red = swept/dangerous, blue/cold = safe/far. z_min filters table-level URDF
    links (default 0.04m) before computing projections.
    Saves to <mdf_path stem>_viz.png by default.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    d = load_mdf_dict(mdf_path)
    occ = d['occupied_voxels_mask'].copy()
    org = np.asarray(d['grid_origin'], dtype=float)
    vs = float(d['voxel_size'])
    nx, ny, nz = occ.shape

    # apply z-floor filter to exclude table/base links in the URDF
    zs = org[2] + (np.arange(nz) + 0.5) * vs
    occ[:, :, zs < z_min] = False

    # recompute the distance field from the z-filtered mask (same EDT as construction)
    mdf = compute_edt_map(occ, vs)

    frac_filtered = occ.mean()

    xs = org[0] + (np.arange(nx) + 0.5) * vs
    ys = org[1] + (np.arange(ny) + 0.5) * vs

    # project: min MDF along perpendicular axis captures closest danger in that column
    proj_xy = np.min(mdf, axis=2)
    proj_xz = np.min(mdf, axis=1)
    proj_yz = np.min(mdf, axis=0)

    vmax = np.percentile(mdf, 95)
    XR = [xs[0], xs[-1]]; YR = [ys[0], ys[-1]]; ZR = [org[2], org[2] + nz * vs]
    cmap = 'RdYlBu'

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    title_base = os.path.basename(mdf_path)

    ax = axes[0]
    im = ax.imshow(proj_xy.T, extent=[XR[0], XR[1], YR[0], YR[1]],
                   origin='lower', cmap=cmap, vmin=0, vmax=vmax, aspect='equal')
    ax.add_patch(Rectangle(
        (patch_x[0], patch_y[0]), patch_x[1] - patch_x[0], patch_y[1] - patch_y[0],
        fill=False, ec='k', lw=1.5, ls='--', label='placement patch'))
    ax.plot(*colobs_init, 'g*', ms=14, label='colObs@init', zorder=5)
    ax.set_title(f'{title_base}  (z_min={z_min}m)\nswept {100*frac_filtered:.1f}%  red=danger  blue=safe  (top-down xy)')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.legend(loc='upper right', fontsize=8); ax.grid(alpha=0.2)
    plt.colorbar(im, ax=ax, label='min MDF dist (m)')

    ax = axes[1]
    im = ax.imshow(proj_xz.T, extent=[XR[0], XR[1], ZR[0], ZR[1]],
                   origin='lower', cmap=cmap, vmin=0, vmax=vmax, aspect='auto')
    ax.set_title('min MDF over y  (side xz)')
    ax.set_xlabel('x'); ax.set_ylabel('z'); ax.grid(alpha=0.2)
    plt.colorbar(im, ax=ax, label='min MDF dist (m)')

    ax = axes[2]
    im = ax.imshow(proj_yz.T, extent=[YR[0], YR[1], ZR[0], ZR[1]],
                   origin='lower', cmap=cmap, vmin=0, vmax=vmax, aspect='auto')
    ax.set_title('min MDF over x  (front yz)')
    ax.set_xlabel('y'); ax.set_ylabel('z'); ax.grid(alpha=0.2)
    plt.colorbar(im, ax=ax, label='min MDF dist (m)')

    plt.tight_layout()
    if out_path is None:
        out_path = os.path.splitext(mdf_path)[0] + '_viz.png'
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved viz to {out_path}  (swept {100*frac_filtered:.2f}% after z_min={z_min}m filter)')
    return out_path


def load_mdf_and_publish():
    saved_pkl_path = get_file_path(__file__, f'mdf_data_clean_cup.pkl')
    publish_mdf_and_occupied_voxels_ros(
        saved_pkl_path, 
        mdf_topic='/mdf_pointcloud',
        occupied_topic='/occupied_voxels',
        frame_id='world', 
        rate=10.0
    )
    rospy.spin()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Construct MDF and optionally limit number of demos processed')
    parser.add_argument('--num_demos', type=int, default=2, help='Use only the first N demonstrations')
    parser.add_argument('--task', type=str, default='insertion',
                        choices=['insertion', 'clean_cup', 'publish', 'viz'],
                        help="'insertion' builds the mj peg-hole MDF; 'clean_cup' the original; "
                             "'publish' ROS viz; 'viz' PNG visualization of a saved pkl")
    parser.add_argument('--frame_stride', type=int, default=5)
    parser.add_argument('--voxel_size', type=float, default=0.02)
    parser.add_argument('--ee_only', action='store_true',
                        help='alternative: gripper-tube occupancy only (default is whole-arm)')
    parser.add_argument('--ee_radius', type=float, default=0.08)
    parser.add_argument('--z_min', type=float, default=0.04,
                        help='drop table/base URDF links below this height from occupancy')
    parser.add_argument('--mdf_path', type=str, default=None,
                        help="pkl path for --task viz (defaults to mdf_data_mj_insertion.pkl)")
    parser.add_argument('--out_path', type=str, default=None,
                        help="output PNG path for --task viz")
    args = parser.parse_args()

    if args.task == 'insertion':
        construct_mdf_insertion(num_demos=args.num_demos, voxel_size=args.voxel_size,
                                frame_stride=args.frame_stride, ee_only=args.ee_only,
                                ee_radius=args.ee_radius, z_min=args.z_min)
    elif args.task == 'clean_cup':
        construct_mdf_and_save_point_cloud(num_demos=args.num_demos)
    elif args.task == 'viz':
        pkl = args.mdf_path or get_file_path(__file__, 'mdf_data_mj_insertion.pkl')
        visualize_mdf(pkl, out_path=args.out_path)
    else:
        load_mdf_and_publish()

