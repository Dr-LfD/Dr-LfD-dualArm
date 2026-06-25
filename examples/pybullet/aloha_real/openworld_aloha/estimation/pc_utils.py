
import numpy as np

from examples.pybullet.utils.pybullet_tools.utils import (
    COLOR_FROM_NAME)

from examples.pybullet.aloha_real.openworld_aloha.estimation.observation import     LabeledPoint

from sklearn.cluster import DBSCAN


def recolored_cluster(cluster, color = 'red'):
    color_1x3 = np.array(COLOR_FROM_NAME[color])[:3]
    new_cluster = [LabeledPoint(lp.point, color_1x3, lp.label) for lp in cluster]
    return new_cluster


def merge_cluster_by_label( old_clusters, new_clusters):
    for label, new_cluster in new_clusters.items():
        if label in old_clusters:
            old_cluster = recolored_cluster(old_clusters[label], 'green')
            old_clusters[label] = old_cluster +  new_cluster
        else:
            old_clusters[label] = new_cluster
    return old_clusters


def filter_pc(pc, use_DBSCAN = True):
    pc = np.asarray(pc)
    if pc.ndim != 2 or pc.shape[1] < 3:
        raise ValueError(f'filter_pc expects an (N, >=3) array; got shape {pc.shape}')

    ### save the point cloud as pcd
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    # Open3D only accepts xyz; keep any extra columns (e.g. rgb) on the side.
    pcd.points = o3d.utility.Vector3dVector(pc[:, :3])


    _, ind = pcd.remove_radius_outlier(nb_points=10, radius=0.05)
    # Select surviving rows from the full array so xyz and rgb stay aligned.
    cl_pts = pc[np.asarray(ind, dtype=int)]
    # cl_pts = cl_pts[cl_pts[:, 2] > 0.03]


    ## use clustering to filter the points
    if not use_DBSCAN:
        return cl_pts

    dbscan = DBSCAN(eps=0.03, min_samples=10)
    labels = dbscan.fit_predict(cl_pts[:, :3])
    valid_mask = labels != -1

    unique_labels = set(labels)
    n_clusters_ = len(unique_labels) - (1 if -1 in labels else 0)
    if n_clusters_ > 1:
        largest_cluster_label = max(unique_labels, key = list(labels).count)
        largest_cluster_mask  = labels == largest_cluster_label
        valid_mask = valid_mask & largest_cluster_mask        

    cl_pts_filtered = cl_pts[valid_mask]


    return cl_pts_filtered

def downsample(pc, num_points = 2048, filter = True):

    if pc.shape[0] > num_points:
        sampled_indices = np.random.choice(pc.shape[0], num_points, replace=False)
        pc = pc[sampled_indices]
    elif pc.shape[0] < num_points:
        if pc.shape[0] < num_points *0.1:
            raise ValueError(f'Input pc shape {pc.shape[0]} is not enough for desired {num_points}!')
        else:
            random_repeated_indices = np.random.choice(pc.shape[0], num_points - pc.shape[0], replace=True)
            pc = np.concatenate([pc, pc[random_repeated_indices]], axis=0)

    if filter:
        ## filter using DBSCAN
        pc = filter_pc(pc, use_DBSCAN=False)

    return pc

def add_projected_point(pc, num_ratio = 0.5):
    """
    Project the point cloud onto the plane at the minimum z value, then downsample to 100 points.
    Args:
        pc (np.ndarray): Input point cloud of shape (N, 3)
    Returns:
        np.ndarray: Downsampled projected point cloud of shape (100, 3)
    """
    # Find the minimum z value
    min_z = np.min(pc[:, 2])
    # Project all points onto the plane z = min_z
    projected_pc = pc.copy()
    projected_pc[:, 2] = min_z + np.random.uniform(0, 0.01, size=pc.shape[0])  # Add per-point random lift
    
    num_points = int(pc.shape[0] * num_ratio)
    projected_pc_down = downsample(projected_pc, num_points=num_points, filter= False)
    return projected_pc_down

def save_np_to_ply(np_arr, file_name = 'test.ply'):
    assert len(np_arr.shape) == 2 and np_arr.shape[1] >= 3, "Input array must be of shape (N, 3) or larger."
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_arr[:, :3])
    o3d.io.write_point_cloud(file_name, pcd)

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