from examples.pybullet.utils.pybullet_tools.utils import set_preview
# from examples.pybullet.aloha_real.openworld_aloha.problem_construction import get_fixed
from examples.pybullet.aloha_real.openworld_aloha.estimation.belief import Belief
from examples.pybullet.aloha_real.openworld_aloha.estimation.dnn import init_seg, init_sc, DEFAULT_VALUE, iterate_array
from examples.pybullet.aloha_real.openworld_aloha.estimation.geometry import cloud_from_depth
from examples.pybullet.aloha_real.openworld_aloha.estimation.observation import save_camera_images, multiply, Pose
from examples.pybullet.aloha_real.openworld_aloha.estimation.tables import estimate_surfaces
import datetime
import pybullet as p
from examples.pybullet.aloha_real.openworld_aloha.entities import Object, get_label_counts, CameraImage

from examples.pybullet.aloha_real.openworld_aloha.robot_entities import ALOHARobot, DUALfrankaRobot

# import cv2
import json
import numpy as np

## camlink to optical
def get_compatible_campose(cam_pose_dict, camlink2optical = True, random_rot_camera = False, **kwargs):
    xfront_cam_pose = [cam_pose_dict['xyz'], cam_pose_dict['wxyz']]

    if random_rot_camera:
        rot_angle = Pose(euler=[0, 0, np.random.uniform(-np.pi/3, np.pi/3)])
        xfront_cam_pose = multiply(rot_angle, xfront_cam_pose)

    if not camlink2optical:
        return xfront_cam_pose
    rotzm90 = Pose(euler=[0, 0, -np.pi / 2])
    rotxm90 = Pose(euler=[-np.pi / 2, 0, 0])
    camera_pose = multiply(xfront_cam_pose, rotzm90)
    camera_pose = multiply(camera_pose, rotxm90) 
    # qw, qx, qy, qz = xfront_cam_pose[1]
    # xfront_cam_pose[1] = [qx, qy, qz, qw]
    # camera_pose = xfront_cam_pose
    return camera_pose


def optical_to_camlink(cam_pose_dict):
    xfront_cam_pose = [cam_pose_dict['xyz'], cam_pose_dict['wxyz']]
    rotxp90 = Pose(euler=[np.pi / 2, 0, 0])
    rotzp90 = Pose(euler=[0, 0, np.pi / 2])
    camera_pose = multiply(xfront_cam_pose, rotxp90)
    camera_pose = multiply(camera_pose, rotzp90)
    return camera_pose

def get_3d_points_from_depth(depth_img, xfront_cam_pose, camera_info_depth):
    rotzm90 = Pose(euler=[0, 0, -np.pi / 2])
    rotxm90 = Pose(euler=[-np.pi / 2, 0, 0])
    camera_pose = multiply(xfront_cam_pose, rotzm90)
    camera_pose = multiply(camera_pose, rotxm90)     

    depth_camera_matrix = np.array(camera_info_depth['K']).reshape(3, 3)
    point_cloud = cloud_from_depth(depth_camera_matrix, depth_img, top_left_origin=True)  # if True, y start from top, otherwise, y start from bottom

    return point_cloud, camera_pose, depth_camera_matrix

#############  estimation  ################
def fuse_predicted_labels(
    seg_network,
    camera_image,
    fuse=False,
    use_depth=False,
    debug=False,
    num_segs=1,
    **kwargs
):
    rgb, depth, bullet_seg, _, camera_matrix = camera_image
    if fuse:
        print("Ground truth:", get_label_counts(bullet_seg))

    point_cloud = None
    if use_depth:
        point_cloud = cloud_from_depth(camera_matrix, depth)

    predicted_seg = seg_network.get_seg(
        rgb[:, :, :3],
        point_cloud=point_cloud,
        depth_image=depth,
        return_int=False,
        num_segs=num_segs,
        **kwargs
    )
    return CameraImage(rgb, depth, predicted_seg, *camera_image[3:])

## TODO: use a list of prompts
class estimation_policy(object):
    def __init__(self, robot_entity,  env_type = 'sim',ws_aabb=None, teleport = False, client = None, seg_branch='sam', text_prompt=None, perception_fn = None, **kwargs):
        self.robot = robot_entity
        real_execution = env_type == 'real'     
        self.robot.reset(reset_pybullet= not real_execution)        
        if perception_fn is not None:
            perception_fn()
        
        self.teleport = teleport
        self.client = client
        self.env_type = env_type
        # if mode == 'common':
        if (env_type == 'file' or env_type == 'real'):            
            print('Segmentation network initialized!')
            self.seg_network = init_seg(branch=seg_branch, text_prompt=text_prompt, **kwargs)
            # self.sc_network = init_sc(branch='msn')
            self.sc_network = init_sc(branch=None)
            print('Completed the initialization of the segmentation network')
        
        elif env_type == 'mj':
            # ## read point cloud instead of image
            # import os
            # if not os.path.exists(mj_pc_path):
            #     raise FileNotFoundError("mh_pc not found in "+ mj_pc_path)
            # ## read dict from npz
            # self.pc_dict  = np.load(mj_pc_path)

            self.seg_network = None
            self.sc_network = None
            # self.sc_network = init_sc(branch='msn')

        # elif mode == 'data_process':
        #     self.seg_network = init_seg(branch= seg_branch, text_prompt=text_prompt)
        #     # self.seg_network = init_seg(branch='ucn')

        #     self.sc_network = init_sc(branch='msn')

        self.estimates = []

        # set_preview(True)

        # self.robot.update_conf()

        self.belief = Belief(
            self.robot,
            surface_beliefs=[
                # SurfaceBelief(table, resolutions=0.04 * np.ones(3), known_objects=real_world.known),
            ],
            ws_aabb=ws_aabb,
            client=self.client,
        )


    # direct feed images and camera info
    def get_image_direct(self, color_img, depth_img, camera_info, cam_pose_json_file, mujoco_seg = None, camlink2optical = True, **kwargs):


        camera_info_depth = camera_info

        with open(cam_pose_json_file, 'r') as f:
            cam_pose_dict = json.load(f)

        camera_pose = get_compatible_campose(cam_pose_dict, camlink2optical = camlink2optical, **kwargs)
        depth_camera_matrix = np.array(camera_info_depth['K']).reshape(3, 3)
        point_cloud = cloud_from_depth(depth_camera_matrix, depth_img, top_left_origin=True)  #

        predicted_seg = self.seg_network.get_seg(
            color_img,
            point_cloud=point_cloud,
            depth_image=depth_img,
            return_int=False,
            num_segs=1,
            mujoco_seg=mujoco_seg,
            **kwargs
        )
        if predicted_seg is None:
            return None
        
        camera_image = CameraImage(color_img, depth_img, predicted_seg, camera_pose, depth_camera_matrix)
        save_camera_images(camera_image)

        return camera_image

    # def get_seg_from_file(self):
    #     import cv2, os, rospy
    #     from std_srvs.srv import Empty
    #     rospy.wait_for_service('trigger_sam')
    #     try:
    #         trigger_sam = rospy.ServiceProxy('trigger_sam', Empty)
    #         response = trigger_sam()

    #         print('Reading binary mask from file')
    #         tempvis_path = '${REPO_ROOT}/examples/pybullet/aloha_real/openworld_aloha/estimation/temp_vis/'
    #         mask_path = os.path.join(tempvis_path, 'franka_cam2_mask/mask.png')        
    #         seg_mask = cv2.imread(mask_path,  cv2.IMREAD_GRAYSCALE) 
    #         if seg_mask is None:
    #             raise ValueError()
    #         str_seg = np.full(seg_mask.shape + (2,), DEFAULT_VALUE , dtype=object)
    #         for r, c in iterate_array(str_seg, dims=[0, 1]):
    #             if seg_mask[r, c] != False:
    #                 str_seg[r, c, 0] = 1
    #                 str_seg[r, c, 1] = "instance_1"
    #         return str_seg
    #     except rospy.ServiceException as e:
    #         rospy.logerr(f"Service call failed: {e}")
    #         return None
        
    
    def get_image(self):
        import cv2
        if self.env_type == 'sim':
            [camera] = self.robot.cameras
            # camera.draw(draw_cone=True)
            camera_image = camera.get_image()
            # do segmentation
            camera_image = fuse_predicted_labels(
                self.seg_network,
                camera_image,
                use_depth=True,
            )
            save_camera_images(camera_image)    
        elif self.env_type == 'file' or self.env_type == 'real':
            color_img = cv2.imread(self.cam_file_dict['color_img'])
            
            depth_img_mm = cv2.imread(self.cam_file_dict['depth_img'], cv2.IMREAD_ANYDEPTH)
            depth_img = depth_img_mm.astype(np.float32) / 1000.0  # realsense

            with open(self.cam_file_dict['camera_info_color'], 'r') as f:
                camera_info_color = json.load(f)
            with open(self.cam_file_dict['camera_info_depth'], 'r') as f:
                camera_info_depth = json.load(f)

            with open(self.cam_file_dict['camera_pose'], 'r') as f:
                cam_pose_dict = json.load(f)

            camera_pose = get_compatible_campose(cam_pose_dict)
            depth_camera_matrix = np.array(camera_info_depth['K']).reshape(3, 3)
            point_cloud = cloud_from_depth(depth_camera_matrix, depth_img, top_left_origin=True)  #

            # pts_ls = point_cloud.reshape(-1, 3)
            # import open3d as o3d
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(pts_ls)
            # o3d.io.write_point_cloud('all_pc_cam_frame.ply', pcd)

            predicted_seg = self.seg_network.get_seg(
                color_img,
                point_cloud=point_cloud,
                depth_image=depth_img,
                return_int=False,
                num_segs=1,
            )
            # predicted_seg = self.get_seg_from_file() ## call sam service

            camera_image = CameraImage(color_img, depth_img, predicted_seg, camera_pose, depth_camera_matrix)
            save_camera_images(camera_image)


        else:
            raise ValueError("Invalid image source")
        
        return camera_image

    def reset_belief(self):
        self.belief.reset_objects()
        self.belief.observations = []
        self.belief.known_objects = []
        return self.belief

    def _save_debug_point_clouds(self, camera_image):
        """Save point clouds in both camera and world frames for debugging"""
        try:
            import open3d as o3d
            import os
            from examples.pybullet.aloha_real.openworld_aloha.estimation.observation import tform_labeled_points
            
            # Extract camera image components
            rgb, depth, seg, camera_pose, camera_matrix = camera_image
            
            # Generate point cloud in camera frame
            from examples.pybullet.aloha_real.openworld_aloha.estimation.geometry import cloud_from_depth
            point_cloud_cam = cloud_from_depth(camera_matrix, depth, top_left_origin=True)
            
            # Filter out invalid points (depth = 0)
            valid_mask = depth > 0
            valid_points_cam = point_cloud_cam[valid_mask]
            
            # Create camera frame point cloud
            pcd_cam = o3d.geometry.PointCloud()
            pcd_cam.points = o3d.utility.Vector3dVector(valid_points_cam)
            
            # Add colors from RGB image
            valid_colors = rgb[valid_mask] / 255.0  # Normalize to [0,1]
            pcd_cam.colors = o3d.utility.Vector3dVector(valid_colors)
            
            # Transform to world frame
            from examples.pybullet.utils.pybullet_tools.utils import tform_points
            valid_points_world = tform_points(camera_pose, valid_points_cam)
            
            # Create world frame point cloud
            pcd_world = o3d.geometry.PointCloud()
            pcd_world.points = o3d.utility.Vector3dVector(valid_points_world)
            pcd_world.colors = o3d.utility.Vector3dVector(valid_colors)
            
            # Create debug directory
            debug_dir = "debug_point_clouds"
            os.makedirs(debug_dir, exist_ok=True)
            
            # Save point clouds with timestamp
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            cam_file = os.path.join(debug_dir, f"point_cloud_camera_frame_{timestamp}.ply")
            world_file = os.path.join(debug_dir, f"point_cloud_world_frame_{timestamp}.ply")
            
            o3d.io.write_point_cloud(cam_file, pcd_cam)
            o3d.io.write_point_cloud(world_file, pcd_world)
            
            print(f"Debug point clouds saved:")
            print(f"  Camera frame: {cam_file}")
            print(f"  World frame: {world_file}")
            print(f"  Points count: {len(valid_points_cam)}")
            
            # Also save camera pose for reference
            pose_file = os.path.join(debug_dir, f"camera_pose_{timestamp}.txt")
            with open(pose_file, 'w') as f:
                f.write(f"Camera pose (position, quaternion):\n")
                f.write(f"Position: {camera_pose[0]}\n")
                f.write(f"Quaternion (w,x,y,z): {camera_pose[1]}\n")
                f.write(f"Camera matrix:\n{camera_matrix}\n")
            
        except ImportError:
            print("Warning: open3d not available, cannot save debug point clouds")
        except Exception as e:
            print(f"Error saving debug point clouds: {e}")

    def estimate_surfaces(self, camera_image, save_debug_pc=True, **kwargs):
        # Save point cloud for debugging
        if save_debug_pc:
            self._save_debug_point_clouds(camera_image)
        
        surfaces = estimate_surfaces(
            self.belief,
            camera_image,
            min_z=self.robot.min_z,
            max_depth=self.robot.max_depth,
            client=self.client,
            **kwargs
        )
        return surfaces

    def estimate_objects(self, camera_image, table, **kwargs):
        objects = self.belief.estimate_objects(
            camera_image,
            use_seg=True,
            surface=table.surface,
            project_base= True,
            sc_network=self.sc_network,
            save_relabeled_img=False,
            surfaces_movable=True,
            max_depth=self.robot.max_depth,
            **kwargs
        )
        return objects

    ## required when the images are already recorded
    def estimate_state_multiview_file(self, cam_dir_mapping, cam_extparam_mapping, phase = 0, **kwargs):
        import cv2
        import os
        color_imgs = {}
        depth_imgs = {}
        camera_infos = {}
        for rs_cam, dir in  cam_dir_mapping.items():
            color_img = cv2.imread(os.path.join(dir, 'color_image.png'))
            color_imgs[rs_cam] = {phase: color_img}

            depth_img_mm = cv2.imread(os.path.join(dir, 'depth_image.png'), cv2.IMREAD_ANYDEPTH)
            depth_img = depth_img_mm.astype(np.float32) / 1000.0
            depth_imgs[rs_cam] = {phase: depth_img}

            intrinsic_file = os.path.join(dir, 'depth_info.json')
            with open(intrinsic_file, 'r') as f:
                camera_info_color = json.load(f)
                camera_info_depth = camera_info_color
            camera_infos[rs_cam] = camera_info_depth

        return self.estimate_state_multiview(color_imgs, depth_imgs, camera_infos, phase, cam_extparam_mapping, cam_dir_mapping = cam_dir_mapping,  **kwargs)

  


    def estimate_state_multiview(self, color_imgs, depth_imgs, camera_infos, phase,   cam_extparam_mapping=None, cam_dir_mapping = None, calibrate_mapping = None, **kwargs):
        self.reset_belief()

        real_imgs = []
        for rs_cam in cam_extparam_mapping.keys():
            color_img = color_imgs[rs_cam][phase]
            depth_img = depth_imgs[rs_cam][phase]
            camera_info = camera_infos[rs_cam]
            camera_extparam = cam_extparam_mapping[rs_cam]  
            cam_dir = cam_dir_mapping[rs_cam]

            real_image = self.get_image_direct(color_img, depth_img, camera_info, camera_extparam, cam_dir = cam_dir, camlink2optical = calibrate_mapping[rs_cam], **kwargs)
            if real_image is not None:
                real_imgs.append(real_image)
            else:
                print("Error in getting image from ", rs_cam)



        if len(real_imgs) == 0:
            raise ValueError("No valid images")

        surfaces = self.estimate_surfaces(real_imgs[0], save_debug_pc = False)
        table = surfaces[0]
        objects = self.belief.estimate_objects_multiview(
            real_imgs,
            use_seg=True,
            surface = table.surface,
            project_base = True,
            sc_network = self.sc_network,
            save_relabeled_img = False,
            surfaces_movable = True,
            filter_outliers=True,
            add_meshpts=False,
            concave=False,
            **kwargs,
        )
        self.estimates.append(
            {
                # TODO: store the mesh *.obj files
                "date": datetime.datetime.now(),
                "surfaces": surfaces,
                "objects": objects,
            }
        )

        return self.belief
    
    def estimate_state(self,  \
                       color_img = None, depth_img = None, camera_info= None, cam_pose_json_file = None):
        self.reset_belief()
        if color_img is None and depth_img is None and camera_info is None and cam_pose_json_file is None:
            real_image = self.get_image()
        else:
            real_image = self.get_image_direct(color_img, depth_img, camera_info, cam_pose_json_file)

        surfaces = self.estimate_surfaces(real_image)
        table = surfaces[0]
        objects = self.estimate_objects(real_image, table,\
                     filter_outliers = True, add_meshpts = False, concave = True)

        self.estimates.append(
            {
                # TODO: store the mesh *.obj files
                "date": datetime.datetime.now(),
                "surfaces": surfaces,
                "objects": objects,
            }
        )

    def esmate_mj_state(self, load_obj_func, mj_pc_dict):
        self.reset_belief()
        for k, pc in mj_pc_dict.items():
            est_obj, obj_pose = load_obj_func(pc, sc_network=self.sc_network, category=k, filter = True, add_bottom = True)
            
            self.belief.estimated_objects.append(est_obj)
        return self.belief