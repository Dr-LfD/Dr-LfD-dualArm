from examples.pybullet.aloha_real.openworld_aloha.estimation.belief import Belief
from examples.pybullet.aloha_real.openworld_aloha.estimation.dnn import init_seg, init_sc
from examples.pybullet.aloha_real.openworld_aloha.estimation.geometry import cloud_from_depth
from examples.pybullet.aloha_real.openworld_aloha.estimation.observation import save_camera_images, multiply, Pose
from examples.pybullet.aloha_real.openworld_aloha.estimation.tables import estimate_surfaces
import datetime
from examples.pybullet.aloha_real.openworld_aloha.entities import CameraImage

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
    return camera_pose


def optical_to_camlink(cam_pose_dict):
    xfront_cam_pose = [cam_pose_dict['xyz'], cam_pose_dict['wxyz']]
    rotxp90 = Pose(euler=[np.pi / 2, 0, 0])
    rotzp90 = Pose(euler=[0, 0, np.pi / 2])
    camera_pose = multiply(xfront_cam_pose, rotxp90)
    camera_pose = multiply(camera_pose, rotzp90)
    return camera_pose


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
        if (env_type == 'file' or env_type == 'real'):
            print('Segmentation network initialized!')
            self.seg_network = init_seg(branch=seg_branch, text_prompt=text_prompt, **kwargs)
            self.sc_network = init_sc(branch=None)
            print('Completed the initialization of the segmentation network')
        elif env_type == 'mj':
            self.seg_network = None
            self.sc_network = None

        self.estimates = []

        self.belief = Belief(
            self.robot,
            surface_beliefs=[],
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

    def reset_belief(self):
        self.belief.reset_objects()
        self.belief.observations = []
        self.belief.known_objects = []
        return self.belief

    def estimate_surfaces(self, camera_image, **kwargs):
        surfaces = estimate_surfaces(
            self.belief,
            camera_image,
            min_z=self.robot.min_z,
            max_depth=self.robot.max_depth,
            client=self.client,
            **kwargs
        )
        return surfaces

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

        surfaces = self.estimate_surfaces(real_imgs[0])
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

    def esmate_mj_state(self, load_obj_func, mj_pc_dict):
        self.reset_belief()
        for k, pc in mj_pc_dict.items():
            est_obj, obj_pose = load_obj_func(pc, sc_network=self.sc_network, category=k, filter = True, add_bottom = True)
            
            self.belief.estimated_objects.append(est_obj)
        return self.belief