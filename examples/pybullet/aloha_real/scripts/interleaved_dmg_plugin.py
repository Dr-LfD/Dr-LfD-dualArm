
import sys
import os
import h5py
import networkx as nx
import json


from interleaved_robosuite_base import robosuite_planner

root_path = next(p for p in (os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *([os.pardir] * k))) for k in range(16)) if os.path.isfile(os.path.join(p, '.repo_root')))

sys.path.append(root_path) if root_path not in sys.path else None
os.chdir(root_path)
from examples.pybullet.aloha_real.openworld_aloha.primitives import  GroupTrajectory


class DMG_planner(robosuite_planner):
    def get_env_options(self, shape_meta):
        cam_keys = [obs_key for obs_key in list(shape_meta['obs'].keys()) if "_image" in obs_key]
        img_shape = shape_meta['obs'][cam_keys[0]].shape
        camera_widths = img_shape[1]
        camera_heights = img_shape[2]
        env_options =   {
            "env_configuration": "single-arm-parallel",
            "robots": ["Panda", "Panda"],
            "camera_names": [cam_key.replace("_image", "") for cam_key in cam_keys],
            "camera_heights": camera_heights,
            "camera_widths": camera_widths,
            "camera_segmentations": "instance",
        }
        # SDP consumes a fused scene point_cloud obs, which the env only emits when output_all_pcds
        # is set. DP is image-only, so leave it off there to avoid the per-step point-cloud cost.
        if self.para['LfD_params']['lfd_alg'] == 'SDP':
            env_options["output_all_pcds"] = True
        return env_options
    
    ## only test the tamp part
    def tmp_pc_from_hdf5(self, hdf5_path):
        mj_pc_dict = {}
        with h5py.File(hdf5_path, 'r') as f:
            for obj_name in self.intereseted_objects:
                obj_pc_list = f[f'data/demo_0/obs/{obj_name}_point_cloud'][()]
                mj_pc_dict[obj_name] = obj_pc_list[0][:, :3]

        return mj_pc_dict







    def get_sgs_from_hdf5(self, hdf5_path):

        skillwise_sg = {}
        with h5py.File(hdf5_path, 'r') as f:
            sg_info = f['sg_info']
        ## get interested primitives
            for skill_name, skill_info in sg_info.items():
                ## get interested primitives
                for primitive_kw in self.sg_param['interested_primitives']:
                    if primitive_kw in skill_name:
                        break
                else:
                    continue

                skillwise_sg[skill_name] = {}
                for sg_phase_key in ['pre']:
                    sg_str = skill_info[f'{sg_phase_key}_sg'][()].decode('utf-8')
                    sg = nx.node_link_graph(json.loads(sg_str))
                    sg.graph['obj_names'] = self.sg_param['interested_objs'].keys()
                    sg.graph['hand_names'] = self.sg_param['robots']                
                    skillwise_sg[skill_name][sg_phase_key] = sg
        return skillwise_sg

