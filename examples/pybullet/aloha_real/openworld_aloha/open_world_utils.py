


import os
import yaml

EXE_FOLDER = next(('/'+ os.path.join(*os.path.dirname(os.path.abspath(__file__)).split(os.sep)[:i+1]) + os.sep
                    for i in range(len(os.path.dirname(os.path.abspath(__file__)).split(os.sep))) 
                    if os.path.dirname(os.path.abspath(__file__)).split(os.sep)[i] == 'pddlstream_aloha'), None)


class obj_info(object):
    def __init__(self,  obj_name, obj_id, \
            x, y, yaw, l, w, h):
        self.obj_name = obj_name
        self.obj_id = obj_id
        self.x = x
        self.y = y
        self.yaw = yaw
        self.l = l
        self.w = w
        self.h = h

    def change_xyyaw(self, x, y, yaw):
        self.x = x
        self.y = y
        self.yaw = yaw
        
SOCKET_DEFAULT = obj_info('socket', 4 , 0.2, -0.2, 0.00, 0.1, 0.1, 0.1)
PEG_DEFAULT = obj_info('peg', 5, -0.25, 0.2, 0.00, 0.1, 0.1, 0.1)
        
def load_insertion_param(yaml_path):
    # load param from yaml
    with open(yaml_path, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
        print(cfg)
        cmd_path_tail = cfg['cmd_path']
        cmd_path = os.path.join(EXE_FOLDER, cmd_path_tail)
        socket_info = obj_info(cfg['socket_info']['obj_name'], cfg['socket_info']['obj_id'], \
            cfg['socket_info']['x'], cfg['socket_info']['y'], cfg['socket_info']['yaw'], \
            cfg['socket_info']['l'], cfg['socket_info']['w'], cfg['socket_info']['h'])
        peg_info = obj_info(cfg['peg_info']['obj_name'], cfg['peg_info']['obj_id'], \
            cfg['peg_info']['x'], cfg['peg_info']['y'], cfg['peg_info']['yaw'], \
            cfg['peg_info']['l'], cfg['peg_info']['w'], cfg['peg_info']['h'])
        colObs_info = obj_info(cfg['colObs_info']['obj_name'], cfg['colObs_info']['obj_id'], \
            cfg['colObs_info']['x'], cfg['colObs_info']['y'], cfg['colObs_info']['yaw'], \
            cfg['colObs_info']['l'], cfg['colObs_info']['w'], cfg['colObs_info']['h'])
        
        is_record = cfg['is_record']
        
    return cmd_path, [socket_info, peg_info,colObs_info], is_record

## construct sg for each ALOHA task
def get_skillwise_sgs(para, env_names):
    def initialize_empty_sg():
        sg = nx.DiGraph()
        sg.add_nodes_from(['table', 'left', 'right'])
        sg.add_edge('table', 'left')
        sg.add_edge('table', 'right')
        return sg

    def sg_from_yaml(obj_names, hand_names):
        empty_sg = initialize_empty_sg()
        empty_sg.graph['obj_names'] = obj_names
        empty_sg.graph['hand_names'] = hand_names
        for i in range(len(obj_names)):
            if i >= len(hand_names):
                continue
            empty_sg.add_edge(hand_names[i], obj_names[i])
        return empty_sg

    import networkx as nx
    skillwise_sgs = {}
    for env_name in env_names:
        per_env_sgs = {}
        skill_sg_param = para[env_name]
        bi_skill_name = f'bimanual_{env_name}'
        per_env_sgs[bi_skill_name] = {
            'pre_sg': None,
            'eff_sg': None,
        }
        per_env_sgs[bi_skill_name] ['skill_type'] = para[env_name]['skill_type']

        pre_sg = sg_from_yaml(skill_sg_param['pre_obj_names'], skill_sg_param['pre_arms'])
        per_env_sgs[bi_skill_name]['pre_sg'] = pre_sg
        
        per_env_sgs[bi_skill_name]['eff_sg'] = None
        ## get eff sg
        num_eff_objs = len(skill_sg_param['eff_obj_names']) 
        if num_eff_objs== 0:
            eff_sg = pre_sg.copy()
        else:
            eff_sg = sg_from_yaml(skill_sg_param['eff_obj_names'], skill_sg_param['eff_arms'])
        per_env_sgs[bi_skill_name]['eff_sg'] = eff_sg

        ## below is order-sensitive. Used for goal literal construction. 
        # per_env_sgs[bi_skill_name]['related_rbts'] = skill_sg_param['pre_arms'] + skill_sg_param['eff_arms']
        per_env_sgs[bi_skill_name]['related_rbts']  = skill_sg_param['related_rbts']
        
        ## get the grasp skill sg (from pre_arms / pre_obj_names)
        initial_sg = initialize_empty_sg()
        pre_arms = skill_sg_param.get('pre_arms', [])
        pre_obj_names = skill_sg_param.get('pre_obj_names', [])
        for pre_arm, pre_obj in zip(pre_arms, pre_obj_names):
            grasp_skill_name = f'{pre_arm}_grasp_{pre_obj}'
            grasp_eff_sg = initialize_empty_sg()
            grasp_eff_sg.add_edge(pre_arm, pre_obj)
            grasp_cur_sg = grasp_eff_sg.copy()
            grasp_cur_sg.add_edge('table', pre_obj)
            per_env_sgs[grasp_skill_name] = {
                'pre_sg': initial_sg,
                'cur_sg': grasp_cur_sg,
                'eff_sg': grasp_eff_sg,
                'related_rbts': [pre_arm],
                'skill_type': 'ATTACH'
            }

        ## get the place skill sg (from eff_arms / eff_obj_names)
        eff_arms = skill_sg_param.get('eff_arms', [])
        eff_obj_names = skill_sg_param.get('eff_obj_names', [])
        for eff_arm, eff_obj in zip(eff_arms, eff_obj_names):
            place_skill_name = f'{eff_arm}_place_{eff_obj}'
            place_pre_sg = eff_sg  # share the bimanual eff_sg object
            place_cur_sg = place_pre_sg.copy()
            place_cur_sg.add_edge('table', eff_obj)
            per_env_sgs[place_skill_name] = {
                'pre_sg': place_pre_sg,
                'cur_sg': place_cur_sg,
                'eff_sg': initial_sg,
                'related_rbts': [eff_arm],
                'skill_type': 'DETACH'
            }
        skillwise_sgs[env_name] = per_env_sgs
    return skillwise_sgs

def load_yaml_params(yaml_path, skill_yaml_paths, task_name=None, mode='sim', is_testing=False):
    import yaml
    with open(yaml_path, 'r') as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)

    # Resolve paths relative to CWD (project root, set by os.chdir at script startup)
    resolved_yaml_paths = [os.path.abspath(p) for p in skill_yaml_paths]
    skills = [os.path.splitext(os.path.basename(p))[0] for p in resolved_yaml_paths]

    parameters['task_name'] = task_name
    parameters['skill_names'] = skills
    interested_objects = set()
    additional_objects = set()
    DP_input_dict = {}

    for skill, skill_yaml in zip(skills, resolved_yaml_paths):
        with open(skill_yaml, 'r') as f:
            skill_params = yaml.load(f, Loader=yaml.FullLoader)
        if skill_params.get('contact_predictor_checkpoint'):
            parameters['contact_predictor_checkpoint'] = skill_params['contact_predictor_checkpoint']
        parameters[skill] = skill_params['sg_params']
        parameters['output_dir'] = skill_params['sg_params']['output_dir']
        interested_objects.update(skill_params['sg_params']['pre_obj_names'])
        additional_objects.update(skill_params['sg_params'].get('additional_objects', []))
        for kw, val in skill_params['LfD_params']['DP_input'].items():
            DP_input_dict[kw] = val

    parameters['LfD_params']['DP_input'] = DP_input_dict
    parameters['skill_yaml_paths'] = resolved_yaml_paths
    parameters['pre_obj_names'] = list(interested_objects)
    parameters['text_prompt'] = '.'.join(parameters['pre_obj_names'] + list(additional_objects)) + '.'
    parameters['env_type'] = mode
    parameters['do_testing'] = is_testing
    parameters['real_execute'] = (mode == 'real')
    return parameters

def get_camera_mappings(para):
    temp_vis_dir = os.path.join(EXE_FOLDER, para['temp_vis_dir'])
    active_cams = para['active_cams']
    cam_extparam_mapping = {}
    cam_dir_mapping = {}
    calibrate_mapping = {}
    for cam in active_cams:
        cam_extparam_mapping[para[f'{cam}_name']] = os.path.join(temp_vis_dir, para[f'{cam}_ext_json'])

        temp_img_dir = os.path.join(temp_vis_dir, para[f'{cam}_img_dir'])
        cam_dir_mapping[para[f'{cam}_name']] = temp_img_dir

        calibrate_mapping[para[f'{cam}_name']] = para[f'{cam}_calink']
    return cam_dir_mapping, cam_extparam_mapping, calibrate_mapping



