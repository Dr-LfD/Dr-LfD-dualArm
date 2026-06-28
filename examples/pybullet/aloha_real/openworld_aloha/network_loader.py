import os, sys

from .open_world_utils import get_skillwise_sgs

def get_lfd_wrapper(cfg,  with_planning = True):
    LfD_params = cfg['LfD_params']
    if LfD_params['lfd_alg'] is None:
        return None


    checkpoint_dict = LfD_params['DP_input']
    if LfD_params['lfd_alg'] == 'ACT':
        raise NotImplementedError("multiskill ACT is not implemented")
        # include ACT
        act_path = os.path.join(root_path, '..', 'ACT')
        sys.path.append(act_path)
        os.chdir(act_path)
        from eval_act_wrapper import ACT_Evaluator

        self.lfd = ACT_Evaluator(task_name = para['task_name'], with_planning=True)
        lfd_env = None
    elif LfD_params['lfd_alg'] == 'DP':
        # include DP
        dp_path = LfD_params['DP_path']
        sys.path.append(dp_path)
        os.chdir(dp_path)
        env_type = LfD_params['env_type']
        output_path = os.path.join(dp_path, LfD_params['DP_output'])
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if env_type != 'dmg':
            raise NotImplementedError(f"env_type {env_type!r} is no longer supported under lfd_alg='DP'; only 'dmg' remains")
        from iterative_dmg_runner import Robosuite_Evaluator as DP_wrapper

        lfd_wrapper = DP_wrapper(checkpoint_dict = checkpoint_dict, output = output_path, max_timesteps = LfD_params['DP_max_timesteps'], num_inference_steps = LfD_params['DP_num_inference_steps'], scale = LfD_params['DP_scale'], with_planning = True, record = LfD_params['vid_record'], render_obs_keys = LfD_params["render_obs_keys"])
    elif LfD_params['lfd_alg'] == 'SDP':
        # include DP
        dp_path = LfD_params['DP_path']
        sys.path.append(dp_path)
        os.chdir(dp_path)
        env_type = LfD_params['env_type']
        output_path = os.path.join(dp_path, LfD_params['DP_output'])
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        from sdp_dmg_runner import SDP_DMG_Evaluator as DP_wrapper

        lfd_wrapper = DP_wrapper(checkpoint_dict = checkpoint_dict, output = output_path, max_timesteps = LfD_params['DP_max_timesteps'], num_inference_steps = LfD_params['DP_num_inference_steps'],  with_planning = True, record = LfD_params['vid_record'], render_obs_keys = LfD_params["render_obs_keys"])
    else:
        raise NotImplementedError(f"lfd_alg {LfD_params['lfd_alg']} is not implemented")
    return lfd_wrapper

def categorize_skill(skill_name):
    is_bimanual = 'bi' in skill_name
    if is_bimanual:
        return 'bimanual'
    
    picking_kw = ['grasp', 'pick'] # 
    placing_kw = ['place', 'release', 'drop']
    for kw in picking_kw:
        if kw in skill_name:
            return 'ATTACH'

    for kw in placing_kw:
        if kw in skill_name:
            return 'DETACH'

    return 'nonprehensile'  ## should output a traj only


def update_alohaMultiSkill_wrapper(para, skill_names, skillwise_sgs):
    """Load one Equibot policy wrapper per skill for the real ALOHA robot (trajectory mode).

    Used by the real-robot plugin. The heavy ``equiv_primitive``/``hydra`` deps are imported lazily so
    importing ``network_loader`` stays free of those packages.
    """
    primitive_learning_path = para['primitive_learning_path']
    sys.path.append(primitive_learning_path)
    from equiv_primitive.policies.aloha_wrapper import pddl_wrapper
    from hydra import compose, initialize
    import hydra
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    equivSkill_info_dict = {}
    task_name = para['task_name']
    for prefix_key in skill_names:
        equivSkill_info_dict[prefix_key] = {}
        equivSkill_info_dict[prefix_key]['task_name'] = task_name

        cfg_name = f'diffGen_{prefix_key}'
        with initialize(version_base=None, config_path="configs", job_name="test_app"):
            cfg = compose(config_name=cfg_name,
                          overrides=["prefix=" + prefix_key, "mode=inference", "use_wandb=false"])

        agent_wrapper = pddl_wrapper(dataset_path=None, cfg=cfg)
        equivSkill_info_dict[prefix_key]['tamp_wrapper'] = agent_wrapper

        skillwise_sgs[prefix_key]['skill_type'] = para[prefix_key]['skill_type']
        equivSkill_info_dict[prefix_key]['skillwise_sgs'] = skillwise_sgs
        equivSkill_info_dict[prefix_key]['skill_names'] = para[prefix_key]['skill_names']

    return equivSkill_info_dict


def update_alohaMultiEquivSkill_wrapper(para, env_names, skillwise_sgs):
    """Load Equibot unimanual + bimanual (biop) policy wrappers per env for the real ALOHA robot.

    Companion of :func:`update_alohaMultiSkill_wrapper`; resolves checkpoints per env and records
    the object-centric mode ('traj' vs 'grasp') inferred from each wrapper's dataset type.
    """
    primitive_learning_path = para['primitive_learning_path']
    sys.path.append(primitive_learning_path)
    from equiv_primitive.policies.aloha_wrapper import pddl_wrapper
    from hydra import compose, initialize
    import hydra
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    equivSkill_info_dict = {}
    task_name = para['task_name']
    for prefix_key in env_names:
        equivSkill_info_dict[prefix_key] = {}
        equivSkill_info_dict[prefix_key]['task_name'] = task_name
        equivSkill_info_dict[prefix_key]['learned_grasp_traj_resolution_deg'] = para[prefix_key].get(
            'learned_grasp_traj_resolution_deg', 2,
        )

        # unimanual ckpt (and biop kp) — fall back to a hydra config when no ckpt is given
        ckpt_path = para[prefix_key].get('equi_ckpt_name', None)
        biop_ckpt_path = para[prefix_key].get('biop_ckpt_name', None)

        if ckpt_path is None:
            equi_config_name = f'diffGen_{prefix_key}'
            with initialize(version_base=None, config_path="configs", job_name=equi_config_name):
                cfg = compose(config_name=equi_config_name,
                              overrides=[f"prefix={prefix_key}", "mode=inference", "use_wandb=false"])
            agent_wrapper = pddl_wrapper(dataset_path=None, cfg=cfg)
        else:
            agent_wrapper = pddl_wrapper(dataset_path=None, ckpt_path=ckpt_path)
        equivSkill_info_dict[prefix_key]['tamp_wrapper'] = agent_wrapper

        if biop_ckpt_path is not None:
            agent_biop_wrapper = pddl_wrapper(dataset_path=None, ckpt_path=biop_ckpt_path)
            equivSkill_info_dict[prefix_key]['biop_wrapper'] = agent_biop_wrapper

        equivSkill_info_dict[prefix_key]['skillwise_sgs'] = skillwise_sgs[prefix_key]
        equivSkill_info_dict[prefix_key]['skill_names'] = list(skillwise_sgs[prefix_key].keys())

        dataset_type = equivSkill_info_dict[prefix_key]['tamp_wrapper'].cfg.data.dataset.dataset_type
        obj_centric_mode = 'traj' if 'traj' in dataset_type else 'grasp'
        equivSkill_info_dict[prefix_key]['obj_centric_mode'] = obj_centric_mode

    return equivSkill_info_dict


def update_equivSkill_wrapper(sg_params, para, get_sgs_from_hdf5_fn = None):
    # if skill_name is not None:
    #     sg_params = para[env_name]
    # else:
    #     sg_params = para['sg_params']
    primitive_learning_path = para['primitive_learning_path']
    sys.path.append(primitive_learning_path) 
    from equiv_primitive.policies.aloha_wrapper import pddl_wrapper
    from hydra import compose, initialize
    import hydra
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    equivSkill_info_dict = {}

    equi_config_name = sg_params['equi_config_name']

    ## if per_skill, then the observation will be 'skill:pc'
    if 'per_skill' in equi_config_name:
        prefix_keys = ['per_skill']
    elif 'two_arm' in equi_config_name:
        prefix_keys = ['bimanual_0']

    task_name = sg_params['task_name']
    ckpt_path = sg_params.get('equi_ckpt_name', None)
    biop_ckpt_path = sg_params.get('biop_ckpt_name', None)
    learned_grasp_traj_resolution_deg = sg_params.get(
        'learned_grasp_traj_resolution_deg',
        1.2,
    ) ## we do not have para['prefix_key'] in robosuite version

    shape_meta = None
    for prefix_key in prefix_keys: 
        dataset_path = os.path.join(primitive_learning_path, 'data', prefix_key)
 
        equivSkill_info_dict[prefix_key] = {}
        equivSkill_info_dict[prefix_key]['task_name'] = task_name
        equivSkill_info_dict[prefix_key]['learned_grasp_traj_resolution_deg'] = learned_grasp_traj_resolution_deg
        ## EE-traj densification controls: copied only when set in sg_params, so the
        ## consumer-side defaults in openworld_streams stay the single source of truth.
        for ee_traj_key in ('ee_traj_step_m', 'ee_traj_step_rad', 'ee_traj_steps_per_waypoint'):
            if ee_traj_key in sg_params:
                equivSkill_info_dict[prefix_key][ee_traj_key] = sg_params[ee_traj_key]

        if ckpt_path is None:
            with initialize(version_base=None, config_path="configs", job_name=equi_config_name):
                cfg = compose(config_name=equi_config_name, overrides=[f"prefix={prefix_key}", "mode=inference", "use_wandb=false"])
            equivSkill_info_dict[prefix_key]['tamp_wrapper'] = pddl_wrapper(dataset_path, cfg = cfg)
        else:
            agent_wrapper = pddl_wrapper(dataset_path, ckpt_path = ckpt_path)
            equivSkill_info_dict[prefix_key]['tamp_wrapper'] = agent_wrapper

        if biop_ckpt_path is not None:
            agent_biop_wrapper = pddl_wrapper(dataset_path, ckpt_path = biop_ckpt_path)
            equivSkill_info_dict[prefix_key]['biop_wrapper'] = agent_biop_wrapper

        ### compatible to different versions. TODO: train with the latest version
        if 'skillwise_sgs_path' in sg_params:
            skill_names = list(sg_params['interested_skills'].keys())
            hdf5_path = os.path.join(para['env_dir'], sg_params['skillwise_sgs_path'])
            skillwise_sgs = get_sgs_from_hdf5_fn(hdf5_path)
        else:
            skillwise_sgs = agent_wrapper.agent.get_skillwise_sgs_from_statistics(task_name)
            # The equi (grasp) checkpoint may not carry the bimanual keypose scene
            # graph; the biop checkpoint does. Each checkpoint owns its own skills'
            # graphs, so fill any skill the equi checkpoint lacks from the biop
            # wrapper (equi wins on overlap, matching the loaded grasp policy).
            if biop_ckpt_path is not None:
                biop_sgs = agent_biop_wrapper.agent.get_skillwise_sgs_from_statistics(task_name)
                for biop_skill_name, biop_sg in biop_sgs.items():
                    skillwise_sgs.setdefault(biop_skill_name, biop_sg)
            skill_names = list(skillwise_sgs.keys())

        for skill_name in skill_names:
            skillwise_sgs[skill_name]['skill_type'] = categorize_skill(skill_name)
        
        equivSkill_info_dict[prefix_key]['skillwise_sgs'] = skillwise_sgs
        equivSkill_info_dict[prefix_key]['skill_names'] = skill_names   

        dataset_type= equivSkill_info_dict[prefix_key]['tamp_wrapper'].cfg.data.dataset.dataset_type
        obj_centric_mode = 'traj' if 'traj' in dataset_type else 'grasp'
        equivSkill_info_dict[prefix_key]['obj_centric_mode'] = obj_centric_mode

        shape_meta = equivSkill_info_dict[prefix_key]['tamp_wrapper'].cfg.shape_meta

    return  equivSkill_info_dict, shape_meta
