import os
import json
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import argparse

from postprocess_template_base import load_yaml_params, parse_prior_graphs,  postprocess_for_sgs


def postprocess_dmg_data(args, target_task_name):

    # target_task_name = 'two_arm_three_piece_assembly'
    # target_task_name = 'two_arm_threading'
    # hdf5_dir = '/home/user/yzchen_ws/imitation_learning/dexmimicgen/datasets/generated/'
    input_hdf5_path = os.path.join(args.hdf5_dir, f"{target_task_name}_pc_instance50.hdf5")
    if not os.path.exists(input_hdf5_path):
        print(f"------{input_hdf5_path.split('/')[-1]} not found!------")
        raise FileNotFoundError(f"------{input_hdf5_path.split('/')[-1]} not found!------")
    
    output_hdf5_path = input_hdf5_path.replace('.hdf5', f'_sg_{args.n_playback}.hdf5')
    if os.path.exists(output_hdf5_path) and args.debug_ep_id is None:
        print(f"Output file {output_hdf5_path} already exists. Please remove it or change the output path.")
        return
    
    cfg_path = os.path.join ('examples/pybullet/aloha_real/openworld_aloha/configs/dmg_cfgs', target_task_name + '.yaml')
    cfg_all = load_yaml_params(cfg_path)
    sg_params = cfg_all['sg_params']
    sg_params['input_hdf5_path'] = input_hdf5_path
    sg_params['output_hdf5_path'] = output_hdf5_path
    sg_params['task_name'] = target_task_name
    # Given JSON data (converted to Python dict)
    per_rbt_sg_json_path = os.path.join ('examples/pybullet/aloha_real/openworld_aloha/configs/dmg_cfgs', f'{target_task_name}_changes.json')

    with open(per_rbt_sg_json_path, 'r') as f:
        per_rbt_sg_data = json.load(f)
    prior_graphs = parse_prior_graphs(per_rbt_sg_data)

    postprocess_for_sgs(sg_params, prior_graphs = prior_graphs, n_playback = args.n_playback, num_workers = args.num_workers, visualize= args.visualize, debug_ep_id = args.debug_ep_id) #, 

    print(f"------{target_task_name} postprocessed!------")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task_suite_name",
        type=str,
        default='DMG',
        help="type of dataset",
    )

    parser.add_argument(
        "--hdf5_dir",
        type=str,
        default='/home/user/yzchen_ws/imitation_learning/dexmimicgen/datasets/generated/',
        help="dir to input hdf5 dataset",
    )
    parser.add_argument(
        "--target_task_name",
        type=str,
        default='two_arm_three_piece_assembly',
        help="path to input hdf5 dataset",
    )

    # specify number of demos to process - useful for debugging conversion with a handful
    # of trajectories
    parser.add_argument(
        "--n_playback",
        type=int,
        default=50,
        help="(optional) stop after n trajectories are processed",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="(optional) visualize the postprocessed skill graphs",
    )

    parser.add_argument(
        "--debug_ep_id",
        type=int,
        default=None,
        help="(optional) debug episode id",
    )

    args = parser.parse_args()

    if args.task_suite_name == 'DMG':
        postprocess_dmg_data(args, args.target_task_name)
    else:
        raise NotImplementedError(
            "Unsupported task_suite_name '{}'; only 'DMG' is handled.".format(
                args.task_suite_name
            )
        )
            
