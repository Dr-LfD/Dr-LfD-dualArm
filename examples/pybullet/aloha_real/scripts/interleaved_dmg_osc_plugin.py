import argparse
import os

from interleaved_dmg_plugin import DMG_planner, root_path
from osc_execution import OSCExecutionMixin


class DMG_OSC_planner(OSCExecutionMixin, DMG_planner):
    """DMG planner that executes via OSC_POSE absolute world-frame control.

    Learned-grasp trajectories are commanded directly as dense Cartesian EE
    poses (no IK branch-flip / "兜一圈" loop); motion-planning trajectories keep
    collision-free joint-space RRT planning but are FK'd to EE poses at execution
    time. Both arms run under a single OSC_POSE abs controller per stage. The
    dual-Panda tool frame matches the default GRIP_SITE_CORRECTION, so only the
    stream-side flag is set here.
    """

    def get_extra_tamp_kwargs(self):
        # Route OSC EE-trajectory generation to the streams via the base hook so
        # get_imitate_traj_fn / get_plan_motion_fn take their ee_traj_mode branch.
        return {'ee_traj_mode': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run interleaved DMG task with OSC_POSE absolute EE execution.'
    )
    parser.add_argument(
        '--task_name', type=str,
        default='two_arm_three_piece_assembly',
        help='Task name to use (default: two_arm_three_piece_assembly)'
    )
    parser.add_argument(
        '--sg', action='append', default=[], metavar='KEY=VALUE',
        help='Override an sg_params field (repeatable), e.g. '
             '--sg biop_ckpt_name=logs/train/dmg_threading/ckpt_perskill_jpose_near.pth'
    )
    parser.add_argument(
        '--dp_ckpt', type=str, default=None,
        help='Override the diffusion-policy checkpoint '
             '(LfD_params.DP_input[<task_name>]).'
    )
    args = parser.parse_args()

    task_name = args.task_name
    yaml_path = os.path.join(
        root_path,
        f'examples/pybullet/aloha_real/openworld_aloha/configs/dmg_cfgs/{task_name}.yaml'
    )

    from repo_paths import load_yaml
    parameters = load_yaml(yaml_path)

    # In-memory checkpoint overrides for sweeps. The DMG route consumes this
    # parameters dict directly (no YAML re-read), so these take effect: biop is
    # read from sg_params (interleaved_robosuite_base.py update_equivSkill_wrapper)
    # and the DP checkpoint from LfD_params.DP_input (network_loader.get_lfd_wrapper).
    # Numeric values are coerced to float; a path like biop_ckpt_name stays a string.
    for kv in args.sg:
        key, val = kv.split('=', 1)
        try:
            val = float(val)
        except ValueError:
            pass
        parameters['sg_params'][key] = val

    if args.dp_ckpt is not None:
        parameters['LfD_params']['DP_input'][task_name] = args.dp_ckpt

    planner = DMG_OSC_planner(parameters)
    exe_info = planner()
    print(f'Execution results: {exe_info}')
