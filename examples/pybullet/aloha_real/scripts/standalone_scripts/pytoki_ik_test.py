import time
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
from yourdfpy import URDF
import numpy as np

import sys
sys.path.append("/home/user/yzchen_ws/env")  
sys.path.append("/home/user/yzchen_ws/env/pyroki/examples")
import pyroki as pk
from viser.extras import ViserUrdf
import pyroki_snippets as pks

import jax
import jaxlie
import jaxls
from jax import lax
from jax import numpy as jnp


def get_jax_se3(wxyz, xyz):

    so3 = jaxlie.SO3.from_quaternion(wxyz)

    # For a full pose (SE3):
    se3 = jaxlie.SE3(jnp.concatenate([wxyz, xyz]))

    return se3

def get_link_pose(robot, jposes, link_name):
    link_poses = robot.forward_kinematics(jposes)
    link_index = robot.links.names.index(link_name)
    pose = link_poses[..., link_index, :]
    wxyz = pose[:4]
    xyz = pose[4:7]
    # se3 = get_jax_se3(wxyz, xyz)
    # return se3
    return wxyz, xyz

def main():
    """Main function for bimanual IK."""

    # urdf = load_robot_description("yumi_description")
    # target_link_names = ["yumi_link_7_r", "yumi_link_7_l"]

    dual_panda_urdf = "examples/pybullet/utils/models/franka_description/robots/dmg_dual_panda.urdf"
    urdf = URDF.load(dual_panda_urdf)
    target_link_names = ['panda2_ee_link', 'panda1_ee_link']

    # Create robot.
    robot = pk.Robot.from_urdf(urdf)

    # Set up visualizer.
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    tgt0_position = np.array([0.11, -0.3, 0.96])
    tgt1_position = np.array([0.01, 0.3, 0.86])

    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    # Create interactive controller with initial position.
    ik_target_0 = server.scene.add_transform_controls(
        "/ik_target_0", scale=0.2, position=tgt0_position, wxyz=(0, 0, 1, 0)
    )
    ik_target_1 = server.scene.add_transform_controls(
        "/ik_target_1", scale=0.2, position=tgt1_position, wxyz=(0, 0, 1, 0)
    )
    while True:


        # Solve IK.
        start_time = time.time()
        solution = pks.solve_ik_with_multiple_targets(
            robot=robot,
            target_link_names=target_link_names,
            target_positions=np.array([ik_target_0.position, ik_target_1.position]),
            target_wxyzs=np.array([ik_target_0.wxyz, ik_target_1.wxyz]),
        )

        ## the problem: if set one arm, another arm will move without control
        # solution = pks.solve_ik(
        #     robot=robot,
        #     target_link_name=target_link_names[0],
        #     target_position=np.array(ik_target_0.position),
        #     target_wxyz=np.array(ik_target_0.wxyz),
        # )

        # Update timing handle.
        elapsed_time = time.time() - start_time
        # print(f"IK solved in {elapsed_time * 1000:.2f} ms")
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)

        eef_pose = get_link_pose(robot, solution, target_link_names[0])

        # Update visualizer.
        urdf_vis.update_cfg(solution)

        # tgt0_position += np.random.uniform(-0.05, 0.05, size=3)
        # tgt1_position += np.random.uniform(-0.05, 0.05, size=3)

def roberts_sequence(num_points, dim, root):
    # From https://gist.github.com/carlosgmartin/1fd4e60bed526ec8ae076137ded6ebab.
    basis = 1 - (1 / root ** (1 + jnp.arange(dim)))

    n = jnp.arange(num_points)
    x = n[:, None] * basis[None, :]
    x, _ = jnp.modf(x)

    return x


class PyrokiIkHelper:
    def __init__(self, urdf_file):
        urdf = URDF.load(urdf_file)
        assert urdf.validate()
        self.robot = pk.Robot.from_urdf(urdf_file)
        
    def solve_ik(self, target_wxyz: jax.Array, target_position: jax.Array) -> jax.Array:
        num_seeds_init: int = 64
        num_seeds_final: int = 4

        total_steps: int = 16
        init_steps: int = 6

        def solve_one(
            initial_q: jax.Array, lambda_initial: float | jax.Array, max_iters: int
        ) -> tuple[jax.Array, jaxls.SolveSummary]:
            """Solve IK problem with a single initial condition. We'll vmap
            over initial_q to solve problems in parallel."""
            joint_var = robot.joint_var_cls(0)
            factors = [
                # pk.costs.pose_cost(
                pk.costs.pose_cost_analytic_jac(
                    robot,
                    joint_var,
                    jaxlie.SE3.from_rotation_and_translation(
                        jaxlie.SO3(target_wxyz), target_position
                    ),
                    self.target_link_index,
                    pos_weight=10.0,
                    ori_weight=5.0,
                ),
                pk.costs.limit_cost(
                    robot,
                    joint_var,
                    weight=50.0,
                ),
            ]
            sol, summary = (
                jaxls.LeastSquaresProblem(factors, [joint_var])
                .analyze()
                .solve(
                    initial_vals=jaxls.VarValues.make(
                        [joint_var.with_value(initial_q)]
                    ),
                    verbose=False,
                    linear_solver="dense_cholesky",
                    termination=jaxls.TerminationConfig(
                        max_iterations=max_iters,
                        early_termination=False,
                    ),
                    trust_region=jaxls.TrustRegionConfig(lambda_initial=lambda_initial),
                    return_summary=True,
                )
            )
            return sol[joint_var], summary

        vmapped_solve = jax.vmap(solve_one, in_axes=(0, 0, None))

        # Create initial seeds, but this time with quasi-random sequence.
        robot = self.robot
        initial_qs = robot.joints.lower_limits + roberts_sequence(
            num_seeds_init, robot.joints.num_actuated_joints, self.root
        ) * (robot.joints.upper_limits - robot.joints.lower_limits)

        # Optimize the initial seeds.
        initial_sols, summary = vmapped_solve(
            initial_qs, jnp.full(initial_qs.shape[:1], 10.0), init_steps
        )

        # Get the best initial solutions.
        best_initial_sols = jnp.argsort(
            summary.cost_history[jnp.arange(num_seeds_init), -1]
        )[:num_seeds_final]

        # Optimize more for the best initial solutions.
        best_sols, summary = vmapped_solve(
            initial_sols[best_initial_sols],
            summary.lambda_history[jnp.arange(num_seeds_init), -1][best_initial_sols],
            total_steps - init_steps,
        )
        return best_sols[
            jnp.argmin(
                summary.cost_history[jnp.arange(num_seeds_final), summary.iterations]
            )
        ]

    def forward_kinematics(self, q: jax.Array | np.ndarray) -> jax.Array:
        return self.robot.forward_kinematics(jnp.asarray(q))[self.target_link_index]
    
        
if __name__ == "__main__":
    main()