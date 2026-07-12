# Dr-LfD-dualArm

**Decompose and Reorganize**: Planning with Primitives and Visuomotor Policies Learned from Demonstrations.

Source code: https://github.com/Dr-LfD/Dr-LfD-dualArm

## What is DR-LfD

DR-LfD (**D**ecomposed and **R**eorganized skills **L**earned **f**rom
**D**emonstrations) fuses task-and-motion planning (TAMP) with imitation
learning. Human demonstrations are decomposed into atomic skills — visuomotor
policies and object-centric primitives — expressed in a TAMP-compatible form,
then reorganized via PDDL schema construction and an online plan-and-execute
loop. This turns long-horizon dexterous bimanual manipulation from a problem
needing exponential demonstration data into one solvable from a few skills.

Project page: https://dr-lfd.github.io/DR-LfD-website ·
Full documentation: [`docs/source`](docs/source/index.rst).

## Installation

Follow [docs/source/Installation.md](docs/source/Installation.md) for the
simulation route (conda env `dr-lfd`, `requirements.txt`, customized forks,
native builds).

## Configuration (paths)

External dependency paths are not hardcoded. Config files and scripts reference
them through `${REPO_ROOT}` (this repo, resolved automatically), `${WS_ROOT}`
(external workspace, only for real-robot data), and `${HOME}`. The learned-policy
dependencies are vendored submodules and auto-discovered, so the simulation route
needs no `.env`; copy `.env.example` to `.env` only for overrides or real-robot
data. Any other unset variable is left literal so a missing path fails loudly.
See [Configuration](docs/source/Configuration.md).

## Run the simulation examples

```bash
conda activate dr-lfd
export WS_ROOT="<your workspace root>"
export CUDA_VISIBLE_DEVICES="0"
export MUJOCO_GL="egl"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/examples/pybullet/aloha_real/scripts"
```

Two runnable bimanual DMG instances:

| Task | Command |
|---|---|
| Threading | `python examples/pybullet/aloha_real/scripts/interleaved_sim_osc_plugin.py --task_name two_arm_threading` |
| Assembly | `python examples/pybullet/aloha_real/scripts/interleaved_sim_osc_plugin.py --task_name two_arm_three_piece_assembly` |

Each run records a video under `lfd_tamp_vids/` and prints
`Execution results: {... 'task_success': True/False ...}`. See
[Quickstart](docs/source/Quickstart.md) for prerequisites and troubleshooting.

## Related modules

The learned-policy and perception components used by DR-LfD are maintained as
separate modules:

- [Customized Diffusion Policy](https://github.com/Dr-LfD/diffusion_policy_sg)
  —  diffusion-policy trained only on bimanual operation.
- [Contact Prediction](https://github.com/Dr-LfD/ContactPrediction) — per-edge
  wrist-camera contact prediction.
- [Equivariant Primitive Learning](https://github.com/Dr-LfD/Equiv_primitive)
  — SIM(3)-equivariant diffusion model for object-centric  primitives.
