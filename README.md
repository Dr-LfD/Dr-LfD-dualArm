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
them through `${WS_ROOT}` (your workspace holding the sibling dependency repos),
`${REPO_ROOT}` (this repo, resolved automatically), and `${HOME}`. Copy
`.env.example` to `.env`, set `WS_ROOT`, and `source .env` before running. Unset
variables are left literal so a missing path fails loudly rather than silently
resolving to the wrong location. See [Configuration](docs/source/Configuration.md).

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
| Threading | `python examples/pybullet/aloha_real/scripts/interleaved_dmg_osc_plugin.py --task_name two_arm_threading` |
| Assembly | `python examples/pybullet/aloha_real/scripts/interleaved_dmg_osc_plugin.py --task_name two_arm_three_piece_assembly` |

Each run records a video under `lfd_tamp_vids/` and prints
`Execution results: {... 'task_success': True/False ...}`. See
[Quickstart](docs/source/Quickstart.md) for prerequisites and troubleshooting.

## TODO

- [x] TAMP for DMG simulation
- [ ] TAMP for real ALOHA robot (constraint check + contact detection)
- [ ] Policy learning
- [ ] Object-centric primitive learning
- [ ] Contact detector
