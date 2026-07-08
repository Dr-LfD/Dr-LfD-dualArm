# Quickstart (simulation)

Run the two bimanual DMG tasks in the robosuite/mujoco simulator. For the
real-robot route see [Real robot](Real_robot.md) (optional).

## Prerequisites

- [Installation](Installation.md) complete: conda env `dr-lfd`, customized forks,
  Fast Downward and IKFast built.
- Submodules initialized (`git submodule update --init`); `diffusion_policy_sg`,
  `primitive_learning` and `contact_detector` are then auto-discovered, so the sim
  route needs no `.env` (see [Configuration](Configuration.md)).
- Checkpoints downloaded: `python download_ckpts.py` (into `ckpt/`).

## Environment

```bash
conda activate dr-lfd
cd <repo root>                               # run commands from the repo root
export CUDA_VISIBLE_DEVICES="0"
export MUJOCO_GL="egl"                       # headless rendering
export PYTHONPATH="$(pwd):$(pwd)/examples/pybullet/aloha_real/scripts"
```

Config `${REPO_ROOT}` placeholders resolve automatically at load time; run from
the repo root so the relative config paths resolve too.

## Run

| Task | Command |
|---|---|
| Two-arm threading | see below |
| Two-arm three-piece assembly | see below |

```bash
# Threading
python examples/pybullet/aloha_real/scripts/interleaved_sim_osc_plugin.py \
    --task_name two_arm_threading

# Three-piece assembly
python examples/pybullet/aloha_real/scripts/interleaved_sim_osc_plugin.py \
    --task_name two_arm_three_piece_assembly
```

### Runtime overrides

- `--sg KEY=VALUE` (repeatable) overrides a `sg_params` field without editing the
  YAML, e.g. `--sg equi_ckpt_name='${REPO_ROOT}/ckpt/dmg_threading/equi.pth'`
  (`${VAR}`/`~` are expanded).
- `--dp_ckpt PATH` overrides the Diffusion-Policy checkpoint.

## Expected output

- A video is recorded under `lfd_tamp_vids/`.
- The run prints `Execution results: {... 'task_success': True/False ...}`.

## Troubleshooting

- Gripper rotated 90° on every command → see [Gripper frame conventions](Gripper_frame_conventions.md)
  and [Troubleshooting](Troubleshooting.md) (PandaGripper `Rz(−90°)` correction).
- `XML Error: ... 'actuatorfrclimited'` → mujoco/robosuite version mismatch; see
  the "Known version pitfalls" in [Installation](Installation.md).
