# Quickstart (simulation)

Run the two bimanual DMG tasks in the robosuite/mujoco simulator. For the
real-robot route see [Real robot](Real_robot.md) (optional).

## Prerequisites

- [Installation](Installation.md) complete: conda env `dr-lfd`, customized forks,
  Fast Downward and IKFast built.
- The learned-policy sibling repos (Diffusion-Policy, equibot) cloned to the
  locations referenced by the task config (see [Configuration](Configuration.md)).
- DMG / Diffusion-Policy checkpoints present at the paths the task config expects.

## Environment

```bash
conda activate dr-lfd
export WS_ROOT="<your workspace root>"      # external repos live here
export CUDA_VISIBLE_DEVICES="0"
export MUJOCO_GL="egl"                       # headless rendering
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/examples/pybullet/aloha_real/scripts"
```

`${REPO_ROOT}` resolves to this repository automatically; `WS_ROOT` you set
yourself (see [Configuration](Configuration.md)).

## Run

| Task | Command |
|---|---|
| Two-arm threading | see below |
| Two-arm three-piece assembly | see below |

```bash
# Threading
python examples/pybullet/aloha_real/scripts/interleaved_dmg_osc_plugin.py \
    --task_name two_arm_threading

# Three-piece assembly
python examples/pybullet/aloha_real/scripts/interleaved_dmg_osc_plugin.py \
    --task_name two_arm_three_piece_assembly
```

### Runtime overrides

- `--sg KEY=VALUE` (repeatable) overrides a `sg_params` field without editing the
  YAML, e.g. `--sg equi_ckpt_name=logs/train/dmg_threading/inhand-fps.pth`.
- `--dp_ckpt PATH` overrides the Diffusion-Policy checkpoint.

## Expected output

- A video is recorded under `lfd_tamp_vids/`.
- The run prints `Execution results: {... 'task_success': True/False ...}`.

## Troubleshooting

- Gripper rotated 90° on every command → see [Gripper frame conventions](Gripper_frame_conventions.md)
  and [Troubleshooting](Troubleshooting.md) (PandaGripper `Rz(−90°)` correction).
- `XML Error: ... 'actuatorfrclimited'` → mujoco/robosuite version mismatch; see
  the "Known version pitfalls" in [Installation](Installation.md).
