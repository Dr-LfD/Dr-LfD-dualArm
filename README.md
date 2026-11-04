# Dr-LfD-dualArm

<p align="center">
  <img src="https://dr-lfd.github.io/DR-LfD-website/paper-figures/tro-teaser.png" alt="DR-LfD system overview" width="100%">
</p>

**Decompose and Reorganize**: Planning with Primitives and Visuomotor Policies Learned from Demonstrations.
Accepted to *IEEE Transactions on Robotics* (T-RO).

[Yizhou Chen](https://jinxer000.github.io/),
[Hang Xu](https://scholar.google.com/citations?user=_tNUciIAAAAJ),
[Dongjie Yu](https://manutdmoon.github.io/),
[Yupu Lu](https://scholar.google.com/citations?user=sodxBZkAAAAJ),
Tengye Xu,
[Zeqing Zhang](https://benjizhang.github.io/),
[Wei Zhang](http://www.wzhanglab.site/),
Yi Ren,
[Ben M. Chen](http://www.mae.cuhk.edu.hk/~bmchen/),
[Jia Pan](https://www.cds.hku.hk/staff/pan-jia/)

[![Project Page](https://img.shields.io/badge/Project-Page-2f80ed)](https://dr-lfd.github.io/DR-LfD-website/)
[![Paper](https://img.shields.io/badge/arXiv-2607.25397-b31b1b)](https://arxiv.org/abs/2607.25397)
[![Code](https://img.shields.io/badge/GitHub-DR--LfD-181717?logo=github)](https://github.com/Dr-LfD/Dr-LfD-dualArm)
[![Checkpoints](https://img.shields.io/badge/Hugging%20Face-Checkpoints-ffcc4d?logo=huggingface)](https://huggingface.co/ConvexBuchi/dr-lfd-ckpt)
[![Video](https://img.shields.io/badge/Demo-Video-8e44ad)](#rollout-videos)

## Rollout videos

Click a preview to watch the full rollout on the
[project website](https://dr-lfd.github.io/DR-LfD-website/).

| Threading | Three-piece assembly |
|:---:|:---:|
| <a href="https://dr-lfd.github.io/DR-LfD-website/libero_dmg/threading.mp4"><img src="https://dr-lfd.github.io/DR-LfD-website/posters/libero_dmg/threading.jpg" alt="DR-LfD threading rollout" width="100%"></a> | <a href="https://dr-lfd.github.io/DR-LfD-website/libero_dmg/assemble.mp4"><img src="https://dr-lfd.github.io/DR-LfD-website/posters/libero_dmg/assemble.jpg" alt="DR-LfD three-piece assembly rollout" width="100%"></a> |

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

## Pretrained checkpoints

Download the released visuomotor-policy, equivariant-primitive, and bimanual
keypose checkpoints from
[Hugging Face](https://huggingface.co/ConvexBuchi/dr-lfd-ckpt):

```bash
python download_ckpts.py                  # threading and assembly
python download_ckpts.py --task threading # one task only
python download_ckpts.py --task assembly
```

The files are placed under `ckpt/dmg_threading/` and `ckpt/dmg_assembly/`,
matching the paths used by the task configurations.

## Configuration (paths)

External dependency paths are not hardcoded. Config files and scripts reference
them through `${REPO_ROOT}` (this repo, resolved automatically), `${WS_ROOT}`
(external workspace, only for real-robot data), and `${HOME}`. The learned-policy
dependencies are vendored submodules and auto-discovered, so the simulation route
needs no `.env`; copy `.env.example` to `.env` only for overrides or real-robot
data. Any other unset variable is left literal so a missing path fails loudly.
See [Configuration](docs/source/Configuration.md).

## Training

Training data is not bundled with the repository. Place each dataset at the
path declared by its task YAML, or update `dataset_path` in that YAML before
launching.

### Visuomotor policies

The threading and assembly visuomotor policies use the scene-graph-conditioned
diffusion-transformer workspace:

```bash
cd diffusion_policy_sg

# threading
python train.py \
    --config-name=train_robomimic_image_sg_transformer \
    task=two_arm_threading_image_sg

# three-piece assembly
python train.py \
    --config-name=train_robomimic_image_sg_transformer \
    task=two_arm_three_piece_assembly_image_sg
```

### Object-centric primitives

The grasp primitives use one SIM(3)-equivariant conditional model per task:

```bash
cd primitive_learning

# threading
python -m equiv_primitive.policies.train_skills \
    --config-name dmg_threading_per_skill \
    prefix=dmg_threading \
    data.dataset.path=/path/to/data/dmg_threading/

# three-piece assembly
python -m equiv_primitive.policies.train_skills \
    --config-name dmg_assembly_per_skill \
    prefix=dmg_assembly \
    data.dataset.path=/path/to/data/dmg_assembly/
```

Weights & Biases logging is enabled by default. Pass `use_wandb=false` to the
primitive commands to disable it. See
[`primitive_learning/README.md`](primitive_learning/README.md) for model and
evaluation details.

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

Each run records a video under `logs/threading/dp/` or `logs/assembly/dp/` and prints
`Execution results: {... 'task_success': True/False ...}`. See
[Quickstart](docs/source/Quickstart.md) for prerequisites and troubleshooting. If everything runs well, you will get around 70%--90% success rate on these two tasks. 

## Related modules

The learned-policy and perception components used by DR-LfD are maintained as
separate modules:

- [Customized Diffusion Policy](https://github.com/Dr-LfD/diffusion_policy_sg)
  —  diffusion-policy trained only on bimanual operation.
- [Contact Prediction](https://github.com/Dr-LfD/ContactPrediction) — per-edge
  wrist-camera contact prediction.
- [Equivariant Primitive Learning](https://github.com/Dr-LfD/Equiv_primitive)
  — SIM(3)-equivariant diffusion model for object-centric  primitives.

## Citation

```bibtex
@article{chen2026decompose,
  title   = {Decompose and Reorganize: Planning with Primitives and Visuomotor Policies Learned from Demonstrations},
  author  = {Chen, Yizhou and Xu, Hang and Yu, Dongjie and Lu, Yupu and Xu, Tengye and Zhang, Zeqing and Zhang, Wei and Ren, Yi and Chen, Ben M. and Pan, Jia},
  journal = {arXiv preprint arXiv:2607.25397},
  year    = {2026}
}
```
