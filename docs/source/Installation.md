# Installation

This guide covers the **simulation** route (robosuite/mujoco DMG tasks). For the
real-robot stack (segmentation perception, GPD grasping, ROS), see
[Real robot](Real_robot.md).

All paths below are relative to the repo root or use `${WS_ROOT}` (your external
workspace holding the sibling dependency repos) and `${REPO_ROOT}` (this repo).
Set these via the `.env` file (see [Configuration](Configuration.md)).

## 1. Clone the repo and submodules

```bash
git clone --recursive https://github.com/Dr-LfD/Dr-LfD-dualArm.git
cd Dr-LfD-dualArm
git submodule update --init --recursive
```

## 2. Create the conda environment

```bash
conda create -n dr-lfd python=3.9
conda activate dr-lfd
pip install -r requirements.txt
```

## 3. PyTorch (CUDA 12.1) and geometry extensions

Ensure CUDA 12.1 or above is available (`nvcc -V`), then install the torch stack
and its compiled extensions (kept out of `requirements.txt` because they must
match your CUDA build):

```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

## 4. Customized forks

The simulation route depends on customized forks of robosuite, dexmimicgen, and
robomimic. Clone them under `${WS_ROOT}` and install editable:

```bash
# robosuite 1.5.1 (switchable branch)
git clone https://github.com/Dr-LfD/robosuite_costomized.git -b switchable ${WS_ROOT}/robosuite
pip install -e ${WS_ROOT}/robosuite

# dexmimicgen  (clone to ${DEXMIMICGEN_ROOT}; configs reference it via env_dir)
git clone https://github.com/Dr-LfD/dexmimicgen -b playback_3d ${DEXMIMICGEN_ROOT}
pip install -e ${DEXMIMICGEN_ROOT}

# robomimic
git clone https://github.com/Dr-LfD/robomimic-customized.git -b pc ${WS_ROOT}/robomimic
pip install -e ${WS_ROOT}/robomimic
```

If a stale `robosuite` is shadowing the editable install, remove it first:

```bash
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
rm -rf "${SITE}/robosuite"
```

## 5. Learned-policy sibling repos

The DMG / Diffusion-Policy skills are imported from sibling repos (added to
`sys.path` at runtime, not pip-installed). Each is located by its own macro set
in `.env` (see [Configuration](Configuration.md)) — clone anywhere, then point
the macro at it:

```bash
git clone https://github.com/Dr-LfD/DP_customized.git <DIFFUSION_POLICY_ROOT>
git clone https://github.com/Dr-LfD/equibot_abstract.git <PRIMITIVE_LEARNING_ROOT>
# .env:
export DIFFUSION_POLICY_ROOT=/abs/path/to/Diffusion-Policy
export PRIMITIVE_LEARNING_ROOT=/abs/path/to/equibot_abstract
```

The SDP variants (`*_sdp.yaml`) additionally use `${SPHERICAL_DP_ROOT}`.

equibot's graph-network extensions are built against the torch+CUDA installed in
step 3:

```bash
pip install torch-scatter torch-cluster torch-geometric \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

## 6. Build native components

```bash
# Fast Downward (classical planner backend for PDDLStream)
./downward/build.py

# IKFast (analytic IK for the ALOHA arms)
cd examples/pybullet/utils/pybullet_tools/ikfast/aloha/
python setup.py
```

Rebuild IKFast whenever you change the Python environment.

## 7. Checkpoints

Place the DMG / Diffusion-Policy checkpoints for the two tasks where the task
configs expect them (see [Configuration](Configuration.md) for the config layout
and the `--dp_ckpt` / `--sg` runtime overrides).

You are now ready to run the simulation examples — see [Quickstart](Quickstart.md).

---

## Known version pitfalls

- **robosuite 1.5.1 requires mujoco 3.x.** The pinned `mujoco==3.3.4` in
  `requirements.txt` satisfies this; older `mujoco==2.3.2` raises
  `XML Error: Schema violation: unrecognized attribute: 'actuatorfrclimited'`.
- The dexmimicgen environment registration is only found when the robosuite fork
  is on its `switchable`/dexmimicgen-compatible branch (step 4).

## Building this documentation

```bash
pip install sphinx sphinx_rtd_theme recommonmark sphinx-autobuild sphinx_markdown_tables
cd docs && make html   # output in docs/build/html/
```
