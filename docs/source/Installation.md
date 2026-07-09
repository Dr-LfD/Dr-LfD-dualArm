# Installation

This guide covers the **simulation** route (robosuite/mujoco DMG tasks). For the
real-robot stack (segmentation perception, GPD grasping, ROS), see
[Real robot](Real_robot.md).

Paths below are relative to the repo root or use `${REPO_ROOT}` (this repo, auto-set)
and `${WS_ROOT}` (your external workspace, only for real-robot data). The
learned-policy dependencies are vendored submodules and need no `.env` for the sim
route (see [Configuration](Configuration.md)).

## 1. Clone the repo and submodules

```bash
git clone --recursive https://github.com/Dr-LfD/Dr-LfD-dualArm.git
cd Dr-LfD-dualArm
git submodule update --init --recursive
```

## 2. Create the conda environment and install PyTorch (CUDA 12.1)

Ensure CUDA 12.1 or above is available (`nvcc -V`). Install the torch stack
**before** the next step, so the git-installed forks in `requirements.txt`
resolve against this CUDA build rather than a default PyPI wheel:

```bash
conda create -n dr-lfd python=3.9
conda activate dr-lfd
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121
```

## 3. Python dependencies (incl. customized robosuite / robomimic forks)

`requirements.txt` pins the PyPI runtime deps and git-installs the customized
robosuite (`switchable`) and robomimic (`pc`) forks. Then build pytorch3d against
the torch+CUDA installed above:

```bash
pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

## 4. dexmimicgen fork

dexmimicgen must be installed **editable from a clone**: the DMG runners import
its top-level `scripts` and `dexmimicgen.utils` packages at runtime, which a plain
wheel install does not expose. Importing dexmimicgen also registers its robosuite
envs.

```bash
git clone https://github.com/JINXER000/dexmimicgen -b playback_3d <path>
pip install -e <path> --config-settings editable_mode=compat
```

`editable_mode=compat` puts the repo root on `sys.path` so the bare top-level
`scripts` package resolves — modern (PEP 660) editable installs otherwise expose
only declared packages. No `${DEXMIMICGEN_ROOT}` env var is needed; that macro is
only referenced by the experimental `*_sdp.yaml` configs.

If a stale `robosuite` is shadowing the editable install, remove it first:

```bash
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
rm -rf "${SITE}/robosuite"
```

## 5. Learned-policy sibling repos

The learned-policy code repos — primitive learning (`primitive_learning`), the
contact predictor (`contact_detector`), and Diffusion-Policy
(`diffusion_policy_sg`) — ship as submodules of the release umbrella and are
imported from `sys.path` at runtime (not pip-installed). Step 1's `--recursive`
clone already populated them; if you cloned without it, initialize them now:

```bash
git submodule update --init
```

`${PRIMITIVE_LEARNING_ROOT}`, `${CONTACT_PREDICTION_ROOT}` and
`${DIFFUSION_POLICY_ROOT}` default to these vendored submodules automatically —
no `.env` entry needed unless you want to point at a separate development
checkout.

The experimental SDP variants (`*_sdp.yaml`) additionally use
`${SPHERICAL_DP_ROOT}` and `${DEXMIMICGEN_ROOT}`.

equiv_primitive's graph-network extensions are built against the torch+CUDA installed in
step 3:

```bash
pip install torch-scatter torch-cluster torch-geometric \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

### Submodule runtime dependencies

Because the submodules are imported from `sys.path` rather than pip-installed,
`requirements.txt` (step 3) does **not** pull their own dependencies. Their
authoritative dependency lists are `primitive_learning/setup.py` and
`diffusion_policy_sg/conda_environment.yaml`; the extras actually needed to run
the DMG tasks are:

```bash
pip install wandb transformers          # equiv_primitive: skill/task BERT + logging
```

Two need special handling:

- **gym** is pinned to `0.21.0` by `diffusion_policy_sg`, and its sdist fails to
  build under modern setuptools. Install it isolated:
  ```bash
  pip install "setuptools<66" "wheel<0.40" && pip install gym==0.21.0
  ```
- **pytorch3d** must match your torch/CUDA build. No prebuilt wheel exists for the
  `torch 2.1.0+cu121` stack above, so build it from source (needs the CUDA
  toolkit / `nvcc`; install `cuda-nvcc` from conda if the system lacks it):
  ```bash
  pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.6"
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

Download the DMG / Diffusion-Policy checkpoints for the two tasks into
`${REPO_ROOT}/ckpt/`, where the task configs already expect them:

```bash
python download_ckpts.py                  # both tasks (~2.9 GB)
python download_ckpts.py --task threading # threading only
```

No manual placement is needed — the configs reference
`${REPO_ROOT}/ckpt/dmg_<task>/{equi.pth,biop.pth,policy.ckpt}`. See
[Configuration](Configuration.md) for the layout and the `--dp_ckpt` / `--sg`
runtime overrides.

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
