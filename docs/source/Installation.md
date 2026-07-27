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

robosuite and robomimic are git-installed by `requirements.txt` (step 3).
dexmimicgen is the exception — its task configs read scene-graph data through
`env_dir=${DEXMIMICGEN_ROOT}`, so it must live on disk at that path rather than
in site-packages:

```bash
git clone https://github.com/JINXER000/dexmimicgen -b playback_3d ${DEXMIMICGEN_ROOT}
pip install -e ${DEXMIMICGEN_ROOT}
```

If a stale `robosuite` is shadowing the editable install, remove it first:

```bash
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
rm -rf "${SITE}/robosuite"
```

## 5. Learned-policy sibling repos

The object-centric primitive-learning repo (`primitive_learning`) ships as a
submodule of the release umbrella — initialize it from the umbrella checkout:

```bash
git submodule update --init primitive_learning
```

The DMG / Diffusion-Policy skills are imported from sibling repos (added to
`sys.path` at runtime, not pip-installed). Each is located by its own macro set
in `.env` (see [Configuration](Configuration.md)). For a development checkout or
a local override, clone anywhere and point the macro at it:

```bash
git clone https://github.com/Dr-LfD/DP_customized.git <DIFFUSION_POLICY_ROOT>
git clone https://github.com/Dr-LfD/Equiv_primitive.git <PRIMITIVE_LEARNING_ROOT>
# .env:
export DIFFUSION_POLICY_ROOT=/abs/path/to/Diffusion-Policy
export PRIMITIVE_LEARNING_ROOT=/abs/path/to/Equiv_primitive
```

equiv_primitive's graph-network extensions are built against the torch+CUDA installed in
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
