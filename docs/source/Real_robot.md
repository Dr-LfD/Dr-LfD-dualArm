# Real Robot (optional)

Installation and running for the **real-robot** route. This is independent of the
simulation route and adds the perception stack (segmentation, grasping) and ROS
execution. Skip this entire page if you only run the simulation tasks.

**Status:** the real route uses the vendored ALOHA hardware adapter in
`diffusion_policy_sg/aloha_pkg`. It must be launched from a ROS Noetic +
Interbotix environment; `--mode file` does not exercise this path.

## Real-robot installation (perception stack)

These components are **not** needed for simulation and are intentionally kept off
the sim [Installation](Installation.md):

### Segmentation perception

- **SAM3 / Grounded-SAM-2 + GroundingDINO** — open-vocabulary segmentation used
  to estimate object state from camera images on the real robot.

  ```bash
  # Segment Anything 2
  cd ${WS_ROOT}/Grounded-SAM-2
  pip install -e .
  # Grounding DINO
  pip install --no-build-isolation -e grounding_dino
  ```

- **vision_utils** — shape completion (AtlasNet/MSN) and segmentation
  (MaskRCNN/UOIS) backends, with git-lfs weights:

  ```bash
  sudo apt-get install git-lfs && git lfs install && git lfs pull
  # move pulled weights into vision_utils/trained_models/
  ```

  See `vision_utils/README.md`.

### Grasp detection (GPD)

```bash
cd examples/pybullet/aloha_real/openworld_aloha/grasp/gpd
mkdir build && cd build
cmake .. && make -j
```

### DP evaluator and ALOHA runtime

Install the real-robot-only Python packages into the same environment that runs
the plugin:

```bash
pip install -r requirements-real.txt
```

Before every `--mode real` run, source the ROS workspaces before activating the
Python environment:

```bash
source /opt/ros/noetic/setup.bash
source ~/interbotix_ws/devel/setup.bash
source /path/to/miniforge3/etc/profile.d/conda.sh
conda activate dr-lfd
```

The Interbotix source step exposes the ALOHA robot messages and controllers;
`requirements-real.txt` supplies `dm_env` for the DP observation bridge and
`modern_robotics` for Interbotix kinematics.

## Real-robot running

Start the ROS master and ALOHA hardware drivers in a terminal after the source
steps above:

```bash
roscore &
roslaunch aloha 4arms_teleop.launch
```

Entry point: `examples/pybullet/aloha_real/scripts/interleaved_real_traj_plugin.py`.
First run its no-motion preflight; `--profile-plan-time` returns before the
execution timer is created:

```bash
python examples/pybullet/aloha_real/scripts/interleaved_real_traj_plugin.py \
  --mode real --online-tamp --profile-plan-time --task-name handoff_cup \
  --skill-yaml-path examples/pybullet/aloha_real/openworld_aloha/configs/skill_cfg/handoff_cup_per_skill.yaml
```

Only after the preflight has completed, the workspace is clear, and an operator
has an immediate power-cutoff path should the same command be run without
`--profile-plan-time`. Validate a single skill before attempting a composed task.
