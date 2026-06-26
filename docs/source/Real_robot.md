# Real Robot (optional)

Installation and running for the **real-robot** route. This is independent of the
simulation route and adds the perception stack (segmentation, grasping) and ROS
execution. Skip this entire page if you only run the simulation tasks.

**Status:** stub — TODO.

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

## Real-robot running

- Entry point: `examples/pybullet/aloha_real/scripts/interleaved_real_traj_plugin.py`
  (standalone ROS lifecycle; same online plan/execute structure as the sim plugin).

TODO: document camera calibration (hand-eye), ROS bring-up, and the real-robot
run command.
