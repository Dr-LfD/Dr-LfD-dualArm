# Overview

## What is DR-LfD

**DR-LfD** — *Decomposed and Reorganized skills Learned from Demonstrations* —
is a framework for long-horizon, dexterous bimanual manipulation that combines
the strengths of task-and-motion planning (TAMP) and imitation learning (IL).

Traditional TAMP excels at symbolic, long-horizon reasoning but is brittle in
contact-rich execution. IL produces smooth, contact-rich behaviors from
demonstrations but generalizes poorly across spatial configurations and scales
badly with horizon length. DR-LfD decomposes human demonstrations into **atomic
skills** — reproduced either as **visuomotor policies** or **object-centric
primitives** — and models each skill's initiation, termination, and constraints
in a TAMP-compatible form. The planner can then **reorganize** skills learned
from different sources to solve tasks with multiple steps, unseen setups, and
physical constraints.

Project page: <https://dr-lfd.github.io/DR-LfD-website>

## The pipeline

```
        Demonstrations
              │  (decompose into atomic skills)
              ▼
  ┌──────────────────────────┐
  │ Schema construction       │  skills → PDDL operators / streams
  │ (schema_construction.py)  │
  └────────────┬─────────────┘
               ▼
  ┌──────────────────────────┐
  │ Online TAMP loop          │  plan → execute → verify → replan
  │ (interleaved_dmg_osc_     │
  │  plugin.py, tamp_workflow)│
  └────────────┬─────────────┘
               ▼
  ┌──────────────────────────┐
  │ Trajectory execution      │  bimanual absolute end-effector control
  │ (interleaved_dmg_osc_     │
  │  plugin.py)               │
  └────────────┬─────────────┘
               ▼
  ┌──────────────────────────┐
  │ Contact / effect feedback │  subgoal verification, retry on failure
  │ (contact_monitoring.py,   │
  │  symbolic_utils.py)       │
  └──────────────────────────┘
```

In the **simulation** route, object state is provided directly by the
robosuite/mujoco environment — there is no segmentation network. SAM3 /
Grounded-SAM perception belongs to the **real-robot** stack and is described on
[Real robot](Real_robot.md), not as a simulation dependency.

## Subsystem map

| Subsystem | Entry file (relative to repo root) |
|---|---|
| Schema construction | `examples/pybullet/aloha_real/scripts/standalone_scripts/schema_construction.py` |
| Problem construction | `examples/pybullet/aloha_real/openworld_aloha/problem_construction.py` |
| Streams / samplers | `examples/pybullet/aloha_real/openworld_aloha/openworld_streams.py`, `aloha_samplers.py` |
| Online TAMP loop | `examples/pybullet/aloha_real/scripts/interleaved_dmg_osc_plugin.py` |
| TAMP workflow helpers | `examples/pybullet/aloha_real/scripts/tamp_workflow.py` |
| Schema executor | `examples/pybullet/aloha_real/openworld_aloha/schema_executor.py` |
| Trajectory execution / gripper frame | `examples/pybullet/aloha_real/scripts/interleaved_dmg_osc_plugin.py` (see [Gripper frame conventions](Gripper_frame_conventions.md)) |
| Contact / effect monitoring | `examples/pybullet/aloha_real/openworld_aloha/contact_monitoring.py`, `symbolic_utils.py` |
| Learned-skill loading | `examples/pybullet/aloha_real/openworld_aloha/network_loader.py` |
| PDDLStream core | `pddlstream/`, `downward/` |

## Where to go next

- [Installation](Installation.md) — set up the `dr-lfd` conda environment.
- [Quickstart](Quickstart.md) — run the two simulation tasks.
- [Configuration](Configuration.md) — paths, task configs, checkpoints, overrides.
