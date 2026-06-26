# Contact & Effect Monitoring

How the simulation route verifies that a skill achieved its intended effect
(attach / detach / insertion) and triggers replanning when it did not.

**Status:** stub — TODO.

## Scope

- Subgoal/effect detection from simulation state (object attachment/detachment,
  insertion success).
- Barrier vs lane action subgoals.

> Object **perception** in simulation comes directly from the robosuite/mujoco
> environment — there is no segmentation network on the sim route. SAM3 /
> Grounded-SAM segmentation is part of the real-robot perception stack; see
> [Real robot](Real_robot.md).

## Source files

- `examples/pybullet/aloha_real/openworld_aloha/contact_monitoring.py`
  (`ContactEffectMonitorGroup`, `ContactPredictorWrapper`).
- `examples/pybullet/aloha_real/openworld_aloha/symbolic_utils.py`
  (`SimPrimitiveSubgoalDetector`, `get_contact_action_subgoal()`,
  `get_barrier_action_subgoal()`).

TODO: document the effect-monitor group, contact prediction, and how subgoals map
back to PDDL effects.
