# Online TAMP Loop

The plan → execute → verify → replan loop that drives bimanual execution.

**Status:** stub — TODO.

## Scope

- Coarse global plan (symbolic ordering) → staged execution (lane batches /
  barriers) → subgoal verification → bounded retry on failure.
- Per-stage retry limit (`MAX_STAGE_RETRIES`).

## Source files

- `examples/pybullet/aloha_real/scripts/interleaved_dmg_osc_plugin.py`
  (`DMG_OSC_planner`, `online_tamp()`, `execute_robosuite_plan()`).
- `examples/pybullet/aloha_real/scripts/tamp_workflow.py`
  (`ArmSchedulerState`, `plan_online_session()`, `plan_detail_mp()`,
  `merge_batch_sequences()`).
- `examples/pybullet/aloha_real/openworld_aloha/symbolic_utils.py`
  (subgoal detection, `simulate_plan_execution()`).

TODO: document the lane/barrier scheduler, the coarse↔detailed planning split,
and subgoal/effect verification.
