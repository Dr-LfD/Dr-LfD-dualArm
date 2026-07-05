# Online TAMP Loop

The plan → execute → verify → replan loop that drives bimanual execution.

**Status:** stub — TODO.

## Scope

- Coarse global plan (symbolic ordering) → staged execution (lane batches /
  barriers) → subgoal verification → bounded retry on failure.
- Per-stage retry limit (`MAX_STAGE_RETRIES`).

## Source files

- `examples/pybullet/aloha_real/scripts/interleaved_sim_osc_plugin.py`
  (`DMG_OSC_planner`, `online_tamp()`, `execute_robosuite_plan()`).
- `examples/pybullet/aloha_real/scripts/tamp_workflow.py`
  (`ArmSchedulerState`, `plan_online_session()`, the shared lane/barrier
  planners `plan_lane_batch_schema()` / `replan_subgoal_detail_mp()` /
  `plan_barrier_schema()`, and the state algebra `state_delta()` /
  `revert_effects()` / `merge_lane_states()`).
- `examples/pybullet/aloha_real/openworld_aloha/symbolic_utils.py`
  (subgoal detection, `simulate_plan_execution()`).

TODO: document the lane/barrier scheduler, the coarse↔detailed planning split,
and subgoal/effect verification.
