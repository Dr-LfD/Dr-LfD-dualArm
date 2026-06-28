# Adding a New Task

Developer guide for defining a new bimanual DMG task.

**Status:** stub — TODO.

## Scope

The pieces a new task needs:

1. **Task config** — a `dmg_cfgs/<task>.yaml` (paths, `LfD_params`, `sg_params`)
   modeled on `two_arm_threading.yaml`.
2. **Schema changes** — a `<task>_changes.json` describing per-skill contact-change
   edges (consumed by [Schema construction](Schema_construction.md)).
3. **Checkpoints** — DP/SDP policy and equiv_primitive keypoint/operator checkpoints,
   referenced from the config (see [Configuration](Configuration.md)).
4. **Run** — `interleaved_dmg_osc_plugin.py --task_name <task>`.

## Source files

- Configs: `examples/pybullet/aloha_real/openworld_aloha/configs/dmg_cfgs/`
- Schema: `.../standalone_scripts/schema_construction.py`

TODO: walk through a concrete end-to-end example with the minimal config/JSON
fields required.
