# Schema Construction

How learned atomic skills are converted into PDDL operators and streams so the
planner can reorganize them.

**Status:** stub — TODO.

## Scope

- Input: per-skill contact-change configs (`*_changes.json`) + skill metadata.
- Output: PDDL domain/stream operators (Attach / Detach / BiOperation) consumed
  by the online TAMP loop.

## Source files

- `examples/pybullet/aloha_real/scripts/standalone_scripts/schema_construction.py`
  (`build_action_schema()` and the operator-build loop).
- `examples/pybullet/aloha_real/openworld_aloha/problem_construction.py`
  (`pddlstream_from_schema_problem()`).
- Templates: `examples/pybullet/aloha_real/.../pddl_templates/`.

TODO: document the JSON schema format (skills / edge_ops), the operator templates,
and the byte-identical golden test under `tests/golden_schema`.
