# schema_construction.py — Correctness and sec_IV_D alignment

## 1. Correctness check

- **Config parsing**: Handles both `contact_changes` (list) and `contact_change` (single), and string-encoded JSON (e.g. `initial_graph` as string). Object types: HAND_PATTERNS → hand, SURFACE_NAMES → surface, else movable.
- **MatchStreams**: Follows doc hierarchy: (1) primitive → LearnedAttach (hand-obj Add or obj-obj add) or LearnedDetach (hand-obj Del); (2) policy → LearnedUniKeyPose (1 arm) or LearnedBiKeyPose (2 arms); (3) if G_eff has hand-obj edge, add LearnedPostGrasp. Involved hands/objects and G_eff are computed from contact_changes and current_graph.
- **Grounding**: For each skill, metadata includes grounding_arm, grounding_object, grounding_surface (detach), and for bimanual grounding_arm1/arm2, grounding_o1/o2 from current graph. Left/right from entity names (robot0, robot_left → left).
- **Domain**: Built from .pddl templates only; learned actions use action_*_grounded.pddl with {{PLACEHOLDER}} substitution. Transit/Transfer from templates; (perceived) injected when --perceive.
- **Streams**: Single universal stream.pddl is copied to output; no per-skill stream generation. Streams use generic ?sk, ?arm, ?o; matching to domain actions is by binding at planning time.
- **Init facts**: Arm, left_arm/right_arm, Movable/Graspable/CanPick, Region, SkillAttach/SkillDetach/Skillbimanual per skill.

## 2. Discrepancy vs ref_doc/sec_IV_D_new.tex

- **Table 1 notation**: Doc uses `?a` for skill identifier; PDDL uses `?sk`. Same role.
- **LearnedDetach inputs**: Doc gives `(?h, ?o_1, ?o_2, ?a, ?g)` (hand, placed object, surface, skill, grasp). Our sample-place-traj has `(?arm ?o ?s ?sp ?sk ?g)` with ?s as surface, ?o as object — aligned.
- **LearnedBiKeyPose**: Doc has `(h_l, h_r, ?a)` (grounded arms). Universal stream has `(?arm1 ?arm2 ?sk)`; init facts set left_arm/right_arm so planner binds arms consistently. Grounded BiOperation actions in the domain use concrete arm names; stream remains generic.
- **LearnedPostGrasp**: Doc describes it as add-on for policy–trajectory switching (contact pose g for post-policy transfer). The script only *detects* LearnedPostGrasp (eff_has_hand_obj_edge) and does not emit a separate stream or action; the doc says it is “linked” to the stream and used for collision checking. If a dedicated LearnedPostGrasp stream/action is required later, it would need to be added.
- **Primitive fallback**: When a primitive has only obj_obj_adds (no hand-obj change), script assigns LearnedAttach. Doc does not explicitly cover this; it is a reasonable default (e.g. “place” where object is placed on surface by the same arm that will later release).

No blocking discrepancies; optional extension: explicit LearnedPostGrasp stream/action if the planner expects it.

## 3. Schema-based problem construction (run_openworld)

When using schema-generated domain/stream with perception-grounded init, set in your task YAML:

```yaml
use_schema: true
domain_path: 'examples/pybullet/aloha_real/openworld_aloha/configs/dmg_cfgs/two_arm_threading_changes_domain.pddl'
stream_path: 'examples/pybullet/aloha_real/openworld_aloha/configs/dmg_cfgs/two_arm_threading_changes_stream.pddl'
schema_config_path: 'examples/pybullet/aloha_real/openworld_aloha/configs/dmg_cfgs/two_arm_threading_changes.json'
task_name: 'two_arm_threading'
match_by_category: true   # optional; match schema object names to body.category
use_perceived: true       # optional; add (perceived) to init
# object_mapping: {}      # optional; explicit schema_name -> perceived body
```

Paths can be relative to the repo root. `compute_TAMP_cmd` and `compute_TAMP_online` will call `pddlstream_from_schema_problem` when `use_schema` and both paths are set.


### debug notes
1. I think the problem construction may be buggy, as tripot_obj is a struct instead of a string. 