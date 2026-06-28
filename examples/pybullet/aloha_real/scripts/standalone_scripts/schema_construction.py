"""BuildActionSchema: generate PDDL domain and stream files from a contact-change config.

Reads a JSON config (initial_graph + per-skill edge_ops) and emits a PDDL domain and
stream file. Learned skills are classified into the stream taxonomy (Attach/Detach ->
pick/place, BiKeyPose -> BiOperation); planned actions (Transit, Transfer, ...) come from
the human-written templates in pddl_templates/.
"""

import json
import ast
import os
import argparse
import re
from dataclasses import dataclass, field

import networkx as nx

from examples.pybullet.aloha_real.openworld_aloha.skill_naming import (
    canonical_bimanual_skill_name,
    canonical_unimanual_skill_name,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.normpath(
    os.path.join(SCRIPT_DIR, "../../openworld_aloha/pddl_templates")
)


def _load_template(name):
    """Load a PDDL template fragment from the pddl_templates/ directory."""
    path = os.path.join(TEMPLATE_DIR, name)
    with open(path) as f:
        return f.read()


def _fill_template(template_content, replacements):
    """Replace {{KEY}} placeholders in template_content. replacements is a dict KEY -> value."""
    out = template_content
    for k, v in replacements.items():
        out = out.replace("{{" + k + "}}", str(v))
    return out


# Optional reachability + MDF policy-safety constraints, emitted only when
# enable_constraints is True (per-task `use_constraints` flag); otherwise the
# {{REACHABLE_PRE}} placeholders fill to "".

# Inline precondition fragment per action template (exact indentation/varnames preserved).
_REACHABLE_PRE = {
    "pick":               "\n                       (Reachable ?a ?o ?p)",
    "pick_coarse":        "\n                       (Reachable ?a ?o ?p)",
    "place":              "\n                       (Reachable ?a ?o ?p)",
    "place_coarse":       "\n                       (Reachable ?a ?o ?p)",
    "learned_pick":       "\n                        (Reachable ?arm ?obj ?p)",
    "learned_pick_coarse":"\n                      (Reachable ?arm ?obj ?p)",
}

_REACHABLE_STREAM_BLOCK = """  (:stream test-reachable
    :inputs (?a ?o ?p )
    :domain (and (Arm ?a) (Pose ?o ?p) (Movable ?o))
    :certified (Reachable ?a ?o ?p )
  )"""


def _reachable_pre(key, enable_constraints):
    """Return the inline (Reachable ...) precondition for an action template, or "" when off."""
    return _REACHABLE_PRE[key] if enable_constraints else ""


def _load_reachable_action_template(template_name, key, enable_constraints):
    """Load a pick/place action template and fill its {{REACHABLE_PRE}} placeholder."""
    return _fill_template(_load_template(template_name),
                          {"REACHABLE_PRE": _reachable_pre(key, enable_constraints)})


# ============================================================
#  1. Config Parsing & Object Type Inference
# ============================================================

HAND_PATTERNS = ("robot0", "robot1", "left_arm", "right_arm")
SURFACE_NAMES = {"table"}

PRIMITIVE_KEYWORDS = ("grasp", "pick", "place", "drop", "release", "detach", "lift")

CONTAINER_NAMES = set()  # O_in container object names → In(obj, container) predicate generated.
# Populate before calling build_action_schema, e.g.: CONTAINER_NAMES.add("base_obj")


def _parse_json_or_string(value):
    """Handle string-encoded JSON values (e.g. in two_arm_lift_tray_changes.json)."""
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


def apply_schema_object_mapping(text, object_mapping):
    """Rewrite abstract schema object names to the active scene's real names.

    A task-agnostic template uses 'placeholder' for the graspable, and each task
    rebinds it (e.g. placeholder -> obj_1) so the generated skills and goal
    reference the perceived object. Substitution is on the raw JSON text so it
    reaches object names, the initial graph, and contact_changes uniformly.
    """
    if object_mapping:
        for schema_name, real_name in object_mapping.items():
            text = text.replace(schema_name, real_name)
    return text


_EDGE_OP_TO_INTERNAL = {"Add": "add", "Del": "remove"}


def parse_config(config_path, object_mapping=None):
    """Parse a skill-schema config JSON into (initial_graph, skills, objects_dict).

    The ``edge_ops`` ``Add``/``Del`` verbs are mapped to the internal ``add``/``remove``
    ``contact_changes`` representation; per-skill extras (``effect_detection``,
    ``timestamps``, ``skill_type``) are carried through verbatim.
    """
    with open(config_path, "r") as f:
        data = json.loads(apply_schema_object_mapping(f.read(), object_mapping))

    for required in ("objects", "skills"):
        if required not in data:
            raise ValueError(f"Missing required {required!r} key in config: {config_path}")
    objects_dict = data["objects"]

    initial_graph = _parse_json_or_string(data["initial_graph"])

    skills = []
    for entry in data["skills"]:
        for required in ("name", "edge_ops"):
            if required not in entry:
                raise ValueError(
                    f"Skill entry missing required {required!r} key in {config_path}: {entry}"
                )
        edge_ops = _parse_json_or_string(entry["edge_ops"])
        contact_changes = []
        for edge, op in edge_ops:
            if op not in _EDGE_OP_TO_INTERNAL:
                raise ValueError(
                    f"Unknown edge op {op!r} in skill {entry.get('name')!r} of "
                    f"{config_path}; expected one of {sorted(_EDGE_OP_TO_INTERNAL)}."
                )
            if isinstance(edge, str):
                edge = ast.literal_eval(edge)
            contact_changes.append([list(edge), _EDGE_OP_TO_INTERNAL[op]])

        skill_entry = {"description": entry["name"], "contact_changes": contact_changes}
        for key in ("effect_detection", "timestamps", "skill_type"):
            if key in entry:
                skill_entry[key] = entry[key]
        skills.append(skill_entry)

    return initial_graph, skills, objects_dict



# ============================================================
#  1b. Schema composition (union of two schema configs)
# ============================================================

def _compose_raw(graph_a, skills_a, graph_b, skills_b, source_a="a", source_b="b"):
    """
    Core composition logic: union initial_graph edges, concatenate skills,
    validate entity type consistency.

    Returns (composed_graph, composed_skills).
    Raises ValueError on type conflicts.
    """
    # Validate entity type consistency across schemas
    obj_types_a = infer_object_types(graph_a, skills_a)
    obj_types_b = infer_object_types(graph_b, skills_b)
    shared_entities = sorted(set(obj_types_a) & set(obj_types_b))
    conflicts = [
        (e, obj_types_a[e], obj_types_b[e])
        for e in shared_entities
        if obj_types_a[e] != obj_types_b[e]
    ]
    if conflicts:
        details = ", ".join(f"{e}: {ta} vs {tb}" for e, ta, tb in conflicts)
        raise ValueError(f"Entity type conflicts between schemas: {details}")

    # Union initial_graph edges (deduplicated, order-preserving)
    seen_edges = set()
    composed_graph = []
    for edge in list(graph_a) + list(graph_b):
        key = tuple(edge)
        if key not in seen_edges:
            seen_edges.add(key)
            composed_graph.append(list(key))

    # Concatenate skills with provenance annotations (preserve existing provenance)
    composed_skills = []
    for source_label, source_graph, source_skills in (
        (source_a, graph_a, skills_a),
        (source_b, graph_b, skills_b),
    ):
        for idx, skill in enumerate(source_skills):
            annotated = dict(skill)
            annotated.setdefault("source_schema", source_label)
            annotated.setdefault("source_index", idx)
            annotated.setdefault("source_initial_graph", [list(e) for e in source_graph])
            composed_skills.append(annotated)

    return composed_graph, composed_skills


def compose_schemas(config_path_a, config_path_b):
    """
    Compose two schema JSON configs into a single (initial_graph, skills) pair.
    Parallel semantics: shared initial state, all skills are independent goals.
    """
    graph_a, skills_a, _ = parse_config(config_path_a)
    graph_b, skills_b, _ = parse_config(config_path_b)
    return _compose_raw(
        graph_a, skills_a, graph_b, skills_b,
        source_a=config_path_a, source_b=config_path_b,
    )


def infer_object_types(initial_graph, skills, objects_dict=None):
    """
    Classify object types from explicit objects_dict (preferred) with HAND_PATTERNS/SURFACE_NAMES fallback.

    objects_dict entries: [] -> hand (if HAND_PATTERNS match) or surface; ["CanPick"] -> movable;
    ["Surface"] / ["Container"] -> surface. Entities absent from objects_dict fall back to
    HAND_PATTERNS / SURFACE_NAMES heuristics.
    """
    all_entities = set()
    for edge in initial_graph:
        all_entities.update(edge)
    for skill in skills:
        for edge, _op in skill["contact_changes"]:
            all_entities.update(edge)

    obj_types = {}
    for entity in sorted(all_entities):
        if objects_dict is not None and entity in objects_dict:
            attrs = objects_dict[entity]
            if attrs == []:
                obj_types[entity] = "hand" if entity in HAND_PATTERNS else "surface"
            elif any(a in ("Surface", "Container") for a in attrs):
                obj_types[entity] = "surface"
            else:
                obj_types[entity] = "movable"
        elif entity in HAND_PATTERNS:
            obj_types[entity] = "hand"
        elif entity in SURFACE_NAMES:
            obj_types[entity] = "surface"
        else:
            obj_types[entity] = "movable"

    return obj_types


def _infer_world_frame(initial_graph, hands):
    """
    Identify the world/root node that arms connect to in the initial graph.

    Config convention: edges are [arm, surface] so the second element of an
    arm-containing edge is the world frame (e.g. "table").
    """
    for edge in initial_graph:
        parent, child = edge
        if parent in hands:
            return child
    return "world"


def DefaultGraphSchema(world_frame, hands):
    """G_0: default state where all hands are free (connected to world_frame, holding nothing)."""
    G0 = nx.DiGraph()
    for h in hands:
        G0.add_edge(h, world_frame)
    return G0


def GraphDiff(G_pre, G_0):
    """Edges present in G_pre but absent from G_0 — the non-default state of the world."""
    return [(u, v) for u, v in G_pre.edges() if not G_0.has_edge(u, v)]


def Derived(G_pre, involved_hands, obj_types):
    """
    Compute ArmEmpty facts from G_pre: return hands that hold no movable object.

    A hand is considered empty if it has no outgoing edge to a node that is
    neither a surface nor another hand (i.e., it is not grasping any movable).
    """
    free_arms = []
    for h in involved_hands:
        if not G_pre.has_node(h):
            free_arms.append(h)
            continue
        holding = [
            nbr for nbr in G_pre.successors(h)
            if obj_types.get(nbr) not in ("surface", "hand")
        ]
        if not holding:
            free_arms.append(h)
    return free_arms


def InvolvedArms(changes, hands, current_graph):
    """``H_i`` -- the arms that skill ``a_i`` engages.

    Steps through the skill's contact changes; at each intermediate graph (with
    surfaces removed) any connected component holding >=2 hands marks those hands
    as jointly involved -- i.e. a bimanual operation.
    """
    involved = set()
    G = current_graph.copy()
    for edge, op in changes:
        u, v = edge
        if op == "add":
            G.add_edge(u, v)
        elif op == "remove" and G.has_edge(u, v):
            G.remove_edge(u, v)

        G_no_surface = G.copy()
        G_no_surface.remove_nodes_from(
            [n for n in G_no_surface.nodes() if n in SURFACE_NAMES]
        )
        G_undirected = G_no_surface.to_undirected()
        if len(G_undirected) > 0:
            for comp_nodes in nx.connected_components(G_undirected):
                comp_hands = hands.intersection(comp_nodes)
                if len(comp_hands) >= 2:
                    involved.update(comp_hands)
    return involved


@dataclass
class _SchemaVarContext:
    """Resolvers from schema entity names to a single action's PDDL variables."""
    arm: object       # hand name -> "?a1"/"?a2"
    ovar: object      # object/surface name -> "?oN"
    gvar: object      # (hand, obj) -> pre-existing grasp var "?gN"
    grefresh: object  # (hand, obj) -> refreshed grasp var "?gN"
    pvar: object      # (obj, surf) -> pose var "?pN"


def EdgeOps2Predicates(category, a, b, ctx):
    """Map one edge operation to its PDDL atom(s).

    ``category`` names the contact transition; ``a``/``b`` are its endpoints and
    ``ctx`` resolves schema names to this action's PDDL variables.
    """
    arm, ovar = ctx.arm, ctx.ovar
    if category == "grasp_pre":            # arm a holds obj b before the action
        return [f"(AtGrasp {arm(a)} {ovar(b)} {ctx.gvar(a, b)})"]
    if category == "surface_pre":          # obj a rests on surface b before
        return [f"(On {ovar(a)} {ovar(b)})", f"(AtPose {ovar(a)} {ctx.pvar(a, b)})"]
    if category == "grasp_del":            # arm a releases obj b
        return [f"(not (AtGrasp {arm(a)} {ovar(b)} {ctx.gvar(a, b)}))",
                f"(not (ArmHolding {arm(a)} {ovar(b)}))",
                f"(ArmEmpty {arm(a)})"]
    if category == "grasp_add":            # arm a grasps obj b
        return [f"(AtGrasp {arm(a)} {ovar(b)} {ctx.gvar(a, b)})",
                f"(ArmHolding {arm(a)} {ovar(b)})",
                f"(not (ArmEmpty {arm(a)}))"]
    if category == "grasp_refresh":        # stale grasp swapped for fresh one
        return [f"(not (AtGrasp {arm(a)} {ovar(b)} {ctx.gvar(a, b)}))",
                f"(AtGrasp {arm(a)} {ovar(b)} {ctx.grefresh(a, b)})"]
    if category == "holding_del":          # obj a no longer held by any hand
        return [f"(not (Holding {ovar(a)}))"]
    if category == "holding_add":          # obj a now held by some hand
        return [f"(Holding {ovar(a)})"]
    if category == "surface_del":          # obj a lifted off surface b
        return [f"(not (AtPose {ovar(a)} {ctx.pvar(a, b)}))"]
    if category == "surface_add":          # obj a placed on surface b
        return [f"(AtPose {ovar(a)} {ctx.pvar(a, b)})"]
    if category == "container_add":        # obj a inserted into container b
        return [f"(In {ovar(a)} {ovar(b)})"]
    raise ValueError(f"Unknown EdgeOps2Predicates category: {category!r}")


def Bound(arm_obj_edges, ctx, *, refresh=False):
    """σ.pre ∧ Bound(σ.pre, F_a.certified) -- bind the learned stream's
    grasp outputs (ImitateGrasp, certified by sample-biop-keypose) into the
    precondition so the planner is forced to call the stream to produce them.
    """
    resolve = ctx.grefresh if refresh else ctx.gvar
    return [
        f"(ImitateGrasp ?sk {ctx.arm(h)} {ctx.ovar(o)} {resolve(h, o)})"
        for h, o in arm_obj_edges
    ]


def SortedArguments(arm_vars, conf_vars, obj_vars, grasp_vars,
                    refresh_grasp_vars, pose_vars, geom_var):
    """σ.args -- collect every schema variable in a deterministic
    order (arms/skill, configs, objects, grasps, refreshed grasps, poses, geom)."""
    params = list(arm_vars) + list(conf_vars)
    params += list(obj_vars.values())
    params += list(grasp_vars.values())
    params += list(refresh_grasp_vars.values())
    params += list(pose_vars.values())
    params.append(geom_var)
    return params


def Applied(skill_var="?sk"):
    """σ.eff ∧ Applied(a_i), realized as the DoneSkill marker."""
    return f"(DoneSkill {skill_var})"


@dataclass
class ActionSchema:
    """Learned-skill schema σ = (args, pre, eff), rendered to PDDL."""
    idx: int
    params: list = field(default_factory=list)   # σ.args
    pre: list = field(default_factory=list)
    eff: list = field(default_factory=list)
    template: str = "action_bioperation_grounded_dynamic.pddl"

    def render(self):
        return _fill_template(_load_template(self.template), {
            "ACTION_IDX": self.idx,
            "PARAMS": " ".join(self.params),
            "PRECONDITIONS": "\n      ".join(self.pre),
            "EFFECTS": "\n      ".join(self.eff),
        })


def _is_primitive(skill):
    """
    Determine whether a skill is an object-centric primitive or a visuomotor policy.
    Uses explicit 'skill_type' field if present, otherwise falls back to description heuristics.
    """
    if "skill_type" in skill:
        return skill["skill_type"] == "primitive"
    return any(kw in skill["description"].lower() for kw in PRIMITIVE_KEYWORDS)


# ============================================================
#  3. MatchStreams
# ============================================================

LEARNED_ATTACH = "LearnedAttach"
LEARNED_DETACH = "LearnedDetach"
LEARNED_UNI_KEYPOSE = "LearnedUniKeyPose"
LEARNED_BI_KEYPOSE = "LearnedBiKeyPose"

DETAILED_MODE = "detailed"
COARSE_MODE = "coarse"
PLANNING_MODES = {DETAILED_MODE, COARSE_MODE}


def _validate_planning_mode(planning_mode):
    if planning_mode not in PLANNING_MODES:
        raise ValueError(
            f"Unknown planning mode {planning_mode!r}. Expected one of "
            f"{sorted(PLANNING_MODES)}"
        )

def _has_downstream_detach(arm, obj, skill_index, classified_skills):
    """Return True if any later skill has LEARNED_DETACH for (arm, obj)."""
    if classified_skills is None:
        return False
    if skill_index is None:
        # Without a meaningful index, we can't reliably look ahead.
        return False
    for j in range(skill_index + 1, len(classified_skills)):
        fwd = classified_skills[j]
        if (LEARNED_DETACH in fwd.get("matched_streams", [])
                and fwd.get("grounding_arm") == arm
                and fwd.get("grounding_object") == obj):
            return True
    return False


def MatchStreams(skill, obj_types, current_graph):
    """Assign network-integrated stream function(s) to a skill.

    Primitives match LearnedAttach/LearnedDetach (hand-obj Add/Del); policies match
    LearnedUniKeyPose (1 arm) or LearnedBiKeyPose (2 arms). Returns
    (matched_streams, metadata).
    """
    changes = skill["contact_changes"]
    hands = {e for e, t in obj_types.items() if t == "hand"}

    hand_obj_adds = []
    hand_obj_removes = []
    obj_obj_adds = []

    for edge, op in changes:
        parent, child = edge
        p_hand, c_hand = parent in hands, child in hands
        if p_hand and op == "add":
            hand_obj_adds.append((parent, child))
        elif c_hand and op == "add":
            hand_obj_adds.append((child, parent))
        elif p_hand and op == "remove":
            hand_obj_removes.append((parent, child))
        elif c_hand and op == "remove":
            hand_obj_removes.append((child, parent))
        elif not p_hand and not c_hand and op == "add":
            obj_obj_adds.append(edge)

    involved_objects = set()
    for h, o in hand_obj_adds + hand_obj_removes:
        involved_objects.add(o)
    for e in obj_obj_adds:
        involved_objects.update(e)

    involved_hands = InvolvedArms(changes, hands, current_graph)

    # Final-state contact graph (after applying all of this skill's changes), used
    # below for the hand-object edges the biop stream / grasp-refresh logic needs.
    G_eff = current_graph.copy()
    for edge, op in changes:
        u, v = edge
        if op == "add":
            G_eff.add_edge(u, v)
        elif op == "remove" and G_eff.has_edge(u, v):
            G_eff.remove_edge(u, v)

    # Collect all hand-object edges in the effect state. These are preserved in
    # metadata so the unified biop-keypose stream can emit grasp outputs and the
    # BiOperation action can refresh stale grasps after a stochastic policy.
    eff_hand_obj_edges = []
    for h in hands:
        for nbr in (G_eff.neighbors(h) if G_eff.has_node(h) else []):
            if nbr not in hands and nbr not in SURFACE_NAMES:
                eff_hand_obj_edges.append((h, nbr))

    matched = []
    is_prim = _is_primitive(skill)

    if is_prim:
        if hand_obj_adds:
            matched.append(LEARNED_ATTACH)
        elif hand_obj_removes:
            matched.append(LEARNED_DETACH)
        elif obj_obj_adds:
            matched.append(LEARNED_ATTACH)
        else:
            matched.append(LEARNED_ATTACH)
    else:
        n_arms = len(involved_hands)
        if n_arms >= 2:
            matched.append(LEARNED_BI_KEYPOSE)
        else:
            matched.append(LEARNED_UNI_KEYPOSE)

    metadata = {
        "description": skill["description"],
        "contact_changes": changes,
        "is_primitive": is_prim,
        "matched_streams": matched,
        "involved_hands": involved_hands,
        "involved_objects": involved_objects,
        "hand_obj_adds": hand_obj_adds,
        "hand_obj_removes": hand_obj_removes,
        "obj_obj_adds": obj_obj_adds,
        "eff_hand_obj_edges": eff_hand_obj_edges,
    }
    # Propagate composition provenance if present
    if "source_schema" in skill:
        metadata["source_schema"] = skill["source_schema"]
    if "source_index" in skill:
        metadata["source_index"] = skill["source_index"]
    return matched, metadata


def _is_left_arm(arm):
    """True if arm is considered left (e.g. robot0, left_arm)."""
    name = (arm or "").lower()
    return "0" in name or "left" in name


def _schema_arm_to_side(arm):
    """Normalize schema arm name to side label used by per-skill SGS."""
    if arm is None:
        return None
    return "left" if _is_left_arm(arm) else "right"


def _detach_surface_from_changes(meta, obj_types):
    """Infer target surface for a LearnedDetach skill from obj_obj_adds if present."""
    surfaces = {e for e, t in obj_types.items() if t == "surface"}
    for edge in meta.get("obj_obj_adds", []):
        for ent in edge:
            if ent in surfaces:
                return ent
    return None


def _classify_skill_sequence(skills, obj_types, initial_graph):
    """
    For each skill, run MatchStreams to determine matched stream functions.
    Enriches metadata with grounding: arm/object/surface and for bimanual arm->object map.
    Returns a list of metadata dicts parallel to skills.
    """
    hands = {e for e, t in obj_types.items() if t == "hand"}
    movables = {e for e, t in obj_types.items() if t == "movable"}
    current_graph = nx.DiGraph()
    current_graph.add_edges_from(tuple(e) for e in initial_graph)
    current_graph.add_nodes_from(obj_types.keys())

    world_frame = _infer_world_frame(initial_graph, hands)

    classified = []
    for skill in skills:
        # Save the pre-state graph (G_pre) before advancing.
        meta_G_pre = current_graph.copy()
        _, meta = MatchStreams(skill, obj_types, current_graph)
        meta["G_pre"] = meta_G_pre
        meta["world_frame"] = world_frame

        # Grounding: arm, object, surface (for detach), bimanual arm->object
        meta["grounding_arm"] = None
        meta["grounding_object"] = None
        meta["grounding_surface"] = None
        meta["grounding_arm1"] = None
        meta["grounding_arm2"] = None
        meta["grounding_o1"] = None
        meta["grounding_o2"] = None

        streams = meta["matched_streams"]
        if LEARNED_ATTACH in streams:
            if meta["hand_obj_adds"]:
                h, o = meta["hand_obj_adds"][0]
                meta["grounding_arm"] = h
                meta["grounding_object"] = o
            else:
                arms = sorted(meta["involved_hands"])
                objs = [x for x in meta["involved_objects"] if obj_types.get(x) == "movable"]
                meta["grounding_arm"] = arms[0] if arms else None
                meta["grounding_object"] = objs[0] if objs else None
        if LEARNED_DETACH in streams:
            if meta["hand_obj_removes"]:
                h, o = meta["hand_obj_removes"][0]
                meta["grounding_arm"] = h
                meta["grounding_object"] = o
            meta["grounding_surface"] = _detach_surface_from_changes(meta, obj_types)
        if LEARNED_BI_KEYPOSE in streams:
            arms_sorted = sorted(meta["involved_hands"], key=lambda a: (0 if _is_left_arm(a) else 1, a))
            if len(arms_sorted) >= 2:
                meta["grounding_arm1"] = arms_sorted[0]
                meta["grounding_arm2"] = arms_sorted[1]
            else:
                meta["grounding_arm1"] = list(meta["involved_hands"])[0] if meta["involved_hands"] else None
                meta["grounding_arm2"] = list(meta["involved_hands"])[1] if len(meta["involved_hands"]) > 1 else None
            # Which object each arm holds at skill start (from current graph)
            o1 = None
            o2 = None
            for h in (meta.get("grounding_arm1"), meta.get("grounding_arm2")):
                if h is None:
                    continue
                for obj in movables:
                    if current_graph.has_edge(h, obj) or current_graph.has_edge(obj, h):
                        if meta.get("grounding_arm1") == h:
                            o1 = obj
                        else:
                            o2 = obj
                        break
            # If handoff: one arm gives, one receives; both "hold" the same object conceptually
            objs_in = meta["involved_objects"]
            if o1 is None and meta["grounding_arm1"]:
                objs_list = [x for x in objs_in if obj_types.get(x) == "movable"]
                o1 = objs_list[0] if objs_list else None
            if o2 is None and meta["grounding_arm2"]:
                objs_list = [x for x in objs_in if obj_types.get(x) == "movable"]
                if len(objs_list) == 1:
                    o2 = objs_list[0]
                elif len(objs_list) >= 2:
                    o2 = objs_list[1] if o1 == objs_list[0] else objs_list[0]
            meta["grounding_o1"] = o1
            meta["grounding_o2"] = o2

        # Advance graph state for next skill
        for edge, op in skill["contact_changes"]:
            u, v = edge
            if op == "add":
                current_graph.add_edge(u, v)
            elif op == "remove" and current_graph.has_edge(u, v):
                current_graph.remove_edge(u, v)

        classified.append(meta)

    return classified


def classify_skills(skills, obj_types, initial_graph):
    """
    Classify skills, isolating composed schemas by their original source graph
    when provenance annotations (source_initial_graph) are present.

    For non-composed inputs (no source_initial_graph), delegates directly to
    _classify_skill_sequence for full backward compatibility.
    """
    if not any("source_initial_graph" in skill for skill in skills):
        return _classify_skill_sequence(skills, obj_types, initial_graph)

    # Group skills by (source_schema, source_initial_graph), preserving original indices.
    # Using graph content as part of the key ensures that composing the same schema
    # twice (A+A) still classifies each instance independently.
    grouped = {}
    for idx, skill in enumerate(skills):
        source_schema = skill.get("source_schema")
        src_graph = skill.get("source_initial_graph")
        if source_schema is None or src_graph is None:
            # Unannotated skills share a single fallback group with sequential semantics
            key = "__fallback__"
        else:
            key = (source_schema, tuple(tuple(e) for e in src_graph))
        grouped.setdefault(key, []).append((idx, skill))

    # Classify each group independently using its source's initial graph
    classified = [None] * len(skills)
    for group_entries in grouped.values():
        group_skills = [skill for _idx, skill in group_entries]
        group_initial_graph = group_skills[0].get("source_initial_graph", initial_graph)
        group_classified = _classify_skill_sequence(group_skills, obj_types, group_initial_graph)
        for (original_idx, _skill), meta in zip(group_entries, group_classified):
            classified[original_idx] = meta

    return classified


def compute_skill_names(classified_skills, env_names, naming_mode=None):
    """
    Compute HDF5-compatible skill names from classified skill metadata.

    Naming convention:
      - LearnedBiKeyPose  -> canonical bimanual name for the env in order
      - LearnedAttach/Detach -> canonical unimanual name for the naming mode
      - Fallback -> "sk_{i}"
    """
    names = []
    biop_counter = 0
    if naming_mode is None:
        use_side_prefix = any(
            "left" in (meta.get("grounding_arm") or "").lower()
            or "right" in (meta.get("grounding_arm") or "").lower()
            for meta in classified_skills
        )
    else:
        use_side_prefix = (naming_mode == "real")
    # Bimanual model keys, in order. canonical_bimanual_skill_name is idempotent on
    # names that are already bimanual, so this selects exactly the bimanual entries
    # regardless of where they sit in env_names (the full skill-key list).
    bimanual_env_names = [n for n in env_names if canonical_bimanual_skill_name(n) == n]
    for i, meta in enumerate(classified_skills):
        streams = meta.get("matched_streams", [])
        if LEARNED_BI_KEYPOSE in streams:
            if biop_counter >= len(bimanual_env_names):
                raise ValueError(
                    f"schema requires bimanual skill #{biop_counter} but the loaded "
                    f"model carries only {len(bimanual_env_names)} bimanual scene "
                    f"graph(s): {bimanual_env_names}. All available skills: "
                    f"{list(env_names)}. Check equi_ckpt_name / biop_ckpt_name."
                )
            names.append(bimanual_env_names[biop_counter])
            biop_counter += 1
        elif LEARNED_ATTACH in streams or LEARNED_DETACH in streams:
            skill_name = canonical_unimanual_skill_name(meta, use_side_prefix)
            names.append(skill_name)
        else:
            names.append(f"sk_{i}")

    # Ensure global uniqueness: append __N suffix for duplicates
    # Use a set of all finalized names to avoid collisions with generated suffixed names
    finalized = set()
    seen_counts = {}
    unique_names = []
    for name in names:
        if name not in seen_counts and name not in finalized:
            seen_counts[name] = 0
            unique_names.append(name)
            finalized.add(name)
        else:
            count = seen_counts.get(name, 0) + 1
            candidate = f"{name}__{count}"
            while candidate in finalized:
                count += 1
                candidate = f"{name}__{count}"
            seen_counts[name] = count
            unique_names.append(candidate)
            finalized.add(candidate)
    return unique_names


# ============================================================
#  3b. Schema metadata for problem construction
# ============================================================

def get_schema_metadata_from_data(initial_graph, skills, env_names, planning_mode=DETAILED_MODE, objects_dict=None, enable_constraints=False):
    """Build schema metadata from in-memory (initial_graph, skills) data.

    Same return structure as get_schema_metadata() but without file I/O.
    """
    _validate_planning_mode(planning_mode)
    obj_types = infer_object_types(initial_graph, skills, objects_dict)
    classified = classify_skills(skills, obj_types, initial_graph)
    instantiated_streams = build_instantiated_stream_specs(
        classified, env_names, planning_mode=planning_mode,
        enable_constraints=enable_constraints,
    )

    hands = [e for e, t in obj_types.items() if t == "hand"]
    movables = [e for e, t in obj_types.items() if t == "movable"]
    surfaces = [e for e, t in obj_types.items() if t == "surface"]

    skill_names = compute_skill_names(classified, env_names)
    skill_goals = [{"sk": name} for name in skill_names]

    return {
        "arm_names": hands,
        "movable_names": movables,
        "surface_names": surfaces,
        "object_names": movables + surfaces,
        "skill_goals": skill_goals,
        "classified": classified,
        "obj_types": obj_types,
        "instantiated_streams": instantiated_streams,
    }


def get_schema_metadata(config_path):
    """
    Load schema config and return metadata needed to build a PDDL problem.

    Delegates to get_schema_metadata_from_data() for core logic, then overlays
    any cached instantiated_streams from the JSON config file.
    """
    initial_graph, skills, objects_dict = parse_config(config_path)
    env_name = os.path.splitext(os.path.basename(config_path))[0].replace("_changes", "")
    metadata = get_schema_metadata_from_data(initial_graph, skills, env_names=[env_name], objects_dict=objects_dict)

    # Prefer cached instantiated_streams from config if already written by build_action_schema()
    try:
        with open(config_path, "r") as f:
            _raw = json.load(f)
        cached = _raw.get("instantiated_streams") or None
    except Exception:
        cached = None
    if cached is not None:
        generated = metadata["instantiated_streams"]
        generated_by_name = {spec["name"]: spec for spec in generated}
        cached_by_name = {spec["name"]: spec for spec in cached}
        if set(generated_by_name) != set(cached_by_name):
            raise ValueError(
                "Cached instantiated_streams do not match generated specs for "
                f"{config_path}. Regenerate the schema config."
            )
        metadata["instantiated_streams"] = [
            {**cached_by_name[name], **generated_by_name[name]}
            for name in cached_by_name
        ]

    return metadata


def load_runtime_schema_metadata(skill_yaml_paths, env_names, root_path=None):
    """Load and compose runtime schema metadata from per-skill YAML configs.

    Shared runtime loader for the online/offline plugin entrypoints: resolves YAML
    paths, loads and composes the referenced schema JSON configs, and returns the
    classified metadata (schema_meta, skill_meta_map, instantiated_streams).
    """
    if not skill_yaml_paths:
        return {
            "schema_meta": None,
            "skill_meta_map": {},
            "instantiated_streams": [],
        }

    schema_inputs = []
    for yaml_path in skill_yaml_paths:
        resolved_yaml_path = yaml_path
        if not os.path.isabs(resolved_yaml_path) and root_path is not None:
            resolved_yaml_path = os.path.join(root_path, resolved_yaml_path)
        resolved_yaml_path = os.path.normpath(resolved_yaml_path)

        from repo_paths import load_yaml
        yaml_data = load_yaml(resolved_yaml_path)

        object_mapping = yaml_data.get("object_mapping")
        schema_relpath = yaml_data.get("schema_config_path") or (
            (yaml_data.get("sg_params", {}).get("schema") or {}).get("path")
        )
        if schema_relpath is None:
            continue

        yaml_dir = os.path.dirname(resolved_yaml_path)
        schema_path = schema_relpath
        if not os.path.isabs(schema_path):
            root_relative_path = None
            if root_path is not None:
                root_relative_path = os.path.normpath(os.path.join(root_path, schema_path))
            direct_relative_path = os.path.normpath(schema_path)
            yaml_relative_path = os.path.normpath(os.path.join(yaml_dir, schema_path))
            if root_relative_path is not None and os.path.exists(root_relative_path):
                schema_path = root_relative_path
            elif os.path.exists(direct_relative_path):
                schema_path = direct_relative_path
            else:
                schema_path = yaml_relative_path

        initial_graph, skills, objects_dict = parse_config(schema_path, object_mapping=object_mapping)
        schema_inputs.append((initial_graph, skills, objects_dict, schema_path))

    if not schema_inputs:
        return {
            "schema_meta": None,
            "skill_meta_map": {},
            "instantiated_streams": [],
        }

    composed_graph, composed_skills, composed_objects_dict, composed_source = schema_inputs[0]
    for graph_b, skills_b, objects_dict_b, source_b in schema_inputs[1:]:
        composed_graph, composed_skills = _compose_raw(
            composed_graph,
            composed_skills,
            graph_b,
            skills_b,
            source_a=composed_source,
            source_b=source_b,
        )
        composed_objects_dict = {**composed_objects_dict, **objects_dict_b}
        composed_source = "composed"

    schema_meta = get_schema_metadata_from_data(
        composed_graph, composed_skills, env_names=env_names, objects_dict=composed_objects_dict,
    )
    classified = schema_meta["classified"]
    skill_names = compute_skill_names(classified, env_names)

    skill_meta_map = {}
    for raw_skill, meta, skill_name in zip(composed_skills, classified, skill_names):
        enriched_meta = dict(meta)
        enriched_meta["skill_name"] = skill_name
        if "effect_detection" in raw_skill:
            enriched_meta["effect_detection"] = raw_skill["effect_detection"]
        skill_meta_map[skill_name] = enriched_meta

    return {
        "schema_meta": schema_meta,
        "skill_meta_map": skill_meta_map,
        "instantiated_streams": schema_meta["instantiated_streams"],
    }


# ============================================================
#  4. Compose Domain and Stream PDDL from templates
# ============================================================

def _predicates_block(has_attach, has_detach, has_bimanual, schema_arm_names=None, schema_object_names=None, schema_skill_names=None, enable_constraints=False):
    """Build the (:predicates ...) block from template fragments.
    Binding predicates: (name ?a) for arms, (name ?o) for objects, (sk_i ?sk) for skills.
    No Grounding predicate in learned parts.
    """
    parts = [_load_template("predicates_base.pddl")]

    if has_attach:
        parts.append(_load_template("predicates_attach.pddl"))
    if has_detach:
        parts.append(_load_template("predicates_detach.pddl"))
    if has_bimanual:
        parts.append(_load_template("predicates_bimanual.pddl"))

    # Constraint predicates: reachability heuristic + MDF policy-safety.
    if enable_constraints:
        parts.append("    (Reachable ?a ?o ?p )")
        if has_bimanual:
            parts.append("    (SkillCheckObj ?sk ?o)")
            parts.append("    (CFreeMDF ?o ?p ?sk)")

    out = "\n".join(parts)
    if schema_arm_names or schema_object_names or schema_skill_names:
        out += "\n    ; Binding: schema name -> variable (no Grounding)"
        if schema_arm_names:
            out += "\n    " + "\n    ".join(f"({c} ?a)" for c in schema_arm_names)
        if schema_object_names:
            out += "\n    " + "\n    ".join(f"({c} ?o)" for c in schema_object_names)
        if schema_skill_names:
            out += "\n    " + "\n    ".join(f"({c} ?sk)" for c in schema_skill_names)
    return out + "\n  )"


def _grounded_learned_pick_action(meta, i, sk, planning_mode=DETAILED_MODE, enable_constraints=False):
    """Generate one grounded learnedPick_i action; arm and object are domain constants."""
    arm = meta["grounding_arm"]
    obj = meta["grounding_object"]
    if arm is None or obj is None:
        return ""
    _validate_planning_mode(planning_mode)
    is_coarse = planning_mode == COARSE_MODE
    template_name = (
        "action_learned_pick_grounded_coarse.pddl"
        if is_coarse
        else "action_learned_pick_grounded.pddl"
    )
    t = _load_template(template_name)
    return _fill_template(t, {
        "REACHABLE_PRE": _reachable_pre("learned_pick_coarse" if is_coarse else "learned_pick", enable_constraints),
        "ACTION_IDX": i,
        "ARM": arm,
        "OBJ": obj,
        "SK": sk,
    })


def _grounded_learned_place_action(meta, i, sk, surface_grounded, planning_mode=DETAILED_MODE):
    """Generate one grounded learnedPlace_i; arm and object bound via ({{ARM}} ?arm), ({{OBJ}} ?obj); surface via ({{SURFACE}} ?s) when fixed."""
    arm = meta["grounding_arm"]
    obj = meta["grounding_object"]
    if arm is None or obj is None:
        return ""
    _validate_planning_mode(planning_mode)
    surf = meta.get("grounding_surface")
    if surface_grounded and surf:
        params = "(?arm ?obj ?g ?sk ?p ?s ?sp ?lg)"
        region_pre = f"\n      ({surf} ?s)"
        # surface_effect = surf
    else:
        raise ValueError(f"Surface grounded but no surface found for {meta}")
    #     params = "(?arm ?obj ?g ?sk ?p ?s ?sp ?aq1 ?aq2 ?at)"
    #     region_pre = "\n      (Region ?s)"
    #     surface_effect = "?s"
    template_name = (
        "action_learned_place_grounded_coarse.pddl"
        if planning_mode == COARSE_MODE
        else "action_learned_place_grounded.pddl"
    )
    t = _load_template(template_name)
    return _fill_template(t, {
        "ACTION_IDX": i,
        "ARM": arm,
        "OBJ": obj,
        "SK": sk,
        "PARAMS": params,
        "REGION_PRE": region_pre,
        "SURFACE": surf or "",
    })


def _build_place_action(planning_mode, enable_constraints=False):
    _validate_planning_mode(planning_mode)
    is_detailed = planning_mode == DETAILED_MODE
    template_name = "action_place.pddl" if is_detailed else "action_place_coarse.pddl"
    return _load_reachable_action_template(
        template_name, "place" if is_detailed else "place_coarse", enable_constraints)



def _grounded_bioperation_action(meta, i, sk, planning_mode=DETAILED_MODE, obj_types=None, classified_skills=None, skill_index=None, enable_constraints=False):
    """Generate one grounded BiOperation_i action.

    Preconditions/effects are derived from the contact-graph difference (G_pre vs. G_0):
    arm/surface/container add/del edges map to AtGrasp/ArmEmpty/AtPose/In atoms, with
    ImitateGrasp and grasp refresh certified by sample-biop-keypose.
    """
    a1 = meta.get("grounding_arm1")
    a2 = meta.get("grounding_arm2")
    if not all([a1, a2]):
        return ""
    _validate_planning_mode(planning_mode)

    if obj_types is None:
        obj_types = {}

    if classified_skills is not None and skill_index is None:
        skill_index = i

    hands = {e for e, t in obj_types.items() if t == "hand"}
    surfaces = {e for e, t in obj_types.items() if t == "surface"}

    # ---- build G_pre and G_post graphs ----
    G_pre_graph = meta.get("G_pre") or nx.DiGraph()
    world_frame = meta.get("world_frame", "world")
    involved_hands = meta.get("involved_hands", [h for h in [a1, a2] if h])

    G_post_graph = G_pre_graph.copy()
    for edge, op in meta.get("contact_changes", []):
        u, v = edge
        if op == "add":
            G_post_graph.add_edge(u, v)
        elif op == "remove" and G_post_graph.has_edge(u, v):
            G_post_graph.remove_edge(u, v)

    # ---- classify NET contact changes via GraphDiff ----
    edges_added = sorted(GraphDiff(G_post_graph, G_pre_graph))
    edges_removed = sorted(GraphDiff(G_pre_graph, G_post_graph))

    arm_del = []        # [(hand, obj)]  arm releases object
    arm_add = []        # [(hand, obj)]  arm grasps object
    surface_add = []    # [(obj, surf)]  object placed on surface
    surface_del = []    # [(obj, surf)]  object lifted from surface
    container_add = []  # [(obj, cont)]  object placed in container

    for p, c in edges_removed:
        if p in hands:
            arm_del.append((p, c))
        elif c in hands:
            arm_del.append((c, p))

    for p, c in edges_added:
        if p in hands:
            arm_add.append((p, c))
        elif c in hands:
            arm_add.append((c, p))
        elif c in CONTAINER_NAMES:
            container_add.append((p, c))

    # Object-level holding transitions. Holding(o) is true iff any hand grasps o.
    def _held_objects(graph):
        held = set()
        for u, v in graph.edges():
            if u in hands and obj_types.get(v) not in ("surface", "hand"):
                held.add(v)
            elif v in hands and obj_types.get(u) not in ("surface", "hand"):
                held.add(u)
        return held

    held_pre = _held_objects(G_pre_graph)
    held_post = _held_objects(G_post_graph)
    holding_del = sorted(held_pre - held_post)
    holding_add = sorted(held_post - held_pre)

    # ---- compute G_pre non-default state via GraphDiff ----
    G_0 = DefaultGraphSchema(world_frame, involved_hands)
    e_pre = sorted(GraphDiff(G_pre_graph, G_0))  # deterministic ordering

    # Partition E_pre into pre-existing grasps and surface contacts
    pre_arm_edges = [
        (u, v) for u, v in e_pre
        if u in hands and obj_types.get(v) not in ("surface", "hand")
    ]
    ## Only movable supported objects: a non-movable entity on a surface can never be
    ## re-placed, so an On precondition for it is unsatisfiable if perception omits the
    ## Supported init fact (and surface-surface edges arrive supporter-first, inverting On).
    pre_surface_edges = [
        (u, v) for u, v in e_pre
        if u not in hands and obj_types.get(u) == "movable"
        and obj_types.get(v) == "surface"
    ]

    # Unchanged holds: arms that hold objects both before and after the policy.
    # Their grasps become stale due to policy stochasticity and need refreshing.
    eff_hand_obj_set = set(map(tuple, meta.get("eff_hand_obj_edges", [])))
    arm_del_set = set(arm_del)
    refresh_arm_edges = sorted([
        (h, o) for h, o in pre_arm_edges
        if (h, o) in eff_hand_obj_set
        and (h, o) not in arm_del_set
        and (classified_skills is None or _has_downstream_detach(h, o, skill_index, classified_skills))
    ])

    # ---- map schema-level object/surface names to PDDL variables ----
    obj_vars = {}   # "tripod_obj" -> "?o1"
    o_ctr = [1]

    def _add_obj(name):
        if name not in hands and name not in obj_vars:
            obj_vars[name] = f"?o{o_ctr[0]}"
            o_ctr[0] += 1

    for h, o in pre_arm_edges:
        _add_obj(o)
    for h, o in arm_add:
        _add_obj(o)
    for h, o in arm_del:
        _add_obj(o)
    for o, s in pre_surface_edges:
        _add_obj(o); _add_obj(s)
    for o, s in surface_add:
        _add_obj(o); _add_obj(s)
    for o, s in surface_del:
        _add_obj(o); _add_obj(s)
    for o, c in container_add:
        _add_obj(o); _add_obj(c)

    # ---- assign unique variable names ----
    grasp_vars = {}          # (hand, obj) -> "?gN"  (old / pre-existing grasps)
    refresh_grasp_vars = {}  # (hand, obj) -> "?gN"  (fresh grasps after policy)
    pose_vars = {}           # (obj, surf) -> "?pN"
    g_ctr, p_ctr = [1], [1]

    def _intern(store, ctr, prefix, *key_parts):
        key = key_parts
        if key not in store:
            store[key] = f"?{prefix}{ctr[0]}"
            ctr[0] += 1
        return store[key]

    def _gvar(h, o):        return _intern(grasp_vars, g_ctr, "g", h, o)
    def _grefresh(h, o):    return _intern(refresh_grasp_vars, g_ctr, "g", h, o)
    def _pvar(o, s):        return _intern(pose_vars, p_ctr, "p", o, s)

    # Allocate vars in deterministic order:
    #   pre-existing grasps from E_pre first (bound in both pre and del effects),
    #   then new grasps from arm_add (bound in ImitateGrasp pre and add effects),
    #   then refresh grasps for unchanged holds (fresh from unified biop stream).
    for h, o in pre_arm_edges:
        _gvar(h, o)
    for h, o in arm_add:
        _gvar(h, o)
    for h, o in refresh_arm_edges:
        _grefresh(h, o)
    # Pre-existing surface contacts from E_pre first (bound in pre and del effects),
    # then new surface contacts from surface_add (bound in add effects only).
    for o, s in pre_surface_edges:
        _pvar(o, s)
    for o, s in surface_add:
        _pvar(o, s)

    def _arm_var(h):
        if h == a1:
            return "?a1"
        if h == a2:
            return "?a2"
        return h

    def _ovar(name):
        return obj_vars.get(name, name)

    ctx = _SchemaVarContext(
        arm=_arm_var, ovar=_ovar, gvar=_gvar, grefresh=_grefresh, pvar=_pvar,
    )

    # ---- σ.pre ----
    # binding header + the biop stream's conf bindings (Bound, F_a.domain/certified)
    pre = [f"({a1} ?a1) ({a2} ?a2) ({sk} ?sk)"]
    if planning_mode == DETAILED_MODE:
        pre.append("(ImitateConf ?sk ?a1 ?q1) (ImitateConf ?sk ?a2 ?q2)")
        pre.append("(AtConf ?a1 ?q1) (AtConf ?a2 ?q2)")
    else:
        pre.append("(Conf ?a1 ?q1) (Conf ?a2 ?q2)")
    pre.append("(GeomState ?sk ?lstate)")

    for name, var in obj_vars.items():
        pre.append(f"({name} {var})")

    # EdgeOps2Predicates(E_pre): grasps the arms already hold before this action
    for h, o in pre_arm_edges:
        pre += EdgeOps2Predicates("grasp_pre", h, o, ctx)
    # Derived(G_pre): hands holding no movable are free
    for h in Derived(G_pre_graph, involved_hands, obj_types):
        pre.append(f"(ArmEmpty {_arm_var(h)})")
    # Bound(F_a.certified): grasp outputs the biop stream must produce/refresh
    pre += Bound(arm_add, ctx)
    pre += Bound(refresh_arm_edges, ctx, refresh=True)
    # EdgeOps2Predicates(E_pre): objects already resting on surfaces
    for o, s in pre_surface_edges:
        pre += EdgeOps2Predicates("surface_pre", o, s, ctx)

    if enable_constraints:
        # IsSafePolicy: inlined as a universal so universal_to_conditional compiles it into a
        # conditional UNSATISFIABLE effect linked to the negated CFreeMDF stream -- this makes
        # "move the blocking obstacle" reachable. CFreeMDF is keypose-free (both planning modes).
        pre.append(
            "(forall (?cfo ?cfp) (imply "
            "(and (AtPose ?cfo ?cfp) (Movable ?cfo) (CanPick ?cfo) (SkillCheckObj ?sk ?cfo)) "
            "(CFreeMDF ?cfo ?cfp ?sk)))"
        )

    # ---- σ.eff : Applied marker + contact transitions (EdgeOps2Predicates) ----
    eff = [Applied("?sk")]
    for h, o in arm_del:
        eff += EdgeOps2Predicates("grasp_del", h, o, ctx)
    for h, o in arm_add:
        eff += EdgeOps2Predicates("grasp_add", h, o, ctx)
    for o in holding_del:
        eff += EdgeOps2Predicates("holding_del", o, None, ctx)
    for o in holding_add:
        eff += EdgeOps2Predicates("holding_add", o, None, ctx)
    # Grasp refresh: swap stale grasp with fresh one for unchanged holds
    for h, o in refresh_arm_edges:
        eff += EdgeOps2Predicates("grasp_refresh", h, o, ctx)
    for o, s in surface_del:
        eff += EdgeOps2Predicates("surface_del", o, s, ctx)
    for o, s in surface_add:
        eff += EdgeOps2Predicates("surface_add", o, s, ctx)
    for o, c in container_add:
        eff += EdgeOps2Predicates("container_add", o, c, ctx)
    if planning_mode == DETAILED_MODE:
        eff.append(f"(CanMove {_arm_var(a1)})")
        eff.append(f"(CanMove {_arm_var(a2)})")

    # ---- σ.args ----
    params = SortedArguments(
        ["?a1", "?a2", "?sk"], ["?q1", "?q2"],
        obj_vars, grasp_vars, refresh_grasp_vars, pose_vars, "?lstate",
    )

    return ActionSchema(idx=i, params=params, pre=pre, eff=eff).render()


def BuildActionSchema(classified_skills, skill_names, obj_types,
                      planning_mode=DETAILED_MODE, enable_constraints=False):
    """Build the learned-skill schemata (the learned actions only).

    For each detected skill, the stream matched by ``MatchStreams`` selects which
    learned schema ``σ = (args, pre, eff)`` to render. The traditional schemata
    (transit/transfer/primitive pick/place) are contributed by :func:`build_domain_pddl`.
    """
    actions = []
    for i, meta in enumerate(classified_skills):
        sk = skill_names[i]
        streams = meta["matched_streams"]
        if LEARNED_ATTACH in streams:
            block = _grounded_learned_pick_action(
                meta, i, sk, planning_mode=planning_mode,
                enable_constraints=enable_constraints,
            )
            if block:
                actions.append(block)
        if LEARNED_DETACH in streams:
            surface_grounded = meta.get("grounding_surface") is not None
            block = _grounded_learned_place_action(
                meta, i, sk, surface_grounded, planning_mode=planning_mode,
            )
            if block:
                actions.append(block)
        if LEARNED_BI_KEYPOSE in streams:
            block = _grounded_bioperation_action(
                meta, i, sk, planning_mode=planning_mode, obj_types=obj_types,
                classified_skills=classified_skills, skill_index=i,
                enable_constraints=enable_constraints,
            )
            if block:
                actions.append(block)
    return actions


def build_domain_pddl(classified_skills, schema_arm_names=None, schema_object_names=None,
                      schema_skill_names=None, obj_types=None, planning_mode=DETAILED_MODE,
                      enable_constraints=False):
    """Compose the domain PDDL with per-skill grounded learned actions.

    The schema_*_names lists supply the arm/object/skill names used to bind the
    learned-action predicates; obj_types drives dynamic BiOp schema generation.
    """
    _validate_planning_mode(planning_mode)
    all_streams = set()
    for meta in classified_skills:
        all_streams.update(meta["matched_streams"])

    has_attach = LEARNED_ATTACH in all_streams
    has_detach = LEARNED_DETACH in all_streams
    has_bimanual = LEARNED_BI_KEYPOSE in all_streams

    parts = []
    header = _load_template("domain_header.pddl")
    # if schema_constants:
    #     header = header.replace("{{SCHEMA_CONSTANTS}}", " ".join(schema_constants))
    # else:
    header = header.replace("{{SCHEMA_CONSTANTS}}", "")
    parts.append(header)
    parts.append(_predicates_block(has_attach, has_detach, has_bimanual, schema_arm_names=schema_arm_names, schema_object_names=schema_object_names, schema_skill_names=schema_skill_names, enable_constraints=enable_constraints))

    parts.append("\n  ;--------------------------------------------------\n")

    # In coarse schema-online mode, only keep contact-changing skeleton actions.
    if planning_mode == DETAILED_MODE:
        transit = _load_template("action_transit.pddl")
        transfer = _load_template("action_transfer.pddl")
        pick = _load_reachable_action_template("action_pick.pddl", "pick", enable_constraints)
        parts.append(transit)
        parts.append("")
        parts.append(transfer)
        parts.append("")
        parts.append(pick)
        parts.append("")
    else:
        # When all picks are learned, the generic coarse pick is redundant: it shares
        # (Grasp ?a ?o ?g) with learnedPick_* but wins search on shorter preconditions,
        # causing learnedPick_* (and its DoneSkill effect) to never be expanded.
        if not has_attach:
            parts.append(_load_reachable_action_template("action_pick_coarse.pddl", "pick_coarse", enable_constraints))
            parts.append("")
    place = _build_place_action(planning_mode, enable_constraints=enable_constraints)
    parts.append(place)

    # 𝔄 \ 𝔄₀ : the learned-skill schemata, one σ per detected skill.
    if schema_skill_names is None:
        raise ValueError("schema_skill_names is required for build_domain_pddl")
    for block in BuildActionSchema(
        classified_skills, schema_skill_names, obj_types,
        planning_mode=planning_mode, enable_constraints=enable_constraints,
    ):
        parts.append("")
        parts.append(block)

    parts.append("\n  ;--------------------------------------------------\n")
    parts.append(_load_template("derived_base.pddl"))

    parts.append("\n\n)")

    return "\n".join(parts)


# ============================================================
#  4b. Instantiated stream PDDL generation
# ============================================================

def build_instantiated_stream_specs(classified_skills, env_names, planning_mode=DETAILED_MODE, enable_constraints=False):
    """Build per-skill instantiated stream spec dicts from classified skill metadata.

    Returns a flat list of spec dicts, one per stream per skill (sample-grasp-traj /
    sample-place-traj / sample-biop-keypose, each tagged with its skill index and arms).
    """
    _validate_planning_mode(planning_mode)
    specs = []
    skill_names = compute_skill_names(classified_skills, env_names)
    for i, meta in enumerate(classified_skills):
        streams = meta.get("matched_streams", [])
        sk = skill_names[i]

        if LEARNED_ATTACH in streams:
            arm = meta.get("grounding_arm")
            obj = meta.get("grounding_object")
            if arm and obj:
                specs.append({
                    "name": f"sample-grasp-traj_{i}",
                    "template": "sample-grasp-traj",
                    "skill_index": i,
                    "skill": sk,
                    "arm": arm,
                    "object": obj,
                    "contact_aware": True,
                    "planning_mode": planning_mode,
                })

        if LEARNED_DETACH in streams:
            arm = meta.get("grounding_arm")
            obj = meta.get("grounding_object")
            if arm and obj:
                specs.append({
                    "name": f"sample-place-traj_{i}",
                    "template": "sample-place-traj",
                    "skill_index": i,
                    "skill": sk,
                    "arm": arm,
                    "object": obj,
                    "surface": meta.get("grounding_surface"),
                    "contact_aware": True,
                    "planning_mode": planning_mode,
                })

        if LEARNED_BI_KEYPOSE in streams:
            arm1 = meta.get("grounding_arm1")
            arm2 = meta.get("grounding_arm2")
            raw_eff_grasps = meta.get("eff_hand_obj_edges", [])
            # Enrich each (arm, obj) with the place skill name from the
            # next LEARNED_DETACH skill that removes this edge.
            eff_grasps = []
            for arm, obj in raw_eff_grasps:
                # Keep only grasp refreshes that can be certified by a later
                # LEARNED_DETACH stream (place/post-grasp network).
                if not _has_downstream_detach(arm, obj, i, classified_skills):
                    continue
                place_skill_name = None
                for j in range(i + 1, len(classified_skills)):
                    fwd = classified_skills[j]
                    if (LEARNED_DETACH in fwd.get("matched_streams", [])
                            and fwd.get("grounding_arm") == arm
                            and fwd.get("grounding_object") == obj):
                        place_skill_name = skill_names[j]
                        break
                if place_skill_name is None:
                    # No downstream placement — grasp refresh not needed.
                    continue
                eff_grasps.append([arm, obj, place_skill_name])
            if arm1 and arm2:
                base = {
                    "skill_index": i,
                    "skill": sk,
                    "arm1": arm1,
                    "arm2": arm2,
                    "eff_grasps": eff_grasps,
                    "planning_mode": planning_mode,
                }
                specs.append({
                    "name": f"sample-biop-keypose_{i}",
                    "template": "sample-biop-keypose",
                    "contact_aware": True,
                    **base,
                })
                # MDF policy-safety stream: keypose-free, emitted in both planning modes.
                if enable_constraints:
                    specs.append({
                        "name": f"test-cfree-bioperation-pose_{i}",
                        "template": "test-cfree-bioperation-pose",
                        "contact_aware": False,
                        **base,
                    })

    return specs


# Map stream template name to template file name (in pddl_templates/)
_STREAM_TEMPLATE_FILES = {
    "sample-grasp-traj": "stream_sample_grasp_traj.pddl",
    "test-cfree-bioperation-pose": "stream_test_cfree_bioperation_pose.pddl",
}


def _render_biop_keypose_stream_block(spec):
    """
    Dynamically generate a grounded biop-keypose stream block.

    The output signature varies with the number of eff_grasps:
      0 grasps: outputs (?lc1 ?lc2 ?effGeom)
      1 grasp:  outputs (?lc1 ?lc2 ?effGeom ?g1)  with (?o1 ?p1) as extra inputs
      2 grasps: outputs (?lc1 ?lc2 ?effGeom ?g1 ?g2) with (?o1 ?p1 ?o2 ?p2) as inputs
    """
    name = spec["name"]
    sk = spec["skill"]
    arm1 = spec.get("arm1", "")
    arm2 = spec.get("arm2", "")
    eff_grasps = list(spec.get("eff_grasps", []))
    planning_mode = spec.get("planning_mode", DETAILED_MODE)

    inputs = ["?a1", "?a2", "?sk"]
    outputs = ["?lc1", "?lc2", "?effGeom"]
    domain_terms = [f"({arm1} ?a1)", f"({arm2} ?a2)", "(Skillbimanual ?sk)", f"({sk} ?sk)"]
    extra_certified_terms = []

    for idx, grasp_entry in enumerate(eff_grasps, start=1):
        arm, obj = grasp_entry[0], grasp_entry[1]
        obj_var = f"?o{idx}"
        pose_var = f"?p{idx}"
        grasp_var = f"?g{idx}"
        arm_var = "?a1" if arm == arm1 else "?a2"

        inputs.extend([obj_var, pose_var])
        outputs.append(grasp_var)
        domain_terms.append(f"({obj} {obj_var})")
        domain_terms.append(f"(Pose {obj_var} {pose_var})")
        domain_terms.append(f"(Graspable {obj_var})")
        extra_certified_terms.append(f"(ImitateGrasp ?sk {arm_var} {obj_var} {grasp_var})")
        extra_certified_terms.append(f"(Grasp {arm_var} {obj_var} {grasp_var})")

    domain_str = " ".join(domain_terms)
    inputs_str = " ".join(inputs)
    outputs_str = " ".join(outputs)

    template_name = (
        "stream_sample_biop_keypose_detailed.pddl"
        if planning_mode == DETAILED_MODE
        else "stream_sample_biop_keypose_coarse.pddl"
    )
    template = _load_template(template_name)
    return _fill_template(template, {
        "NAME": name,
        "INPUTS": inputs_str,
        "DOMAIN_TERMS": domain_str,
        "OUTPUTS": outputs_str,
        "EXTRA_CERTIFIED_TERMS": "\n      ".join(extra_certified_terms),
    })


def _render_instantiated_stream_block(spec):
    """
    Render a single grounded PDDL stream block from an instantiated stream spec.
    Loads the stream definition from pddl_templates/ and fills {{NAME}}, {{SK}},
    {{ARM}}, {{OBJ}}, and grounded place stream terms when needed.
    """
    name = spec["name"]
    template = spec["template"]
    sk = spec["skill"]
    planning_mode = spec["planning_mode"]

    # biop-keypose is generated dynamically (variable grasp outputs)
    if template == "sample-biop-keypose":
        return _render_biop_keypose_stream_block(spec)

    if template == "sample-place-traj":
        s = spec.get("surface")
        if s:
            surface_domain = f"({s} ?s) (Region ?s) (Pose ?s ?sp)"
        else:
            surface_domain = "(Region ?s) (Pose ?s ?sp)"
        template_name = (
            "stream_sample_place_traj_grounded_detailed.pddl"
            if planning_mode == DETAILED_MODE
            else "stream_sample_place_traj_grounded_coarse.pddl"
        )
        grounded_template = _load_template(template_name)
        return _fill_template(grounded_template, {
            "NAME": name,
            "ARM": spec.get("arm", ""),
            "OBJ": spec.get("object", ""),
            "SURFACE_DOMAIN": surface_domain,
            "SK": sk,
        })

    template_file = _STREAM_TEMPLATE_FILES.get(template)
    if template_file is None:
        raise ValueError(f"Unknown stream template for instantiation: {template!r}")

    content = _load_template(template_file)
    replacements = {"NAME": name, "SK": sk}

    if template == "sample-grasp-traj":
        replacements["ARM"] = spec.get("arm", "")
        replacements["OBJ"] = spec.get("object", "")
    # test-cfree-bioperation-pose only needs NAME + SK (keypose-free MDF check).

    return _fill_template(content, replacements)


def _remove_named_stream_blocks(stream_pddl, stream_names):
    """Remove top-level ``(:stream NAME ...)`` blocks by name.

    This is more robust than regex removal because several template blocks close
    inline (for example ``:certified (...)``), which can cause regex spans to
    swallow following stream declarations such as ``plan-motion``.
    """
    names = tuple(stream_names)
    out = stream_pddl
    for name in names:
        pattern = re.compile(rf"\(:stream\s+{re.escape(name)}\b")
        while True:
            match = pattern.search(out)
            if match is None:
                break
            start = match.start()
            depth = 0
            end = None
            for index in range(start, len(out)):
                char = out[index]
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                    if depth == 0:
                        end = index + 1
                        break
            if end is None:
                raise ValueError(f"Failed to locate end of stream block {name!r}")

            block_start = start
            while block_start > 0 and out[block_start - 1].isspace():
                block_start -= 1
            block_end = end
            while block_end < len(out) and out[block_end].isspace():
                block_end += 1
            out = out[:block_start] + "\n\n" + out[block_end:]
    return out


def _build_base_stream_pddl(planning_mode, enable_constraints=False):
    base_pddl = _load_template("stream.pddl")
    _validate_planning_mode(planning_mode)
    if planning_mode != DETAILED_MODE:
        base_pddl = _remove_named_stream_blocks(base_pddl, [
            "test-cfree-traj-pose",
            "plan-learned-pick",
            "plan-motion",
            "plan-place",
        ])
    if enable_constraints:
        # Insert the reachability test stream before the template's final ')'.
        close_idx = base_pddl.rfind(")")
        if close_idx < 0:
            raise ValueError("Invalid stream PDDL template: missing final closing ')'")
        base_pddl = (base_pddl[:close_idx].rstrip()
                     + "\n\n" + _REACHABLE_STREAM_BLOCK + "\n\n)")
    return base_pddl


def build_stream_pddl(classified_skills, env_names, planning_mode=DETAILED_MODE, enable_constraints=False):
    """Compose the full stream PDDL: the universal stream.pddl template plus per-skill
    instantiated stream blocks. Returns (stream_pddl_str, instantiated_specs).
    """
    _validate_planning_mode(planning_mode)
    base_pddl = _build_base_stream_pddl(planning_mode, enable_constraints=enable_constraints)

    instantiated_specs = build_instantiated_stream_specs(
        classified_skills, env_names, planning_mode=planning_mode,
        enable_constraints=enable_constraints,
    )
    if not instantiated_specs:
        return base_pddl, instantiated_specs

    rendered_blocks = "\n\n".join(
        block for block in (_render_instantiated_stream_block(s) for s in instantiated_specs)
        if block
    )

    close_idx = base_pddl.rfind(")")
    if close_idx < 0:
        raise ValueError("Invalid stream PDDL template: missing final closing ')'")

    stream_pddl = (
        base_pddl[:close_idx].rstrip()
        + "\n\n  ;-- per-skill instantiated streams --\n\n"
        + rendered_blocks
        + "\n\n)"
    )
    return stream_pddl, instantiated_specs


def _save_instantiated_streams_to_config(config_path, instantiated_specs):
    """
    Write instantiated stream specs into the source schema JSON config.
    Adds/overwrites the top-level "instantiated_streams" field in-place.
    """
    with open(config_path, "r") as f:
        config_data = json.load(f)
    config_data["instantiated_streams"] = instantiated_specs
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)
        f.write("\n")


# ============================================================
#  4c. Init facts for problem file
# ============================================================

def build_init_facts(classified_skills, obj_types, skill_names):
    """
    Build init facts for the problem file: arm roles, object types, skill type facts.

    Returns list of PDDL fact strings, e.g. ["(Arm robot0)", "(left_arm robot0)", ...].
    """
    facts = []


    for i, meta in enumerate(classified_skills):
        sk = skill_names[i]
        if LEARNED_BI_KEYPOSE in meta["matched_streams"]:
            facts.append(f"(Skillbimanual {sk})")

    return facts


def _partition_schema_entities(obj_types):
    """Return schema entities grouped by type in insertion order."""
    hands = [entity for entity, entity_type in obj_types.items() if entity_type == "hand"]
    movables = [entity for entity, entity_type in obj_types.items() if entity_type == "movable"]
    surfaces = [entity for entity, entity_type in obj_types.items() if entity_type == "surface"]
    return hands, movables, surfaces


def _build_schema_outputs(initial_graph, skills, env_names, planning_mode, objects_dict=None, enable_constraints=False):
    """Run the shared schema classification and PDDL generation pipeline."""
    _validate_planning_mode(planning_mode)
    if not env_names:
        raise ValueError("env_names is required for schema construction")

    obj_types = infer_object_types(initial_graph, skills, objects_dict)
    classified = classify_skills(skills, obj_types, initial_graph)
    schema_skill_names = compute_skill_names(classified, env_names)
    init_facts = build_init_facts(classified, obj_types, skill_names=schema_skill_names)
    print_summary(initial_graph, skills, classified, obj_types, init_facts=init_facts)

    hands, movables, surfaces = _partition_schema_entities(obj_types)
    domain_pddl = build_domain_pddl(
        classified,
        schema_arm_names=hands,
        schema_object_names=movables + surfaces,
        schema_skill_names=schema_skill_names,
        obj_types=obj_types,
        planning_mode=planning_mode,
        enable_constraints=enable_constraints,
    )
    stream_pddl, instantiated_specs = build_stream_pddl(
        classified, env_names, planning_mode=planning_mode,
        enable_constraints=enable_constraints,
    )
    return {
        "domain_pddl": domain_pddl,
        "stream_pddl": stream_pddl,
        "instantiated_specs": instantiated_specs,
        "init_facts": init_facts,
    }


def _write_schema_outputs(output_dir, domain_name, domain_pddl, stream_pddl, init_facts):
    """Persist generated schema artifacts and return their file paths."""
    os.makedirs(output_dir, exist_ok=True)
    domain_path = os.path.join(output_dir, f"{domain_name}_domain.pddl")
    stream_path = os.path.join(output_dir, f"{domain_name}_stream.pddl")
    init_facts_path = os.path.join(output_dir, f"{domain_name}_init_facts.txt")

    with open(domain_path, "w") as f:
        f.write(domain_pddl)
    print(f"Domain PDDL written to: {domain_path}")

    with open(stream_path, "w") as f:
        f.write(stream_pddl)
    print(f"Stream PDDL (instantiated) written to: {stream_path}")

    with open(init_facts_path, "w") as f:
        f.write("\n".join(init_facts))
    print(f"Init facts written to: {init_facts_path}")

    return domain_path, stream_path, init_facts_path


# ============================================================
#  5. Summary / Debug Output
# ============================================================

def print_summary(initial_graph, skills, classified, obj_types, init_facts=None):
    """Print a human-readable summary of the parsed config and MatchStreams results."""
    print("=" * 60)
    print("BuildActionSchema -- Summary")
    print("=" * 60)

    print(f"\nObject types inferred from config:")
    for entity, etype in sorted(obj_types.items()):
        print(f"  {entity:25s} -> {etype}")

    print(f"\nInitial graph edges ({len(initial_graph)}):")
    for e in initial_graph:
        print(f"  {e[0]} -> {e[1]}")

    print(f"\nSkills detected ({len(skills)}):")
    for i, (sk, cl) in enumerate(zip(skills, classified)):
        hands_str = ", ".join(sorted(cl["involved_hands"])) or "(none)"
        objs_str = ", ".join(sorted(cl["involved_objects"])) or "(none)"
        streams_str = ", ".join(cl["matched_streams"])
        prim_str = "primitive" if cl["is_primitive"] else "policy"
        print(
            f"  [{i+1}] {sk['description']:25s}  "
            f"type={prim_str:10s}  "
            f"streams=[{streams_str}]  "
            f"hands=[{hands_str}]  objects=[{objs_str}]"
        )

    all_streams = set()
    for cl in classified:
        all_streams.update(cl["matched_streams"])
    print(f"\nAction templates to include:")
    print(f"  Transit, Transfer (always)")
    if LEARNED_ATTACH in all_streams:
        print(f"  learnedPick   (LearnedAttach)")
    if LEARNED_DETACH in all_streams:
        print(f"  learnedPlace  (LearnedDetach)")
    if LEARNED_BI_KEYPOSE in all_streams:
        print(f"  BiOperation   (LearnedBiKeyPose)")

    print(f"\nStream templates to include:")
    print(f"  test-cfree-*, plan-motion, plan-pick, plan-place, sample-placement (always)")
    if LEARNED_ATTACH in all_streams:
        print(f"  sample-grasp-traj           (LearnedAttach)")
    if LEARNED_DETACH in all_streams:
        print(f"  sample-place-traj           (LearnedDetach)")
    if LEARNED_BI_KEYPOSE in all_streams:
        print(f"  sample-biop-keypose         (LearnedBiKeyPose, with eff_grasps)")

    if init_facts:
        print(f"\nInit facts (for problem file):")
        for fact in init_facts:
            print(f"  {fact}")

    print()


# ============================================================
#  6. Main -- BuildActionSchema entry point
# ============================================================

def build_action_schema_from_data(initial_graph, skills, env_names, output_dir=None,
                                  domain_name="composed",
                                  planning_mode=DETAILED_MODE, objects_dict=None,
                                  enable_constraints=False):
    """
    Like build_action_schema() but from in-memory (initial_graph, skills) data.
    Returns (domain_pddl, stream_pddl). Writes files only if output_dir is provided.
    """
    outputs = _build_schema_outputs(initial_graph, skills, env_names, planning_mode, objects_dict=objects_dict, enable_constraints=enable_constraints)
    domain_pddl = outputs["domain_pddl"]
    stream_pddl = outputs["stream_pddl"]

    if output_dir is not None:
        _write_schema_outputs(
            output_dir,
            domain_name,
            domain_pddl,
            stream_pddl,
            outputs["init_facts"],
        )

    return domain_pddl, stream_pddl


def build_action_schema(config_path, output_dir=None, domain_name=None,
                        planning_mode=DETAILED_MODE, enable_constraints=False):
    """Main entry point: parse the JSON config and write the domain/stream PDDL files.

    output_dir defaults to the config's directory and domain_name to its filename stem.
    Returns (domain_pddl, stream_pddl).
    """
    # Parse config inputs
    initial_graph, skills, objects_dict = parse_config(config_path)
    env_name = os.path.splitext(os.path.basename(config_path))[0].replace("_changes", "")
    outputs = _build_schema_outputs(initial_graph, skills, [env_name], planning_mode, objects_dict=objects_dict, enable_constraints=enable_constraints)
    domain_pddl = outputs["domain_pddl"]
    stream_pddl = outputs["stream_pddl"]
    instantiated_specs = outputs["instantiated_specs"]

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(config_path))
    if domain_name is None:
        domain_name = os.path.splitext(os.path.basename(config_path))[0]

    _write_schema_outputs(
        output_dir,
        domain_name,
        domain_pddl,
        stream_pddl,
        outputs["init_facts"],
    )
    _save_instantiated_streams_to_config(config_path, instantiated_specs)

    return domain_pddl, stream_pddl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BuildActionSchema: generate PDDL domain and stream files from a contact-change config."
    )
    parser.add_argument(
        "config",
        nargs="?",
        # default="examples/pybullet/aloha_real/openworld_aloha/configs/dmg_cfgs/two_arm_three_piece_assembly_changes.json",
        # default="examples/pybullet/aloha_real/openworld_aloha/configs/dmg_cfgs/two_arm_threading_changes.json",
        default="examples/pybullet/aloha_real/openworld_aloha/configs/aloha_cfgs/handoff_cup_changes.json",
        # default="examples/pybullet/aloha_real/openworld_aloha/configs/aloha_cfgs/screwdriver_changes.json",
        help="Path to the JSON config file.",
    )
    parser.add_argument(
        "--output-dir", "-o", default=None,
        help="Output directory (default: same as config).",
    )
    parser.add_argument(
        "--name", "-n", default=None,
        help="Base name for output files (default: config filename stem).",
    )
    args = parser.parse_args()

    build_action_schema(
        args.config,
        output_dir=args.output_dir,
        domain_name=args.name,
    )
