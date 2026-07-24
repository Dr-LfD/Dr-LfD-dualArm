def _lower(value):
    return (value or "").lower()


def arm_name_for_mode(schema_arm, use_side_prefix):
    arm_norm = _lower(schema_arm)
    is_left = ("0" in arm_norm) or ("left" in arm_norm)
    if use_side_prefix:
        return "left" if is_left else "right"
    return "robot0" if is_left else "robot1"


def canonical_bimanual_skill_name(env_name):
    env_norm = env_name or ""
    if env_norm == "bimanual_0":
        return env_norm
    if env_norm.startswith("bimanual_"):
        return env_norm
    return f"bimanual_{env_norm}"


def canonical_unimanual_skill_name(meta, use_side_prefix):
    arm = meta.get("grounding_arm")
    obj = meta.get("grounding_object")
    desc = "_".join((meta.get("description") or "").lower().split())
    if not (arm and obj and desc):
        raise ValueError('invalid skill name')
    arm_prefix = arm_name_for_mode(arm, use_side_prefix)
    surf = meta.get("grounding_surface")
    if surf:
        return f"{arm_prefix}_{desc}_{obj}_{surf}"
    return f"{arm_prefix}_{desc}_{obj}"


_POLICY_ARM_PREFIXES = (
    "left_arm_",
    "right_arm_",
    # "robot0_",
    # "robot1_",
    "left_",
    "right_",
)


def policy_skill_name(skill_name):
    """Map PDDL/schema skill names to equiv_primitive checkpoint embedding keys."""
    for prefix in _POLICY_ARM_PREFIXES:
        if skill_name.startswith(prefix):
            return skill_name[len(prefix):]
    return skill_name


_PRIMITIVE_KEYWORDS = ("grasp", "pick", "place", "drop", "release", "detach", "lift")


def _is_sub_multiset(sub_tokens, super_tokens):
    """True if every token appears in super_tokens at least as often as in sub_tokens."""
    return all(sub_tokens.count(tok) <= super_tokens.count(tok) for tok in set(sub_tokens))


def resolve_policy_skill_name(skill_name, available_keys=None):
    """Resolve a schema skill name to the embedding key a checkpoint carries.

    Schema names may append grounding-only tokens the checkpoint omits (e.g. the
    placement surface in ``place_cup_table`` vs. the trained ``place_cup``) and may
    order object tokens differently. An exact key always wins; otherwise resolve to
    the unique, most-specific checkpoint key that shares the skill's primitive
    keyword and whose tokens are a sub-multiset of the schema name's. An ambiguous
    match resolves to nothing so the caller fails loudly rather than silently
    invoking the wrong policy.
    """
    name = policy_skill_name(skill_name)
    if not available_keys or name in available_keys:
        return name

    name_tokens = name.split("_")
    primitives = set(name_tokens) & set(_PRIMITIVE_KEYWORDS)
    if len(primitives) != 1:
        return name
    primitive = next(iter(primitives))

    candidates = [
        key for key in available_keys
        if set(key.split("_")) & set(_PRIMITIVE_KEYWORDS) == {primitive}
        and _is_sub_multiset(key.split("_"), name_tokens)
    ]
    if not candidates:
        return name

    best_len = max(len(key.split("_")) for key in candidates)
    best = [key for key in candidates if len(key.split("_")) == best_len]
    return best[0] if len(best) == 1 else name


def resolve_skill_env_key(skill_name, equiv_skill_info_dict):
    if skill_name in equiv_skill_info_dict:
        return skill_name
    for env_key, env_info in equiv_skill_info_dict.items():
        skillwise_sgs = env_info.get("skillwise_sgs", {})
        if skill_name in skillwise_sgs:
            return env_key
        if skill_name in env_info.get("skill_names", []):
            return env_key
    return skill_name


def build_skill_to_env_map(equiv_skill_info_dict):
    mapping = {}
    for env_key, env_info in equiv_skill_info_dict.items():
        mapping[env_key] = env_key
        for skill_name in env_info.get("skillwise_sgs", {}).keys():
            mapping[skill_name] = env_key
        for skill_name in env_info.get("skill_names", []):
            mapping[skill_name] = env_key
    return mapping
