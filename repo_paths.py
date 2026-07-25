"""Portable repository path resolution.

Single source of truth for locating the repository root and for expanding
environment-variable placeholders inside configuration files. Nothing here
depends on the repository directory being named ``pddlstream_aloha`` -- the
root is identified by the committed ``.repo_root`` marker -- so the repository
can be renamed or vendored without breaking path resolution.

Placeholders understood by :func:`load_yaml`:

* ``${REPO_ROOT}`` -- absolute path of this repository (always available).
* ``${WS_ROOT}``   -- the user's external workspace (SAM and other shared
                      assets). Set it in your environment.
* ``${HOME}``      -- standard home directory (for ``~/.cache`` etc.).

Each external dependency module has its own dedicated macro rather than a
``${WS_ROOT}/<module>`` sub-path, so a repo can live anywhere:

* ``${PRIMITIVE_LEARNING_ROOT}``  -- equiv_primitive; defaults to ``<repo>/primitive_learning``.
* ``${DIFFUSION_POLICY_ROOT}``    -- Diffusion-Policy code repo; defaults to ``<repo>/diffusion_policy_sg``.
* ``${SPHERICAL_DP_ROOT}``        -- Spherical Diffusion Policy (experimental ``*_sdp`` configs).
* ``${DEXMIMICGEN_ROOT}``         -- dexmimicgen (experimental ``*_sdp`` configs only).
* ``${CONTACT_PREDICTION_ROOT}``  -- contact-prediction; defaults to ``<repo>/contact_detector``.

An explicit environment variable wins over the submodule defaults. Any other
unset variable is left verbatim so a missing path fails loudly at use time.
"""

import os
from typing import Any

import yaml

_ROOT_MARKER = ".repo_root"
_FALLBACK_MARKERS = (_ROOT_MARKER, "package.xml", "CMakeLists.txt")


def find_repo_root(start=None):
    """Walk upward from ``start`` (default: this file) until a root marker is found."""
    current = os.path.abspath(start or __file__)
    if os.path.isfile(current):
        current = os.path.dirname(current)
    while True:
        if any(os.path.exists(os.path.join(current, m)) for m in _FALLBACK_MARKERS):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            raise FileNotFoundError(
                "Could not locate the repository root: none of "
                f"{_FALLBACK_MARKERS} found above {start or __file__}."
            )
        current = parent


# Use this module's directory as the root, unless the module was relocated.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(REPO_ROOT, _ROOT_MARKER)):
    REPO_ROOT = find_repo_root()


def repo_path(*parts):
    """Absolute path under the repository root."""
    return os.path.join(REPO_ROOT, *parts)


def env_path(var, default=None, required=False):
    """Resolve an external-dependency path from an environment variable.

    Raises ``EnvironmentError`` when ``required`` is set and neither the
    variable nor ``default`` provides a value, so missing configuration is
    surfaced explicitly rather than defaulting silently.
    """
    value = os.environ.get(var) or default
    if not value:
        if required:
            raise EnvironmentError(
                f"Environment variable {var!r} is not set. Define it in your shell "
                "or copy .env.example and source it (see the README)."
            )
        return None
    return os.path.expanduser(value)


def _expand(node) -> Any:
    if isinstance(node, str):
        return os.path.expanduser(os.path.expandvars(node))
    if isinstance(node, list):
        return [_expand(item) for item in node]
    if isinstance(node, dict):
        return {key: _expand(item) for key, item in node.items()}
    return node


# Map environment variables to their default submodule paths and sentinels.
_SUBMODULE_DEFAULTS = {
    "PRIMITIVE_LEARNING_ROOT": ("primitive_learning", "equiv_primitive"),
    "CONTACT_PREDICTION_ROOT": ("contact_detector", "contact_pred"),
    "DIFFUSION_POLICY_ROOT": ("diffusion_policy_sg", "diffusion_policy"),
}


def _apply_submodule_defaults():
    for var, (subdir, sentinel) in _SUBMODULE_DEFAULTS.items():
        candidate = os.path.join(REPO_ROOT, subdir)
        if os.path.isdir(os.path.join(candidate, sentinel)):
            os.environ.setdefault(var, candidate)


def load_yaml(path) -> Any:
    """Load a YAML file, recursively expanding ``${VAR}`` placeholders in strings."""
    os.environ.setdefault("REPO_ROOT", REPO_ROOT)
    _apply_submodule_defaults()
    with open(os.path.expanduser(path), "r") as stream:
        return _expand(yaml.load(stream, Loader=yaml.FullLoader))
