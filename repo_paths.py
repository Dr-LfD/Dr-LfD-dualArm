"""Portable repository path resolution.

Single source of truth for locating the repository root and for expanding
environment-variable placeholders inside configuration files. Nothing here
depends on the repository directory being named ``pddlstream_aloha`` -- the
root is identified by the committed ``.repo_root`` marker -- so the repository
can be renamed or vendored without breaking path resolution.

Placeholders understood by :func:`load_yaml`:

* ``${REPO_ROOT}`` -- absolute path of this repository (always available).
* ``${WS_ROOT}``   -- the user's external workspace holding sibling
                      dependency repos (Diffusion-Policy, equibot, SAM, ...).
                      Set it in your environment; see ``.env.example``.
* ``${HOME}``      -- standard home directory (for ``~/.cache`` etc.).

Unset variables are left verbatim so a missing path fails loudly at use time
instead of resolving to a silently wrong location.
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


# This module lives at the repository root, so its own directory is the root in
# the common case; fall back to a marker walk if the file was relocated.
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


def load_yaml(path) -> Any:
    """Load a YAML file, recursively expanding ``${VAR}`` placeholders in strings."""
    os.environ.setdefault("REPO_ROOT", REPO_ROOT)
    with open(os.path.expanduser(path), "r") as stream:
        return _expand(yaml.load(stream, Loader=yaml.FullLoader))
