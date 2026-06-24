import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_IMPL = os.path.join(_ROOT, "examples", "pybullet", "utils", "pybullet_tools")

if not os.path.isdir(_IMPL):
    raise ModuleNotFoundError(
        "Missing pybullet_tools sources at examples/pybullet/utils/pybullet_tools. "
        "Initialize or restore examples/pybullet/utils."
    )

__path__ = [_IMPL]
