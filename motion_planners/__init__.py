import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_IMPL = os.path.join(_ROOT, "examples", "pybullet", "utils", "motion", "motion_planners")

if not os.path.isdir(_IMPL):
    raise ModuleNotFoundError(
        "Missing motion_planners sources at examples/pybullet/utils/motion/motion_planners."
    )

__path__ = [_IMPL]
