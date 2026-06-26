# Troubleshooting

## PandaGripper Rz(−90°) gripper offset (simulation)

**Symptom:** When executing trajectories in simulation via
`interleaved_dmg_osc_plugin.py`, the gripper rotates 90° on every command — both
motion-planning segments (joint-space FK) and learned-grasp segments (EE-traj
from the DMG policy). This is specific to the simulation controller; the real
ALOHA executor does not use it.

**Root cause:** `panda_gripper.xml` mounts the gripper root body with a −90°
rotation around Z:

```xml
<!-- robosuite .../models/assets/grippers/panda_gripper.xml -->
<body name="right_gripper" pos="0 0 0" quat="0.707107 0 0 -0.707107">
```

The simulation controller targets `grip_site` (inside this body), so its world
orientation is `right_hand_world × Rz(−90°)`. The two sources that feed the
controller are both in the `right_hand` frame (without the −90°):

- **pybullet FK** of `panda2_ee_link` / `panda1_ee_link` — these links share the
  same orientation as `right_hand`.
- **DMG policy output** — training data uses `eef_quat = get_body_xquat(eef_name)`
  with `eef_name = "right_hand"`.

**Fix:** Post-multiply every outgoing orientation by `Rz(−90°)` inside
`pose_to_osc_slice` in `interleaved_dmg_osc_plugin.py`:

```python
_GRIP_SITE_CORRECTION = Rotation.from_euler('z', -np.pi / 2)

def pose_to_osc_slice(pose, gripper_scalar):
    pos, quat_xyzw = pose
    corrected = Rotation.from_quat(quat_xyzw) * _GRIP_SITE_CORRECTION
    axisangle = quat2axisangle(corrected.as_quat())
    return list(pos) + list(axisangle) + [float(gripper_scalar)]
```

See [Gripper frame conventions](Gripper_frame_conventions.md) for the full frame
discussion.

## `XML Error: ... unrecognized attribute: 'actuatorfrclimited'`

mujoco/robosuite version mismatch. robosuite 1.5.1 requires mujoco 3.x; the
pinned `mujoco==3.3.4` in `requirements.txt` satisfies this. See "Known version
pitfalls" in [Installation](Installation.md).

## dexmimicgen environment not found

The dexmimicgen task envs are only registered when the robosuite fork is on its
`switchable` / dexmimicgen-compatible branch — see [Installation](Installation.md)
step 4.
