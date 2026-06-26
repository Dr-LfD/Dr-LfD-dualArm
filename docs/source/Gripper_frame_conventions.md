# Gripper frame conventions

> Scope: this page concerns the **simulation** execution path (robosuite/mujoco).
> The corrections below are specific to that controller; the real ALOHA executor
> uses joint-space trajectories and does not need them.

Several gripper/tool frames coexist in this project and they do **not** agree on
where the finger-closing axis points. Mixing them up shows as a grasp that is
rotated 90° about the approach axis (gripper descends correctly but the jaws
close across the wrong direction). This page records every convention and the
exact corrections used in code.

All Panda-style frames here share the same approach axis (+Z points out of the
flange toward the fingertips); they differ by a **yaw about Z** and, between
the pybullet and robosuite models, by a small **translation along Z**
(section 3 below).

## Frame definitions

### PyBullet (planning side)

| Frame | Robot | Mounted on | Yaw about Z | Source |
|---|---|---|---|---|
| `tool_link` | single Panda (LIBERO) | `panda_link8` | **+135°** | `examples/pybullet/utils/models/franka_description/robots/panda_arm_hand.urdf`, `tool_joint` `rpy="0 0 2.35619449"`, z-offset 0.1 m |
| `panda1_ee_link` / `panda2_ee_link` | dual Panda (DMG) | hand | 0° (orientation-equivalent to robosuite `right_hand`; position is hand + 0.100 m, see section 3) | `dmg_dual_panda.urdf` |
| `panda_hand` | single Panda | `panda_link8` | −45° | same URDF, `panda_hand_joint` `rpy="0 0 -0.785398"` |

The IK/planning target everywhere is the **tool frame** (`Manipulator.tool_name`),
so grasp poses, `ee_path` waypoints and FK results are all expressed in it.

### Robosuite / MuJoCo (execution side)

| Frame | Mounted on | Yaw about Z | Source |
|---|---|---|---|
| `right_hand` | link7 (≈ link8) | **−45°** | `robosuite/models/assets/robots/panda/robot.xml`, quat `0.924 0 0 -0.383` |
| PandaGripper root | `right_hand` | **−90°** | `robosuite/models/assets/grippers/panda_gripper.xml`, quat `0.707107 0 0 -0.707107` |
| `grip_site` | gripper `eef` body | 0° (relative to gripper root) | same XML |

So the frame the OSC_POSE controller tracks is
`grip_site = link8 ∘ Rz(−45°) ∘ Rz(−90°) = link8 ∘ Rz(−135°)`.
The fingers of the robosuite PandaGripper separate along the **local Y axis of
`grip_site`**.

### Grasp predictors

| Source | Closing axis | Approach axis | Note |
|---|---|---|---|
| M2T2 (contact-graspnet convention) | local **X** | +Z | jaws separate along X of the predicted grasp pose |
| Panda EE frame used by planning/IK | local **Y** | +Z | what `get_jspace_path` / IK expects |
| GPD | converted at the backend boundary to the tool frame (`world_from_tool`) | +Z | no extra roll needed downstream |

## Corrections applied in code

### 1. M2T2 → EE roll (upstream, M2T2 only)

`_M2T2_TO_EE_ROLL` (+90° roll about local Z) in
`examples/pybullet/aloha_real/openworld_aloha/openworld_streams.py` is applied to
every M2T2 candidate in `gen_m2t2_attach_traj` **before** the grasp-depth shift.
It converts the contact-graspnet X-closing convention to the Y-closing EE frame.
This is shared by joint-space and OSC execution; do not touch it when debugging
OSC-only symptoms.

### 2. tool frame → grip_site (downstream, OSC execution only)

`DMG_OSC_planner.GRIP_SITE_CORRECTION`
(`examples/pybullet/aloha_real/scripts/interleaved_dmg_osc_plugin.py`) is right-multiplied
onto every FK'd / learned tool orientation in `pose_to_osc_slice` so the
commanded pose lands in the `grip_site` frame the OSC controller tracks:

| Robot | pybullet tool frame | required correction `C` (`tool ∘ C = grip_site`) | Set in |
|---|---|---|---|
| DMG dual Panda | `panda*_ee_link`, orientation ≡ `right_hand` | **Rz(−90°)** | class default, `interleaved_dmg_osc_plugin.py` |
| LIBERO single Panda | `tool_link` = link8 ∘ Rz(+135°) | **Rz(+90°)** (= Rz(−135° − 135°) mod 360°) | `interleaved_libero_plugin.py` (`Libero_OSC_planner`); `generic_grasp_plugin.py` aliases it |

Joint-space execution needs **no** such correction — it commands joints
directly, so the tool/grip_site mismatch never enters the loop. That is why a
wrong `GRIP_SITE_CORRECTION` shows up only under `--osc`, and for **all** grasp
backends at once (M2T2 *and* GPD), whereas a wrong `_M2T2_TO_EE_ROLL` affects
M2T2 in both execution modes.

### 3. tool frame → grip_site translation (downstream, OSC execution only)

The yaw corrections above fix orientation, but the two models also disagree on
**where** the tracked point sits along the approach axis. The planning and
execution models place their end-effector frames at different distances from
the wrist, and OSC_POSE servos the robosuite point, not the pybullet one.

Walking both kinematic chains from `panda_link7` (the last joint-bearing link,
identical in both models) out to the frame each side actually uses:

| Side | Chain from link7 | Distance along approach axis | Sources |
|---|---|---|---|
| PyBullet (planning) | link7 → link8 (**+0.107**) → hand/tool mount (**+0**) → tool frame (**+0.100**) | **0.207 m** | `dmg_dual_panda.urdf` `panda*_joint8` + `panda*_ee_joint`; `panda_arm_hand.urdf` `tool_joint` (identical numbers for both robots) |
| Robosuite (execution) | link7 → `right_hand` (**+0.1065**) → PandaGripper root (**+0**) → `eef` body / `grip_site` (**+0.097**) | **0.2035 m** | `robosuite/models/assets/robots/panda/robot.xml:221`; `robosuite/models/assets/grippers/panda_gripper.xml` (`<body name="eef" pos="0 0 0.097">`) |

The mismatch has **two independent contributions**:

1. the wrist mount: robosuite's `right_hand` sits at link7 + 0.1065 m, but the
   URDF `link8` (which the hand/tool mount on) sits at link7 + 0.107 m → 0.5 mm;
2. the gripper depth: the URDF tool frames are 0.100 m past the hand, but
   robosuite's `grip_site` is only 0.097 m past it → 3.0 mm.

Net: the pybullet tool frame is **3.5 mm further out along +Z (toward the
fingertips)** than the `grip_site` the OSC controller tracks. Expressed from
link8: `tool = link8 + 0.100 m`, `grip_site = link8 + 0.0965 m`. The chain is
numerically identical for the DMG dual Panda and the LIBERO single Panda — the
robots differ only in the yaw conventions of section 2.

**Failure mode when uncorrected.** Planning, IK, and the learned `ee_path` all
express grasp poses in the tool frame. If the executor passes the FK'd tool
*position* straight through as the OSC target, the controller drives
`grip_site` — a point 3.5 mm *short* of the tool frame — onto the spot the
planner chose for the tool frame. To get `grip_site` there, the whole hand must
advance 3.5 mm, so the executed grasp lands **3.5 mm too deep along the
approach axis**. Observed consequences on the DMG tasks (top-down grasps):
slightly inaccurate grasp poses, and after the grasp the *perceived* arm
configuration sat low enough that pybullet replanning rejected it as in
collision with the table (`initial configuration is in collision` → barrier
planning abort).

**Correction.** `DMG_OSC_planner.GRIP_SITE_POS_OFFSET = (0, 0, −0.0035)`
(`interleaved_dmg_osc_plugin.py`), a translation in the **local tool frame**, added to every
commanded position in `pose_to_osc_slice`:

```
cmd_pos = tool_pos + R_tool · (0, 0, −0.0035)
```

i.e. the commanded `grip_site` target is placed 3.5 mm short of the planned
tool position along the approach axis, so that when `grip_site` converges
there, the tool frame ends up exactly where the planner intended. Because all
yaw corrections in section 2 rotate about local Z, they leave the local Z axis
invariant — so applying the offset with the *uncorrected* tool rotation is
exact, and the same class default is correct for both robots (it is **not**
overridden per robot, unlike `GRIP_SITE_CORRECTION`).

As with the yaw corrections, joint-space execution is immune: it never converts
through Cartesian space, so this offset must stay OSC-only.

## Verification record

2026-06-12: with `GRIP_SITE_CORRECTION = Rz(+90°)` for the single Panda, both
`generic_grasp_plugin.py --osc --grasp_backend m2t2` and `--grasp_backend gpd`
completed `pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate`
(`task_success: True`); frame dumps at gripper closure show the bowl rim
spanning between the finger pads. Previously (identity correction) both
backends grasped 90° off.

2026-06-12 (later the same day): added `GRIP_SITE_POS_OFFSET = (0, 0, −0.0035)`.
A/B on DMG `two_arm_threading` OSC runs (one run each, so indicative rather
than statistical): with the gripper settle/freeze/dwell hold but *without* the
offset, the run aborted at the first replan with
`initial configuration is in collision` (arm driven 3.5 mm into the table at
the grasp); with the offset added, the pick phase and the following subgoal
executed cleanly and the abort did not occur — consistent with the geometry
above.
