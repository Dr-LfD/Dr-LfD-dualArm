# Configuration

## Path placeholders (`.env`)

External dependency paths are never hardcoded. Configs and scripts reference them
through `${VAR}` placeholders, expanded by `repo_paths.py`:

| Placeholder | Meaning |
|---|---|
| `${REPO_ROOT}` | This repository. Resolved automatically (via a `.repo_root` sentinel). |
| `${WS_ROOT}` | Your external workspace for shared assets (e.g. SAM). You set this. |
| `${HOME}` | Standard home directory. |

Each external dependency **module** has its own dedicated macro (rather than a
`${WS_ROOT}/<module>` sub-path), so a checkout can live anywhere:

| Macro | Module |
|---|---|
| `${PRIMITIVE_LEARNING_ROOT}` | equiv_primitive (object-centric primitive learning) |
| `${DIFFUSION_POLICY_ROOT}` | Diffusion-Policy (visuomotor policies + checkpoints) |
| `${SPHERICAL_DP_ROOT}` | Spherical Diffusion Policy (SDP configs) |
| `${DEXMIMICGEN_ROOT}` | dexmimicgen (sim env / data) |
| `${CONTACT_PREDICTION_ROOT}` | contact-prediction (real-robot) |

Set up:

```bash
cp .env.example .env       # then set WS_ROOT and each module macro
source .env
```

Macros have **no default** — an unset one is left literal so a missing path fails
loudly rather than silently resolving to the wrong location.

## Task configs

Per-task configuration lives under
`examples/pybullet/aloha_real/openworld_aloha/configs/dmg_cfgs/`:

| File | Role |
|---|---|
| `two_arm_threading.yaml` | Threading task config (paths, skills, LfD params) |
| `two_arm_three_piece_assembly.yaml` | Assembly task config |
| `*_sdp.yaml` | Spherical-Diffusion-Policy variant of the above |
| `*_changes.json` | Per-skill contact-change edges consumed by schema construction |

### Key fields

- **Sibling-repo paths** — `DP_path` / `DP_dir` (`${DIFFUSION_POLICY_ROOT}`, or
  `${SPHERICAL_DP_ROOT}` in `*_sdp.yaml`), `primitive_learning_path`
  (`${PRIMITIVE_LEARNING_ROOT}`, equiv_primitive), `env_dir` (`${DEXMIMICGEN_ROOT}`),
  `contact_prediction_root` (`${CONTACT_PREDICTION_ROOT}`). Point each macro at
  your checkout in `.env` (see [Installation](Installation.md) step 5).
- **`LfD_params`** — `lfd_alg` (`DP` or `SDP`), `env_type: dmg`, and `DP_input`
  (per-task checkpoint paths).
- **`sg_params`** — learned-skill checkpoints: `equi_ckpt_name` (equiv_primitive grasp
  keypoints), `biop_ckpt_name` (bimanual operator). Overridable at runtime with
  `--sg KEY=VALUE`.

## Checkpoints

Checkpoint locations are config-driven (`DP_input`, `equi_ckpt_name`,
`biop_ckpt_name`), typically under the Diffusion-Policy repo's
`data/outputs/<task>/` directory. Override per run without editing YAML:

- `--dp_ckpt PATH` — Diffusion-Policy checkpoint.
- `--sg equi_ckpt_name=PATH` — equiv_primitive keypoint checkpoint (and any other
  `sg_params` field).

## Environment variables (runtime)

| Variable | Purpose |
|---|---|
| `WS_ROOT` | Workspace root for sibling repos (consumed by `${WS_ROOT}` in configs) |
| `CUDA_VISIBLE_DEVICES` | GPU selection |
| `MUJOCO_GL=egl` | Headless rendering |
| `PYTHONPATH` | Must include the repo root and `examples/pybullet/aloha_real/scripts` |
