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

| Macro | Module | Required? |
|---|---|---|
| `${DIFFUSION_POLICY_ROOT}` | Diffusion-Policy code repo | Optional — defaults to `<repo>/diffusion_policy_sg` submodule |
| `${PRIMITIVE_LEARNING_ROOT}` | equiv_primitive (object-centric primitive learning) | Optional — defaults to `<repo>/primitive_learning` submodule |
| `${CONTACT_PREDICTION_ROOT}` | contact-prediction (real-robot) | Optional — defaults to `<repo>/contact_detector` submodule |
| `${SPHERICAL_DP_ROOT}` | Spherical Diffusion Policy | Experimental `*_sdp.yaml` only |
| `${DEXMIMICGEN_ROOT}` | dexmimicgen | Experimental `*_sdp.yaml` only |

The sim route needs no `.env` at all once submodules are initialized: the three
learned-policy macros default to the vendored submodules
(`<repo>/{diffusion_policy_sg,primitive_learning,contact_detector}`) after `git
submodule update --init`. Set a macro only to point at a separate development
checkout (an explicit environment variable always wins); `WS_ROOT` is only
needed for real-robot data paths. Copy the template if you need any override:

```bash
cp .env.example .env && source .env
```

Any other unset macro is left literal so a missing path fails loudly rather than
silently resolving to the wrong location.

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

- **Sibling-repo paths** — `DP_path` (`${DIFFUSION_POLICY_ROOT}`, or
  `${SPHERICAL_DP_ROOT}` in `*_sdp.yaml`), `primitive_learning_path`
  (`${PRIMITIVE_LEARNING_ROOT}`). Both default to their vendored submodules (see
  [Installation](Installation.md) step 5).
- **`LfD_params`** — `lfd_alg` (`DP` or `SDP`), `env_type: dmg`, and `DP_input`
  (per-task checkpoint paths).
- **`sg_params`** — learned-skill checkpoints: `equi_ckpt_name` (equiv_primitive grasp
  keypoints), `biop_ckpt_name` (bimanual operator). Overridable at runtime with
  `--sg KEY=VALUE`.

## Checkpoints

The two runnable DMG configs reference their checkpoints under
`${REPO_ROOT}/ckpt/dmg_<task>/{equi.pth,biop.pth,policy.ckpt}`, fetched by:

```bash
python download_ckpts.py            # both tasks
python download_ckpts.py --task threading
```

Override per run without editing YAML (both expand `${VAR}`/`~`):

- `--dp_ckpt PATH` — Diffusion-Policy checkpoint.
- `--sg equi_ckpt_name=PATH` — equiv_primitive keypoint checkpoint (and any other
  `sg_params` field), e.g. `--sg equi_ckpt_name='${REPO_ROOT}/ckpt/dmg_threading/equi.pth'`.

## Environment variables (runtime)

| Variable | Purpose |
|---|---|
| `WS_ROOT` | Workspace root for sibling repos (consumed by `${WS_ROOT}` in configs) |
| `CUDA_VISIBLE_DEVICES` | GPU selection |
| `MUJOCO_GL=egl` | Headless rendering |
| `PYTHONPATH` | Must include the repo root and `examples/pybullet/aloha_real/scripts` |
