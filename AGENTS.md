# Log output

- Write all runtime logs, evaluation results, diagnostic reports, and their
  attachments under the repository-root `logs/` directory.
- Use a task or run subdirectory within `logs/`; resolve its path from the
  repository root rather than the current working directory.
- Keep `docs/` for maintained documentation. Put captured command output and
  temporary investigation artifacts in `logs/`.
- Configure file logging and command redirection to use `logs/` before running
  a task. Keep generated logs out of version control.
- Keep experiment outputs, including generated schemas and videos, under
  `logs/<task-or-run>/`; do not write them under `scripts/eval_results/`.

# Training checkpoints

- Delete checkpoints from abandoned training runs and candidates superseded by
  a better checkpoint; keep an unqualified run's best checkpoint until its
  failure diagnosis is done. Retain only the best checkpoint from the active experiment,
  selected by the agreed evaluation metric. Keep experiment configs, metrics,
  and evaluation records under `logs/`.

# Policy training objectives

- Keep policy losses task-independent. DP uses the canonical diffusion action
  prediction loss. Do not add task-specific object keypoints, poses, alignment,
  contact, or success objectives to the loss.
- Task-specific labels and simulator measurements may be used for offline
  diagnosis and evaluation, including checkpoint selection by task success.

# Controlled experiments

- Keep the existing 100 DP training demos (source0-99) fixed. Do not add or
  replace demonstrations, including recovery demonstrations. This is a hard
  constraint for all improvement proposals and experiments.
- DP retraining starts from random initialization. Initialize a new model, EMA,
  optimizer and scheduler; fit the normalizer on the training demos. Existing DP
  checkpoints provide configuration and evaluation references only.
- Train for 2000 complete epochs. Resume or warm-start training only when the
  user explicitly requests it; never infer permission from an existing checkpoint.
- Define the baseline explicitly and change one experimental factor at a time.
  Fix training demos, initialization, training budget, optimizer/scheduler/EMA,
  preprocessing, inference settings, evaluation starts, and checkpoint selection
  across paired runs except for the named factor. Report differences from the
  original checkpoint separately from differences between experimental runs.
