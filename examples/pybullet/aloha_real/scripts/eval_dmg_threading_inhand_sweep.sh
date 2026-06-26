#!/usr/bin/env bash
# Sweep two_arm_threading over the two in-hand keypoint checkpoints, N runs each,
# recording task_success per run. Overrides only sg_params.equi_ckpt_name via the
# plugin's --sg hook (no YAML edit). Env mirrors release-route verification:
# conda env sdp_dmg, scripts/ on PYTHONPATH, MUJOCO_GL=egl, GPU0.
set -u

REPO_ROOT="/home/user/yzchen_ws/TAMP-ubuntu22/pddlstream_aloha"
SCRIPTS_DIR="${REPO_ROOT}/examples/pybullet/aloha_real/scripts"
ENV_NAME="sdp_dmg"
TASK="two_arm_threading"
RUNS="${RUNS:-20}"
PER_RUN_TIMEOUT="${PER_RUN_TIMEOUT:-600}"   # seconds; bounds the rare resample hang

CKPTS=(
  "logs/train/dmg_threading/inhand-saliency.pth"
  "logs/train/dmg_threading/inhand-fps.pth"
)

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${SCRIPTS_DIR}/eval_out/inhand_sweep_${TS}"
mkdir -p "${OUT_DIR}"
CSV="${OUT_DIR}/results.csv"
echo "ckpt,run,success,exit_code,seconds,logfile" > "${CSV}"

export WS_ROOT="/home/user/yzchen_ws"
export CUDA_VISIBLE_DEVICES="0"
export MUJOCO_GL="egl"
export PYTHONPATH="${REPO_ROOT}:${SCRIPTS_DIR}"

run_one() {
  local ckpt="$1" idx="$2"
  local tag; tag="$(basename "${ckpt}" .pth)"
  local log="${OUT_DIR}/${tag}_run${idx}.log"
  local start end secs rc success

  start=$(date +%s)
  timeout "${PER_RUN_TIMEOUT}" conda run --no-capture-output -n "${ENV_NAME}" \
    python "${SCRIPTS_DIR}/interleaved_dmg_osc_plugin.py" \
      --task_name "${TASK}" \
      --sg "equi_ckpt_name=${ckpt}" \
    > "${log}" 2>&1
  rc=$?
  end=$(date +%s); secs=$((end - start))

  # task_success printed as part of: Execution results: {... 'task_success': True ...}
  if [ ${rc} -eq 124 ]; then
    success="TIMEOUT"
  elif grep -q "'task_success': True" "${log}"; then
    success="True"
  elif grep -q "'task_success': False" "${log}"; then
    success="False"
  else
    success="ERROR"
  fi

  echo "${tag},${idx},${success},${rc},${secs},${log}" >> "${CSV}"
  printf '[%s] %-16s run %2d/%d -> %-7s (%ds, rc=%d)\n' \
    "$(date +%H:%M:%S)" "${tag}" "${idx}" "${RUNS}" "${success}" "${secs}" "${rc}"
}

echo "Sweep: ${#CKPTS[@]} ckpts x ${RUNS} runs -> ${OUT_DIR}"
for ckpt in "${CKPTS[@]}"; do
  for i in $(seq 1 "${RUNS}"); do
    run_one "${ckpt}" "${i}"
  done
done

echo "=== summary ==="
for ckpt in "${CKPTS[@]}"; do
  tag="$(basename "${ckpt}" .pth)"
  ntrue=$(grep -c "^${tag},.*,True," "${CSV}")
  echo "${tag}: ${ntrue}/${RUNS} success"
done | tee "${OUT_DIR}/summary.txt"
echo "CSV: ${CSV}"
