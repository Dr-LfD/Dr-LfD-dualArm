#!/usr/bin/env bash
# Success-rate sweep for the DMG OSC two_arm_threading task across a grid of
# biop (handoff jpose) checkpoints x diffusion-policy checkpoints.
#
# Checkpoints are injected purely in-memory via the plugin's --sg / --dp_ckpt
# flags, so the shared two_arm_threading.yaml is never mutated.
#
# Smoke test (one combo, one run, measure duration first):
#   RUNS=1 BIOP_LABELS=far DP_LABELS=1000demo bash eval_dmg_threading_sweep.sh
#
# Full sweep (2 biop x 5 dp x 20 = 200 runs):
#   RUNS=20 bash eval_dmg_threading_sweep.sh
#
# Env-var overrides:
#   RUNS             trials per combo                         (default: 20)
#   PER_RUN_TIMEOUT  per-run wall-clock cap in seconds        (default: 1200)
#   BIOP_LABELS      space-separated subset of biop labels    (default: all)
#   DP_LABELS        space-separated subset of dp labels      (default: all)

set -euo pipefail

TASK_NAME=two_arm_threading
RUNS=${RUNS:-20}
PER_RUN_TIMEOUT=${PER_RUN_TIMEOUT:-1200}
PYTHON=${PYTHON:-/home/user/miniforge3/envs/sdp_dmg/bin/python}

PLUGIN_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$PLUGIN_DIR/../../../.." && pwd)"
SCRIPT="$PLUGIN_DIR/interleaved_dmg_osc_plugin.py"
DP_DIR=/home/user/yzchen_ws/TAMP-ubuntu22/ALOHA/Diffusion-Policy/data/outputs/two_arm_threading

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYTHONPATH="$REPO_ROOT:$PLUGIN_DIR${PYTHONPATH:+:$PYTHONPATH}"

# label -> biop checkpoint (relative path, resolved like the yaml's value)
declare -A BIOP=(
    [far]=logs/train/dmg_threading/ckpt_perskill_jpose_far.pth
    [near]=logs/train/dmg_threading/ckpt_perskill_jpose_near.pth
)
ALL_BIOP=(far near)

# label -> DP checkpoint (absolute path, as in the yaml)
declare -A DP=(
    [1000demo]="$DP_DIR/1000demo0.500.ckpt"
    [500demos]="$DP_DIR/500demos_0.340.ckpt"
    [100demo]="$DP_DIR/100demo_0.240.ckpt"
    [sg_singleview]="$DP_DIR/sg_singleview_100demo_0.780.ckpt"
    [sg_100demo]="$DP_DIR/sg_100demo_0.6.ckpt"
)
ALL_DP=(1000demo 500demos 100demo sg_singleview sg_100demo)

# Optional subset selection.
read -ra biop_sel <<< "${BIOP_LABELS:-${ALL_BIOP[*]}}"
read -ra dp_sel <<< "${DP_LABELS:-${ALL_DP[*]}}"

# Per-combo result, filled as the sweep runs and read back for the summary matrix
# (matrix.tsv is the on-disk artifact; the matrix is printed from memory).
declare -A CELL

STAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$PLUGIN_DIR/eval_results/dmg_threading_sweep/$STAMP"
SUMMARY="$RESULTS_DIR/summary.txt"
MATRIX="$RESULTS_DIR/matrix.tsv"
mkdir -p "$RESULTS_DIR"

log() { echo "$*" | tee -a "$SUMMARY"; }

printf 'biop\tdp\tsuccess\truns\n' > "$MATRIX"

log "=== DMG OSC $TASK_NAME sweep | $(date) ==="
log "RUNS=$RUNS  PER_RUN_TIMEOUT=${PER_RUN_TIMEOUT}s  PYTHON=$PYTHON"
log "biop=${biop_sel[*]}  dp=${dp_sel[*]}"
log "Results in: $RESULTS_DIR"
log ""

total_succ=0
total_runs=0

for biop_label in "${biop_sel[@]}"; do
    biop_ckpt="${BIOP[$biop_label]:?unknown biop label '$biop_label'}"
    for dp_label in "${dp_sel[@]}"; do
        dp_ckpt="${DP[$dp_label]:?unknown dp label '$dp_label'}"
        combo="${biop_label}__${dp_label}"
        combo_dir="$RESULTS_DIR/$combo"
        mkdir -p "$combo_dir"

        combo_succ=0
        log "--- biop=$biop_label  dp=$dp_label ---"

        for ((run = 1; run <= RUNS; run++)); do
            logfile="$combo_dir/run${run}.log"

            set +e
            timeout -k 30 "$PER_RUN_TIMEOUT" \
                "$PYTHON" "$SCRIPT" \
                    --task_name "$TASK_NAME" \
                    --sg "biop_ckpt_name=$biop_ckpt" \
                    --dp_ckpt "$dp_ckpt" \
                &> "$logfile"
            exit_code=$?
            set -e

            if grep -q "'task_success': True" "$logfile"; then
                status=PASS
                ((combo_succ++)) || true
            elif [[ $exit_code -eq 124 || $exit_code -eq 137 ]]; then
                status=TIMEOUT
            else
                status=FAIL
            fi
            echo "  run $run/$RUNS: $status  exit=$exit_code  (log: $logfile)"
        done

        ((total_succ += combo_succ)) || true
        ((total_runs += RUNS)) || true
        printf '%s\t%s\t%s\t%s\n' "$biop_label" "$dp_label" "$combo_succ" "$RUNS" >> "$MATRIX"
        CELL["$biop_label,$dp_label"]="$combo_succ/$RUNS"
        log "  RESULT: $combo  $combo_succ / $RUNS"
        log ""
    done
done

log "=== SUMMARY MATRIX (success/runs) ==="
{
    printf '%-10s' 'biop\dp'
    for dp_label in "${dp_sel[@]}"; do printf '  %-16s' "$dp_label"; done
    printf '\n'
    for biop_label in "${biop_sel[@]}"; do
        printf '%-10s' "$biop_label"
        for dp_label in "${dp_sel[@]}"; do
            printf '  %-16s' "${CELL[$biop_label,$dp_label]:--}"
        done
        printf '\n'
    done
} | tee -a "$SUMMARY"

log ""
log "=== OVERALL: $total_succ / $total_runs ==="
log "TSV matrix: $MATRIX"
