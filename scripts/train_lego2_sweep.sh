#!/usr/bin/env bash
# Sequential Lego2 training for use inside screen/tmux.
#
# Usage:
#   screen -S lego2_sweep   # or: screen -r yhy1
#   cd ~/yhy/mjlab
#   bash scripts/train_lego2_sweep.sh
#   # Ctrl+A D to detach
#
# Logs:
#   scripts/logs/lego2_sweep_<timestamp>.log          (full console output)
#   logs/rsl_rl/lego2_velocity/rough/<run_name>/      (checkpoints + TB)
#
# Shared baseline (env_cfgs):
#   - no command_vel curriculum; fixed lin_x=(-1,1), ang_z=(-1,1)
#   - track_ang.w=3 std=sqrt(0.25); fric promote/demote 2.55/2.1
#   - require_spin_inplace; cmd_gate=0.2; rel_spin_envs=0.2
#   - flat=1.0; side sync off
#
# This sweep (3 runs, sequential):
#   0) baseline: track_lin.w=3, std=sqrt(0.5)  (env default)
#   1) A: track_lin std=sqrt(0.25), weight=3
#   2) B: track_lin weight=4, std=sqrt(0.5)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PATH="${HOME}/.local/bin:${PATH}"

NUM_ENVS=4096
MAX_ITERS=10000
LOGGER="tensorboard"

LOG_DIR="${ROOT}/scripts/logs"
mkdir -p "$LOG_DIR"
SWEEP_LOG="${LOG_DIR}/lego2_sweep_$(date +%Y%m%d_%H%M%S).log"

FLAT1="--env.scene.terrain.terrain-generator.sub-terrains.flat.proportion 1.0 --env.scene.terrain.terrain-generator.sub-terrains.random-rough.proportion 0.0"
NOSIDE="--env.rewards.wheel-side-sync-exp.weight 0.0"

# math.sqrt(0.25) == 0.5
LIN_STD_TIGHT="--env.rewards.track-lin-vel-xy-exp.params.std 0.5"
LIN_W4="--env.rewards.track-lin-vel-xy-exp.weight 4.0"

declare -a JOBS=(
  # 0) Baseline replay: lin=(-1,1) fixed, default track_lin (w=3, std=√0.5).
  "posw1c15_friccurr_v6_spin02_lin11_base_flat|Mjlab-Velocity-Rough-Lego2|${FLAT1} ${NOSIDE}"
  # 1) Tighter lin tracking std; weight stays 3.
  "posw1c15_friccurr_v6_spin02_lin11_std025_flat|Mjlab-Velocity-Rough-Lego2|${FLAT1} ${NOSIDE} ${LIN_STD_TIGHT}"
  # 2) Heavier lin tracking weight; std stays sqrt(0.5).
  "posw1c15_friccurr_v6_spin02_lin11_linw4_flat|Mjlab-Velocity-Rough-Lego2|${FLAT1} ${NOSIDE} ${LIN_W4}"
)

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$SWEEP_LOG"
}

if ! command -v uv >/dev/null 2>&1; then
  log "ERROR: uv not found. Add ~/.local/bin to PATH or install uv."
  exit 1
fi

log "Sweep started: ${#JOBS[@]} jobs"
log "num_envs=${NUM_ENVS} max_iters=${MAX_ITERS}"
log "Base: no lin curriculum; lin_x=(-1,1); spin gate; flat=1"
log "0: baseline track_lin  1: std=0.5(=√0.25)  2: track_lin.w=4"
log "Full log: ${SWEEP_LOG}"

job_idx=0
for job in "${JOBS[@]}"; do
  job_idx=$((job_idx + 1))
  run_name="${job%%|*}"
  rest="${job#*|}"
  task="${rest%%|*}"
  extra="${rest#*|}"

  log "========== JOB ${job_idx}/${#JOBS[@]}: ${run_name} =========="
  log "Task: ${task}"
  log "Extra args: ${extra:-<none>}"

  # shellcheck disable=SC2086
  if uv run train "$task" \
    --env.scene.num-envs "$NUM_ENVS" \
    --agent.max-iterations "$MAX_ITERS" \
    --agent.run-name "$run_name" \
    --agent.logger "$LOGGER" \
    $extra 2>&1 | tee -a "$SWEEP_LOG"; then
    log "========== DONE: ${run_name} =========="
  else
    log "========== FAILED: ${run_name} (exit $?) — continuing to next job =========="
  fi
done

log "All jobs finished."
