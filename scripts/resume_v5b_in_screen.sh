#!/usr/bin/env bash
# Resume posw1c15_friccurr_v5b from latest checkpoint (for use inside screen yhy1).
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="${HOME}/.local/bin:${PATH}"

RUN_DIR="2026-09-04_15-47-34_posw1c15_friccurr_v5b_ang1_lin3_rough06"
LOG_ROOT="logs/rsl_rl/lego2_velocity/rough"
CKPT_DIR="${LOG_ROOT}/${RUN_DIR}"

LATEST_ITER=$(ls "${CKPT_DIR}"/model_*.pt | sed 's/.*model_//;s/.pt//' | sort -n | tail -1)
CKPT="model_${LATEST_ITER}.pt"
REMAIN=$((10000 - LATEST_ITER))

if (( REMAIN <= 0 )); then
  echo "Already at or past 10000 (iter=${LATEST_ITER}); nothing to resume."
  exit 0
fi

echo "[INFO] Resuming from ${CKPT_DIR}/${CKPT}"
echo "[INFO] current_iter=${LATEST_ITER} remaining=${REMAIN} (target 10000)"

exec uv run train Mjlab-Velocity-Rough-Lego2 \
  --env.scene.num-envs 4096 \
  --agent.max-iterations "${REMAIN}" \
  --agent.run-name posw1c15_friccurr_v5b_ang1_lin3_rough06 \
  --agent.logger tensorboard \
  --agent.resume True \
  --agent.load-run "${RUN_DIR}" \
  --agent.load-checkpoint "${CKPT}" \
  --env.scene.terrain.terrain-generator.sub-terrains.flat.proportion 0.4 \
  --env.scene.terrain.terrain-generator.sub-terrains.random-rough.proportion 0.6 \
  --env.rewards.wheel-side-sync-exp.weight 0.0 \
  --env.rewards.track-lin-vel-xy-exp.weight 3.0
