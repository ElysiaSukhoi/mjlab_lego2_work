#!/usr/bin/env bash
# Run only the first 128-env / 200-iteration PPO smoke in screen zrq.
set -euo pipefail

case "${STY:-}" in
  *.zrq) ;;
  *) printf '请先进入screen zrq，再执行本脚本。\n' >&2; exit 1 ;;
esac
if command -v uv >/dev/null 2>&1; then
  PHASE3_SMOKE_UV="$(command -v uv)"
else
  PHASE3_SMOKE_UV="$HOME/.local/bin/uv"
fi
test -x "$PHASE3_SMOKE_UV"
cd "$HOME/zrq/mjlab_lego2"
sha256sum -c "$HOME/zrq/phase3_smoke_sources.sha256"

# GPU 0 here is the first device visible to this shell, as used by train --gpu-ids '[0]'.
"$PHASE3_SMOKE_UV" run --no-sync python - <<'PY'
from pathlib import Path
import torch
assert torch.cuda.is_available(), 'CUDA不可用，未开训'
free, total = torch.cuda.mem_get_info(0)
print(torch.cuda.get_device_name(0), 'free MiB:', free // 2**20)
assert free >= 8 * 2**30, '空闲显存不足8 GiB，未开训'
mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
assert int(mem['MemAvailable'].split()[0]) >= 4 * 2**20, '可用内存不足4 GiB，未开训'
PY
free -m
nvidia-smi --query-gpu=name,memory.free,driver_version --format=csv
PHASE3_SMOKE_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$"
PHASE3_SMOKE_EVIDENCE="$HOME/zrq/backups/phase3_smoke_$PHASE3_SMOKE_STAMP"
mkdir "$PHASE3_SMOKE_EVIDENCE"
PHASE3_SMOKE_RUN="phase3_smoke_128x200_v1_$PHASE3_SMOKE_STAMP"
cp "$HOME/zrq/phase3_smoke_sources.sha256" "$PHASE3_SMOKE_EVIDENCE/"
cp "$HOME/zrq/start_phase3_smoke_20261031.sh" "$PHASE3_SMOKE_EVIDENCE/"
free -m > "$PHASE3_SMOKE_EVIDENCE/memory_before.txt"
nvidia-smi > "$PHASE3_SMOKE_EVIDENCE/gpu_before.txt"
"$PHASE3_SMOKE_UV" --version > "$PHASE3_SMOKE_EVIDENCE/uv.txt"
# Explicitly override a stale FORCE_CPU from a diagnostic shell.
export FORCE_CPU=0
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$PHASE3_SMOKE_EVIDENCE/cache/matplotlib"
# Reuse the cache from the successful phase3 checks, retaining all prior evidence.
export XDG_CACHE_HOME="$HOME/zrq/backups/phase3_20261031_214701_2152912/cache"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
printf '新run后缀：%s\n短训证据：%s\n计划采样量：128×24×200=614400（实际完成量以训练输出为准）\n' \
  "$PHASE3_SMOKE_RUN" "$PHASE3_SMOKE_EVIDENCE"
PHASE3_SMOKE_CMD=(
  "$PHASE3_SMOKE_UV" run --no-sync train Mjlab-Blind-Pit-Lego2
  --env.scene.num-envs 128
  --agent.max-iterations 200
  --agent.seed 7
  --agent.resume False
  --agent.logger tensorboard
  --agent.run-name "$PHASE3_SMOKE_RUN"
  --gpu-ids '[0]'
)
printf '%q ' "${PHASE3_SMOKE_CMD[@]}" > "$PHASE3_SMOKE_EVIDENCE/command.sh"
printf '\n' >> "$PHASE3_SMOKE_EVIDENCE/command.sh"
set +e
"${PHASE3_SMOKE_CMD[@]}" 2>&1 | tee "$PHASE3_SMOKE_EVIDENCE/train_stdout.txt"
PHASE3_SMOKE_STATUS="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$PHASE3_SMOKE_STATUS" > "$PHASE3_SMOKE_EVIDENCE/exit_code.txt"
PHASE3_SMOKE_LOGROOT="logs/rsl_rl/lego2_blind_pit"
if [ -d "$PHASE3_SMOKE_LOGROOT" ]; then
  find "$PHASE3_SMOKE_LOGROOT" -mindepth 1 -maxdepth 1 -type d \
    -name "*_$PHASE3_SMOKE_RUN" -print > "$PHASE3_SMOKE_EVIDENCE/run_dirs.txt"
  while IFS= read -r PHASE3_SMOKE_DIR; do
    find "$PHASE3_SMOKE_DIR" -type f \
      \( -name 'model_*.pt' -o -name 'agent.yaml' -o -name 'env.yaml' \) -print
  done < "$PHASE3_SMOKE_EVIDENCE/run_dirs.txt" \
    | tee "$PHASE3_SMOKE_EVIDENCE/run_files.txt"
fi
printf '短训进程退出码：%s\n证据目录：%s\n' "$PHASE3_SMOKE_STATUS" "$PHASE3_SMOKE_EVIDENCE"
exit "$PHASE3_SMOKE_STATUS"
