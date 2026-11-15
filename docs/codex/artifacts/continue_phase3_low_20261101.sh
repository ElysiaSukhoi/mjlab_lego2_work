#!/usr/bin/env bash
# Continue the verified smoke policy with 1024 environments, 1000 new updates.
set -euo pipefail
case "${STY:-}" in
  *.zrq) ;;
  *) printf '请先进入screen zrq。\n' >&2; exit 1 ;;
esac
PHASE3_LOW_UV="$HOME/.local/bin/uv"
test -x "$PHASE3_LOW_UV"
cd "$HOME/zrq/mjlab_lego2"
PHASE3_LOW_SOURCE="$HOME/zrq/phase3_eval_sources_20261101.sha256"
test -f "$PHASE3_LOW_SOURCE"
# Validate train sources/assets/tests; the local-only progress tool is not required.
awk '$2 ~ /^(src|tests)\// {print}' "$PHASE3_LOW_SOURCE" | sha256sum -c -
PHASE3_LOW_PARENT="2026-10-31_21-58-03_phase3_smoke_128x200_v1_"
PHASE3_LOW_PARENT+="20261031_215751_2154252"
PHASE3_LOW_CKPT="logs/rsl_rl/lego2_blind_pit/$PHASE3_LOW_PARENT/model_199.pt"
"$PHASE3_LOW_UV" run --no-sync python - "$PHASE3_LOW_CKPT" <<'PY'
from pathlib import Path
import hashlib
import sys
import torch
p = Path(sys.argv[1])
assert p.is_file(), p
expected = '901cc02246fcd47517a0791ceec4513cc6725a1d8d0786955ab3c945ddb1b2f4'
assert hashlib.sha256(p.read_bytes()).hexdigest() == expected, 'checkpoint hash不符'
assert torch.cuda.is_available(), 'CUDA不可用，未启动续训'
free, _ = torch.cuda.mem_get_info(0)
print(torch.cuda.get_device_name(0), 'free MiB:', free // 2**20)
assert free >= 12 * 2**30, '空闲显存不足12 GiB，未启动1024环境训练'
lines = Path('/proc/meminfo').read_text().splitlines()
mem = dict(line.split(':', 1) for line in lines)
assert int(mem['MemAvailable'].split()[0]) >= 8 * 2**20, '可用内存不足8 GiB'
PY
PHASE3_LOW_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$"
PHASE3_LOW_EVIDENCE="$HOME/zrq/backups/phase3_low_$PHASE3_LOW_STAMP"
mkdir "$PHASE3_LOW_EVIDENCE"
PHASE3_LOW_RUN="phase3_low_1024x1000_v1_$PHASE3_LOW_STAMP"
cp "$PHASE3_LOW_SOURCE" "$PHASE3_LOW_EVIDENCE/"
cp "$HOME/zrq/continue_phase3_low_20261101.sh" "$PHASE3_LOW_EVIDENCE/"
sha256sum "$PHASE3_LOW_CKPT" > "$PHASE3_LOW_EVIDENCE/parent_checkpoint.sha256"
free -m | tee "$PHASE3_LOW_EVIDENCE/memory_before.txt"
nvidia-smi | tee "$PHASE3_LOW_EVIDENCE/gpu_before.txt"
export FORCE_CPU=0
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="$HOME/zrq/backups/phase3_20261031_214701_2152912/cache"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export MPLCONFIGDIR="$PHASE3_LOW_EVIDENCE/cache/matplotlib"
printf '续训父模型：%s\n新run后缀：%s\n证据目录：%s\n' \
  "$PHASE3_LOW_CKPT" "$PHASE3_LOW_RUN" "$PHASE3_LOW_EVIDENCE"
printf '计划新增1000次更新、24576000样本；原迭代编号会恢复。\n'
printf '权重/优化器状态恢复，仿真重新reset，高度从3 cm起步；保留原run。\n'
PHASE3_LOW_CMD=(
  "$PHASE3_LOW_UV" run --no-sync train Mjlab-Blind-Pit-Lego2
  --env.scene.num-envs 1024
  --agent.max-iterations 1000
  --agent.seed 7
  --agent.resume True
  --agent.load-run "^$PHASE3_LOW_PARENT$"
  --agent.load-checkpoint '^model_199[.]pt$'
  --agent.logger tensorboard
  --agent.run-name "$PHASE3_LOW_RUN"
  --gpu-ids '[0]'
)
printf '%q ' "${PHASE3_LOW_CMD[@]}" > "$PHASE3_LOW_EVIDENCE/command.sh"
printf '\n' >> "$PHASE3_LOW_EVIDENCE/command.sh"
set +e
"${PHASE3_LOW_CMD[@]}" 2>&1 | tee "$PHASE3_LOW_EVIDENCE/train_stdout.txt"
PHASE3_LOW_STATUS="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$PHASE3_LOW_STATUS" > "$PHASE3_LOW_EVIDENCE/exit_code.txt"
PHASE3_LOW_LOGROOT=logs/rsl_rl/lego2_blind_pit
if [ -d "$PHASE3_LOW_LOGROOT" ]; then
  find "$PHASE3_LOW_LOGROOT" -mindepth 1 -maxdepth 1 -type d \
    -name "*_$PHASE3_LOW_RUN" -print > "$PHASE3_LOW_EVIDENCE/run_dirs.txt"
  while IFS= read -r PHASE3_LOW_DIR; do
    find "$PHASE3_LOW_DIR" -type f \
      \( -name 'model_*.pt' -o -name 'env.yaml' -o -name 'agent.yaml' \) -print
  done < "$PHASE3_LOW_EVIDENCE/run_dirs.txt" \
    | tee "$PHASE3_LOW_EVIDENCE/run_files.txt"
fi
printf '续训退出码：%s\n证据目录：%s\n' \
  "$PHASE3_LOW_STATUS" "$PHASE3_LOW_EVIDENCE"
exit "$PHASE3_LOW_STATUS"
