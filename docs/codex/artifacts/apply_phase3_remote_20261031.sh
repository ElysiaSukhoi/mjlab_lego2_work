#!/usr/bin/env bash
# Stage 3 checks only; no PPO, dependency sync, or old-result deletion.
set -euo pipefail

# Preflight BEFORE modifying the checkout (non-interactive SSH PATH is minimal).
if [ -n "${PHASE3_UV_BIN:-}" ]; then
  test -x "$PHASE3_UV_BIN"
elif command -v uv >/dev/null 2>&1; then
  PHASE3_UV_BIN="$(command -v uv)"
else
  for PHASE3_CANDIDATE in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
    if [ -x "$PHASE3_CANDIDATE" ]; then
      PHASE3_UV_BIN="$PHASE3_CANDIDATE"
      break
    fi
  done
fi
if [ -z "${PHASE3_UV_BIN:-}" ]; then
  printf '未找到uv，请通过PHASE3_UV_BIN传入绝对路径。\n' >&2
  exit 127
fi
printf '使用uv：%s\n' "$PHASE3_UV_BIN"
"$PHASE3_UV_BIN" --version
if [ "${1:-}" = '--check-uv' ]; then exit 0; fi

cd "$HOME/zrq/mjlab_lego2"
if [ "${1:-}" = '--checks-only' ]; then
  PHASE3_BACKUP="${2:?请传入本批实际备份目录}"
  test -d "$PHASE3_BACKUP"
else
  PHASE3_BUNDLE="$HOME/zrq/lego2_phase3_20261031_v1.tar.gz"
  PHASE3_BACKUP="$HOME/zrq/backups/phase3_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$"
  mkdir -p "$PHASE3_BACKUP/files"
  printf '阶段3备份位置：%s\n' "$PHASE3_BACKUP"
  tar -xOf "$PHASE3_BUNDLE" docs/codex/artifacts/phase3_prerequisites.sha256 \
    > "$PHASE3_BACKUP/prerequisites.sha256"
  sha256sum -c "$PHASE3_BACKUP/prerequisites.sha256" \
    | tee "$PHASE3_BACKUP/prerequisites_check.txt"
  tar -xOf "$PHASE3_BUNDLE" docs/codex/artifacts/phase3_files.txt \
    > "$PHASE3_BACKUP/incoming_files.txt"
  while IFS= read -r PHASE3_FILE; do
    if [ -e "$PHASE3_FILE" ] || [ -L "$PHASE3_FILE" ]; then
      cp -a --parents -- "$PHASE3_FILE" "$PHASE3_BACKUP/files/"
    else
      printf '%s\n' "$PHASE3_FILE" >> "$PHASE3_BACKUP/previously_absent.txt"
    fi
  done < "$PHASE3_BACKUP/incoming_files.txt"
  tar -xzf "$PHASE3_BUNDLE" --no-same-owner --verbatim-files-from \
    --files-from "$PHASE3_BACKUP/incoming_files.txt"
  sha256sum -c docs/codex/artifacts/phase3_sync.sha256 \
    | tee "$PHASE3_BACKUP/sync_check.txt"
fi

PHASE3_CHECK="$PHASE3_BACKUP/check_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$PHASE3_CHECK"
printf '阶段3验证目录：%s\n' "$PHASE3_CHECK"
# Subsequent check-only invocations verify code, without reverting context receipts.
while read -r PHASE3_HASH PHASE3_FILE; do
  case "$PHASE3_FILE" in
    src/*|tests/*|scripts/*) printf '%s  %s\n' "$PHASE3_HASH" "$PHASE3_FILE" ;;
  esac
done < docs/codex/artifacts/phase3_sync.sha256 > "$PHASE3_CHECK/source.sha256"
test "$(wc -l < "$PHASE3_CHECK/source.sha256")" -ge 6
sha256sum -c "$PHASE3_CHECK/source.sha256"
export UV_CACHE_DIR="$PHASE3_BACKUP/cache/uv"
export MPLCONFIGDIR="$PHASE3_BACKUP/cache/matplotlib"
export XDG_CACHE_HOME="$PHASE3_BACKUP/cache"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export FORCE_CPU=1
free -m | tee "$PHASE3_CHECK/memory.txt"
"$PHASE3_UV_BIN" run --no-sync python -m compileall -q \
  src/mjlab/tasks/velocity/config/lego2/blind_pit_env_cfg.py \
  src/mjlab/tasks/velocity/config/lego2/blind_pit_curriculums.py \
  src/mjlab/tasks/velocity/config/lego2/blind_pit_metrics.py \
  scripts/tools/evaluate_lego2_blind_pit.py \
  tests/test_lego2_blind_pit_stage3.py
"$PHASE3_UV_BIN" run --no-sync python -m mjlab.scripts.list_envs --keyword Lego2 \
  | tee "$PHASE3_CHECK/list_envs.txt"
"$PHASE3_UV_BIN" run --no-sync pytest -q tests/test_lego2_blind_pit_terrain.py \
  tests/test_lego2_blind_pit.py tests/test_observation_history.py \
  tests/test_observation_manager.py tests/test_lego2_blind_pit_stage3.py | tee "$PHASE3_CHECK/cpu_tests.txt"

if [ "${PHASE3_GPU_CHECK:-0}" = 1 ]; then
  "$PHASE3_UV_BIN" run --no-sync python - <<'PY' | tee "$PHASE3_CHECK/gpu.txt"
import torch
assert torch.cuda.is_available(), 'GPU验证待确认：CUDA不可用'
free, total = torch.cuda.mem_get_info()
print(torch.cuda.get_device_name(), 'free MiB:', free // 2**20)
assert free >= 3 * 2**30, 'GPU验证待确认：空闲显存不足3 GiB，未启动仿真'
PY
  FORCE_CPU=0 "$PHASE3_UV_BIN" run --no-sync pytest -q \
    tests/test_lego2_blind_pit_terrain.py \
    tests/test_lego2_blind_pit_stage3.py \
    -k 'real_height_rewards_state or inference_checkpoint or real_rim_support or fall_demotes' \
    | tee "$PHASE3_CHECK/gpu_tests.txt"
fi
"$PHASE3_UV_BIN" run --no-sync python scripts/tools/evaluate_lego2_blind_pit.py \
  --output "$PHASE3_CHECK/eval_capped" --heights 0.03 --episodes 1 --max-steps 2 \
  | tee "$PHASE3_CHECK/eval_capped.txt"
"$PHASE3_UV_BIN" run --no-sync python - "$PHASE3_CHECK/eval_capped/report.json" <<'PY'
import json
import sys
r = json.load(open(sys.argv[1]))
assert r['mode'] == 'zero_action_diagnostic'
assert r['heights'][0]['n'] == 0
assert r['heights'][0]['samples'][0]['steps'] == 2
print('PASS: capped diagnostic excluded from evaluation n')
PY
printf '阶段3校验结束（GPU仅在gpu_tests通过时计入）；配置核验后再在screen zrq开始新run短训；本脚本不训练：%s\n' "$PHASE3_CHECK"
