#!/usr/bin/env bash
# Stage 2 checks only; no PPO, dependency sync, or old-result deletion.
set -euo pipefail

# Preflight BEFORE modifying the checkout (non-interactive SSH PATH is minimal).
if [ -n "${PHASE2_UV_BIN:-}" ]; then
  test -x "$PHASE2_UV_BIN"
elif command -v uv >/dev/null 2>&1; then
  PHASE2_UV_BIN="$(command -v uv)"
else
  for PHASE2_CANDIDATE in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
    if [ -x "$PHASE2_CANDIDATE" ]; then
      PHASE2_UV_BIN="$PHASE2_CANDIDATE"
      break
    fi
  done
fi
if [ -z "${PHASE2_UV_BIN:-}" ]; then
  printf '未找到uv，请通过PHASE2_UV_BIN传入绝对路径。\n' >&2
  exit 127
fi
printf '使用uv：%s\n' "$PHASE2_UV_BIN"
"$PHASE2_UV_BIN" --version
if [ "${1:-}" = '--check-uv' ]; then exit 0; fi

cd "$HOME/zrq/mjlab_lego2"
if [ "${1:-}" = '--checks-only' ]; then
  PHASE2_BACKUP="${2:?请传入本批实际备份目录}"
  test -d "$PHASE2_BACKUP"
else
  PHASE2_BUNDLE="$HOME/zrq/lego2_phase2_20261031_v2.tar.gz"
  PHASE2_BACKUP="$HOME/zrq/backups/phase2_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$"
  mkdir -p "$PHASE2_BACKUP/files"
  printf '阶段2备份位置：%s\n' "$PHASE2_BACKUP"
  tar -xOf "$PHASE2_BUNDLE" docs/codex/artifacts/phase2_prerequisites.sha256 \
    > "$PHASE2_BACKUP/prerequisites.sha256"
  sha256sum -c "$PHASE2_BACKUP/prerequisites.sha256" \
    | tee "$PHASE2_BACKUP/prerequisites_check.txt"
  tar -xOf "$PHASE2_BUNDLE" docs/codex/artifacts/phase2_files.txt \
    > "$PHASE2_BACKUP/incoming_files.txt"
  while IFS= read -r PHASE2_FILE; do
    if [ -e "$PHASE2_FILE" ] || [ -L "$PHASE2_FILE" ]; then
      cp -a --parents -- "$PHASE2_FILE" "$PHASE2_BACKUP/files/"
    else
      printf '%s\n' "$PHASE2_FILE" >> "$PHASE2_BACKUP/previously_absent.txt"
    fi
  done < "$PHASE2_BACKUP/incoming_files.txt"
  tar -xzf "$PHASE2_BUNDLE" --no-same-owner --verbatim-files-from \
    --files-from "$PHASE2_BACKUP/incoming_files.txt"
  sha256sum -c docs/codex/artifacts/phase2_sync.sha256 \
    | tee "$PHASE2_BACKUP/sync_check.txt"
fi

PHASE2_CHECK="$PHASE2_BACKUP/check_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$PHASE2_CHECK"
printf '阶段2验证目录：%s\n' "$PHASE2_CHECK"
# Subsequent check-only invocations verify code, without reverting context receipts.
while read -r PHASE2_HASH PHASE2_FILE; do
  case "$PHASE2_FILE" in
    src/*|tests/*) printf '%s  %s\n' "$PHASE2_HASH" "$PHASE2_FILE" ;;
  esac
done < docs/codex/artifacts/phase2_sync.sha256 > "$PHASE2_CHECK/source.sha256"
test "$(wc -l < "$PHASE2_CHECK/source.sha256")" -eq 3
sha256sum -c "$PHASE2_CHECK/source.sha256"
export UV_CACHE_DIR="$PHASE2_BACKUP/cache/uv"
export MPLCONFIGDIR="$PHASE2_BACKUP/cache/matplotlib"
export XDG_CACHE_HOME="$PHASE2_BACKUP/cache"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export FORCE_CPU=1
free -m | tee "$PHASE2_CHECK/memory.txt"
"$PHASE2_UV_BIN" run --no-sync python -m compileall -q \
  src/mjlab/tasks/velocity/config/lego2/blind_pit_env_cfg.py \
  src/mjlab/tasks/velocity/config/lego2/blind_pit_curriculums.py \
  tests/test_lego2_blind_pit_terrain.py
"$PHASE2_UV_BIN" run --no-sync python -m mjlab.scripts.list_envs --keyword Lego2 \
  | tee "$PHASE2_CHECK/list_envs.txt"
"$PHASE2_UV_BIN" run --no-sync pytest -q tests/test_lego2_blind_pit_terrain.py \
  tests/test_lego2_blind_pit.py tests/test_observation_history.py \
  tests/test_observation_manager.py | tee "$PHASE2_CHECK/cpu_tests.txt"

if [ "${PHASE2_GPU_CHECK:-0}" = 1 ]; then
  "$PHASE2_UV_BIN" run --no-sync python - <<'PY' | tee "$PHASE2_CHECK/gpu.txt"
import torch
assert torch.cuda.is_available(), 'GPU验证待确认：CUDA不可用'
free, total = torch.cuda.mem_get_info()
print(torch.cuda.get_device_name(), 'free MiB:', free // 2**20)
assert free >= 3 * 2**30, 'GPU验证待确认：空闲显存不足3 GiB，未启动仿真'
PY
  FORCE_CPU=0 "$PHASE2_UV_BIN" run --no-sync pytest -q \
    tests/test_lego2_blind_pit_terrain.py \
    -k 'all_ten_heights or real_rim_support or fall_demotes' \
    | tee "$PHASE2_CHECK/gpu_tests.txt"
fi
printf '阶段2校验结束（GPU仅在gpu_tests通过时计入）；尚不可开训：%s\n' "$PHASE2_CHECK"
