#!/usr/bin/env bash
# Resume checks after a successful phase 1 transfer; never reapply the old bundle.
set -euo pipefail

# Non-interactive SSH need not inherit the interactive shell's uv PATH.
if [ -n "${PHASE1_UV_BIN:-}" ]; then
  if [ ! -x "$PHASE1_UV_BIN" ]; then
    printf 'PHASE1_UV_BIN不可执行：%s\n' "$PHASE1_UV_BIN" >&2
    exit 127
  fi
elif command -v uv >/dev/null 2>&1; then
  PHASE1_UV_BIN="$(command -v uv)"
else
  for PHASE1_CANDIDATE in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
    if [ -x "$PHASE1_CANDIDATE" ]; then
      PHASE1_UV_BIN="$PHASE1_CANDIDATE"
      break
    fi
  done
fi
if [ -z "${PHASE1_UV_BIN:-}" ]; then
  printf '%s\n' '未找到uv。交互登录后执行 command -v uv，' \
    '将输出的绝对路径通过 PHASE1_UV_BIN 传入本脚本；不需要安装依赖或重传源码。' >&2
  exit 127
fi
printf '使用uv：%s\n' "$PHASE1_UV_BIN"
"$PHASE1_UV_BIN" --version
if [ "${1:-}" = '--check-uv' ]; then
  exit 0
fi

cd "$HOME/zrq/mjlab_lego2"
PHASE1_PARENT="${1:-$HOME/zrq/backups/phase1_20261031_194321_2147957}"
test -d "$PHASE1_PARENT"
PHASE1_RESUME="$PHASE1_PARENT/resume_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$PHASE1_RESUME"
printf '本次续验目录：%s\n' "$PHASE1_RESUME"

# Check installed code without treating context receipt updates as code drift.
while read -r PHASE1_HASH PHASE1_FILE; do
  case "$PHASE1_FILE" in
    src/*|tests/*)
      printf '%s  %s\n' "$PHASE1_HASH" "$PHASE1_FILE" ;;
  esac
done < docs/codex/artifacts/phase1_sync.sha256 \
  > "$PHASE1_RESUME/source.sha256"
test "$(wc -l < "$PHASE1_RESUME/source.sha256")" -eq 3
sha256sum -c "$PHASE1_RESUME/source.sha256" \
  | tee "$PHASE1_RESUME/source_check.txt"

export UV_CACHE_DIR="$PHASE1_PARENT/cache/uv"
export MPLCONFIGDIR="$PHASE1_PARENT/cache/matplotlib"
export XDG_CACHE_HOME="$PHASE1_PARENT/cache"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export FORCE_CPU=1
free -m | tee "$PHASE1_RESUME/memory.txt"
"$PHASE1_UV_BIN" run --no-sync python -m compileall -q \
  src/mjlab/tasks/velocity/config/lego2/blind_pit_env_cfg.py \
  src/mjlab/tasks/velocity/config/lego2/__init__.py tests/test_lego2_blind_pit.py
"$PHASE1_UV_BIN" run --no-sync python -m mjlab.scripts.list_envs --keyword Lego2 \
  | tee "$PHASE1_RESUME/list_envs.txt"
"$PHASE1_UV_BIN" run --no-sync pytest -q tests/test_lego2_blind_pit.py \
  tests/test_observation_history.py tests/test_observation_manager.py \
  | tee "$PHASE1_RESUME/tests.txt"
printf '阶段1远程CPU续验结束；尚不可开训。证据目录：%s\n' "$PHASE1_RESUME"
