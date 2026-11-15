#!/usr/bin/env bash
# Stage 1 only: preserve existing files and validate the two-env CPU scaffold.
set -euo pipefail
cd "$HOME/zrq/mjlab_lego2"
PHASE1_BUNDLE="$HOME/zrq/lego2_phase1_20261031_v1.tar.gz"
PHASE1_BACKUP="$HOME/zrq/backups/phase1_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$PHASE1_BACKUP/files"
printf '阶段1备份及验证目录：%s\n' "$PHASE1_BACKUP"

# Check source prerequisites BEFORE overwriting any repository files.
tar -xOf "$PHASE1_BUNDLE" docs/codex/artifacts/phase1_prerequisites.sha256 \
  > "$PHASE1_BACKUP/prerequisites.sha256"
sha256sum -c "$PHASE1_BACKUP/prerequisites.sha256" \
  | tee "$PHASE1_BACKUP/prerequisites_check.txt"
tar -xOf "$PHASE1_BUNDLE" docs/codex/artifacts/phase1_files.txt \
  > "$PHASE1_BACKUP/incoming_files.txt"
while IFS= read -r PHASE1_FILE; do
  if [ -e "$PHASE1_FILE" ] || [ -L "$PHASE1_FILE" ]; then
    cp -a --parents -- "$PHASE1_FILE" "$PHASE1_BACKUP/files/"
  else
    printf '%s\n' "$PHASE1_FILE" >> "$PHASE1_BACKUP/previously_absent.txt"
  fi
done < "$PHASE1_BACKUP/incoming_files.txt"
tar -xzf "$PHASE1_BUNDLE" --no-same-owner --verbatim-files-from \
  --files-from "$PHASE1_BACKUP/incoming_files.txt"
sha256sum -c docs/codex/artifacts/phase1_sync.sha256 \
  | tee "$PHASE1_BACKUP/sync_check.txt"

export UV_CACHE_DIR="$PHASE1_BACKUP/cache/uv"
export MPLCONFIGDIR="$PHASE1_BACKUP/cache/matplotlib"
export XDG_CACHE_HOME="$PHASE1_BACKUP/cache"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export FORCE_CPU=1
# This batch deliberately validates CPU parity, without a GPU workload or PPO.
free -m | tee "$PHASE1_BACKUP/memory.txt"
uv run --no-sync python -m compileall -q \
  src/mjlab/tasks/velocity/config/lego2/blind_pit_env_cfg.py \
  src/mjlab/tasks/velocity/config/lego2/__init__.py tests/test_lego2_blind_pit.py
uv run --no-sync python -m mjlab.scripts.list_envs --keyword Lego2 \
  | tee "$PHASE1_BACKUP/list_envs.txt"
uv run --no-sync pytest -q tests/test_lego2_blind_pit.py \
  tests/test_observation_history.py tests/test_observation_manager.py \
  | tee "$PHASE1_BACKUP/tests.txt"
printf '阶段1远程CPU校验结束；尚不可开训。证据目录：%s\n' "$PHASE1_BACKUP"
