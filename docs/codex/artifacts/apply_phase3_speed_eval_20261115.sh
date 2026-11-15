#!/usr/bin/env bash
# SSH增量安装入口：校验旧文件、备份、按清单更新；不运行测试或训练。
set -euo pipefail
PHASE3_SPEED_ROOT="${PHASE3_SPEED_ROOT:-$HOME/zrq/mjlab_lego2}"
PHASE3_SPEED_BASE="${PHASE3_SPEED_BASE:-$HOME/zrq/backups}"
PHASE3_SPEED_BUNDLE="${PHASE3_SPEED_BUNDLE:-$HOME/zrq/lego2_phase3_speed_eval_20261115_v1.tar.gz}"
printf '%s  %s\n' 'da1b0d95ee2f66a7c250404294dd0006f55ef8c1040098e58b8a2f58ad8b20cf' "$PHASE3_SPEED_BUNDLE" | sha256sum -c -
cd "$PHASE3_SPEED_ROOT"
mkdir -p "$PHASE3_SPEED_BASE"
PHASE3_SPEED_BACKUP="$(mktemp -d "$PHASE3_SPEED_BASE/phase3_speed_sync_20261115_XXXXXX")"
printf '本批旧源码备份目录：%s\n' "$PHASE3_SPEED_BACKUP"
tar -xOf "$PHASE3_SPEED_BUNDLE" \
  docs/codex/artifacts/phase3_speed_prerequisites_20261115.sha256 \
  > "$PHASE3_SPEED_BACKUP/prerequisites.sha256"
sha256sum -c "$PHASE3_SPEED_BACKUP/prerequisites.sha256" \
  | tee "$PHASE3_SPEED_BACKUP/prerequisites_check.txt"
tar -xOf "$PHASE3_SPEED_BUNDLE" \
  docs/codex/artifacts/phase3_speed_20261115_files.txt \
  > "$PHASE3_SPEED_BACKUP/incoming_files.txt"
: > "$PHASE3_SPEED_BACKUP/existing_files.txt"
while IFS= read -r PHASE3_SPEED_FILE; do
  if [[ -e "$PHASE3_SPEED_FILE" || -L "$PHASE3_SPEED_FILE" ]]; then
    printf '%s\n' "$PHASE3_SPEED_FILE" >> "$PHASE3_SPEED_BACKUP/existing_files.txt"
  else
    printf '%s\n' "$PHASE3_SPEED_FILE" >> "$PHASE3_SPEED_BACKUP/previously_absent.txt"
  fi
done < "$PHASE3_SPEED_BACKUP/incoming_files.txt"
tar -czf "$PHASE3_SPEED_BACKUP/files_before.tar.gz" --verbatim-files-from \
  --files-from "$PHASE3_SPEED_BACKUP/existing_files.txt"
tar -xzf "$PHASE3_SPEED_BUNDLE" --no-same-owner --verbatim-files-from \
  --files-from "$PHASE3_SPEED_BACKUP/incoming_files.txt"
sha256sum -c docs/codex/artifacts/phase3_speed_sources_20261115.sha256 \
  | tee "$PHASE3_SPEED_BACKUP/installed_check.txt"
printf '增量同步完成；旧模型和结果保留，尚未测试或训练。\n'
printf '随后在screen zrq运行：bash docs/codex/artifacts/check_phase3_speed_eval_20261115.sh\n'
