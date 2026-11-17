#!/usr/bin/env bash
# SSH增量安装；验证旧源码并备份，不运行Git、测试或训练。
set -euo pipefail
REWARD_V2_ROOT="${REWARD_V2_ROOT:-$HOME/zrq/mjlab_lego2}"
REWARD_V2_BASE="${REWARD_V2_BASE:-$HOME/zrq/backups}"
REWARD_V2_BUNDLE="${REWARD_V2_BUNDLE:-$HOME/zrq/lego2_reward_v2_20261115_v1.tar.gz}"
printf '%s  %s\n' 'a11777d6efe5105aa1e03177f502b47eb0bbf09e5a26c977cd2ce4459d7aa0d9' "$REWARD_V2_BUNDLE" | sha256sum -c -
cd "$REWARD_V2_ROOT"
mkdir -p "$REWARD_V2_BASE"
REWARD_V2_BACKUP="$(mktemp -d "$REWARD_V2_BASE/reward_v2_sync_20261115_XXXXXX")"
printf '本批备份/安装目录：%s\n' "$REWARD_V2_BACKUP"
mkdir "$REWARD_V2_BACKUP/staged"
tar -xzf "$REWARD_V2_BUNDLE" --no-same-owner -C "$REWARD_V2_BACKUP/staged"
REWARD_V2_STAGE="$REWARD_V2_BACKUP/staged"
REWARD_V2_MANIFEST="docs/codex/artifacts/reward_v2_sources_20261115.sha256"
if sha256sum --quiet -c "$REWARD_V2_STAGE/$REWARD_V2_MANIFEST" 2>/dev/null; then
  printf '所有目标源码已是本批版本，无需覆盖。\n'
  exit 0
fi
sha256sum -c "$REWARD_V2_STAGE/docs/codex/artifacts/reward_v2_prerequisites_20261115.sha256" \
  | tee "$REWARD_V2_BACKUP/prerequisites_check.txt"
cp "$REWARD_V2_STAGE/docs/codex/artifacts/reward_v2_20261115_files.txt" \
  "$REWARD_V2_BACKUP/incoming_files.txt"
: > "$REWARD_V2_BACKUP/existing_files.txt"
while IFS= read -r REWARD_V2_FILE; do
  if [[ -L "$REWARD_V2_FILE" ]]; then
    printf '目标是符号链接，停止覆盖：%s\n' "$REWARD_V2_FILE" >&2
    exit 1
  fi
  if [[ -e "$REWARD_V2_FILE" ]]; then
    # 唯一覆盖的既有源码__init__已由前置hash核验；其他同名文件必须与新包相同。
    if [[ "$REWARD_V2_FILE" != src/mjlab/tasks/velocity/config/lego2/__init__.py ]] && \
      ! cmp -s "$REWARD_V2_FILE" "$REWARD_V2_STAGE/$REWARD_V2_FILE"; then
      printf '发现未知同名文件，停止覆盖：%s\n' "$REWARD_V2_FILE" >&2
      exit 1
    fi
    printf '%s\n' "$REWARD_V2_FILE" >> "$REWARD_V2_BACKUP/existing_files.txt"
  else
    printf '%s\n' "$REWARD_V2_FILE" >> "$REWARD_V2_BACKUP/previously_absent.txt"
  fi
done < "$REWARD_V2_BACKUP/incoming_files.txt"
tar -czf "$REWARD_V2_BACKUP/files_before.tar.gz" --verbatim-files-from \
  --files-from "$REWARD_V2_BACKUP/existing_files.txt"
while IFS= read -r REWARD_V2_FILE; do
  mkdir -p "$(dirname "$REWARD_V2_FILE")"
  cp "$REWARD_V2_STAGE/$REWARD_V2_FILE" "$REWARD_V2_FILE"
done < "$REWARD_V2_BACKUP/incoming_files.txt"
sha256sum -c "$REWARD_V2_MANIFEST" | tee "$REWARD_V2_BACKUP/installed_check.txt"
printf '增量同步完成，未训练；旧模型/logs/.venv及上下文文件保持。\n'
printf '随后在screen zrq运行REWARD_V2_GPU_CHECK=1 bash docs/codex/artifacts/check_phase3_reward_v2_20261115.sh\n'
