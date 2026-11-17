#!/usr/bin/env bash
# SSH增量安装；验证旧源码并备份，不运行Git、测试或训练。
set -euo pipefail
V2_TRIAL_ROOT="${V2_TRIAL_ROOT:-$HOME/zrq/mjlab_lego2}"
V2_TRIAL_BASE="${V2_TRIAL_BASE:-$HOME/zrq/backups}"
V2_TRIAL_BUNDLE="${V2_TRIAL_BUNDLE:-$HOME/zrq/lego2_reward_v2_trial_20261116_v2.tar.gz}"
printf '%s  %s\n' 'b593ff441092af184e7b7a7961c4e1507ea844265f0aeef3577005e593e024d3' "$V2_TRIAL_BUNDLE" | sha256sum -c -
cd "$V2_TRIAL_ROOT"
mkdir -p "$V2_TRIAL_BASE"
V2_TRIAL_BACKUP="$(mktemp -d "$V2_TRIAL_BASE/v2_trial_sync_20261116_XXXXXX")"
printf '本批备份/安装目录：%s\n' "$V2_TRIAL_BACKUP"
mkdir "$V2_TRIAL_BACKUP/staged"
tar -xzf "$V2_TRIAL_BUNDLE" --no-same-owner -C "$V2_TRIAL_BACKUP/staged"
V2_TRIAL_STAGE="$V2_TRIAL_BACKUP/staged"
V2_TRIAL_MANIFEST="docs/codex/artifacts/v2_trial_sources_20261116.sha256"
if sha256sum --quiet -c "$V2_TRIAL_STAGE/$V2_TRIAL_MANIFEST" 2>/dev/null; then
  printf '所有目标源码已是本批版本，无需覆盖。\n'
  exit 0
fi
sha256sum -c "$V2_TRIAL_STAGE/docs/codex/artifacts/v2_trial_prerequisites_20261116.sha256" \
  | tee "$V2_TRIAL_BACKUP/prerequisites_check.txt"
cp "$V2_TRIAL_STAGE/docs/codex/artifacts/v2_trial_20261116_files.txt" \
  "$V2_TRIAL_BACKUP/incoming_files.txt"
: > "$V2_TRIAL_BACKUP/existing_files.txt"
while IFS= read -r V2_TRIAL_FILE; do
  if [[ -L "$V2_TRIAL_FILE" ]]; then
    printf '目标是符号链接，停止覆盖：%s\n' "$V2_TRIAL_FILE" >&2
    exit 1
  fi
  if [[ -e "$V2_TRIAL_FILE" ]]; then
    # 唯一覆盖的既有评估源码已由前置hash核验；其他同名文件必须与新包相同。
    if [[ "$V2_TRIAL_FILE" != scripts/tools/evaluate_lego2_blind_pit.py ]] && \
      ! cmp -s "$V2_TRIAL_FILE" "$V2_TRIAL_STAGE/$V2_TRIAL_FILE"; then
      printf '发现未知同名文件，停止覆盖：%s\n' "$V2_TRIAL_FILE" >&2
      exit 1
    fi
    printf '%s\n' "$V2_TRIAL_FILE" >> "$V2_TRIAL_BACKUP/existing_files.txt"
  else
    printf '%s\n' "$V2_TRIAL_FILE" >> "$V2_TRIAL_BACKUP/previously_absent.txt"
  fi
done < "$V2_TRIAL_BACKUP/incoming_files.txt"
tar -czf "$V2_TRIAL_BACKUP/files_before.tar.gz" --verbatim-files-from \
  --files-from "$V2_TRIAL_BACKUP/existing_files.txt"
while IFS= read -r V2_TRIAL_FILE; do
  mkdir -p "$(dirname "$V2_TRIAL_FILE")"
  cp "$V2_TRIAL_STAGE/$V2_TRIAL_FILE" "$V2_TRIAL_FILE"
done < "$V2_TRIAL_BACKUP/incoming_files.txt"
sha256sum -c "$V2_TRIAL_MANIFEST" | tee "$V2_TRIAL_BACKUP/installed_check.txt"
printf '增量同步完成，未训练；旧模型/logs/.venv及上下文文件保持。\n'
printf '随后在screen zrq运行bash docs/codex/artifacts/run_phase3_reward_v2_trial_20261116.sh\n'
