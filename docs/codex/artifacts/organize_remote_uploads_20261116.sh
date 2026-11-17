#!/usr/bin/env bash
# 整理~/zrq顶层的已知项目上传物；不执行旧脚本，不覆盖artifacts已有文件。
# 默认仅预览；--apply先备份，再按原文件名移入独立历史目录。
set -euo pipefail
umask 077
ORGANIZE_MODE="${1:---preview}"
case "$ORGANIZE_MODE" in
  --preview|--apply) ;;
  *) printf '用法：bash %s [--preview|--apply]\n' "$0" >&2; exit 2 ;;
esac
ORGANIZE_ROOT="${ORGANIZE_ROOT:-$HOME/zrq}"
ORGANIZE_ROOT="$(realpath -- "$ORGANIZE_ROOT")"
ORGANIZE_ARTIFACTS="$ORGANIZE_ROOT/mjlab_lego2/docs/codex/artifacts"
test -d "$ORGANIZE_ARTIFACTS"
ORGANIZE_NAMES=(
  apply_phase1_remote_20261031.sh
  apply_phase2_remote_20261031.sh
  apply_phase3_remote_20261031.sh
  apply_phase3_reset_fix_20261101.sh
  apply_phase3_reward_v2_20261115.sh
  apply_phase3_reward_v2_trial_20261116.sh
  apply_phase3_speed_eval_20261115.sh
  bootstrap_remote_git_20261115.sh
  bootstrap_remote_git_20261115_v2.sh
  context_20261031_remote_acceptance.sha256
  context_20261031_remote_acceptance.txt
  continue_phase3_baseline_20261115.sh
  continue_phase3_low_20261101.sh
  continue_reward_v2_2200_20261116.sh
  continue_reward_v2_2200_20261116.sha256
  evaluate_lego2_blind_pit.py
  evaluate_phase3_low_1198_20261101.sh
  evaluate_phase3_model2197_20261115.sh
  evaluate_phase3_smoke_20261031.sh
  evaluate_phase3_smoke_20261101.sh
  evaluate_phase3_smoke_progress_20261101.sh
  evaluate_phase3_upper_1198_20261101.sh
  lego2_phase0_20261030_v1.tar.gz
  lego2_phase1_20261031_v1.tar.gz
  lego2_phase2_20261031_v2.tar.gz
  lego2_phase3_20261031_v1.tar.gz
  lego2_phase3_reset_fix_20261101_v1.tar.gz
  lego2_phase3_speed_eval_20261115_v1.tar.gz
  lego2_reward_v2_20261115_v1.tar.gz
  lego2_reward_v2_trial_20261116_v2.tar.gz
  phase3_continue_sources_20261115.sha256
  phase3_eval_sources_20261101.sha256
  phase3_smoke_sources.sha256
  resume_phase1_checks_20261031.sh
  resume_phase3_reward_v2_checks_20261116.sh
  resume_phase3_reward_v2_checks_20261116.sha256
  start_phase3_smoke_20261031.sh
  sync_remote_git_stage1_20261115.sh
)
ORGANIZE_FILES=()
ORGANIZE_HASHES=()
ORGANIZE_STATUS=()
for name in "${ORGANIZE_NAMES[@]}"; do
  src="$ORGANIZE_ROOT/$name"
  if [[ -L "$src" ]]; then
    printf '跳过符号链接：%s\n' "$src" >&2
    continue
  fi
  [[ -f "$src" ]] || continue
  hash="$(sha256sum -- "$src")"
  hash="${hash%% *}"
  canonical="$ORGANIZE_ARTIFACTS/$name"
  status=no_canonical_file
  if [[ -e "$canonical" || -L "$canonical" ]]; then
    status=canonical_conflict_preserved
    if [[ -f "$canonical" && ! -L "$canonical" ]]; then
      other="$(sha256sum -- "$canonical")"
      if [[ "${other%% *}" == "$hash" ]]; then status=identical_copy; fi
    fi
  fi
  ORGANIZE_FILES+=("$name")
  ORGANIZE_HASHES+=("$hash")
  ORGANIZE_STATUS+=("$status")
  printf '%s  %s  %s\n' "$status" "$hash" "$name"
done
printf '发现%s个已知顶层文件；模式：%s\n' \
  "${#ORGANIZE_FILES[@]}" "$ORGANIZE_MODE"
if [[ "$ORGANIZE_MODE" == --preview || "${#ORGANIZE_FILES[@]}" == 0 ]]; then
  exit 0
fi
mkdir -p "$ORGANIZE_ARTIFACTS/_legacy_uploads" "$ORGANIZE_ROOT/backups"
ORGANIZE_DEST="$(mktemp -d \
  "$ORGANIZE_ARTIFACTS/_legacy_uploads/20261116_XXXXXX")"
ORGANIZE_BACKUP="$(mktemp -d "$ORGANIZE_ROOT/backups/upload_cleanup_XXXXXX")"
printf 'source\tdestination\tsha256\tcanonical_status\n' \
  > "$ORGANIZE_DEST/manifest.tsv"
# 在任何移动之前保存完整快照；备份失败会停止，顶层文件不动。
tar -czf "$ORGANIZE_BACKUP/top_level_uploads.tar.gz" \
  -C "$ORGANIZE_ROOT" -- "${ORGANIZE_FILES[@]}"
sha256sum "$ORGANIZE_BACKUP/top_level_uploads.tar.gz" \
  > "$ORGANIZE_BACKUP/snapshot.sha256"
printf '%s\n' "$ORGANIZE_BACKUP" > "$ORGANIZE_DEST/backup_dir.txt"
for i in "${!ORGANIZE_FILES[@]}"; do
  name="${ORGANIZE_FILES[$i]}"
  src="$ORGANIZE_ROOT/$name"
  dst="$ORGANIZE_DEST/$name"
  hash="$(sha256sum -- "$src")"
  [[ "${hash%% *}" == "${ORGANIZE_HASHES[$i]}" && ! -L "$src" ]]
  test ! -e "$dst"
  mv -n -- "$src" "$dst"
  test ! -e "$src"
  hash="$(sha256sum -- "$dst")"
  [[ "${hash%% *}" == "${ORGANIZE_HASHES[$i]}" ]]
  printf '%s\t%s\t%s\t%s\n' "$src" "$dst" \
    "${ORGANIZE_HASHES[$i]}" "${ORGANIZE_STATUS[$i]}" \
    >> "$ORGANIZE_DEST/manifest.tsv"
done
cp "$ORGANIZE_DEST/manifest.tsv" "$ORGANIZE_BACKUP/manifest.tsv"
printf '整理完成：%s\n备份：%s\n' "$ORGANIZE_DEST" "$ORGANIZE_BACKUP"
printf 'artifacts根目录现有文件保持；历史目录仅留存，旧路径脚本不再执行。\n'
