#!/usr/bin/env bash
# 在本地执行：SSH只读打包两轮评估证据，不传模型、不改服务器文件。
set -euo pipefail
umask 077
PHASE3_COLLECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PHASE3_COLLECT_DEFAULT="$PHASE3_COLLECT_ROOT/docs/codex/diagnostics"
PHASE3_COLLECT_DEFAULT+="/20261115_model2197_comparison_remote"
PHASE3_COLLECT_DIR="${PHASE3_COLLECT_DIR:-$PHASE3_COLLECT_DEFAULT}"
mkdir -p "$PHASE3_COLLECT_DIR"
PHASE3_COLLECT_PART="$(
  mktemp "$PHASE3_COLLECT_DIR/remote_evidence_XXXXXX.tar.gz.partial"
)"
PHASE3_COLLECT_OLD=phase3_speed_20261115_185511_3295996
PHASE3_COLLECT_NEW=phase3_eval2197_20261115_201606_3299957
printf '正在读取服务器原始证据，下载到：%s\n' "$PHASE3_COLLECT_PART"
if ! ssh -p 10395 ajifang@27.128.203.222 \
  "tar -czf - -C /home/ajifang/zrq/backups \
  $PHASE3_COLLECT_OLD/full $PHASE3_COLLECT_OLD/checkpoint.sha256 \
  $PHASE3_COLLECT_NEW/full $PHASE3_COLLECT_NEW/comparison.json \
  $PHASE3_COLLECT_NEW/checkpoint.sha256 \
  $PHASE3_COLLECT_NEW/train_env.yaml $PHASE3_COLLECT_NEW/train_agent.yaml" \
  > "$PHASE3_COLLECT_PART"; then
  printf '下载未完成，保留partial文件。重跑会创建新文件，不覆盖旧数据。\n' >&2
  exit 1
fi
tar -tzf "$PHASE3_COLLECT_PART" > "$PHASE3_COLLECT_PART.list.txt"
PHASE3_COLLECT_FINAL="${PHASE3_COLLECT_PART%.partial}"
test ! -e "$PHASE3_COLLECT_FINAL"
mv -- "$PHASE3_COLLECT_PART" "$PHASE3_COLLECT_FINAL"
printf '证据下载完成：%s\n' "$PHASE3_COLLECT_FINAL"
