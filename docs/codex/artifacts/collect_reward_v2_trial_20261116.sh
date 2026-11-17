#!/usr/bin/env bash
# 在本地执行：只读下载本次v2试验的报告、轨迹、训练日志和配置，不传模型或缓存。
set -euo pipefail
umask 077
V2_COLLECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
V2_COLLECT_DEFAULT="$V2_COLLECT_ROOT/docs/codex/diagnostics"
V2_COLLECT_DEFAULT+="/20261116_reward_v2_trial_remote"
V2_COLLECT_DIR="${V2_COLLECT_DIR:-$V2_COLLECT_DEFAULT}"
mkdir -p "$V2_COLLECT_DIR"
V2_COLLECT_PART="$(mktemp "$V2_COLLECT_DIR/remote_evidence_XXXXXX.tar.gz.partial")"
printf '正在下载到：%s\n' "$V2_COLLECT_PART"
if ! ssh -p 10395 ajifang@27.128.203.222 'bash -s' > "$V2_COLLECT_PART" <<'REMOTE'
set -euo pipefail
cd /home/ajifang/zrq
trial=backups/reward_v2_trial_20261116_AuHs5A
run=mjlab_lego2/logs/rsl_rl/lego2_blind_pit_reward_v2/20261116_121444_3330412_actor2197_512x300
tar -czf - \
  "$trial/paired" \
  "$trial/comparison_0.3.json" "$trial/comparison_0.5.json" \
  "$trial/v2_trial_exit_code.txt" "$trial/checkpoints.sha256" \
  "$trial/run_dir.txt" "$trial/train_stdout.txt" "$trial/transfer_check.txt" \
  "$trial/source_check.txt" "$trial/v2_trial_sources_20261116.sha256" \
  "$trial/run_phase3_reward_v2_trial_20261116.sh" \
  "$trial/cpu_tests.txt" "$trial/gpu_tests.txt" \
  "$trial/gpu_before.txt" "$trial/memory_before.txt" "$trial/resources.txt" \
  "$trial/parent_transfer_check/warmstart.json" \
  "$trial/parent_transfer_check/invocation.json" \
  "$trial/parent_transfer_check/params" \
  "$run/warmstart.json" "$run/invocation.json" "$run/params"
REMOTE
then
  printf '下载未完成；保留partial文件。重新运行会创建新文件。\n' >&2
  exit 1
fi
tar -tzf "$V2_COLLECT_PART" > "$V2_COLLECT_PART.list.txt"
V2_COLLECT_FINAL="${V2_COLLECT_PART%.partial}"
test ! -e "$V2_COLLECT_FINAL"
mv -- "$V2_COLLECT_PART" "$V2_COLLECT_FINAL"
sha256sum "$V2_COLLECT_FINAL" > "$V2_COLLECT_FINAL.sha256"
printf '证据下载完成：%s\n' "$V2_COLLECT_FINAL"
