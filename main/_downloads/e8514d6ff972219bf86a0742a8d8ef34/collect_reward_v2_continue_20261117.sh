#!/usr/bin/env bash
# 本地执行：只读下载v2续训/首次短训的证据，核对训练过程和重复评估差异。
# 不上传脚本，不读取权重或缓存，不更改服务器文件；中断保留partial。
set -euo pipefail
umask 077
V2_EVIDENCE_SCRIPT="$(realpath -- "${BASH_SOURCE[0]}")"
V2_EVIDENCE_ROOT="$(cd "$(dirname "$V2_EVIDENCE_SCRIPT")/../../.." && pwd)"
V2_EVIDENCE_DEFAULT="$V2_EVIDENCE_ROOT/docs/codex/diagnostics"
V2_EVIDENCE_DEFAULT+="/20261117_reward_v2_continue_remote"
V2_EVIDENCE_DIR="${V2_EVIDENCE_DIR:-$V2_EVIDENCE_DEFAULT}"
mkdir -p "$V2_EVIDENCE_DIR"
V2_EVIDENCE_PART="$(
  mktemp "$V2_EVIDENCE_DIR/remote_evidence_XXXXXX.tar.gz.partial"
)"
printf '正在下载到：%s\n' "$V2_EVIDENCE_PART"
if ! ssh -p 10395 ajifang@27.128.203.222 'bash -s' \
  > "$V2_EVIDENCE_PART" <<'REMOTE'
set -euo pipefail
cd /home/ajifang/zrq
trial=backups/reward_v2_trial_20261116_AuHs5A
continued=backups/reward_v2_continue_20261116_scKcS4
logroot=mjlab_lego2/logs/rsl_rl/lego2_blind_pit_reward_v2
first="$logroot/20261116_121444_3330412_actor2197_512x300"
run="$logroot/2026-11-16_22-25-03_v2_continue_1024x1900_"
run+=20261116_222451_3365307
tar -czf - \
  "$continued/paired" \
  "$continued/comparison_0.3.json" "$continued/comparison_0.5.json" \
  "$continued/exit_code.txt" "$continued/train_exit_code.txt" \
  "$continued/checkpoint.sha256" "$continued/input_checkpoints.sha256" \
  "$continued/run_dir.txt" "$continued/train_stdout.txt" \
  "$continued/resume_precheck.json" "$continued/command.sh" \
  "$continued/source_check.txt" "$continued/v2_trial_sources_20261116.sha256" \
  "$continued/launcher.sh" \
  "$continued/gpu_before.txt" "$continued/memory_before.txt" \
  "$continued/parent_agent.yaml" "$continued/parent_env.yaml" "$run/params" \
  "$trial/paired" \
  "$trial/comparison_0.3.json" "$trial/comparison_0.5.json" \
  "$trial/v2_trial_exit_code.txt" "$trial/checkpoints.sha256" \
  "$trial/run_dir.txt" "$trial/train_stdout.txt" "$trial/transfer_check.txt" \
  "$trial/source_check.txt" "$trial/v2_trial_sources_20261116.sha256" \
  "$trial/run_phase3_reward_v2_trial_20261116.sh" \
  "$trial/parent_transfer_check/warmstart.json" \
  "$trial/parent_transfer_check/invocation.json" \
  "$trial/parent_transfer_check/params" \
  "$first/warmstart.json" "$first/invocation.json" "$first/params"
REMOTE
then
  printf '下载未完成，保留partial；重跑使用新文件，不覆盖旧证据。\n' >&2
  exit 1
fi
tar -tzf "$V2_EVIDENCE_PART" > "$V2_EVIDENCE_PART.list.txt"
V2_EVIDENCE_FINAL="${V2_EVIDENCE_PART%.partial}"
test ! -e "$V2_EVIDENCE_FINAL"
mv -- "$V2_EVIDENCE_PART" "$V2_EVIDENCE_FINAL"
sha256sum "$V2_EVIDENCE_FINAL" > "$V2_EVIDENCE_FINAL.sha256"
printf '证据下载完成：%s\n' "$V2_EVIDENCE_FINAL"
