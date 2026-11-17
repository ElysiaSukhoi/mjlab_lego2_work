#!/usr/bin/env bash
# v2从model299恢复训练状态，新增1900次更新，再做原条件的小样本配对。
# 不改奖励/PPO；新run保存结果。仿真和地形课程重新从3厘米开始。
set -euo pipefail
# 在切换工作目录前固定脚本自身路径，兼容相对路径和绝对路径启动。
V2_CONT_LAUNCHER="$(realpath -- "${BASH_SOURCE[0]}")"
v2_continue_error() {
  local rc="$1" line="$2"
  if [[ -n "${V2_CONT_OUT:-}" ]]; then
    printf '%s\n' "$rc" > "$V2_CONT_OUT/exit_code.txt"
  fi
  printf '失败 line=%s exit=%s；证据：%s\n' \
    "$line" "$rc" "${V2_CONT_OUT:-未创建}" >&2
  exit "$rc"
}
trap 'v2_continue_error "$?" "$LINENO"' ERR
case "${STY:-}" in
  *.zrq) ;;
  *) printf '请先进入screen zrq。\n' >&2; exit 1 ;;
esac
V2_CONT_UV="${V2_CONT_UV:-$HOME/.local/bin/uv}"
test -x "$V2_CONT_UV"
cd "${V2_CONT_ROOT:-$HOME/zrq/mjlab_lego2}"
V2_CONT_BASE="${V2_CONT_BASE:-$HOME/zrq/backups}"
mkdir -p "$V2_CONT_BASE"
V2_CONT_OUT="$(mktemp -d "$V2_CONT_BASE/reward_v2_continue_20261116_XXXXXX")"
printf '证据目录：%s\n' "$V2_CONT_OUT"
cp -- "$V2_CONT_LAUNCHER" "$V2_CONT_OUT/launcher.sh"
V2_CONT_MANIFEST=docs/codex/artifacts/v2_trial_sources_20261116.sha256
printf '%s  %s\n' \
  eefba622156bc8717e2fbe233718f51ea770daf5097b47517b23ac9471e46261 \
  "$V2_CONT_MANIFEST" | sha256sum -c -
sha256sum -c "$V2_CONT_MANIFEST" | tee "$V2_CONT_OUT/source_check.txt"
cp "$V2_CONT_MANIFEST" "$V2_CONT_OUT/"
V2_CONT_PARENT=20261116_121444_3330412_actor2197_512x300
V2_CONT_CKPT="logs/rsl_rl/lego2_blind_pit_reward_v2/$V2_CONT_PARENT/model_299.pt"
V2_CONT_REFERENCE=logs/rsl_rl/lego2_blind_pit/
V2_CONT_REFERENCE+=2026-11-15_19-28-14_phase3_continue_1024x1000_v1_
V2_CONT_REFERENCE+=20261115_192803_3297676/model_2197.pt
export FORCE_CPU=0 PYTHONUNBUFFERED=1
V2_CONT_DEFAULT_CACHE="$V2_CONT_BASE/phase3_20261031_214701_2152912/cache"
export XDG_CACHE_HOME="${V2_CONT_CACHE:-$V2_CONT_DEFAULT_CACHE}"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export MPLCONFIGDIR="$V2_CONT_OUT/cache/matplotlib"
free -m | tee "$V2_CONT_OUT/memory_before.txt"
nvidia-smi | tee "$V2_CONT_OUT/gpu_before.txt"
"$V2_CONT_UV" run --no-sync python - \
  "$V2_CONT_CKPT" "$V2_CONT_REFERENCE" "$V2_CONT_OUT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
import torch

candidate, reference, output = map(Path, sys.argv[1:])
expected = [
    'cb06e04dd5528ada05b730b687444ab82a0f0a03ca26c9b5f2e94a0005ea742c',
    'b8bc52a8ee370eb94e359bc71d310f5f1cd0ec013cae9e5982b44fb13b8a54a4',
]
for path, digest in zip((candidate, reference), expected, strict=True):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, path
state = torch.load(candidate, map_location='cpu', weights_only=False)
assert state['iter'] == 299
assert state['actor_state_dict']['mlp.0.weight'].shape[1] == 144
assert state['critic_state_dict']['mlp.0.weight'].shape[1] == 254
assert state['optimizer_state_dict']['state'], '优化器状态为空，拒绝重置式续训'
counter = state['infos']['env_state']['common_step_counter']
assert counter == 300 * 24, '实际采样步数不符，先核对旧训练日志'
assert torch.cuda.is_available(), 'CUDA不可用'
free, _ = torch.cuda.mem_get_info(0)
assert free >= 12 * 2**30, '空闲显存不足12 GiB'
mem = dict(
    line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines()
)
assert int(mem['MemAvailable'].split()[0]) >= 8 * 2**20, '可用内存不足8 GiB'
audit = {
    'resume_checkpoint': str(candidate),
    'resume_checkpoint_sha256': expected[0],
    'reference_checkpoint_sha256': expected[1],
    'saved_iteration': state['iter'],
    'saved_common_step_counter': counter,
    'saved_optimizer_lrs': [
        g['lr'] for g in state['optimizer_state_dict']['param_groups']
    ],
    'planned_additional_updates': 1900,
    'planned_num_envs': 1024,
    'planned_rollout_steps': 24,
    'planned_additional_samples': 1900 * 1024 * 24,
    'planned_v2_total_updates': 300 + 1900,
    'planned_v2_total_samples': 300 * 512 * 24 + 1900 * 1024 * 24,
    'v1_samples_inferred_from_reported_budgets': 200 * 128 * 24 + 2000 * 1024 * 24,
    'limits': (
        'actor inherited v1 learning; not a same-start reward ablation; '
        'simulator and curriculum reset'
    ),
}
(output / 'resume_precheck.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps(audit, indent=2), flush=True)
PY
sha256sum "$V2_CONT_CKPT" "$V2_CONT_REFERENCE" \
  > "$V2_CONT_OUT/input_checkpoints.sha256"
cp "logs/rsl_rl/lego2_blind_pit_reward_v2/$V2_CONT_PARENT/params/agent.yaml" \
  "$V2_CONT_OUT/parent_agent.yaml"
cp "logs/rsl_rl/lego2_blind_pit_reward_v2/$V2_CONT_PARENT/params/env.yaml" \
  "$V2_CONT_OUT/parent_env.yaml"
if [[ "${V2_CONT_CHECK_ONLY:-0}" == 1 ]]; then
  printf '预检结束，未训练：%s\n' "$V2_CONT_OUT"
  exit 0
fi
V2_CONT_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$"
V2_CONT_RUN="v2_continue_1024x1900_$V2_CONT_STAMP"
V2_CONT_CMD=(
  "$V2_CONT_UV" run --no-sync train Mjlab-Blind-Pit-Lego2-RewardV2
  --env.scene.num-envs 1024
  --agent.num-steps-per-env 24
  --agent.max-iterations 1900
  --agent.seed 7
  --agent.resume True
  --agent.load-run "^$V2_CONT_PARENT$"
  --agent.load-checkpoint '^model_299[.]pt$'
  --agent.logger tensorboard
  --agent.upload-model False
  --agent.run-name "$V2_CONT_RUN"
  --gpu-ids '[0]'
)
printf '%q ' "${V2_CONT_CMD[@]}" > "$V2_CONT_OUT/command.sh"
printf '\n' >> "$V2_CONT_OUT/command.sh"
printf '恢复v2 actor/critic/归一化/优化器及计数；新增1900更新，课程重新起步。\n'
"${V2_CONT_CMD[@]}" 2>&1 | tee "$V2_CONT_OUT/train_stdout.txt"
printf '0\n' > "$V2_CONT_OUT/train_exit_code.txt"
mapfile -t V2_CONT_DIRS < <(
  find logs/rsl_rl/lego2_blind_pit_reward_v2 -mindepth 1 -maxdepth 1 \
    -type d -name "*_$V2_CONT_RUN"
)
[[ "${#V2_CONT_DIRS[@]}" == 1 ]]
V2_CONT_NEW="${V2_CONT_DIRS[0]}/model_2198.pt"
# runner从已保存的299开始执行1900次，末尾标签2198；标签不是累计更新次数。
test -f "$V2_CONT_NEW"
printf '%s\n' "${V2_CONT_DIRS[0]}" > "$V2_CONT_OUT/run_dir.txt"
sha256sum "$V2_CONT_NEW" > "$V2_CONT_OUT/checkpoint.sha256"
mkdir "$V2_CONT_OUT/paired"
for V2_CONT_LABEL in parent candidate; do
  V2_CONT_EVAL_CKPT="$V2_CONT_REFERENCE"
  if [[ "$V2_CONT_LABEL" == candidate ]]; then
    V2_CONT_EVAL_CKPT="$V2_CONT_NEW"
  fi
  for V2_CONT_SPEED in 0.5 0.3; do
    V2_CONT_HEIGHTS=(0.30)
    if [[ "$V2_CONT_SPEED" == 0.5 ]]; then V2_CONT_HEIGHTS=(0.03 0.30); fi
    "$V2_CONT_UV" run --no-sync python scripts/tools/evaluate_lego2_blind_pit.py \
      --checkpoint "$V2_CONT_EVAL_CKPT" \
      --output "$V2_CONT_OUT/paired/${V2_CONT_LABEL}_${V2_CONT_SPEED}" \
      --reward-version v2 --heights "${V2_CONT_HEIGHTS[@]}" \
      --forward-speed "$V2_CONT_SPEED" --paired-seed-base 11 --episodes 3 \
      --seed 7 --device cuda:0 \
      2>&1 | tee "$V2_CONT_OUT/paired/${V2_CONT_LABEL}_${V2_CONT_SPEED}.txt"
  done
done
for V2_CONT_SPEED in 0.5 0.3; do
  "$V2_CONT_UV" run --no-sync python scripts/tools/compare_lego2_reward_v2.py \
    --reference "$V2_CONT_OUT/paired/parent_${V2_CONT_SPEED}/report.json" \
    --candidate "$V2_CONT_OUT/paired/candidate_${V2_CONT_SPEED}/report.json" \
    --output "$V2_CONT_OUT/comparison_${V2_CONT_SPEED}.json" \
    | tee "$V2_CONT_OUT/comparison_${V2_CONT_SPEED}.txt"
done
printf '0\n' > "$V2_CONT_OUT/exit_code.txt"
printf 'v2累计2200更新计划结束；仍是小样本初筛。\n模型：%s\n证据：%s\n' \
  "$V2_CONT_NEW" "$V2_CONT_OUT"
