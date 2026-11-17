#!/usr/bin/env bash
# 独立v2迁移试验：新软件检查、真实父模型迁移核对、512x300短训与小样本配对复评。
set -euo pipefail
trap 'V2_TRIAL_RC=$?; if [[ -n "${V2_TRIAL_OUT:-}" ]]; then printf "%s\n" "$V2_TRIAL_RC" > "$V2_TRIAL_OUT/v2_trial_exit_code.txt"; fi; printf "阶段失败 line=%s exit=%s；目录：%s\n" "$LINENO" "$V2_TRIAL_RC" "${V2_TRIAL_OUT:-未创建}" >&2; exit "$V2_TRIAL_RC"' ERR
case "${STY:-}" in
  *.zrq) ;;
  *) printf '请先进入screen zrq。\n' >&2; exit 1 ;;
esac
V2_TRIAL_UV="${V2_TRIAL_UV:-$HOME/.local/bin/uv}"
cd "${V2_TRIAL_ROOT:-$HOME/zrq/mjlab_lego2}"
V2_TRIAL_BASE="${V2_TRIAL_BASE:-$HOME/zrq/backups}"
mkdir -p "$V2_TRIAL_BASE"
V2_TRIAL_OUT="$(mktemp -d "$V2_TRIAL_BASE/reward_v2_trial_20261116_XXXXXX")"
printf '本批证据目录：%s\n' "$V2_TRIAL_OUT"
sha256sum -c docs/codex/artifacts/v2_trial_sources_20261116.sha256 \
  | tee "$V2_TRIAL_OUT/source_check.txt"
cp docs/codex/artifacts/v2_trial_sources_20261116.sha256 "$V2_TRIAL_OUT/"
cp docs/codex/artifacts/run_phase3_reward_v2_trial_20261116.sh "$V2_TRIAL_OUT/"
export XDG_CACHE_HOME="${V2_TRIAL_CACHE:-$HOME/zrq/backups/phase3_20261031_214701_2152912/cache}"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export MPLCONFIGDIR="$V2_TRIAL_OUT/cache/matplotlib"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONUNBUFFERED=1
V2_TRIAL_PARENT="${V2_TRIAL_PARENT:-logs/rsl_rl/lego2_blind_pit/2026-11-15_19-28-14_phase3_continue_1024x1000_v1_20261115_192803_3297676/model_2197.pt}"
printf '[1/6] 新迁移/评估工具CPU检查\n'
free -m | tee "$V2_TRIAL_OUT/memory_before.txt"
FORCE_CPU=1 "$V2_TRIAL_UV" run --no-sync pytest -q \
  tests/test_lego2_blind_pit_warmstart.py \
  2>&1 | tee "$V2_TRIAL_OUT/cpu_tests.txt"
printf '[2/6] 资源与GPU软件检查；不终止其他进程\n'
nvidia-smi | tee "$V2_TRIAL_OUT/gpu_before.txt"
"$V2_TRIAL_UV" run --no-sync python - <<'PY' | tee "$V2_TRIAL_OUT/resources.txt"
from pathlib import Path
import torch
assert torch.cuda.is_available(), 'CUDA不可用'
free, _ = torch.cuda.mem_get_info(0)
assert free >= 12 * 2**30, '空闲显存不足12 GiB'
mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
assert int(mem['MemAvailable'].split()[0]) >= 8 * 2**20, '可用内存不足8 GiB'
print(torch.cuda.get_device_name(0), 'free MiB:', free // 2**20)
PY
FORCE_CPU=0 "$V2_TRIAL_UV" run --no-sync pytest -q \
  tests/test_lego2_blind_pit_warmstart.py -k 'not quality_comparison' \
  2>&1 | tee "$V2_TRIAL_OUT/gpu_tests.txt"
printf '[3/6] 真实父模型actor迁移预检：仅2环境，不更新PPO\n'
"$V2_TRIAL_UV" run --no-sync python scripts/tools/train_lego2_blind_pit_reward_v2.py \
  --checkpoint "$V2_TRIAL_PARENT" --output "$V2_TRIAL_OUT/parent_transfer_check" \
  --num-envs 2 --iterations 1 --device cuda:0 --validate-only \
  2>&1 | tee "$V2_TRIAL_OUT/transfer_check.txt"
if [[ "${V2_TRIAL_CHECK_ONLY:-0}" == 1 ]]; then
  printf '预检结束，未启动真实父策略PPO微调：%s\n' "$V2_TRIAL_OUT"
  exit 0
fi
printf '[4/6] 独立512环境、300次更新；计划3686400新样本\n'
V2_TRIAL_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$"
V2_TRIAL_RUN="logs/rsl_rl/lego2_blind_pit_reward_v2/${V2_TRIAL_STAMP}_actor2197_512x300"
printf '%s\n' "$V2_TRIAL_RUN" > "$V2_TRIAL_OUT/run_dir.txt"
"$V2_TRIAL_UV" run --no-sync python scripts/tools/train_lego2_blind_pit_reward_v2.py \
  --checkpoint "$V2_TRIAL_PARENT" --output "$V2_TRIAL_RUN" \
  --num-envs 512 --iterations 300 --seed 7 --device cuda:0 \
  2>&1 | tee "$V2_TRIAL_OUT/train_stdout.txt"
V2_TRIAL_NEW="$V2_TRIAL_RUN/model_299.pt"
test -f "$V2_TRIAL_NEW"
sha256sum "$V2_TRIAL_PARENT" "$V2_TRIAL_NEW" > "$V2_TRIAL_OUT/checkpoints.sha256"
printf '[5/6] 配对复评：3/30 cm各vx=0.5，另30 cm vx=0.3；seeds11~13\n'
mkdir "$V2_TRIAL_OUT/paired"
for V2_TRIAL_LABEL in parent candidate; do
  V2_TRIAL_CHECKPOINT="$V2_TRIAL_PARENT"
  if [[ "$V2_TRIAL_LABEL" == candidate ]]; then
    V2_TRIAL_CHECKPOINT="$V2_TRIAL_NEW"
  fi
  for V2_TRIAL_SPEED in 0.5 0.3; do
    V2_TRIAL_HEIGHTS=(0.30)
    if [[ "$V2_TRIAL_SPEED" == 0.5 ]]; then V2_TRIAL_HEIGHTS=(0.03 0.30); fi
    "$V2_TRIAL_UV" run --no-sync python scripts/tools/evaluate_lego2_blind_pit.py \
      --checkpoint "$V2_TRIAL_CHECKPOINT" --output "$V2_TRIAL_OUT/paired/${V2_TRIAL_LABEL}_${V2_TRIAL_SPEED}" \
      --reward-version v2 --heights "${V2_TRIAL_HEIGHTS[@]}" --forward-speed "$V2_TRIAL_SPEED" \
      --paired-seed-base 11 --episodes 3 --seed 7 --device cuda:0 \
      2>&1 | tee "$V2_TRIAL_OUT/paired/${V2_TRIAL_LABEL}_${V2_TRIAL_SPEED}.txt"
  done
done
printf '[6/6] 核对真实reset初态与运动质量指标\n'
for V2_TRIAL_SPEED in 0.5 0.3; do
  "$V2_TRIAL_UV" run --no-sync python scripts/tools/compare_lego2_reward_v2.py \
    --reference "$V2_TRIAL_OUT/paired/parent_${V2_TRIAL_SPEED}/report.json" \
    --candidate "$V2_TRIAL_OUT/paired/candidate_${V2_TRIAL_SPEED}/report.json" \
    --output "$V2_TRIAL_OUT/comparison_${V2_TRIAL_SPEED}.json" \
    | tee "$V2_TRIAL_OUT/comparison_${V2_TRIAL_SPEED}.txt"
done
printf '0\n' > "$V2_TRIAL_OUT/v2_trial_exit_code.txt"
printf 'v2短训/小样本配对评估结束，不代表正式验收。\n模型：%s\n证据目录：%s\n' "$V2_TRIAL_NEW" "$V2_TRIAL_OUT"
