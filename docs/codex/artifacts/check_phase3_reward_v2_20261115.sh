#!/usr/bin/env bash
# 仅验证v2奖励候选；不更新源码、不加载策略训练、不改旧结果。
set -euo pipefail
case "${STY:-}" in
  *.zrq) ;;
  *) printf '请先进入screen zrq再运行本批验证。\n' >&2; exit 1 ;;
esac
REWARD_V2_UV="$HOME/.local/bin/uv"
test -x "$REWARD_V2_UV"
cd "$HOME/zrq/mjlab_lego2"
REWARD_V2_OUT="$(mktemp -d "$HOME/zrq/backups/reward_v2_checks_20261115_XXXXXX")"
printf '奖励v2验证目录：%s\n' "$REWARD_V2_OUT"
sha256sum -c docs/codex/artifacts/reward_v2_sources_20261115.sha256 \
  | tee "$REWARD_V2_OUT/source_check.txt"
cp docs/codex/artifacts/reward_v2_sources_20261115.sha256 "$REWARD_V2_OUT/"
export XDG_CACHE_HOME="$HOME/zrq/backups/phase3_20261031_214701_2152912/cache"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export MPLCONFIGDIR="$REWARD_V2_OUT/cache/matplotlib"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONUNBUFFERED=1
free -m | tee "$REWARD_V2_OUT/memory_before.txt"
"$REWARD_V2_UV" run --no-sync list-envs | tee "$REWARD_V2_OUT/tasks.txt"
FORCE_CPU=1 "$REWARD_V2_UV" run --no-sync pytest -q \
  tests/test_lego2_blind_pit_reward_v2.py tests/test_lego2_blind_pit_stage3.py \
  tests/test_lego2_blind_pit.py tests/test_lego2_blind_pit_terrain.py \
  tests/test_lego2_blind_pit_speed_eval.py \
  2>&1 | tee "$REWARD_V2_OUT/cpu_tests.txt"

if [[ "${REWARD_V2_GPU_CHECK:-0}" == 1 ]]; then
  nvidia-smi | tee "$REWARD_V2_OUT/gpu_before.txt"
  "$REWARD_V2_UV" run --no-sync python - <<'PY' \
    | tee "$REWARD_V2_OUT/gpu_resources.txt"
import torch
assert torch.cuda.is_available(), 'CUDA不可用，停止GPU核验'
free, total = torch.cuda.mem_get_info(0)
assert free >= 3 * 2**30, '空闲显存不足3 GiB'
print(torch.cuda.get_device_name(0), 'free MiB:', free // 2**20)
PY
  FORCE_CPU=0 "$REWARD_V2_UV" run --no-sync pytest -q \
    tests/test_lego2_blind_pit_reward_v2.py \
    -k real_v2_rewards_observations_and_inference_reset \
    2>&1 | tee "$REWARD_V2_OUT/gpu_tests.txt"
fi
printf '候选奖励软件核验结束；GPU仅在gpu_tests通过时计入；尚无v2训练/效果结论。\n'
printf '证据目录：%s\n' "$REWARD_V2_OUT"
