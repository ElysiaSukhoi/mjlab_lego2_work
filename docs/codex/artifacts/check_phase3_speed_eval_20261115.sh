#!/usr/bin/env bash
# 仅做验证/评估；源码传输、备份与安装由独立SSH同步步骤完成。
set -euo pipefail
case "${STY:-}" in
  *.zrq) ;;
  *) printf '请先进入screen zrq再运行本批验证。\n' >&2; exit 1 ;;
esac
PHASE3_SPEED_UV="$HOME/.local/bin/uv"
test -x "$PHASE3_SPEED_UV"
cd "$HOME/zrq/mjlab_lego2"
sha256sum -c docs/codex/artifacts/phase3_speed_sources_20261115.sha256
PHASE3_SPEED_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$"
PHASE3_SPEED_OUT="$HOME/zrq/backups/phase3_speed_$PHASE3_SPEED_STAMP"
mkdir "$PHASE3_SPEED_OUT"
printf '本批验证/评估目录：%s\n' "$PHASE3_SPEED_OUT"
cp docs/codex/artifacts/phase3_speed_sources_20261115.sha256 "$PHASE3_SPEED_OUT/"
export XDG_CACHE_HOME="$HOME/zrq/backups/phase3_20261031_214701_2152912/cache"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export MPLCONFIGDIR="$PHASE3_SPEED_OUT/cache/matplotlib"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONUNBUFFERED=1
free -m | tee "$PHASE3_SPEED_OUT/memory_before.txt"
FORCE_CPU=1 "$PHASE3_SPEED_UV" run --no-sync pytest -q \
  tests/test_lego2_blind_pit.py tests/test_lego2_blind_pit_terrain.py \
  tests/test_lego2_blind_pit_stage3.py tests/test_lego2_blind_pit_speed_eval.py \
  2>&1 | tee "$PHASE3_SPEED_OUT/cpu_tests.txt"

PHASE3_SPEED_RUN="2026-11-01_20-17-22_phase3_low_1024x1000_v1_"
PHASE3_SPEED_RUN+="20261101_201712_2200851"
PHASE3_SPEED_CKPT="$HOME/zrq/mjlab_lego2/logs/rsl_rl/lego2_blind_pit/"
PHASE3_SPEED_CKPT+="$PHASE3_SPEED_RUN/model_1198.pt"
"$PHASE3_SPEED_UV" run --no-sync python - "$PHASE3_SPEED_CKPT" <<'PY' \
  | tee "$PHASE3_SPEED_OUT/model_and_resources.txt"
from pathlib import Path
import hashlib
import sys
import torch
p = Path(sys.argv[1])
expected = '7780cf0d2557c86f51c0486f870e7d471b13703777e9cee5e1854427def05027'
assert hashlib.sha256(p.read_bytes()).hexdigest() == expected, '模型内容校验失败'
assert torch.cuda.is_available(), 'CUDA不可用，未开始策略评估'
free, _ = torch.cuda.mem_get_info(0)
assert free >= 3 * 2**30, '空闲显存不足3 GiB'
mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
assert int(mem['MemAvailable'].split()[0]) >= 4 * 2**20, '可用内存不足4 GiB'
print('Checkpoint:', p)
print('SHA256:', expected)
print('GPU:', torch.cuda.get_device_name(0), 'free MiB:', free // 2**20)
PY
sha256sum "$PHASE3_SPEED_CKPT" > "$PHASE3_SPEED_OUT/checkpoint.sha256"
nvidia-smi > "$PHASE3_SPEED_OUT/gpu_before.txt"

# 先用真实checkpoint做6个两步GPU片段，配对通过后才开始30个完整episode。
mkdir "$PHASE3_SPEED_OUT/gpu_smoke" "$PHASE3_SPEED_OUT/full"
for PHASE3_SPEED_VX in 0.30 0.40 0.50; do
  FORCE_CPU=0 "$PHASE3_SPEED_UV" run --no-sync python \
    scripts/tools/evaluate_lego2_blind_pit.py --checkpoint "$PHASE3_SPEED_CKPT" \
    --device cuda:0 --heights 0.30 --forward-speed "$PHASE3_SPEED_VX" \
    --paired-seed-base 7 --episodes 2 --seed 7 --max-steps 2 \
    --output "$PHASE3_SPEED_OUT/gpu_smoke/vx_$PHASE3_SPEED_VX" \
    2>&1 | tee "$PHASE3_SPEED_OUT/gpu_smoke/vx_$PHASE3_SPEED_VX.txt"
done
"$PHASE3_SPEED_UV" run --no-sync python scripts/tools/summarize_lego2_speed_eval.py \
  --directory "$PHASE3_SPEED_OUT/gpu_smoke" --episodes 2 --mode smoke \
  | tee "$PHASE3_SPEED_OUT/gpu_smoke/summary.txt"

for PHASE3_SPEED_VX in 0.30 0.40 0.50; do
  FORCE_CPU=0 "$PHASE3_SPEED_UV" run --no-sync python \
    scripts/tools/evaluate_lego2_blind_pit.py --checkpoint "$PHASE3_SPEED_CKPT" \
    --device cuda:0 --heights 0.30 --forward-speed "$PHASE3_SPEED_VX" \
    --paired-seed-base 7 --episodes 10 --seed 7 \
    --output "$PHASE3_SPEED_OUT/full/vx_$PHASE3_SPEED_VX" \
    2>&1 | tee "$PHASE3_SPEED_OUT/full/vx_$PHASE3_SPEED_VX.txt"
done
"$PHASE3_SPEED_UV" run --no-sync python scripts/tools/summarize_lego2_speed_eval.py \
  --directory "$PHASE3_SPEED_OUT/full" --episodes 10 --mode full \
  | tee "$PHASE3_SPEED_OUT/full/summary.txt"
printf '固定速度配对评估完成；本批没有PPO更新。证据目录：%s\n' "$PHASE3_SPEED_OUT"
