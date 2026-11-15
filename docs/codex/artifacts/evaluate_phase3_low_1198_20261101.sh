#!/usr/bin/env bash
# Evaluate model_1198 with the same fixed heights, seed and episode count.
set -euo pipefail
case "${STY:-}" in
  *.zrq) ;;
  *) printf '请先进入screen zrq再运行评估。\n' >&2; exit 1 ;;
esac
PHASE3_EVAL_UV="$HOME/.local/bin/uv"
test -x "$PHASE3_EVAL_UV"
cd "$HOME/zrq/mjlab_lego2"
PHASE3_EVAL_MANIFEST="$HOME/zrq/phase3_eval_sources_20261101.sha256"
awk '$2 ~ /^(src|tests)\// {print}' "$PHASE3_EVAL_MANIFEST" | sha256sum -c -
PHASE3_EVAL_RUN="2026-11-01_20-17-22_phase3_low_1024x1000_v1_"
PHASE3_EVAL_RUN+="20261101_201712_2200851"
PHASE3_EVAL_CKPT="$HOME/zrq/mjlab_lego2/logs/rsl_rl/lego2_blind_pit/"
PHASE3_EVAL_CKPT+="$PHASE3_EVAL_RUN/model_1198.pt"
test -f "$PHASE3_EVAL_CKPT"
"$PHASE3_EVAL_UV" run --no-sync python - <<'PY'
from pathlib import Path
import torch
assert torch.cuda.is_available(), 'CUDA不可用，未启动评估'
free, _ = torch.cuda.mem_get_info(0)
print(torch.cuda.get_device_name(0), 'free MiB:', free // 2**20)
assert free >= 3 * 2**30, '空闲显存不足3 GiB，未启动评估'
lines = Path('/proc/meminfo').read_text().splitlines()
mem = dict(line.split(':', 1) for line in lines)
assert int(mem['MemAvailable'].split()[0]) >= 4 * 2**20, '可用内存不足4 GiB'
PY
PHASE3_EVAL_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$"
PHASE3_EVAL_EVIDENCE="$HOME/zrq/backups/phase3_eval1198_$PHASE3_EVAL_STAMP"
mkdir "$PHASE3_EVAL_EVIDENCE"
# Install only the already-validated progress tool, backing up the old file.
PHASE3_EVAL_TOOL=scripts/tools/evaluate_lego2_blind_pit.py
PHASE3_EVAL_INCOMING="$HOME/zrq/evaluate_lego2_blind_pit.py"
PHASE3_EVAL_WANTED="d409dce20da0bfb6c32bbe46988258ea386b9802a1562220ee0abf2a8fe0ff79"
PHASE3_EVAL_CURRENT="$(sha256sum "$PHASE3_EVAL_TOOL" | awk '{print $1}')"
if [ "$PHASE3_EVAL_CURRENT" != "$PHASE3_EVAL_WANTED" ]; then
  PHASE3_EVAL_OLD="$(awk -v f="$PHASE3_EVAL_TOOL" '$2 == f {print $1}' \
    "$PHASE3_EVAL_MANIFEST")"
  test "$PHASE3_EVAL_CURRENT" = "$PHASE3_EVAL_OLD"
  test "$(sha256sum "$PHASE3_EVAL_INCOMING" | awk '{print $1}')" \
    = "$PHASE3_EVAL_WANTED"
  cp -a "$PHASE3_EVAL_TOOL" "$PHASE3_EVAL_EVIDENCE/evaluator_before.py"
  cp "$PHASE3_EVAL_INCOMING" "$PHASE3_EVAL_TOOL"
fi
printf '%s  %s\n' "$PHASE3_EVAL_WANTED" "$PHASE3_EVAL_TOOL" \
  | sha256sum -c -
sha256sum "$PHASE3_EVAL_TOOL" > "$PHASE3_EVAL_EVIDENCE/evaluator.sha256"
cp "$HOME/zrq/phase3_eval_sources_20261101.sha256" "$PHASE3_EVAL_EVIDENCE/"
cp "$HOME/zrq/evaluate_phase3_low_1198_20261101.sh" "$PHASE3_EVAL_EVIDENCE/"
sha256sum "$PHASE3_EVAL_CKPT" > "$PHASE3_EVAL_EVIDENCE/checkpoint.sha256"
free -m > "$PHASE3_EVAL_EVIDENCE/memory_before.txt"
nvidia-smi > "$PHASE3_EVAL_EVIDENCE/gpu_before.txt"
export FORCE_CPU=0
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="$HOME/zrq/backups/phase3_20261031_214701_2152912/cache"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export MPLCONFIGDIR="$PHASE3_EVAL_EVIDENCE/cache/matplotlib"
printf '固定高度评估证据目录：%s\n' "$PHASE3_EVAL_EVIDENCE"
"$PHASE3_EVAL_UV" run --no-sync python scripts/tools/evaluate_lego2_blind_pit.py \
  --checkpoint "$PHASE3_EVAL_CKPT" \
  --device cuda:0 --heights 0.03 0.06 0.09 --episodes 10 --seed 7 \
  --output "$PHASE3_EVAL_EVIDENCE/results" \
  2>&1 | tee "$PHASE3_EVAL_EVIDENCE/eval_stdout.txt"
"$PHASE3_EVAL_UV" run --no-sync python - \
  "$PHASE3_EVAL_EVIDENCE/results/report.json" <<'PY' \
  | tee "$PHASE3_EVAL_EVIDENCE/summary.txt"
import json
import sys
r = json.load(open(sys.argv[1]))
assert r['mode'] == 'policy'
print('Checkpoint:', r['checkpoint'])
print('SHA256:', r['checkpoint_sha256'])
for h in r['heights']:
    assert h['n'] == 10, h
    samples = [s for s in h['samples'] if s['completed']]
    print(f"height={h['height_m']:.2f}m n={h['n']} "
          f"passed={h['passed']} rate={h['pass_rate']:.3f}")
    keys = ('root_out', 'front_up', 'rear_up', 'complete',
            'fell_over', 'failed', 'stalled')
    for key in keys:
        print(f'  {key}: {sum(bool(s[key]) for s in samples)}/{len(samples)}')
    rmse = sum(s['speed_rmse'] for s in samples) / len(samples)
    print('  mean_speed_rmse:', rmse)
PY
printf '评估结束，证据目录：%s\n' "$PHASE3_EVAL_EVIDENCE"
