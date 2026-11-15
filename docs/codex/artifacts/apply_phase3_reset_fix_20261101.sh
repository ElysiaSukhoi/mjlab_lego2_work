#!/usr/bin/env bash
# Patch pit diagnostic buffers, validate resets, retain checkpoints and failed reports.
set -euo pipefail
if command -v uv >/dev/null 2>&1; then
  PHASE3_RESET_UV="$(command -v uv)"
else
  PHASE3_RESET_UV="$HOME/.local/bin/uv"
fi
test -x "$PHASE3_RESET_UV"
printf '使用uv：%s\n' "$PHASE3_RESET_UV"
cd "$HOME/zrq/mjlab_lego2"
if [ "${1:-}" = '--checks-only' ]; then
  PHASE3_RESET_BACKUP="${2:?请传入本次实际修复备份目录}"
  test -d "$PHASE3_RESET_BACKUP"
else
  PHASE3_RESET_BUNDLE="$HOME/zrq/lego2_phase3_reset_fix_20261101_v1.tar.gz"
  PHASE3_RESET_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$"
  PHASE3_RESET_BACKUP="$HOME/zrq/backups/phase3_reset_fix_$PHASE3_RESET_STAMP"
  mkdir "$PHASE3_RESET_BACKUP"
  printf '修复备份目录：%s\n' "$PHASE3_RESET_BACKUP"
  tar -xOf "$PHASE3_RESET_BUNDLE" \
    docs/codex/artifacts/phase3_reset_fix_prerequisites.sha256 \
    > "$PHASE3_RESET_BACKUP/prerequisites.sha256"
  sha256sum -c "$PHASE3_RESET_BACKUP/prerequisites.sha256" \
    | tee "$PHASE3_RESET_BACKUP/prerequisites_check.txt"
  tar -xOf "$PHASE3_RESET_BUNDLE" \
    docs/codex/artifacts/phase3_reset_fix_files.txt \
    > "$PHASE3_RESET_BACKUP/incoming_files.txt"
  : > "$PHASE3_RESET_BACKUP/existing_files.txt"
  while IFS= read -r PHASE3_RESET_FILE; do
    if [ -e "$PHASE3_RESET_FILE" ] || [ -L "$PHASE3_RESET_FILE" ]; then
      printf '%s\n' "$PHASE3_RESET_FILE" \
        >> "$PHASE3_RESET_BACKUP/existing_files.txt"
    else
      printf '%s\n' "$PHASE3_RESET_FILE" \
        >> "$PHASE3_RESET_BACKUP/previously_absent.txt"
    fi
  done < "$PHASE3_RESET_BACKUP/incoming_files.txt"
  tar -czf "$PHASE3_RESET_BACKUP/files_before.tar.gz" --verbatim-files-from \
    --files-from "$PHASE3_RESET_BACKUP/existing_files.txt"
  tar -xzf "$PHASE3_RESET_BUNDLE" --no-same-owner --verbatim-files-from \
    --files-from "$PHASE3_RESET_BACKUP/incoming_files.txt"
  sha256sum -c docs/codex/artifacts/phase3_reset_fix_sync.sha256 \
    | tee "$PHASE3_RESET_BACKUP/sync_check.txt"
fi
# This is a new evaluation manifest; never rewrite the smoke run's source manifest.
sha256sum -c docs/codex/artifacts/phase3_eval_sources_20261101.sha256
PHASE3_RESET_CHECK_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$"
PHASE3_RESET_CHECK="$PHASE3_RESET_BACKUP/check_$PHASE3_RESET_CHECK_STAMP"
mkdir "$PHASE3_RESET_CHECK"
printf '修复验证目录：%s\n' "$PHASE3_RESET_CHECK"
free -m | tee "$PHASE3_RESET_CHECK/memory.txt"
export XDG_CACHE_HOME="$HOME/zrq/backups/phase3_20261031_214701_2152912/cache"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export MPLCONFIGDIR="$PHASE3_RESET_CHECK/cache/matplotlib"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export FORCE_CPU=1
"$PHASE3_RESET_UV" run --no-sync python -m compileall -q \
  src/mjlab/tasks/velocity/config/lego2/blind_pit_metrics.py \
  tests/test_lego2_blind_pit_stage3.py
"$PHASE3_RESET_UV" run --no-sync pytest -q tests/test_lego2_blind_pit_stage3.py \
  | tee "$PHASE3_RESET_CHECK/cpu_tests.txt"
PHASE3_RESET_DEVICE=cpu
if [ "${PHASE3_RESET_GPU_CHECK:-0}" = 1 ]; then
  "$PHASE3_RESET_UV" run --no-sync python - <<'PY' \
    | tee "$PHASE3_RESET_CHECK/gpu.txt"
import torch
assert torch.cuda.is_available(), 'CUDA不可用，GPU修复验证待确认'
free, _ = torch.cuda.mem_get_info(0)
print(torch.cuda.get_device_name(0), 'free MiB:', free // 2**20)
assert free >= 3 * 2**30, '空闲显存不足3 GiB'
PY
  FORCE_CPU=0 "$PHASE3_RESET_UV" run --no-sync pytest -q \
    tests/test_lego2_blind_pit_stage3.py \
    -k 'inference_checkpoint or real_height_rewards_state' \
    | tee "$PHASE3_RESET_CHECK/gpu_tests.txt"
  PHASE3_RESET_DEVICE=cuda:0
fi
# Two short episodes at each height exercise the evaluator's actual reset loop.
FORCE_CPU=0 "$PHASE3_RESET_UV" run --no-sync python \
  scripts/tools/evaluate_lego2_blind_pit.py \
  --device "$PHASE3_RESET_DEVICE" --heights 0.03 0.06 --episodes 2 \
  --max-steps 2 --output "$PHASE3_RESET_CHECK/eval_reset_smoke" \
  2>&1 | tee "$PHASE3_RESET_CHECK/eval_reset_smoke.txt"
"$PHASE3_RESET_UV" run --no-sync python - \
  "$PHASE3_RESET_CHECK/eval_reset_smoke/report.json" <<'PY'
import json
import sys
r = json.load(open(sys.argv[1]))
assert len(r['heights']) == 2
for h in r['heights']:
    assert h['n'] == 0 and len(h['samples']) == 2
    assert all(s['steps'] == 2 and not s['completed'] for s in h['samples'])
print('PASS: repeated inference/reset across two episodes and two heights')
PY
# Deploy the launcher and manifest under new names, preserving the old evidence.
cp docs/codex/artifacts/evaluate_phase3_smoke_20261101.sh "$HOME/zrq/"
cp docs/codex/artifacts/phase3_eval_sources_20261101.sha256 "$HOME/zrq/"
printf '修复核验结束；旧训练与评估目录保留。验证目录：%s\n' "$PHASE3_RESET_CHECK"
printf '在screen zrq内重新评估：bash ~/zrq/evaluate_phase3_smoke_20261101.sh\n'
