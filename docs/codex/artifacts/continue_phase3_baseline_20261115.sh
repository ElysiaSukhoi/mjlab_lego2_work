#!/usr/bin/env bash
# 原基线续训：1024环境、新增1000次更新；不改奖励/指令分布，不覆盖父run。
set -euo pipefail
case "${STY:-}" in
  *.zrq) ;;
  *) printf '请先进入screen zrq。\n' >&2; exit 1 ;;
esac
PHASE3_CONT_UV="$HOME/.local/bin/uv"
test -x "$PHASE3_CONT_UV"
cd "$HOME/zrq/mjlab_lego2"
PHASE3_CONT_SOURCES="$HOME/zrq/phase3_continue_sources_20261115.sha256"
sha256sum -c "$PHASE3_CONT_SOURCES"
sha256sum -c docs/codex/artifacts/phase3_speed_sources_20261115.sha256
PHASE3_CONT_PARENT="2026-11-01_20-17-22_phase3_low_1024x1000_v1_"
PHASE3_CONT_PARENT+="20261101_201712_2200851"
PHASE3_CONT_CKPT="logs/rsl_rl/lego2_blind_pit/$PHASE3_CONT_PARENT/model_1198.pt"
PHASE3_CONT_BASELINE="$HOME/zrq/backups/phase3_speed_20261115_185511_3295996"
"$PHASE3_CONT_UV" run --no-sync python - \
  "$PHASE3_CONT_CKPT" "$PHASE3_CONT_BASELINE" <<'PY'
from pathlib import Path
import hashlib
import json
import sys
import torch
checkpoint, baseline = map(Path, sys.argv[1:])
expected = '7780cf0d2557c86f51c0486f870e7d471b13703777e9cee5e1854427def05027'
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == expected
summary = json.loads((baseline / 'full/paired_summary.json').read_text())
assert summary['mode'] == 'full' and summary['pairing_verified']
assert summary['checkpoint_sha256'] == expected
assert [r['speed_m_s'] for r in summary['results']] == [.3, .4, .5]
assert [r['n'] for r in summary['results']] == [10, 10, 10]
for name, digest in summary['input_report_sha256'].items():
    report = baseline / 'full' / name
    assert hashlib.sha256(report.read_bytes()).hexdigest() == digest
    data = json.loads(report.read_text())
    assert data['checkpoint_sha256'] == expected
    assert data['paired_seed_base'] == 7
    for source, wanted in data['source_sha256'].items():
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == wanted, source
assert torch.cuda.is_available(), 'CUDA不可用，未启动训练'
free, _ = torch.cuda.mem_get_info(0)
print(torch.cuda.get_device_name(0), 'free MiB:', free // 2**20)
assert free >= 12 * 2**30, '空闲显存不足12 GiB'
lines = Path('/proc/meminfo').read_text().splitlines()
mem = dict(line.split(':', 1) for line in lines)
assert int(mem['MemAvailable'].split()[0]) >= 8 * 2**20, '可用内存不足8 GiB'
print('父模型及原始配对评估报告校验通过。')
PY

PHASE3_CONT_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$"
PHASE3_CONT_OUT="$HOME/zrq/backups/phase3_continue_$PHASE3_CONT_STAMP"
PHASE3_CONT_RUN="phase3_continue_1024x1000_v1_$PHASE3_CONT_STAMP"
mkdir "$PHASE3_CONT_OUT"
cp "$PHASE3_CONT_SOURCES" "$PHASE3_CONT_OUT/"
cp "$HOME/zrq/continue_phase3_baseline_20261115.sh" "$PHASE3_CONT_OUT/"
cp "$PHASE3_CONT_BASELINE/full/paired_summary.json" "$PHASE3_CONT_OUT/baseline.json"
cp "logs/rsl_rl/lego2_blind_pit/$PHASE3_CONT_PARENT/params/env.yaml" \
  "$PHASE3_CONT_OUT/parent_env.yaml"
cp "logs/rsl_rl/lego2_blind_pit/$PHASE3_CONT_PARENT/params/agent.yaml" \
  "$PHASE3_CONT_OUT/parent_agent.yaml"
sha256sum "$PHASE3_CONT_CKPT" > "$PHASE3_CONT_OUT/parent_checkpoint.sha256"
free -m | tee "$PHASE3_CONT_OUT/memory_before.txt"
nvidia-smi | tee "$PHASE3_CONT_OUT/gpu_before.txt"
export FORCE_CPU=0
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="$HOME/zrq/backups/phase3_20261031_214701_2152912/cache"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export MPLCONFIGDIR="$PHASE3_CONT_OUT/cache/matplotlib"
PHASE3_CONT_CMD=(
  "$PHASE3_CONT_UV" run --no-sync train Mjlab-Blind-Pit-Lego2
  --env.scene.num-envs 1024
  --agent.max-iterations 1000
  --agent.seed 7
  --agent.resume True
  --agent.load-run "^$PHASE3_CONT_PARENT$"
  --agent.load-checkpoint '^model_1198[.]pt$'
  --agent.logger tensorboard
  --agent.run-name "$PHASE3_CONT_RUN"
  --gpu-ids '[0]'
)
printf '%q ' "${PHASE3_CONT_CMD[@]}" > "$PHASE3_CONT_OUT/command.sh"
printf '\n' >> "$PHASE3_CONT_OUT/command.sh"
printf '父模型：%s\n新run后缀：%s\n证据目录：%s\n' \
  "$PHASE3_CONT_CKPT" "$PHASE3_CONT_RUN" "$PHASE3_CONT_OUT"
printf '新增1000次更新，计划24576000样本；恢复优化器/迭代，仿真和课程重新reset。\n'
printf '训练vx仍为0.3～0.6 m/s，课程从3 cm起步；原模型/结果保留。\n'
set +e
"${PHASE3_CONT_CMD[@]}" 2>&1 | tee "$PHASE3_CONT_OUT/train_stdout.txt"
PHASE3_CONT_STATUS="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$PHASE3_CONT_STATUS" > "$PHASE3_CONT_OUT/exit_code.txt"
find logs/rsl_rl/lego2_blind_pit -mindepth 1 -maxdepth 1 -type d \
  -name "*_$PHASE3_CONT_RUN" -print > "$PHASE3_CONT_OUT/run_dirs.txt"
while IFS= read -r PHASE3_CONT_DIR; do
  find "$PHASE3_CONT_DIR" -type f \
    \( -name 'model_*.pt' -o -name 'env.yaml' -o -name 'agent.yaml' \) -print
done < "$PHASE3_CONT_OUT/run_dirs.txt" | tee "$PHASE3_CONT_OUT/run_files.txt"
printf '续训退出码：%s\n证据目录：%s\n' "$PHASE3_CONT_STATUS" "$PHASE3_CONT_OUT"
exit "$PHASE3_CONT_STATUS"
