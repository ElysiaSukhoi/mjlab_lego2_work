#!/usr/bin/env bash
# 复用既有评估工具：新模型与父模型同场景/同配对初态，独立目录保存。
set -euo pipefail
case "${STY:-}" in
  *.zrq) ;;
  *) printf '请先进入screen zrq。\n' >&2; exit 1 ;;
esac
PHASE3_REEVAL_UV="$HOME/.local/bin/uv"
test -x "$PHASE3_REEVAL_UV"
cd "$HOME/zrq/mjlab_lego2"
sha256sum -c "$HOME/zrq/phase3_continue_sources_20261115.sha256"
sha256sum -c docs/codex/artifacts/phase3_speed_sources_20261115.sha256
PHASE3_REEVAL_RUN="2026-11-15_19-28-14_phase3_continue_1024x1000_v1_"
PHASE3_REEVAL_RUN+="20261115_192803_3297676"
PHASE3_REEVAL_CKPT="logs/rsl_rl/lego2_blind_pit/$PHASE3_REEVAL_RUN/model_2197.pt"
PHASE3_REEVAL_TRAIN="$HOME/zrq/backups/phase3_continue_20261115_192803_3297676"
PHASE3_REEVAL_BASE="$HOME/zrq/backups/phase3_speed_20261115_185511_3295996"
"$PHASE3_REEVAL_UV" run --no-sync python - \
  "$PHASE3_REEVAL_CKPT" "$PHASE3_REEVAL_TRAIN" <<'PY'
from pathlib import Path
import sys
import torch
checkpoint, evidence = map(Path, sys.argv[1:])
assert (evidence / 'exit_code.txt').read_text().strip() == '0'
runs = {Path(p).resolve() for p in (evidence / 'run_dirs.txt').read_text().splitlines()}
assert checkpoint.parent.resolve() in runs, '模型不属于本轮记录的run'
for name in ('env.yaml', 'agent.yaml'):
    assert (checkpoint.parent / 'params' / name).is_file()
state = torch.load(checkpoint, map_location='cpu', weights_only=False)
assert int(state['iter']) == 2197, 'checkpoint内部迭代号不符'
del state
assert torch.cuda.is_available(), 'CUDA不可用，未开始评估'
free, _ = torch.cuda.mem_get_info(0)
assert free >= 3 * 2**30, '空闲显存不足3 GiB'
lines = Path('/proc/meminfo').read_text().splitlines()
mem = dict(line.split(':', 1) for line in lines)
assert int(mem['MemAvailable'].split()[0]) >= 4 * 2**20, '可用内存不足4 GiB'
print('GPU:', torch.cuda.get_device_name(0), 'free MiB:', free // 2**20)
PY
test -f "$PHASE3_REEVAL_BASE/full/paired_summary.json"
PHASE3_REEVAL_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$"
PHASE3_REEVAL_OUT="$HOME/zrq/backups/phase3_eval2197_$PHASE3_REEVAL_STAMP"
mkdir "$PHASE3_REEVAL_OUT"
cp "$HOME/zrq/evaluate_phase3_model2197_20261115.sh" "$PHASE3_REEVAL_OUT/"
cp "$PHASE3_REEVAL_TRAIN/exit_code.txt" "$PHASE3_REEVAL_OUT/train_exit_code.txt"
cp "${PHASE3_REEVAL_CKPT%/*}/params/env.yaml" "$PHASE3_REEVAL_OUT/train_env.yaml"
cp "${PHASE3_REEVAL_CKPT%/*}/params/agent.yaml" "$PHASE3_REEVAL_OUT/train_agent.yaml"
cp "$PHASE3_REEVAL_BASE/full/paired_summary.json" "$PHASE3_REEVAL_OUT/baseline.json"
cp "$HOME/zrq/phase3_continue_sources_20261115.sha256" "$PHASE3_REEVAL_OUT/"
cp docs/codex/artifacts/phase3_speed_sources_20261115.sha256 "$PHASE3_REEVAL_OUT/"
sha256sum "$PHASE3_REEVAL_CKPT" | tee "$PHASE3_REEVAL_OUT/checkpoint.sha256"
free -m > "$PHASE3_REEVAL_OUT/memory_before.txt"
nvidia-smi > "$PHASE3_REEVAL_OUT/gpu_before.txt"
export FORCE_CPU=0
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="$HOME/zrq/backups/phase3_20261031_214701_2152912/cache"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export MPLCONFIGDIR="$PHASE3_REEVAL_OUT/cache/matplotlib"
mkdir "$PHASE3_REEVAL_OUT/gpu_smoke" "$PHASE3_REEVAL_OUT/full"
printf 'model2197复评目录：%s\n' "$PHASE3_REEVAL_OUT"
for PHASE3_REEVAL_MODE in gpu_smoke full; do
  PHASE3_REEVAL_EPISODES=10
  PHASE3_REEVAL_LIMIT=()
  PHASE3_REEVAL_SUMMARY_MODE=full
  if [[ "$PHASE3_REEVAL_MODE" = gpu_smoke ]]; then
    PHASE3_REEVAL_EPISODES=2
    PHASE3_REEVAL_LIMIT=(--max-steps 2)
    PHASE3_REEVAL_SUMMARY_MODE=smoke
  fi
  for PHASE3_REEVAL_VX in 0.30 0.40 0.50; do
    PHASE3_REEVAL_CMD=(
      "$PHASE3_REEVAL_UV" run --no-sync python scripts/tools/evaluate_lego2_blind_pit.py
      --checkpoint "$PHASE3_REEVAL_CKPT" --device cuda:0 --heights 0.30
      --forward-speed "$PHASE3_REEVAL_VX" --paired-seed-base 7
      --episodes "$PHASE3_REEVAL_EPISODES" --seed 7
      "${PHASE3_REEVAL_LIMIT[@]}"
      --output "$PHASE3_REEVAL_OUT/$PHASE3_REEVAL_MODE/vx_$PHASE3_REEVAL_VX"
    )
    "${PHASE3_REEVAL_CMD[@]}" 2>&1 \
      | tee "$PHASE3_REEVAL_OUT/$PHASE3_REEVAL_MODE/vx_$PHASE3_REEVAL_VX.txt"
  done
  "$PHASE3_REEVAL_UV" run --no-sync python \
    scripts/tools/summarize_lego2_speed_eval.py \
    --directory "$PHASE3_REEVAL_OUT/$PHASE3_REEVAL_MODE" \
    --episodes "$PHASE3_REEVAL_EPISODES" --mode "$PHASE3_REEVAL_SUMMARY_MODE" \
    | tee "$PHASE3_REEVAL_OUT/$PHASE3_REEVAL_MODE/summary.txt"
done

# 不只检查新模型内部配对；跨模型也核对原始reset状态，避免初态变化造成假改善。
"$PHASE3_REEVAL_UV" run --no-sync python - \
  "$PHASE3_REEVAL_BASE/full" "$PHASE3_REEVAL_OUT/full" <<'PY' \
  | tee "$PHASE3_REEVAL_OUT/comparison.txt"
from pathlib import Path
import hashlib
import json
import math
import sys
baseline, current = map(Path, sys.argv[1:])
summaries = [json.loads((p / 'paired_summary.json').read_text())
             for p in (baseline, current)]
for directory, summary in zip((baseline, current), summaries, strict=True):
    assert summary['mode'] == 'full' and summary['pairing_verified']
    assert [r['speed_m_s'] for r in summary['results']] == [.3, .4, .5]
    assert all(r['n'] == 10 for r in summary['results'])
    for name, digest in summary['input_report_sha256'].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
comparisons = []
for speed in (.3, .4, .5):
    name = f'vx_{speed:.2f}/report.json'
    old, new = [json.loads((directory / name).read_text())
                for directory in (baseline, current)]
    assert old['source_sha256'] == new['source_sha256']
    assert old['paired_seed_base'] == new['paired_seed_base'] == 7
    assert old['seed'] == new['seed'] == 7
    before, after = old['heights'][0], new['heights'][0]
    assert before['height_m'] == after['height_m'] == .3
    assert before['n'] == after['n'] == 10
    gained, lost = [], []
    for episode, (left, right) in enumerate(
        zip(before['samples'], after['samples'], strict=True)
    ):
        assert left['reset_seed'] == right['reset_seed'] == 7 + episode
        assert left['initial_state'].keys() == right['initial_state'].keys()
        for field, values in left['initial_state'].items():
            other = right['initial_state'][field]
            if len(values) != len(other) or not all(
                math.isclose(a, b, rel_tol=0, abs_tol=1e-6)
                for a, b in zip(values, other, strict=True)
            ):
                raise ValueError(f'跨模型初态不匹配：vx={speed}, seed={7+episode}')
        if not left['passed'] and right['passed']:
            gained.append(7 + episode)
        if left['passed'] and not right['passed']:
            lost.append(7 + episode)
    item = {'speed_m_s': speed, 'old_passed': before['passed'],
            'new_passed': after['passed'], 'n': 10,
            'improved_seeds': gained, 'regressed_seeds': lost}
    comparisons.append(item)
    print(json.dumps(item))
result = {'cross_model_initial_states_verified': True,
          'baseline_checkpoint_sha256': summaries[0]['checkpoint_sha256'],
          'new_checkpoint_sha256': summaries[1]['checkpoint_sha256'],
          'comparisons': comparisons}
with (current.parent / 'comparison.json').open('x') as stream:
    json.dump(result, stream, indent=2)
    stream.write('\n')
print('PASS: 新/父模型的10组初态跨模型核对通过，容差1e-6。')
PY
printf '复评完成，无PPO更新。证据目录：%s\n' "$PHASE3_REEVAL_OUT"
