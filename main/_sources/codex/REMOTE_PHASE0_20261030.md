# 阶段0远程同步与校验

状态：本地已验证；下列远程命令尚未执行。详细结果见
[阶段0记录](PHASE0_DIAGNOSTICS_20261030.md)。本批不启动训练/回放。
旧模型和`logs/rsl_rl`只读；同步包只有明确清单中的诊断、测试、文档和本地备份。
不传机器人资产或旧任务源码，不重复覆盖用户已同步的资产修复。

## 1. 本地终端：传送已准备好的同步包

Codex已在本地创建并检查压缩包，不需要用户重复本地测试：

```bash
cd /home/asus/work/mjlab_lego2_work
scp -P 10395 docs/codex/artifacts/lego2_phase0_20261030_v1.tar.gz \
  ajifang@27.128.203.222:~/zrq/
ssh -p 10395 ajifang@27.128.203.222
```

包内文件以[清单](phase0_sync_files.txt)为准，校验和见
[同步hash](phase0_sync.sha256)及[当前业务输入hash](phase0_expected_sources.sha256)。
`backups/20261030_phase0_before_v1`是本地批前记录；下一步还会单独备份远程原文件，
两者来源不同，不能互相冒充。

## 2. 远程终端：先备份，再只提取清单文件并校验

整段执行即可。`set -euo pipefail`只在子bash中生效；中途失败会停止本批操作，
请保留并回传错误及已打印的备份位置，不要补传整套业务源码。

```bash
bash <<'BASH'
set -euo pipefail
cd "$HOME/zrq/mjlab_lego2"
PHASE0_BUNDLE="$HOME/zrq/lego2_phase0_20261030_v1.tar.gz"
PHASE0_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
PHASE0_BACKUP="$HOME/zrq/backups/phase0_${PHASE0_STAMP}_$$"
mkdir -p "$PHASE0_BACKUP/files"
printf '本批远程备份与证据目录：%s\n' "$PHASE0_BACKUP"

tar -xOf "$PHASE0_BUNDLE" docs/codex/phase0_sync_files.txt \
  > "$PHASE0_BACKUP/incoming_files.txt"
while IFS= read -r PHASE0_FILE; do
  if [ -f "$PHASE0_FILE" ]; then
    cp --parents --preserve=mode,timestamps -- \
      "$PHASE0_FILE" "$PHASE0_BACKUP/files/"
  else
    printf '%s\n' "$PHASE0_FILE" >> "$PHASE0_BACKUP/previously_absent.txt"
  fi
done < "$PHASE0_BACKUP/incoming_files.txt"

tar -xzf "$PHASE0_BUNDLE" --no-same-owner --verbatim-files-from \
  --files-from "$PHASE0_BACKUP/incoming_files.txt"
sha256sum -c docs/codex/phase0_sync.sha256 \
  | tee "$PHASE0_BACKUP/sync_hash_check.txt"
sha256sum -c docs/codex/phase0_expected_sources.sha256 \
  | tee "$PHASE0_BACKUP/source_hash_check.txt"

export UV_CACHE_DIR="$PHASE0_BACKUP/cache/uv"
export MPLCONFIGDIR="$PHASE0_BACKUP/cache/matplotlib"
export XDG_CACHE_HOME="$PHASE0_BACKUP/cache"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export FORCE_CPU=1

uv run --no-sync python -m compileall -q \
  scripts/tools/inspect_lego2_actions.py tests/test_lego2_action_diagnostics.py
uv run --no-sync python -m mjlab.scripts.list_envs --keyword Lego2 \
  | tee "$PHASE0_BACKUP/list_envs.txt"
uv run --no-sync python - <<'PY'
import mjlab.tasks
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

expected = {
    'Mjlab-Platform-Climb-Lego2',
    'Mjlab-Velocity-Flat-Lego2',
    'Mjlab-Velocity-Rough-Lego2',
}
assert {t for t in list_tasks() if 'Lego2' in t} == expected
for task in sorted(expected):
    for play in (False, True):
        cfg = load_env_cfg(task, play=play)
        assert len(cfg.actions['joint_pos'].actuator_names) == 8
        assert len(cfg.actions['joint_vel'].actuator_names) == 4
        assert cfg.actions['joint_pos'].scale == 0.45
        assert cfg.actions['joint_vel'].scale == 5.0
        assert all(term.clip is None for term in cfg.actions.values())
        print(task, 'play=', play, 'config OK')
    assert load_rl_cfg(task).clip_actions is None
print('PASS: 6 env configs, 3 agent configs; no Blind-Pit registration')
PY

uv run --no-sync pytest -q tests/test_lego2_action_diagnostics.py \
  tests/test_asset_zoo.py -k 'diagnostic or Lego2' \
  | tee "$PHASE0_BACKUP/tests.txt"
uv run --no-sync python scripts/tools/inspect_lego2_actions.py \
  --output "$PHASE0_BACKUP/phase0_remote_native_cpu.json" \
  | tee "$PHASE0_BACKUP/diagnostic_stdout.txt"

# 只读定位用户指定的旧run，不猜日期目录，也不读取/执行pickle。
if [ -d logs/rsl_rl/lego2_platform_climb ]; then
  find logs/rsl_rl/lego2_platform_climb -type f \
    -path '*_platform_climb_front_lift_full_150mm_2048_v1/*' \
    \( -name 'model_9999.pt' -o -path '*/params/*' \) \
    | sort | tee "$PHASE0_BACKUP/old_run_files.txt"
else
  printf '待确认：旧Platform-Climb日志目录不存在\n' \
    | tee "$PHASE0_BACKUP/old_run_files.txt"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.free,driver_version --format=csv \
    | tee "$PHASE0_BACKUP/gpu_state.txt"
fi
printf '阶段0远程CPU校验结束，证据目录：%s\n' "$PHASE0_BACKUP"
BASH
```

该CPU诊断在有GPU的服务器上仍执行原生MuJoCo CPU动态部分，不能标为Warp动态通过。
`nvidia-smi`只记录当前状态，不启动GPU作业。
如果源码/资产hash不一致，命令会在运行诊断前停止；先回传不一致文件，核对差异归属。
这不会自动要求重做已确认完成的惯量修复。

回传：hash校验结果、pytest摘要、诊断末尾映射、旧run文件清单及最后的证据目录。
如需进一步核对，JSON中保存了每帧数据，无需手工从Viser估算动作。
旧run清单为空时完整checkpoint目录/参数仍标为待确认，不能猜一个新路径。

## 3. 回退方法

本批没有训练任务/资产变更。如需撤回同步，用实际打印的备份路径中的`files/`
恢复相同相对路径文件。`previously_absent.txt`记录原先不存在的文件，先逐项核对
再撤下本批新增项；不要对源码或日志执行整目录删除。
原本地434个已有源码/资产/依赖的hash保存在批前manifest中，可用于核对。

远程验证通过后再把状态更新为“已同步远程、远程CPU已验证”，继续阶段1。
后续正式训练仍使用screen `zrq`；本批没有新任务train命令。
