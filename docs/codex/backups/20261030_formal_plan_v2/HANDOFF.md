# Codex 交接与操作手册

更新时间：2026-10-30（Asia/Shanghai）

## 当前边界（2026-10-29）

用户已确认资产纠偏完成且远程同步。**本轮只做方案研究，禁止自行改训练业务代码或开训**。
当前目标是盲走。大体五阶段规划已经制定，见 `REFACTOR_PLAN.md`，尚未实施。旧操作示例仅供未来获准实施时参考。
研究报告：`/home/asus/work/Jiche_Project/高台训练/2026-10-29_盲走轮足高台开源对比报告.md`。
源码来源：`/home/asus/work/Jiche_Project/OpenSource/SOURCES_20261029.md`。

## 新聊天的读取顺序

先读本文件与 `REFACTOR_PLAN.md`，再读 `CURRENT_STATUS.md`、`DECISIONS.md`、`PROJECT_CONTEXT.md` 和
`EXPERIMENT_LOG.md`。当前需额外读取 `INERTIA_AUDIT_20261027.md`。早期说明保留在
外层项目的指引手册和交接 txt 中，旧记录中提及的 `task.txt`/`task_2.txt` 是否仍存在
需先检查，不能把旧文档路径当作已确认可用。

## 开始工作前

本地：

```bash
cd /home/asus/work/mjlab_lego2_work
uv run python -m compileall -q <本次修改的.py文件>
uv run list-envs | grep -E 'Mjlab-(Platform-Climb|Velocity-(Flat|Rough))-Lego2'
```

远程：

```bash
ssh -p 10395 ajifang@27.128.203.222
cd ~/zrq/mjlab_lego2
```

同步时只传本次明确修改的文件，并先在远程保留可回滚副本。不要使用未经确认的 `rsync --delete`。远程同步后执行同样的 `compileall`、`list-envs` 和配置加载校验；本地校验由 Codex 完成，用户只需要执行远程命令并回传输出。

## 远程长训练

新训练必须在远程 screen 会话 `zrq` 中启动：

```bash
screen -S zrq
cd ~/zrq/mjlab_lego2
uv run train Mjlab-Platform-Climb-Lego2 \
  --env.scene.num-envs 2048 \
  --agent.max-iterations 1000 \
  --agent.run-name <新的_run_name> \
  --agent.logger tensorboard
```

按 `Ctrl-A`、`D` 脱离；查看用 `screen -r zrq`。如需续训，先确认 checkpoint，再使用 Tyro 接受的大小写布尔值：

```bash
--agent.resume True
```

`--agent.load-run`、`--agent.load-checkpoint` 的实际路径必须从远程 `find logs/rsl_rl/lego2_platform_climb -name 'model_*.pt'` 确认，不能凭日期猜测。

## 监控与回放

远程启动 TensorBoard：

```bash
uv run tensorboard --logdir logs/rsl_rl/lego2_velocity --bind_all
```

本地另开终端建立隧道：

```bash
ssh -N -p 10395 \
  -L 8080:localhost:8080 -L 6006:localhost:6006 \
  ajifang@27.128.203.222
```

浏览器访问 `http://localhost:8080`（Viser）和 `http://localhost:6006`（TensorBoard）。回放命令的 checkpoint 必须先在远程确认。

### 2026-10-28 当前操作：只回放已有高台策略

用户明确要求先不训练。已在本地通过 `uv run --no-sync play
Mjlab-Platform-Climb-Lego2 --help` 核实 CLI 参数，未启动仿真或远程操作。

本轮用户指定目标 run：`platform_climb_front_lift_full_150mm_2048_v1`。
只在该 run 内查找实际 checkpoint，不猜测时间目录或 `model_9999.pt`。
在远程 `zrq` 会话的空闲窗口执行以下命令；如有多个同名后缀的日期目录，
按排序选择最新目录中的最大 checkpoint 编号。没有匹配时不启动回放：

```bash
cd ~/zrq/mjlab_lego2
CHECKPOINT="$(find logs/rsl_rl/lego2_platform_climb -type f \
  -path '*_platform_climb_front_lift_full_150mm_2048_v1/model_*.pt' \
  | sort -V | tail -n 1)"

if [ -n "${CHECKPOINT}" ]; then
  printf '回放 checkpoint：%s\n' "${CHECKPOINT}"
  uv run play Mjlab-Platform-Climb-Lego2 \
    --checkpoint-file "${CHECKPOINT}" \
    --num-envs 1 \
    --viewer viser
else
  echo '未找到该实验的 checkpoint，暂未启动回放。'
fi
```

本地另开终端建立仅回放用的 SSH 隧道：

```bash
ssh -N -o ExitOnForwardFailure=yes -p 10395 \
  -L 8080:localhost:8080 ajifang@27.128.203.222
```

浏览器打开 `http://localhost:8080`。若 Viser 实际监听其他端口，隧道右侧服务器
端口需按启动日志调整；已有 8080 隧道则直接复用。

`play` 加载当前源码的 play 配置/资产，不会自动恢复 checkpoint 训练时的模型。
若远程已替换惯量，旧 checkpoint 配新惯量的表现不能当作旧实验的原样复现。
2026-10-29 用户确认远程资产已同步；具体训练/回放版本对应仍待确认。观察接近、前足跨边缘、机身/后足是否登顶、
有无穿透或弹飞；记录回放所用 checkpoint 和资产版本，不生成新训练结果。

## 下一次工作的具体起点

大体规划已完成且未实施，详见 `REFACTOR_PLAN.md`；不要再次从零研究或直接开训。
用户要求实施时，从阶段1建立独立任务和盲走actor/历史开始；阶段2一起接入坑、出生、
指令与初始高度课程；阶段3完成通用奖励基线和评估后才开始首轮短训。阶段4接触反应
仅在基线出现明确瓶颈时探索；阶段5再做完整训练和独立直线高台泛化评估。

本轮无业务代码变更、无新实验。候选参数如5帧历史、3～30cm/10级和约3m升级距离
尚未验证，不能当已经实现。用户已确认视频使用全量训练checkpoint编号9999；
完整路径、训练参数与资产对应仍待确认。

本地源码 `/home/asus/work/mjlab_lego2_work`，外层项目 `/home/asus/work/Jiche_Project`。
未来每阶段结合核心代码讲解；本地uv验证由Codex完成，用户只执行远程同步/校验。
远程训练统一screen `zrq`；保留旧结果。用户未要求实施时不修改训练代码或启动训练。

## 状态标签

记录时明确使用：`本地已验证`、`已同步远程`、`远程已验证`、`训练已验证`、`待确认`。这能防止把本地代码、远程代码和训练产物混为同一版本。

## 2026-10-30最新澄清（继续工作先读）

- 原始配置actor不读取height_scan；用户后来注释了删除语句，当前修改版本才保留扫描。
- 视频来自全量训练checkpoint 9999；多次Reset均未通过，每次碰台前轮不抬、后腿反抬。
  不再把它描述成只有单个短片段，或据front_lift均值认定前腿抬升已学到。
- 精确测试次数、checkpoint完整目录/实际配置及训练与回放资产对应待确认。
- 用户尚未要求实施：只记录和讨论，不改业务代码、不运行训练。后续实施前
  先核对足端索引、动作关节映射/符号/限幅与前腿清出台沿的运动学可行性，
  不能仅凭失败现象认定前后腿颠倒或机械上绝对不能抬前轮。
