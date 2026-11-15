# Lego2 高台攀爬项目上下文

本文件保存长期项目事实；当前阶段和同步状态见 [CURRENT_STATUS.md](CURRENT_STATUS.md)。
跨会话唯一交接入口为 [HANDOFF.md](HANDOFF.md)，维护规则见 [AGENTS.md](../../AGENTS.md)。

更新时间：2026-11-15（Asia/Shanghai）。

## 项目目标与平台

目标是让 Lego2 四足轮足机器人依靠本体感知稳定越过高台，既有目标参考为300 mm。
任务已明确为盲走：部署 actor 不读取高度扫描、地图位置或训练接触真值；
critic、奖励和评估可以使用训练特权信息。其他机器人80 cm演示不证明Lego2可达同一高度。

原始Lego2配置关闭actor的height_scan。用户后来注释了删除扫描的语句，
使修改后的Rough/Platform-Climb保留扫描；不能将这一修改说成原始设计。
新路线的选择依据和适用状态由 [DECISIONS.md](DECISIONS.md) 管理；
具体设计与验收参数查阅 [REFACTOR_PLAN.md](REFACTOR_PLAN.md)，不从本页推断已实现阶段。

## 框架与总体架构

- 环境：mjlab `ManagerBasedRlEnv`，以manager组织观测、动作、指令、事件、奖励、课程和终止。
- 仿真：MuJoCo + MuJoCo-Warp；原生CPU诊断与Warp动态接触验证属于不同证据。
- 强化学习：RSL-RL PPO，现有actor/critic为MLP；配置入口为 `lego2_ppo_runner_cfg()`。
- 控制：8个腿关节位置目标＋4个轮关节速度目标，控制周期0.02 s。
  任务配置的腿/轮scale为0.45/5；缩放不等于动作裁剪。
- 机器人实际加载：`lego2_constants.py:get_spec()` → `xmls/lego2.xml`。
  URDF不是训练时的实时转换入口；两格式的资产修复需要分别保持一致。
- Rough、Flat和Platform-Climb共享机器人资产，资产变化可能影响三个任务。

## 控制与PPO默认配置速查

本节整合自[旧核心代码解析](/home/asus/work/mjlab_doc_quarantine/20261031_approved_cleanup_b5b1f0c3/files/Jiche_Project/粗糙地面速度跟踪/核心代码解析.txt)，并于2026-10-31只读核对
`env_cfgs.py`和`rl_cfg.py`。下表是当前源码默认值，不证明任一历史run实际采用它们；
运行配置仍须查该run的`params/env.yaml`和`params/agent.yaml`。

| 配置 | 当前默认值及含义 |
| --- | --- |
| 腿/轮控制 | 腿为位置目标，轮为速度目标；动作、物理关节目标和观测缩放是不同量 |
| actor观测缩放 | 角速度0.25、腿关节位置偏差1.0、关节速度0.05；旧任务当前保留扫描的原因见前文 |
| actor / critic | 均为512→256→128 MLP，ELU，启用观测归一化 |
| 探索分布 | GaussianDistribution，初始标准差1.0，scalar标准差类型 |
| PPO更新 | clip_param=0.2，5个learning epochs，4个mini batches |
| 学习率与探索项 | learning_rate=1e-3，adaptive schedule，entropy_coef=0.01 |
| 折扣与GAE | gamma=0.99，lam=0.95 |
| 更新约束 | desired_kl=0.01，max_grad_norm=1.0，value_loss_coef=1.0，use_clipped_value_loss=True |
| rollout与保存 | 每环境每轮24步；save_interval=50，默认max_iterations=10000；不据此猜测实际模型编号 |

采样量按“环境数×每环境rollout步数×实际完成的更新轮数”理解。
以24步rollout为例，4096环境每轮98,304条、256环境每轮6,144条；
若4096环境确实完成10,000轮，理论总量为983,040,000条，而非实际完成量的证明。

旧解析里的腿scale常量0.25、旧奖励表及`rl_cfgs.py`写法属于历史描述；
当前生效腿scale与实际文件名分别见前文及关键路径表。
这些历史说明不作为新盲走任务的已实现配置，也不据奖励正负名称推断函数语义。

历史已确认的本地环境为Ubuntu 22.04、RTX 4060 Laptop GPU（8 GB）、
项目隔离的uv `.venv`、Python 3.13.16、PyTorch 2.9.0+cu128、MuJoCo 3.11.0。
这些是既有记录，不代表每个执行环境都能访问CUDA；当次能力和验证范围记录在CURRENT_STATUS。
远程GPU为RTX 4090（24 GB），可用显存需在实际试训前确认。

## 源码、服务器与历史副本

| 用途 | 路径或位置 |
| --- | --- |
| 本地日常编辑与uv环境 | `/home/asus/work/mjlab_lego2_work` |
| 可选GitHub版本管理仓库 | `https://github.com/ElysiaSukhoi/mjlab_lego2_work`（私有）；不作服务器传输中转 |
| 外层研究项目 | `/home/asus/work/Jiche_Project`；旧中文项目路径已迁移，非Jicha_Project |
| 远程SSH | `ajifang@27.128.203.222`，端口 `10395` |
| 远程训练源码 | `~/zrq/mjlab_lego2` |
| 历史Git接入候选目录 | `~/zrq/mjlab_lego2_git*`；接入计划已停用，已有目录或失败现场保留，不作为训练入口 |
| 旧远程只读快照 | `/home/asus/work/Jiche_Project/高台训练/remote_snapshot/20261015_204549/mjlab_lego2` |
| 历史工作副本，非编辑入口 | `/home/asus/work/Jiche_Project/高台训练/working_copy/mjlab_lego2` |
| 远程源码备份记录 | `~/zrq/backups/mjlab_lego2_pre_platform_climb_20261015_215244/`，内有 `source.tar.gz` 和 `remote_state.txt` |

快照没有后来新增的platform_climb专用模块，不能代表当前远程源码。
源码拷贝可能不含Git元数据；当次分支/HEAD可否取得及版本比较依据见CURRENT_STATUS。
快照、备份和实验日志保留其原有来源，不把本地备份冒充远程备份。

## 关键源码路径

下列路径相对本地源码根目录；旧高台专用模块实际位于 `config/lego2/`。

| 功能 | 源码路径 |
| --- | --- |
| 任务注册 | `src/mjlab/tasks/velocity/config/lego2/__init__.py` |
| Rough/Flat配置 | `src/mjlab/tasks/velocity/config/lego2/env_cfgs.py` |
| 盲走任务配置 | `src/mjlab/tasks/velocity/config/lego2/blind_pit_env_cfg.py` |
| 旧高台配置 | `src/mjlab/tasks/velocity/config/lego2/platform_climb_env_cfg.py` |
| 旧高台奖励/指标/终止 | 同目录 `platform_climb_rewards.py`、`platform_climb_metrics.py`、`platform_climb_terminations.py` |
| PPO配置 | `src/mjlab/tasks/velocity/config/lego2/rl_cfg.py` |
| 机器人加载与驱动 | `src/mjlab/asset_zoo/robots/lego2/lego2_constants.py` |
| MJCF / URDF | 同目录 `xmls/lego2.xml` / `urdf/urdf_legowheel2.urdf` |
| 环境生命周期 | `src/mjlab/envs/manager_based_rl_env.py` |
| 观测历史 | `src/mjlab/managers/observation_manager.py`、`src/mjlab/utils/buffers/circular_buffer.py` |
| 地形及等级 | `src/mjlab/terrains/primitive_terrains.py`、`terrain_generator.py`、`terrain_entity.py` |
| 通用奖励/课程/终止 | `src/mjlab/tasks/velocity/mdp/` |
| 训练 / 回放 | `src/mjlab/scripts/train.py` / `src/mjlab/scripts/play.py` |

旧Platform-Climb通过deepcopy派生自Rough，保留独立注册：
`Mjlab-Platform-Climb-Lego2`；另有 `Mjlab-Velocity-Rough-Lego2`、
`Mjlab-Velocity-Flat-Lego2`。新任务的实施和注册状态只查CURRENT_STATUS。

独立入口`Mjlab-Blind-Pit-Lego2`使用`lego2_blind_pit_env_cfg()`，日志名
`lego2_blind_pit`。actor与critic分别deepcopy，避免基础工厂共享term造成历史污染。
actor按角速度/重力/腿位置/全关节速度排列，各5帧；动作12维和xz命令2维只保留当前值。
原生历史按term-major、各term内旧到新拼接，actor共144维；部分reset按环境回填新首帧。
train/play共用工厂，play关闭actor噪声及高度课程。保持12维动作和现有PPO参数，
不直接加载旧checkpoint继续训练；阶段验收及是否具备开训条件以CURRENT_STATUS为准。

新任务地形由十行独立10×10 m地块组成，每块中央3×3 m单级坑，深度3～30 cm。
origin已在坑底，出生采用origin加默认root高度0.47 m，无二次减深度。
20 s episode保持固定vx命令，heading反馈修正yaw；训练从等级0开始。
`blind_pit_curriculums.py`提供独立课程和局部边界：按名称匹配轮子/接触，
用完整轮包络、轮底/接触点高度及向下接触法向确认坑沿支撑，再按单环境结果升降级。
课程只处理已终止episode；play关闭课程，中途手动reset不更新等级。
root超过局部4 m或轮子包络到达4.9 m时truncate，此信号不等于越障成功。

## 训练、续训与回放流程

2026-11-15按用户最新要求，服务器同步恢复SSH直传，GitHub仅用于用户自行决定的本地代码版本管理：
按批修改并解释核心代码 → Codex完成本地uv验证并说明范围/未通过项 →
提供明确文件清单、旧源码备份与SCP/SSH传输指令 → 服务器核对文件内容校验值、运行远程验证 →
配置与功能验收通过后在screen内按阶段试训 → 记录实际结果后决定下一步。
是否commit/push及何时进行均由用户决定，不是上述同步或验证的前置条件；Codex不自动commit/push。
不要求每个小批次建立单独Git版本阶段，也不要求用户回传提交号后才继续机器人工作。
GitHub连接、CI状态及Git提交号均不充当服务器传输门槛，本地与远程uv验证约定仍保留。

远程编辑/训练入口继续为`~/zrq/mjlab_lego2`，用户已确认该目录无Git元数据，不要求初始化。
此前GitHub中转、固定SHA bootstrap以及stage1下载方案均已停用，历史脚本/归档/失败现场保留。
新/更新的远程测试.sh只执行当批验证，不包含Git下载/更新，不硬编码期望commit SHA。
同步与验证分开；使用传输清单、实际文件SHA256和真实run配置追溯实验，不需要为同步创建提交。
同步前按清单备份旧源码，传输不覆盖logs/checkpoint/.venv或无关文件，不自动解压旧文档快照。
代码/观测/资产变更须按实际模型来源核对；运行中的训练或评估使用的源码在进程结束前不更新。
依赖清单变化时另行安排环境核验，不因传输源码而自动更新依赖。
GitHub仍可用于本地代码版本管理；轻量脚本/清单可以跟踪，训练logs/.venv及历史同步压缩包仍忽略。
旧实验结果、已跟踪诊断/历史资料及Git失败现场不自动删除。

- 本地终端与远程SSH终端分开标注；远程通过 `ssh -p 10395 ajifang@27.128.203.222`
  连接后进入 `~/zrq/mjlab_lego2`。禁止未经核对的全目录覆盖或 `rsync --delete`。
- 长训练固定screen名 `zrq`：先 `screen -ls`；存在时 `screen -r zrq`，
  不存在才 `screen -S zrq`。用Ctrl-A、D脱离会话。
- 相同配置续训先确认实际checkpoint；Tyro布尔参数为 `--agent.resume True`，
  不能写小写true。`--agent.load-run`、`--agent.load-checkpoint`取自真实日志文件。
  不保证Ctrl-C恰好保存中断时的迭代，只能从实际存在的checkpoint恢复。
- 观测/奖励/地形/资产变化须按相应技术决策核对新run和版本归属，
  不把旧checkpoint与新配置混成一个可比较实验。

| 旧任务 | 远程日志根目录 |
| --- | --- |
| Platform-Climb | `logs/rsl_rl/lego2_platform_climb/` |
| Rough / Flat | `logs/rsl_rl/lego2_velocity/rough/` / `logs/rsl_rl/lego2_velocity/flat/` |

实验名称、参数、完整checkpoint及结果只由 [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) 保存。
查看实际模型可用远程 `find logs/rsl_rl -type f -name 'model_*.pt'`；
不能由run后缀或编号猜日期目录、环境数和训练资产。

远程TensorBoard命令：`uv run tensorboard --logdir logs/rsl_rl --bind_all`。
本地另开终端建立隧道：

```bash
ssh -N -o ExitOnForwardFailure=yes -p 10395 \
  -L 8080:localhost:8080 -L 6006:localhost:6006 ajifang@27.128.203.222
```

浏览器使用 `http://localhost:8080`（Viser）及 `http://localhost:6006`。
回放使用实际注册任务、`--checkpoint-file`、`--num-envs 1 --viewer viser`；
若Viser监听端口不同，按日志调整隧道。回放加载当前源码配置/资产，
不会自动还原模型训练时的资产；须先核对输入与资产版本。

## 技术证据与研究位置

- [本地接口核查](IMPLEMENTATION_READINESS_20261030.md)：历史、reset、课程和地形原点接口证据。
- [资产审计](INERTIA_AUDIT_20261027.md)：URDF/MJCF圆柱近似修复、CAD及雷达质量待确认项；
  [修复前资产](backups/20261027_lego2_inertia_before_v1/manifest.json) 保留回滚依据。
- [开源交叉对比报告](/home/asus/work/Jiche_Project/高台训练/2026-10-29_盲走轮足高台开源对比报告.md)
  与 [来源/版本清单](/home/asus/work/Jiche_Project/OpenSource/SOURCES_20261029.md) 管理研究证据。
- 参考副本在 `/home/asus/work/Jiche_Project/OpenSource/`：
  Dreamwaq-highplatform、LocoWheeledLegged、DeepRobotics_rl_training、robot_lab；
  仅用于研究，不作为当前训练环境安装或替换。
- 用户原有 [Dreamwaq分析](/home/asus/work/Jiche_Project/Dreamwaq高台训练代码分析.md)、
  [保留的部署手册](../../指引手册.md)及高台训练目录早期交接txt继续保留。
- 2026-10-31获批移出的4份旧文本，原文、原始路径、SHA-256及恢复映射见
  [隔离恢复清单](/home/asus/work/mjlab_doc_quarantine/20261031_approved_cleanup_b5b1f0c3/manifest.json)；当前操作仍用本页SSH与screen约定。
  旧服务器指南中的`ssh mjlab-server`别名和`zrq_lego2`会话名只作历史信息，未核实为当前设置。
- [原人工实验日志](../../日志.md)被`log_viewer.html`读取，原文继续保留；
  档案覆盖范围与实验归属说明见EXPERIMENT_LOG，不把其中“待测”改为已完成。
  旧文档不是新的执行授权；当前工作从HANDOFF进入，不从额外索引或历史单篇报告进入。
