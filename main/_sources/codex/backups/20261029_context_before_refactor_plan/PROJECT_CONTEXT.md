# Lego2 高台攀爬项目上下文

更新时间：2026-10-29（Asia/Shanghai）

## 目标与边界

最终目标仍是Lego2轮足机器人稳定越过高台。2026-10-29用户明确任务为**盲走**，
并提出中央坑式训练、从开始启用低高度课程、速度跟踪主导且尽量复用已有奖励的新方向。
目前只做开源研究和可行性讨论；未获准开始实施重构或新训练。

当前已实现的旧任务是从低地正面攀爬固定150mm平台，它的actor仍有height_scan，
因此不能称当前代码已经实现盲走。300mm是既有目标参考，其他机器人80cm～1m结果
不能直接作为Lego2硬件可行性结论。独立直线高台可继续作为未来泛化评估场景。

## 软件与硬件

- 框架：mjlab，环境管理器为 ManagerBased RL。
- 仿真：MuJoCo + MuJoCo-Warp。
- 算法：RSL-RL PPO，配置在 `src/mjlab/tasks/velocity/config/lego2/rl_cfg.py`。
- 机器人：Lego2 轮足机器人；资产在 `src/mjlab/asset_zoo/robots/lego2/`。
- 训练通过 `lego2_constants.py:get_spec()` 直接加载 `xmls/lego2.xml`，URDF
  `urdf/urdf_legowheel2.urdf` 不是实时训练加载入口；二者需人工同步资产修复。
- 本地：Ubuntu 22.04（聊天中确认），RTX 4060 Laptop GPU，uv 隔离环境，Python 3.13.16，PyTorch 2.9.0+cu128，MuJoCo 3.11.0，Warp 可导入。
- 远程：`ajifang@27.128.203.222:10395`，项目 `~/zrq/mjlab_lego2`，GPU 为 RTX 4090。
- 长时间远程训练统一使用 screen 会话名 `zrq`。

## 代码、快照与备份

| 用途 | 路径 |
| --- | --- |
| 本地编辑工作副本 | `/home/asus/work/mjlab_lego2_work` |
| 远程源码只读快照 | `/home/asus/work/Jiche_Project/高台训练/remote_snapshot/20261015_204549/mjlab_lego2` |
| 远程训练源码 | `~/zrq/mjlab_lego2` |
| 远程源码 tar 备份 | `~/zrq/backups/mjlab_lego2_pre_platform_climb_20261015_215244/source.tar.gz` |
| 远程状态记录 | `~/zrq/backups/mjlab_lego2_pre_platform_climb_20261015_215244/remote_state.txt` |

截至整理时，没有从 Git 提交历史得到足够可靠的版本依据；快照、tar 备份和训练日志是主要回滚依据。

远程源码快照 `remote_snapshot/20261015_204549/mjlab_lego2` 只包含当时的原始 Lego2 配置文件，没有当前新增的 `platform_climb_env_cfg.py`、`platform_climb_rewards.py`、`platform_climb_metrics.py` 和 `platform_climb_terminations.py`。因此快照只能作为早期回滚参考，不能用来判断当前高台代码是否已同步远程。

关键路径：

- 任务注册：`src/mjlab/tasks/velocity/config/lego2/__init__.py`
- Rough/Flat 基础环境：`src/mjlab/tasks/velocity/config/lego2/env_cfgs.py`
- 高台环境：`src/mjlab/tasks/velocity/config/lego2/platform_climb_env_cfg.py`
- 高台奖励：`src/mjlab/tasks/velocity/config/lego2/platform_climb_rewards.py`
- 高台指标：`src/mjlab/tasks/velocity/config/lego2/platform_climb_metrics.py`
- 高台成功终止：`src/mjlab/tasks/velocity/config/lego2/platform_climb_terminations.py`
- PPO：`src/mjlab/tasks/velocity/config/lego2/rl_cfg.py`
- 训练入口：`src/mjlab/scripts/train.py`
- 回放入口：`src/mjlab/scripts/play.py`

当前高台专用模块位于 `config/lego2/`，不要根据旧手册把它们误认为已经位于 `tasks/velocity/mdp/`。

## 任务注册与训练流程

任务 ID：

```text
Mjlab-Platform-Climb-Lego2
Mjlab-Velocity-Rough-Lego2
Mjlab-Velocity-Flat-Lego2
```

高台任务通过 `deepcopy(lego2_rough_env_cfg(...))` 建立独立配置，日志实验名为 `lego2_platform_climb`。典型流程是：本地修改 → 本地 `uv run` 编译/加载检查 → 同步远程 → 远程编译和任务注册检查 → screen 中短训练 → 记录结果 → 再决定下一次修改。

远程训练日志根目录为 `logs/rsl_rl/lego2_platform_climb/`；Rough 基线为 `logs/rsl_rl/lego2_velocity/rough/`。TensorBoard 使用 `uv run tensorboard --logdir logs/rsl_rl/lego2_velocity --bind_all`，Viser 默认使用 8080；本地端口转发命令和回放命令见 `HANDOFF.md`。

## 当前高台配置摘要

- 当前工作副本的固定平台高度为 `0.15 m`，用于动作诊断；300 mm 目标尚未撤销。
- 地形为单行单列固定 BoxPyramidStairs，平台宽 3.0 m，出生点在平台负 x 方向，x 约 `(-3.2, -2.8)`，y 约 `(-0.15, 0.15)`，yaw 为 0。
- 指令仅保留正向速度：`lin_vel_x=(0.30,0.80)`，横向和角速度为 0。
- 已建立接近奖励、边缘抬升奖励、前足抬升奖励、前足接触后继续前移奖励、顶部奖励、顶部稳定指标和诊断指标。
- 高台基线关闭了会掩盖失败原因的多数动力学/推力随机化及 terrain/friction curriculum；固定的 `foot_friction_slide` 事件仍存在。
- 共同 MuJoCo 接触配置已恢复为 `ccd_iterations=300`、`contact_sensor_maxmatch=500`、`nconmax=99`。Flat 任务自己的 `50/64/None` 覆盖保持不变。

## 工作约定

本地 ASCII 路径优先，原始快照不直接编辑。每轮先做小范围、可回退的代码变更；本地校验由 Codex 完成，用户只接收远程同步和远程校验命令。训练参数、checkpoint、成功指标必须与“本地已验证”“远程已验证”“训练已验证”区分记录。

2026-10-27 起优先核查共享机器人质量属性。第一批圆柱轴向/轮子质心近似修复和
CPU 验证详见 `INERTIA_AUDIT_20261027.md`，原文件保存在
`docs/codex/backups/20261027_lego2_inertia_before_v1/`。它影响三个 Lego2 任务，
不能继续沿用“只有高台任务发生变化”的假设。未确认数据继续标为待确认。

## 2026-10-29研究资料

外层项目已迁移到 `/home/asus/work/Jiche_Project`，OpenSource下有用户提供的
Dreamwaq-highplatform，以及本轮下载的LocoWheeledLegged、DeepRobotics_rl_training、robot_lab。
只用于静态研究，来源/提交见 `OpenSource/SOURCES_20261029.md`；报告见
`高台训练/2026-10-29_盲走轮足高台开源对比报告.md`。原有分析报告保留未覆盖。
用户确认资产纠偏完成且远程同步；旧checkpoint训练时资产与回放资产对应仍待确认。
