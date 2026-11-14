# Lego2 项目资料索引与新聊天入口

核查日期：2026-10-30（Asia/Shanghai）。本页负责资料位置和关联，不替代当前状态或修改计划。
新聊天即使看不到旧对话，也应从源码根目录的[AGENTS.md](../../AGENTS.md)开始。
AGENTS.md目前是指向[CLAUDE.md](../../CLAUDE.md)的符号链接，保留链接即可，不要拆成两份规则。

## 1. 先确认工作位置

| 位置 | 用途 |
| --- | --- |
| [本地训练源码根目录](/home/asus/work/mjlab_lego2_work) | 唯一日常编辑、uv验证的位置，虚拟环境为其.venv |
| [外层项目](/home/asus/work/Jiche_Project) | 旧中文目录已迁移到此；不是Jicha_Project |
| [OpenSource](/home/asus/work/Jiche_Project/OpenSource) | 参考代码研究副本，不作为当前训练环境安装/替换 |
| [旧远程快照](/home/asus/work/Jiche_Project/高台训练/remote_snapshot/20261015_204549/mjlab_lego2) | 2026-10-15历史回滚参考，保持只读；不代表当前远程状态 |
| [旧working_copy](/home/asus/work/Jiche_Project/高台训练/working_copy/mjlab_lego2) | 历史副本，不是当前维护入口 |

远程训练目录为 `ajifang@27.128.203.222:10395` 的 `~/zrq/mjlab_lego2`。
screen名固定为 `zrq`。远程位置和日志元数据来自既有记录；本次未连接服务器重新核验。
不要在已不存在的 `/home/asus/work/机车学院项目` 下重建一份训练源码。

## 2. 六个长期上下文文件

| 文件 | 负责的信息 | 何时维护 |
| --- | --- | --- |
| [AGENTS.md](../../AGENTS.md) | 开发和文档维护规则，入口导航 | 工作约定变化时，保持简洁 |
| [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) | 总体目标、框架/机器人、目录、关键源码与流程 | 长期目标、路径或架构变化时 |
| [CURRENT_STATUS.md](CURRENT_STATUS.md) | 已实现/已验证/远程同步状态、当前阶段和未解决项 | 每个重要开发或验证阶段结束后 |
| [DECISIONS.md](DECISIONS.md) | 技术选择、依据、旧决策的适用范围与替代关系 | 关键选择或证据变化时 |
| [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) | 实际训练参数、checkpoint、结果和失败反馈 | 实验结束/中止或用户补充实际结果后 |
| [HANDOFF.md](HANDOFF.md) | 新聊天启动顺序、下一步、操作边界 | 停止/切换工作阶段和聊天前 |

建议读取顺序：AGENTS → HANDOFF → CURRENT_STATUS → 正式计划 → 接口核查，
再按需要阅读PROJECT_CONTEXT、DECISIONS、EXPERIMENT_LOG及下列研究证据。
上述六个文件是当前上下文来源；旧的单一“平台攀爬训练任务记录.md”已退役，不再持续更新。

## 3. 当前正式计划与本地核查

| 资料 | 内容与使用边界 |
| --- | --- |
| [正式修改计划v2](REFACTOR_PLAN.md) | 阶段0诊断→1盲走历史→2坑/出生/指令/课程→3通用奖励/评估/首训→4条件性接触反应→5完整训练/泛化；包含文件、起始参数和验收条件 |
| [接口就绪核查](IMPLEMENTATION_READINESS_20261030.md) | 实际历史/reset、动作缩放、课程生命周期、坑原点和CPU检查结果；不能当新任务已实现 |

当前停在**计划完成、阶段0待实施**。新任务 `Mjlab-Blind-Pit-Lego2` 尚未注册，
144维actor、5帧历史、3～30cm/10级和起始权重都属于计划，不能直接给新任务训练命令。
阶段1、2逐批验收；阶段3的配置/功能检查通过后在该阶段开始首轮短训，不等阶段4/5。
首轮训练必须已经接通高度课程。

## 4. 开源研究与来源

| 资料 | 位置与用途 |
| --- | --- |
| [交叉对比报告](/home/asus/work/Jiche_Project/高台训练/2026-10-29_盲走轮足高台开源对比报告.md) | 当前方案依据；地形、课程、实际生效奖励、盲走观测和旧奖励问题的对照，已补2026-10-30用户澄清 |
| [源码来源清单](/home/asus/work/Jiche_Project/OpenSource/SOURCES_20261029.md) | 仓库URL、提交、浅克隆/稀疏范围及未下载资料；不能把所有参考项目称为80cm高台复现 |
| [用户原有Dreamwaq分析](/home/asus/work/Jiche_Project/Dreamwaq高台训练代码分析.md) | 先前分析保留，具体判断以实际源码与后续交叉核查为准 |
| [Dreamwaq-highplatform](/home/asus/work/Jiche_Project/OpenSource/Dreamwaq-highplatform) | 用户提供的主要坑式参考，无.git，精确上游提交待确认 |
| [LocoWheeledLegged](/home/asus/work/Jiche_Project/OpenSource/LocoWheeledLegged) | 历史盲走与轮腿分离正则参考 |
| [DeepRobotics_rl_training](/home/asus/work/Jiche_Project/OpenSource/DeepRobotics_rl_training) | 官方M20通用课程/奖励与条件正则参考 |
| [robot_lab](/home/asus/work/Jiche_Project/OpenSource/robot_lab) | 通用轮足速度跟踪对照 |

这些资料无需重新下载或安装。Dreamwaq的历史估计器和特定命令组织不能被简化成
“只靠几个通用奖励”，参考机器人演示高度也不能当作Lego2已验证能力。

## 5. 资产审计、工具和历史反馈

| 资料 | 内容与当前状态 |
| --- | --- |
| [惯量审计](INERTIA_AUDIT_20261027.md) | 2026-10-27圆柱轴向/轮子近似质心的修复依据、假设与数值；用户随后确认资产纠偏和远程同步完成，旧“待同步”段落只反映当时状态 |
| [惯量CPU校验工具](../../scripts/tools/check_lego2_inertia.py) | 检查两格式的圆柱近似自洽性，不证明实物惯量或攀爬能力 |
| [惯量前后对比图](assets/lego2_wheel_inertia_before_after.png) | 几何与等效惯量椭球示意 |
| [全量训练回放视频](</home/asus/视频/录屏/录屏 2026年10月28日 21时58分42秒.webm>) | 用户确认来自checkpoint编号9999，多次Reset均失败，前轮不抬/后腿反抬；完整run路径、次数、资产对应见实验记录中的待确认项 |

需保留的纠正：原始actor没有height_scan，用户后来注释删除语句才保留扫描。
前轮奖励使用轮心高度、静态位置×接触的奖励可在不前移时得分，不能由均值上升认定正确抬腿。
先诊断实际足端/动作映射和响应；目前没有证据证明前后腿接反。

## 6. 备份与早期资料

| 资料 | 用途 |
| --- | --- |
| [修复前资产备份](backups/20261027_lego2_inertia_before_v1)及[manifest](backups/20261027_lego2_inertia_before_v1/manifest.json) | 原URDF/MJCF及SHA-256；按保存的源码相对路径恢复，不自动覆盖当前资产 |
| [开源研究前上下文](backups/20261029_context_before_open_source_research) | 保留研究前状态 |
| [大体规划前上下文](backups/20261029_context_before_refactor_plan) | 保留旧路线与阶段记录 |
| [用户澄清前文档](backups/20261030_user_clarifications) | 保留height_scan/checkpoint/重复失败澄清前内容 |
| [正式计划v2前文档](backups/20261030_formal_plan_v2) | 保留正式计划细化前内容 |
| [本次交接核查前文档](backups/20261030_handoff_link_audit) | 本轮修订前原文和变更manifest |
| [旧指引手册](/home/asus/work/Jiche_Project/指引手册.txt)、[服务器指南](/home/asus/work/Jiche_Project/服务器使用指南.txt) | 历史部署/训练背景，不替代当前计划 |
| [早期GPT交接](/home/asus/work/Jiche_Project/高台训练/gpt网页版交接.txt)、[早期其他交接](/home/asus/work/Jiche_Project/高台训练/谷歌交接.txt) | 历史过程，包含已纠正的upward、地形等解释；不能直接当当前实施指令 |

远程源码备份记录位置是
`~/zrq/backups/mjlab_lego2_pre_platform_climb_20261015_215244/`，
其中有 `source.tar.gz` 和 `remote_state.txt`；未在本轮访问远程核验。
不把不存在/已迁移的 `task.txt`、`task_2.txt` 当成必须重新索取的前置资料。

## 7. 更新和证据规则

最新用户说明优先。当前实现以实际源码/验证为准；规划参数不能反推为已实现。
CURRENT_STATUS写当前，DECISIONS保留依据和替代关系，EXPERIMENT_LOG保留原始数据并在
旧解释旁标明更正。研究报告、早期审计和旧操作示例均按各自日期理解。
用户在新聊天明确要求开始实施时，可以按阶段0继续，不必把本轮“只审文档”当作永久禁止开发。

新增/移动资料应更新本页与HANDOFF的链接，并检查文件存在；历史快照不跟随本轮改写。
本轮校验记录见 [CONTEXT_LINK_CHECK_20261030.json](CONTEXT_LINK_CHECK_20261030.json)：
仅验证本地链接/目录、关键备份和业务文件未变，未验证网络URL、远程文件或训练成功率。
