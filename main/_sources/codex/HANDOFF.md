# Lego2 项目交接

更新时间：2026-11-17（Asia/Shanghai）。唯一交接入口，先读[AGENTS](../../AGENTS.md)。

## 当前目标

本体盲走轮足越障，中央坑高度课程，参考300 mm。
用户新录屏已查看并做离线姿态核查；当前评估先修改还是长训，建议先小幅补平地稳定姿态目标，尚待用户选择。
用户最关注后腿借关节撑台与出坑后低机身；v1由用户commit/push保留，v2独立任务/日志。

## 当前进展

- 阶段0～2和原阶段3软件核验/训练已有回传；v1两模型原始60回合/33718帧已离线核对。
- model1198原30 cm三速度4/10、10/10、10/10；model2197为9/10、10/10、8/10，速度误差降低但未全面改善。
- v2独立13项：俯仰±25°死区/侧倾保留，非轮接触-0.5/通道，平整地面最低root净空0.34 m、权重-2。
- v2保持actor144/critic254、动作12及资产/课程/PPO配置；阶段4未触发，正式泛化/大样本未验收。
- 初始v2仅继承父2197 actor/归一化/分布，critic/优化器/计数新建；512环境×300更新后为model299。
- 用户要求充分训练再比较，随后从299完整恢复训练状态，1024环境新增1900次，实际2198路径已回传。
- stdout核对300+1900次、3686400+46694400=50380800样本；初始actor继承v1，非同起点奖励消融。

## 最近一次开发

本批查看20.22 s新录屏并抽帧，起步后腿抬落/左右前腿伸展差异/出坑后前后构型不同可见。
录像20 s处约222步（约4.44 s仿真），画面无模型SHA或腿名；不直接证明每个关节的量值/因果。
归档9父/9候选作FK；候选高速六回合首次complete+1 s后，RF相对自身髋部的前向偏移比LF多约7.9 cm，
持续3.4～3.6 s。所取1059帧无非轮接触/饱和，实际硬限余量≥14.16°；未推进物理时间。
左右差异是持续构型偏置，但未发现上述窗口有必须立即中断的证据；前后原始关节角不能强制相等。
建议先一项弱姿态回归，仅局部平地/直行/稳定时启用，过沿和转向保留自由；起步抬腿暂不罚。
此为建议而非已采纳技术决策，未改奖励/src/PPO或开训；若用户优先v2能力上限，可选+1000受控对照。
详见EXPERIMENT_LOG最新条目及新姿态核查JSON；原2198保持参照，后续先有限微调/评估再长训。

最新用户证据目录：`/home/ajifang/zrq/backups/reward_v2_continue_20261116_scKcS4`。
模型：`logs/rsl_rl/lego2_blind_pit_reward_v2/2026-11-16_22-25-03_v2_continue_1024x1900_20261116_222451_3365307/model_2198.pt`。
新SHA：`27c1bdf3a05fa3455a16932dd4d5493fef526680575caee01e31a5c0d169248c`（仅回传，未下载权重）。
30 cm、0.3 m/s各3回合：父2197通过2/3，v2为3/3；完整出坑均3/3、failed均0。
低净空/有效帧70.7173%→0.7126%，有效覆盖58.2882%→60.2289%；
首次complete后非轮接触帧占比39.2894%→0，最低root-台面高度0.215999→0.338516 m。
速度RMSE为0.025294→0.029422 m/s，略增；不能称速度也改善。
出坑后指标不能证明后腿爬台全程无刮擦，root高度不能保证关节绝对安全。

两轮归档已到diagnostics/20261117_reward_v2_continue_remote/remote_evidence_tCcuUA.tar.gz。
SHA为775e111e57cf2e9296578f46eadcd058868059b52bccf9c709065bd80b248f4f，98成员/85文件，不含模型。
四组比较报告hash/来源清单/实际reset/汇总重算通过，尚未逐帧动力学重建。
本轮v2三条件均3/3（共9/9），父为3/3、1/3、2/3；顺序为3 cm/.5、30 cm/.5、30 cm/.3。
0.5 m/s两组v2低净空约72%，出坑后最低root-台面约0.325 m；非轮接触均0，不能只推广低速结果。
同父跨轮初态差0、来源hash同，但高台两速度通过仍1↔2，原因待轨迹核查。
已核对临时注册固定30 cm/0.50 m/s、1环境、自动reset、Viser默认8080入口，给用户进程内指令。
仅配置/API验证，未本地启动仿真或网页；不写远程脚本，不改src/奖励/PPO，不更新参数。
27文件注释仍仅本地，执行AST与旧源码一致，但文本hash不同；旧清单不改，不混入远程来源证据。
长期中文注释要求统一见[CLAUDE规范](../../CLAUDE.md#中文代码注释规范)。

## 当前问题

- 高速0.34 m以下净空和左右约7.9 cm偏置持续；无当前姿态偏好奖励，不能保证长训自然去除。
- 同父高台复评波动存在，跨轮实际初态差0且来源hash一致；接触复现根因仍待核查。
- 每条件3组为初筛；本轮接触/净空方向改善，尚非正式验收、单项奖励因果或整体所有条件更优。
- 首次trial的model299不出坑，低净空0和post-complete null不能当改善；历史结果保留，不再当最终退化结论。
- v2首个续训尝试zNd3XT在自备份失败，未训练；已修正cd前realpath。新scKcS4为用户已回传完成的实际运行。
- 旧顶层上传物整理是否完成未回传；不要因续训完成而假定整理已完成。
- 全库旧类型/lint问题未修复；本地注释批Ruff13/ty5/Pyright5问题未增减，不声称全库通过。

## 下一步任务

1. 等用户选择先补平地稳定姿态目标或原v2受控续训；本次只核查/建议，不视为开训授权。
2. 若选新候选，核对适用条件/量级/加载边界，独立run有限验证，再决定长训；不全程强制对称。
3. 配对筛查先扩样本约10/条件，继续关注接触/净空/限位/力矩与姿态；正式验收仍待后续。
4. 保留父2197、v2模型299/2198及旧结果；正式独立高台/平地/更多命令验收待后续。

## Git 与实验状态

- GitHub只供用户可选版本管理，是否commit/push由用户决定；服务器使用SSH/SCP直传。
- 远程执行.sh及配套清单/包统一到`~/zrq/mjlab_lego2/docs/codex/artifacts/`，同名先备份，不再散放~/zrq。
- 旧上传物先快照、按已知清单移入artifacts/_legacy_uploads/独立目录；冲突/相同副本都保留，不覆盖当前入口。
- 历史Git接入方案停用；恢复资料`artifacts/git_history_before_20261115_jlp8m7cp/`禁止自动清理。
- 源码继续~/zrq/mjlab_lego2，不删除logs/.venv/checkpoint；原始实验由EXPERIMENT_LOG维护。
- 历史phase3/reset_fix包含旧上下文，不重放覆盖当前交接；trial_v2冻结包无上下文，保留原50来源hash。
- 本地注释文本不同于远程；本批无源码同步或commit/push，Git工作区已有改动均保留。

## 需要进一步查阅的文档

- [CURRENT_STATUS](CURRENT_STATUS.md)：详细当前状态、工具核验和限制。
- [PROJECT_CONTEXT](PROJECT_CONTEXT.md)：架构、SSH/screen、统一上传目录与自主版本管理。
- [DECISIONS](DECISIONS.md)：v2奖励/加载边界/延长训练和目录决策。
- [EXPERIMENT_LOG](EXPERIMENT_LOG.md)：实际run、SHA、回传及历史证据。
- [REFACTOR_PLAN](REFACTOR_PLAN.md)：正式阶段及验收要求。
- [最新回执](diagnostics/20261117_reward_v2_continue_remote_receipt.json)与
  [新收集工具核验](diagnostics/20261117_v2_continue_collection_validation.json)、
  [已下载报告核对](diagnostics/20261117_v2_downloaded_reports_analysis.json)及
  [回放配置核验](diagnostics/20261117_v2_play_entry_validation.json)、
  [新视频/姿态核查](diagnostics/20261117_v2_video_posture_review.json)及
  [关键帧](diagnostics/20261117_v2_video_keyframes.jpg)。
- [v1逐帧核验](diagnostics/20261115_model2197_comparison_analysis.json)及
  [v2初始迁移核验](diagnostics/20261116_v2_trial_validation.json)、
  [完整续训加载核验](diagnostics/20261116_v2_continue_validation.json)。
