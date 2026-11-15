# 当前状态

更新时间：2026-11-15（Asia/Shanghai）。唯一跨会话入口：[HANDOFF.md](HANDOFF.md)。

## 当前阶段

**model2197原始配对证据已下载并离线核验：三速度9/10、10/10、8/10成立；三个未通过样本均由轮接触记录间歇使连续支撑不足0.5 s，暂不追加训练。**
用户当前要求先看Viser实际效果并反馈，再进行下一步；暂不开展子步回放开发或后续训练。
原model_1198的3/6/9/12/15/20 cm初筛各10/10，30 cm为8/10；尚非正式大样本/直线高台验收。
阶段0～2及阶段3原核验、首次短训/低阶续训已有用户回传，原始实验路径及参数归
[EXPERIMENT_LOG](EXPERIMENT_LOG.md)。阶段4未触发，不引入平地预训练或接触引导机制。

## 最新配对结果与解释

同30 cm/三速度/10组初态，原始报告中的组内/跨模型初态已本地复核，最大差0（排除actor的vx输入）。
model2197稳定通过9/10、10/10、8/10，父model1198为4/10、10/10、10/10。
三速度速度RMSE均下降至0.02533/0.02921/0.03117，但稳定通过未全面改善。
新模型各速度完整出坑均10/10，停滞/翻倒/failed均0。
0.30 m/s退步seed11，新增通过seed7/9/10/12/13/14；0.50 m/s退步seed11/13；0.40无变动。
详情归[EXPERIMENT_LOG](EXPERIMENT_LOG.md)最新条目，
[用户摘要](diagnostics/20261115_phase3_model2197_user_eval_summary.json)不替代原始report。
原始报告、逐步trace、comparison、两份SHA文本及训练params已下载；模型权重未下载。
model2197 SHA为`b8bc52a8ee370eb94e359bc71d310f5f1cd0ec013cae9e5982b44fb13b8a54a4`，
三份新报告与SHA文本一致；不冒充对权重文件本身重新计算hash。

[离线分析](diagnostics/20261115_model2197_comparison_analysis.json)逐帧重建60回合/33718帧，
阶段、连续计时、锁存通过和RMSE均匹配，轮底FK最大误差1.399e-7 m；六报告记录的源码hash均匹配本地。
三个未通过样本的首次complete之后均有足够行程时间（约3.9～6.4 s），倾角≤3.25°、轮底距台面≤1.61 mm；
全部中断帧均可由至少一轮found=0/force=0解释，最长连续支撑分别0.48/0.34/0.28 s。
两模型均存在RF/LF前大腿box接触台面（力范数>1 N，重建box最低点近台面）；
该传感器force是接触坐标系，不能把其第3分量称为世界竖直力。
大腿接触不是新模型独有，也未证明是轮接触间歇的原因；20 ms采样不能区分真实卸载与传感/求解现象。
先让用户查看30 cm交互回放并反馈；后续再判断是否需要两模型小规模视频/5 ms子步接触核对。
保持原判据，不先改奖励。
两模型均保留，各10组初态不是正式大样本/多训练seed/直线高台泛化验收。

## 已完成的配对工具与核验

- [固定高度工厂](../../src/mjlab/tasks/velocity/config/lego2/blind_pit_env_cfg.py)：
  可选forward_speed只固定评估vx范围，合法范围0.3～0.6 m/s；保留heading反馈。
  未传参数时保留原评估随机速度，训练工厂不变；新增src关键代码有中文注释。
- [评估工具](../../scripts/tools/evaluate_lego2_blind_pit.py)：
  可选逐episode seed=base+episode，记录实际reset状态/actor首帧，避免只凭seed假定已配对。
  补记接触点、连续支撑时长、相对root位置和非轮接触；原n统计/截断/稳定判据不变。
- [配对汇总](../../scripts/tools/summarize_lego2_speed_eval.py)：
  核对三组出生/历史一致，只排除vx输入，wz仍比较；容差1e-6，失败时不给出有效配对结论。
- [新回归](../../tests/test_lego2_blind_pit_speed_eval.py)：
  三组用不同截断步数测试下一回合配对；调试片段n=0，且故意改动初态可被汇总器拒绝。

actor144/critic254、12维动作、奖励权重、训练速度分布、课程、机器人资产及PPO均保持。
评估工具不更新PPO；已有零动作/随机权重软件核验不当作真实模型策略能力。

## 本地核验

- 本次只读配置核对：普通注册play为3～30 cm十级、初始level0且课程关闭，出生在3 cm，非固定30 cm。
  已验证临时进程内注册30 cm/0.50 m/s/单环境/自动reset配置，复用标准run_play及model2197；
  不改源码或原注册任务。Viser本地依赖默认监听8080，实际服务器监听端口以启动日志为准。
  配置/调用连接已离线核对，未在本机启动GPU回放；实际观看和反馈待用户执行。
- 新增[离线分析脚本](artifacts/analyze_phase3_paired_comparison_20261115.py)：
  原归档只读，不解压覆盖源码；60回合一致性断言通过，定向Ruff/格式/ty/Pyright通过。
  [连续支撑对照图](diagnostics/20261115_model2197_comparison_analysis.png)已渲染检查，无仿真/PPO或远程操作。
- [阶段1～3及本批CPU测试](diagnostics/20261115_phase3_speed_tests.txt)：19 passed，12.36 s；
  4个既有ONNX弃用警告。实际配对跨不同回合步数成立，输入维数及checkpoint推理/reset回归通过。
- 4个新增/修改Python文件定向Ruff/格式/ty/Pyright通过；未声称全库旧类型错误已修复。
- 真实CLI固定速度/配对seed两步CPU诊断及汇总CLI核对；截断片段不计入n。
- 本机GPU驱动不可用，验证前可用内存约22 GiB，只做小规模CPU运行。
- [SSH安装测试](diagnostics/20261115_phase3_speed_sync_validation.json)：
  旧文件hash通过才安装，先归档旧文件；未知/已更新源码会停止；旧模型/环境/额外源码保持。
- [改前源码归档](backups/20261115_phase3_speed_eval_before.tar.gz)及
  [改前hash](backups/20261115_phase3_speed_eval_before.json)保留；旧报告/模型/归档未改写。

## 训练/复评状态与下一步

第二轮续训退出码0、34:02、model2197.pt及params均有用户回传，详情归EXPERIMENT_LOG。
末尾2197符合恢复1198后新增1000更新；24576000仍为计划采样推算，完整Total timesteps未下载。
训练pit_complete=0.3408是阶段时间占比，不与固定评估的episode稳定率混用。
model2197的GPU组内/跨模型配对原始证据已核对，本地模拟报告测试本身仍只证明工具逻辑。

已准备[本地SSH读取入口](artifacts/collect_phase3_comparison_20261115.sh)，
从两个实际评估目录只读tar取回full报告/逐步轨迹、checkpoint.sha256及新模型comparison/训练params。
实际归档为`diagnostics/20261115_model2197_comparison_remote/remote_evidence_kO71wR.tar.gz`，
47个成员，归档SHA256为`501d3b382997b001c98eb8e11a3a24d0670392c497697526852848eb5d4e8aaf`。
不传模型权重或更改服务器源码/环境/结果；原归档保持不变。
[本地核验](diagnostics/20261115_phase3_collection_validation.json)包含两次独立下载与中断保留，
采用模拟SSH，仅证明指令连线/本地归档保护；本次实际下载证据另见离线分析JSON。

1. 用户通过SSH隧道/Viser观看model2197的30 cm中央坑出坑效果（提供临时进程内固定配置指令）。
2. 收到用户反馈后，再决定是否实现两模型配对视频/5 ms子步接触核对，不自动放宽原0.5 s判据。
3. 暂不替换父模型或追加第三轮训练；正式大样本/直线高台/平地能力仍待后续。

## Git、同步与实验边界

**GitHub仅供用户自行版本管理；是否commit/push由用户决定。服务器固定采用SCP/SSH直传，
GitHub连通、CI、提交号或push均不作为机器人开发/同步/验证前置条件。**
规范归AGENTS链接目标及PROJECT_CONTEXT，DECISIONS已移除冲突的旧操作规则，只留失效历史说明。
用户回传GitHub已回退initial；本地HEAD与origin/main跟踪引用也为3592c9e，未另行联网核实。
后续小提交的文件改动仍保留在工作区，本批未commit/push；不要求每批生成Git版本阶段。
恢复资料仍在`artifacts/git_history_before_20261115_jlp8m7cp/`，禁止自动清理。
服务器继续使用`~/zrq/mjlab_lego2`及其现有环境；Git候选目录/失败现场保留，不再接入或切换环境。
训练完成来自用户回传；两模型配对评估原始明细已下载核验。完整训练stdout/Total timesteps仍未下载。
归档检查：历史`lego2_phase3_20261031_v1.tar.gz`及`lego2_phase3_reset_fix_20261101_v1.tar.gz`
含旧HANDOFF/CURRENT_STATUS，仅作历史保留，不重放覆盖当前交接；新speed_eval包无上下文快照。
