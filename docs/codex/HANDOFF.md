# Lego2 项目交接

更新时间：2026-11-15（Asia/Shanghai）。唯一交接入口，先读[AGENTS](../../AGENTS.md)。

## 当前目标

本体盲走轮足越障，中央坑高度课程，目标参考300 mm。
阶段3继续：model2197配对复评9/10、10/10、8/10已核验，轮接触间歇使三个样本连续支撑不足；不立即续训。
当前用户先看Viser实际效果并反馈，再推进下一步；子步诊断入口开发暂不开展。

## 当前进展

- 阶段0～2及原阶段3远程CPU/GPU已有回传；首次短训及两次1024环境新增1000更新续训已完成。
- model_1198在3/6/9/12/15/20 cm各10/10，30 cm为8/10；原报告与30 cm轨迹已下载核查。
- 两模型原始60回合/33718帧已核验：model2197三速度9/10、10/10、8/10，父模型4/10、10/10、10/10。
- 组内/跨模型实际初态最大差0（排除vx输入），六报告源码hash匹配本地。
- actor144/critic254、12维动作、11项通用奖励、课程与机器人资产保持；阶段4未触发。
- 每高度10次仍是初筛，正式大样本/独立直线高台/平地保留能力尚未验收。

## 最近一次开发

新模型全部30次曾完整出坑且无停滞/翻倒/failed，速度RMSE均下降，但0.50 m/s稳定通过从10降至8。
0.30 m/s唯一未稳定通过为seed11，0.50 m/s退步seed11/13；原始数值/路径归EXPERIMENT_LOG。
原始report/trace/comparison、SHA文本及训练params已下载；权重文件未下载。
新增离线FK分析，逐帧阶段/保持计时/锁存/RMSE均匹配，FK误差≤1.399e-7 m；定向四项检查通过。
三个未通过样本最长连续支撑0.48/0.34/0.28 s；出坑后倾角≤3.25°、轮底距台面≤1.61 mm。
中断均对应至少一轮found=0/force=0；20 ms采样不能判定真实卸载或接触求解/记录问题。
旧/新模型均有前大腿box近台面且非轮接触力范数>1 N；非轮force在接触坐标系，因果仍待核对。
未改src、奖励/判据，未启动第三轮训练或覆盖旧模型；两模型均保留。
普通注册play初始level0是3 cm，非固定30 cm；本次给出进程内临时注册30 cm/0.50 m/s回放指令。
复用标准run_play、model2197、单环境及自动reset；本地配置/调用核对通过，实际远程观看待用户执行。

## 当前问题

- 新模型低速改善伴随0.50 m/s支撑连续性退步；先等用户观看反馈，保持判据。
- 正式泛化/大样本验收未完成；旧全库类型错误未在本批处理，不把定向检查当全库通过。
- 模型SHA已核对报告/文本一致，但没有权重文件独立rehash；完整训练stdout仍未下载。
- 原训练完整stdout/Total timesteps/参数细节及旧失败根因、EPA影响、CAD属性仍有待确认项。

## 下一步任务

1. 用户SSH隧道访问Viser，查看model2197的固定30 cm中央坑回放（非独立直线高台），回传观感。
2. 按反馈决定是否实现配对视频/5 ms子步接触核对；不预先认定真实跳跃或传感器bug。
3. 再据原因决定后续训练，保留两模型和原通过判据，暂不追加第三轮训练。
4. 正式大样本/直线高台/平地验证仍待后续。

## Git 与实验状态

- **GitHub仅为用户自行版本管理工具，是否commit/push由用户决定；服务器固定SSH直传。**
  不等待GitHub连通、CI、提交SHA或push；不要求每个小批次形成Git版本阶段，Codex不自动commit/push。
- 用户回传GitHub已回退initial，本地HEAD及origin/main跟踪引用也为3592c9e；本批为未提交改动。
- 旧GitHub中转/接入/下载方案全部停用，冲突指令已从当前规范清理，历史文件与失败现场保留。
- Git恢复目录为`artifacts/git_history_before_20261115_jlp8m7cp/`；禁止自动清理。
- 服务器继续`~/zrq/mjlab_lego2`及既有.venv/logs，不切换Git候选目录，不删除任何训练结果。
- 实际run/checkpoint与实验归EXPERIMENT_LOG；第二轮续训已回传，两模型复评原始证据已离线核验。
- 远程脚本只比较文件/模型内容hash，不硬编码Git commit；新speed_eval包无上下文快照。
- 历史phase3原包/reset_fix包含旧HANDOFF/CURRENT_STATUS，仅保留历史，不重放覆盖当前交接。

## 需要进一步查阅的文档

- [CURRENT_STATUS](CURRENT_STATUS.md)：本批代码、核验、SSH交付与限制。
- [PROJECT_CONTEXT](PROJECT_CONTEXT.md)：长期架构、SSH/screen与用户自主版本管理约定。
- [DECISIONS](DECISIONS.md)：有效技术决策及固定速度配对定义，旧Git方案已失效。
- [EXPERIMENT_LOG](EXPERIMENT_LOG.md)：真实训练与评估，原始模型SHA和证据路径。
- [REFACTOR_PLAN](REFACTOR_PLAN.md)：正式阶段设计与验收。
- [30 cm离线分析](diagnostics/20261114_phase3_upper1198_analysis.json)及
  [本批CPU输出](diagnostics/20261115_phase3_speed_tests.txt)。
- [两模型逐帧分析](diagnostics/20261115_model2197_comparison_analysis.json)及
  [连续支撑图](diagnostics/20261115_model2197_comparison_analysis.png)。
