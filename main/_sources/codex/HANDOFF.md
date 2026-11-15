# Lego2 项目交接

更新时间：2026-11-15（Asia/Shanghai）。唯一交接入口，先读[AGENTS](../../AGENTS.md)。

## 当前目标

本体盲走轮足越障，目标参考300 mm；中央坑、首训即启用低高度课程。
低高度3/6/9 cm各10/10；新回传12/15/20 cm各10/10、30 cm为8/10，原始轨迹已核查，正式大样本/直线高台验收待完成。

## 当前进展

- 阶段0～1远程CPU、阶段2远程CPU 28项/GPU 3项通过（用户回传）。
- 阶段3输入/同步hash、CPU 33项/GPU 4项及两步固定高度诊断通过（用户回传）。
- 阶段2保持十级3～30 cm坑/坑底出生、20 s固定vx/heading、固定摩擦、支撑过滤课程。
- 阶段3显式11项通用奖励，无高度/腿姿态/贴地/滑动成本或坑坐标奖金。
- 新增前后轮支撑/完整出坑指标；四轮有效坑沿支撑且倾角<45°连续0.5 s记录稳定通过。
- 固定高度工具保留训练输入/动作/终止，精确统计完整episode；输出动作/接触/奖励轨迹及配置/hash。
- actor144/critic254、12维动作、PPO和旧任务/资产保持；已完成首次流程短训及一次低阶续训，阶段4～5未实施。

## 最近一次开发

用户下载高高度report、30 cm trace及checkpoint.sha256，已确认同model_1198.pt/seed7/cuda:0。
14项源码/资产hash全部匹配；未下载模型本体，不声称本地重算checkpoint SHA。
第2个episode（原始id=1）低速0.3195 m/s，20 s未上沿；第3个（id=2）低速0.3005 m/s，
已完整出坑，但接触通道间歇未检出，连续支撑最长0.22 s，最大倾角15.5°，到边界截断。
运动学重建轮底与trace吻合，轮子距两侧壁至少0.633 m，未见轮子坑角借力。
数值/方法/限制归EXPERIMENT_LOG最新条目与分析JSON。本批未改src或训练，只做离线分析。

2026-11-15后续同步采用本地→GitHub→服务器；首次服务器接入待只读核查，流程归PROJECT_CONTEXT。
修正.gitignore以放行docs/codex/artifacts的小脚本/清单，压缩包/logs/.venv仍忽略；
9项规则与diff空白核验通过。仅修改忽略规则和文档，尚未commit/push或操作服务器。

## 当前问题

- 30 cm未通过的两类表现已定位；低速依赖和接触中断原因仍待固定速度/回放对照，正式验收未完成。
- 高高度report和30 cm trace已下载；其余高度trace、数字退出码和独立修复核验回执未回传。
- 全项目ty/Pyright仍有旧文件类型错误；本批定向检查通过，不声称全库通过。
- 用户已推送GitHub，截图显示Actions多项失败；本地已读CI配置，具体失败日志待获取，未改工作流。
- 完整训练stdout/Total timesteps/实际params未读取；30 cm结果只支持当前坑式初筛，不替代正式验收。
- 旧失败根因/旧参数资产对应、EPA警告影响、CAD质量属性仍待确认。
- 旧报告/索引只作历史支撑；最新同步和实验状态以本页及CURRENT_STATUS为准。

## 下一步任务

1. 先只读核对服务器现有源码目录的Git根目录/分支/HEAD/改动与来源，准备首次GitHub接入，保留所有结果。
2. 准备30 cm固定vx=0.30/0.40/0.50 m/s、相同初态和seed的小批对照及回放，入口尚未实现。
3. 核对低速受阻/台面接触间歇是否可重复；必要时补接触点/非轮接触，保持现有0.5 s判据。
4. 再决定训练预算及正式每高度至少100次/独立直线高台/平地验证。
   当前不改奖励、不引入预训练或阶段4，不直接扩训10000轮。

## Git 与实验状态

- 本地已有Git，main/3592c9e（Initial commit）；当前有忽略规则/文档未提交更改，Codex不自动commit/push。
  GitHub代码已显示，CI失败根因待日志确认；Codex未自行访问训练服务器。
- 阶段2远程证据：
  `/home/ajifang/zrq/backups/phase2_20261031_211214_2150255/check_20261031_211214_2150255`。
- 阶段3远程备份：`/home/ajifang/zrq/backups/phase3_20261031_214701_2152912`，
  检查目录为其下`check_20261031_214701_2152912`。
- 新短训证据：`/home/ajifang/zrq/backups/phase3_smoke_20261031_215751_2154252`；
  实际run/model_199.pt及成功完成的评估目录见EXPERIMENT_LOG；早先失败目录仍待确认。
- 低阶续训证据：`/home/ajifang/zrq/backups/phase3_low_20261101_201712_2200851`；
  新run/model_1198.pt及低/高高度复评路径见EXPERIMENT_LOG；原始报告及30 cm轨迹已核查。
- SSH/screen约定归PROJECT_CONTEXT，旧run路径归EXPERIMENT_LOG；训练结果全部保留。
- AGENTS符号链接及中文注释规范保留本地；不随本批覆盖远程CLAUDE。

## 需要进一步查阅的文档

- [CURRENT_STATUS](CURRENT_STATUS.md)：本批范围、检查限制和证据路径。
- [PROJECT_CONTEXT](PROJECT_CONTEXT.md)：长期架构、输入及操作约定。
- [DECISIONS](DECISIONS.md)：阶段3奖励权重、尺度、稳定通过/评估定义。
- [EXPERIMENT_LOG](EXPERIMENT_LOG.md)：真实训练及checkpoint证据。
- [REFACTOR_PLAN](REFACTOR_PLAN.md)：阶段3短训与后续阶段设计/验收。
- [修复验证JSON](diagnostics/20261101_phase3_reset_validation.json)及
  [改前源码](backups/20261101_phase3_inference_reset_before.tar.gz)。

- [离线分析JSON](diagnostics/20261114_phase3_upper1198_analysis.json)及其同名PNG：数值、哈希和运动学核查。
