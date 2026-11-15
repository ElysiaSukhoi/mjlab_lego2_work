# 当前状态

更新时间：2026-11-15（Asia/Shanghai）。唯一跨会话入口：[HANDOFF.md](HANDOFF.md)。

## 当前阶段

**用户回传12/15/20 cm各10/10稳定通过、30 cm为8/10；高高度原始证据已核对，30 cm未通过样本分为未上沿和连续支撑不足两类；下一步固定速度对照。**
阶段0～1远程CPU及阶段2远程CPU/GPU已有用户回传通过。
阶段2验证目录：`/home/ajifang/zrq/backups/phase2_20261031_211214_2150255/check_20261031_211214_2150255`。
回传为CPU 28 passed（88.93 s），RTX 4090 GPU 3 passed（225.49 s）；没有自行连接远程。
阶段4未触发；已有30 cm坑式初筛成功样本，阶段5完整训练/正式大样本及直线高台泛化验收未实施。

此前用户回传[完整阶段3输出](diagnostics/20261031_phase3_remote_receipt.txt)：
输入/同步hash及任务注册成功，CPU 33 passed（85.19 s），GPU 4 passed、7 deselected（57.20 s），
RTX 4090空闲23806 MiB；两步固定高度诊断n=0检查成功。两个ONNX弃用提示及已知MULTICCD提示未阻断检查。
远程备份：`/home/ajifang/zrq/backups/phase3_20261031_214701_2152912`；
验证目录为其下`check_20261031_214701_2152912`。本轮未自行访问远程。

## 最新复评结果与解释

用户已下载[原始report](diagnostics/20261114_phase3_upper1198_remote/report.json)、30 cm逐步trace和checkpoint.sha256。
已核对model_1198.pt路径/模型哈希与此前记录一致，seed7、cuda:0；14项记录的源码/资产hash均匹配本地。
未下载checkpoint本体，模型校验是两份远程校验记录的一致性核对，不冒充本地重算模型SHA。
12/15/20 cm各10/10、30 cm为8/10；原始样本记录及逐步trace相互吻合。
低高度3/6/9 cm各10/10仍为先前独立批次结果，不混算正式单高度成功率。

[分析JSON](diagnostics/20261114_phase3_upper1198_analysis.json)和[图](diagnostics/20261114_phase3_upper1198_analysis.png)保存本地核查。
按人读1起编号（原始episode分别1、2）：
- 第2个episode：vx=0.3195 m/s，20 s超时；前后轮上沿均未达到，root最终仅前进至坑中心+x约0.99 m。
  前轮RF/LF轮底相对坑底最高约0.075/0.235 m，均未越过0.30 m坑沿；非翻倒退出。
- 第3个episode：vx=0.3005 m/s，11.42 s首次四轮完整出坑，17.16 s到地块边界截断；
  连续有效支撑最长0.22 s，小于0.5 s，最大倾角15.50°未触及45°门槛。
  四轮均几何出坑且轮底接近台面共288步，其中95步至少一轮无有效支撑；
  对应接触通道短时found=0/force=0，不能单凭此判为真实腾空、传感器问题或求解器问题。

用匹配XML及记录的root/joint姿态做纯运动学重建，轮底与trace最大差1.45e-7 m。
全部10次轮子Y包络距侧壁最小0.633 m，未见轮子坑角借力；不替代独立直线高台泛化测试，
trace也缺少非轮接触/接触点坐标，不能声称全面排除所有机身借力。
记录采样点力矩饱和为0；两未通过样本膝目标越硬限帧数198/263，成功样本也有34～66帧，
属于待跟踪相关现象，尚不能归因为力矩上限或目标越限。

两未通过样本恰为本批最低vx；成功样本vx为0.3545～0.5362 m/s，速度与初始扰动混杂。
当前建议固定30 cm、vx=0.30/0.40/0.50 m/s配对初态和seed的小批对照，并做回放，
入口尚未实现/执行；保持奖励、0.5 s判据及原任务不变，不直接加训练或阶段4机制。
正式每高度至少100次、多seed条件测试、独立直线高台和平地能力仍未完成。

## model_199.pt评估基线与解释

用户已回传最终summary及“评估结束”；实际checkpoint/hash/证据目录及全部数值归
[EXPERIMENT_LOG](EXPERIMENT_LOG.md)最新初筛条目维护，
[结构化回传](diagnostics/20261101_phase3_model199_user_eval_summary.json)只转录用户摘要，非原始report。
3/6/9 cm分别稳定通过5/10、1/10、0/10；翻倒/failed均为0，停滞分别3/10、3/10、10/10。
6 cm曾完整四轮出坑7/10，但稳定通过仅1/10；9 cm前轮上沿4/10、后轮上沿0/10。
说明已有初步低阶能力，稳定保持和更高台阶完整通过仍不足；确切受阻原因需要轨迹，不据此立即改奖励。
每高度仅10个样本、单seed，不能当作可靠收敛结论或300 mm能力。

此前静默评估已结束；新版入口包含进度工具安装/hash检查，本次用户已回传复评结束。
高高度report/30 cm trace已下载；其余高度trace、数字退出码及独立reset修复部署/CPU/GPU回执未回传，不推定全部检查通过。

## 本次修复与计划确认

用户确认继续原计划；同144维接口平地预训练仅在有明确瓶颈证据后考虑，当前不迁移旧Flat或开新分支。
用户回传评估reset报错，本地复现为PitProgress.values在inference_mode内被替换为推理tensor，
模式外reset原地清零时异常。此前测试推理后直接关闭环境，遗漏了跨模式reset。
现改为首次分配时关闭推理模式，其后只更新普通缓存的fresh行；关键代码已加中文注释。
奖励/actor144/critic254/12维动作/资产和稳定0.5 s定义均未改变；无需重新训练model_199.pt。
本次只修诊断缓存与回归测试，旧训练hash清单和失败评估目录均保留。

## 本批实现与核心代码

- [blind_pit_env_cfg.py](../../src/mjlab/tasks/velocity/config/lego2/blind_pit_env_cfg.py)：
  显式重建11项通用奖励，权重/函数尺度冻结于[DECISIONS](DECISIONS.md)阶段3条目。
  去掉继承的高度/腿姿态/贴地/滑动及额外能耗成本；不加入坑坐标或登顶奖金。
  保留阶段2课程、20 s指令/终止、actor144/critic254、12维动作及原PPO参数。
- [blind_pit_metrics.py](../../src/mjlab/tasks/velocity/config/lego2/blind_pit_metrics.py)：
  轮子按名称分前后组，复用圆柱包络与有效坑沿支撑过滤。
  root离坑/前轮上沿/后轮上沿/完整出坑记录时间占比；稳定通过按episode锁存。
  四轮有效支撑、倾角<45°、无失败连续0.5 s才通过；每控制步只累计一次，partial reset隔离。
  课程不依赖metrics，诊断不入actor/奖励。关键新增src代码已加中文注释。
- [evaluate_lego2_blind_pit.py](../../scripts/tools/evaluate_lego2_blind_pit.py)：
  固定真实深度两行地形、关闭升降级和auto_reset，保留20 s和原终止。
  每高度精确结算已结束episode；调试max_steps片段不进入n，人工reset不算样本。
  输出seed、配置、源码/资产/checkpoint hash、每高度通过/失败/翻倒/停滞/速度误差，
  逐步记录前后轮净空/接触、动作目标/实际角、目标越限、力矩饱和和奖励原值/加权速率/dt贡献。
  未传checkpoint时明确标为零动作诊断，不当作策略评估结果。

## 本次修复验收

- 改前真实推理后模式外reset复现1项失败；修复后定向7项通过、阶段1～3相关回归35项通过。
  [修复验证JSON](diagnostics/20261101_phase3_reset_validation.json)、
  [改前错误](diagnostics/20261101_phase3_reset_before.txt)、
  [改后测试](diagnostics/20261101_phase3_reset_tests.txt)保存证据；4个既有ONNX弃用警告。
- Ruff/格式、定向Pyright/ty、编译通过；本批未修改全库旧类型错误所在文件。
- 原评估工具CPU完整运行3/6/9 cm各2个零动作episode，共6次均完整结束（每次1000步），
  [报告](diagnostics/20261101_phase3_reset_eval_report.json)确认跨episode及跨高度reset可执行。
  此为软件回归，不是远程model_199.pt的成功率；没有PPO更新。
- 本批GPU驱动仍不可用；远程复验脚本另含GPU推理/重置及两高度重复短episode检查。
- [改前源码归档](backups/20261101_phase3_inference_reset_before.tar.gz)和
  [改前hash](backups/20261101_phase3_inference_reset_before.json)保留回滚依据。

## 原阶段3本地证据与范围

- [验证JSON](diagnostics/20261031_phase3_validation.json)、
  [测试输出](diagnostics/20261031_phase3_tests.txt)：33项通过；两个既有ONNX导出弃用警告。
  涵盖阶段1/2回归、5 cm真实坑深射线、稳定保持/断触/失败帧、同一步重复读取、
  manager部分reset、腿actuator选择、reward dt积分及随机权重保存/加载/推理（无PPO）。
- 新增/修改Python文件Ruff、格式、Pyright、ty和编译通过。
  全项目检查未通过：ty有8错误/2警告，Pyright有9错误，均位于本批未修改的旧文件，
  包括env_cfgs、通用奖励索引类型及旧测试；不能将定向检查写成全库通过。
- 26项既有源码/测试/资产hash与阶段2前置清单一致；旧任务/资产未改。
- [完整零动作诊断](diagnostics/20261031_phase3_eval/full/report.json)：3 cm、1000步、
  n=1、通过0、超时/停滞；[两步诊断](diagnostics/20261031_phase3_eval/capped/report.json)：n=0。
  两者仅检查工具流程，未证明任何自主越障能力；真实支撑测试是人工放到坑沿。
- 本地GPU驱动不可用；修复前阶段3远程GPU核验已有用户回传通过，不替代本次修复复验。
- [改前源码](backups/20261031_phase3_source_before.tar.gz)与
  [改前hash](backups/20261031_phase3_before.json)已保存，无新增上下文Markdown。

## 下一步

1. 准备30 cm固定vx=0.30/0.40/0.50 m/s、相同初态/seed的小批对照和策略回放；
   先核对CLI/配置与模型接口，再给远程指令，当前尚未实施或执行。
2. 检查低速未上沿与台面接触中断是否可重复；需要时补记接触点和非轮接触，
   区分真实卸载与检测影响，保持现行稳定通过判据。
3. 据结果决定续训预算，再推进正式3/5/10/15/20/30 cm每高度至少100次及直线高台/平地验证。

本批新增离线分析脚本、派生JSON/图，并核对原始证据；只运行运动学，无仿真/PPO/远程访问。
分析脚本Ruff/格式/ty/Pyright通过；10个episode的步数/阶段/通过/速度RMSE与原始report一致。
未修改src、规范或历史实验结果；全部旧训练和远程原件保留。

## Git、同步与实验边界

本地main及origin/main跟踪引用现为be4f42a（远程服务器git pull），用户已提交此前接入准备；
首版接入准备前工作区干净，当前有本批脚本/验证记录/文档更改，Codex未commit/push。
用户截图显示GitHub已有代码，Actions为10失败、2成功、1跳过、6取消；推送认证已解决。
已只读核对ci.yml和docker.yml，检查覆盖全库格式/类型/多Python版本测试/打包与Docker。
尚未读取远程失败步骤日志，不能从检查名称推断具体根因；本轮未修改工作流/源码或重跑CI。
Codex不自动commit/push，本轮未自行执行PPO或访问远程服务器。

2026-11-15按用户意向，后续采用本地→GitHub→训练服务器的代码同步流程，
长期流程及首次接入前置检查见PROJECT_CONTEXT，依据归DECISIONS最新条目。
发现docs/codex/artifacts脚本/清单此前全部被artifacts/规则忽略且未跟踪；
本批只为.sh/.py/.sha256/*_files.txt放行，压缩包/logs/.venv仍忽略。
9项Git忽略规则核验及git diff --check通过；未执行仿真、修改运行源码、commit/push或服务器接入。
已有轻量脚本已由用户提交进14038e3；服务器端GitHub访问认证仍待验证。
CI失败不撤销上传，但运行影响待日志定因；每批本地验证和远程小场景验收仍保留。

用户回传服务器原目录`~/zrq/mjlab_lego2`不是Git仓库，不能直接pull。
首版[接入入口](artifacts/bootstrap_remote_git_20261115.sh)已远程执行到clone阶段，
用户回传curl 92 HTTP/2 CANCEL、unexpected disconnect、EOF，下载未完成，尚无comparison.json。
失败核查目录：`/home/ajifang/zrq/backups/git_bootstrap_20261115_utkVcK`；
按脚本顺序已经过旧源码归档命令，但未读取远程归档及hash原件，不冒充已独立核验。
保留首版脚本/验证/失败现场；原目录、.venv和logs/checkpoint未由脚本修改，尚未切换训练入口。

v2仅在本地准备/核验，未回传远程执行；本次用户要求下载与版本验收拆分，首版/v2固定SHA入口均停用。
改用[第一步下载入口](artifacts/sync_remote_git_stage1_20261115.sh)：
临时HTTP/1.1、depth=1克隆至`~/zrq/mjlab_lego2_git_http1`；已有干净候选仓库时pull --ff-only。
不写死期望commit，仅记录实际HEAD/origin/main/状态至receipt.txt；不做源码hash比对、安装环境或训练。
[7项本地核验](diagnostics/20261115_git_stage1_local_validation.json)确认首次克隆、无需修改脚本快进到
新提交、重复执行通过；有未审阅改动、非仓库目标、旧目录本身或其子目录时停止。
旧源码/模拟checkpoint/旧环境/失败现场保持不变。Bash语法与diff检查通过；远程HTTP/1.1链路尚未验证。
下一步用户上传执行第一步，回传receipt.txt终端版本信息/记录目录，之后再给比对和验收指令。
用户已回传首次短训及一次低阶续训结果，记录于EXPERIMENT_LOG；旧日志/checkpoint和本次新run均保留。
阶段3包纳入本地src注释及最新交接；AGENTS→CLAUDE链接及长期中文注释规范本地保留，
CLAUDE不在远程训练增量清单内。旧v1/v2包、hash和原验证JSON保留历史用途，不能覆盖当前交接。
阶段3v1包保留交付时文档快照，旧包不包含本次修复；修复/复验状态以最新交接为准，
不重打旧包覆盖远程状态。
旧训练失败根因、旧参数与资产对应、EPA警告影响、CAD仍待确认。
