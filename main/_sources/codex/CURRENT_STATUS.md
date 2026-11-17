# 当前状态

更新时间：2026-11-17（Asia/Shanghai）。唯一跨会话入口：[HANDOFF.md](HANDOFF.md)。

## 当前阶段

**用户新录屏已查看；起步单腿抬落、左右前腿伸展差异与出坑后前后构型差异可见。离线轨迹确认高速前轮相对髋部前伸差约7.9 cm持续存在；建议先小幅、带适用条件地补姿态目标，再长训，尚待用户选择。**
前批远程候选奖励软件核验通过（用户回传）：CPU24/GPU1；迁移试验已回传结束行，新工具预检明细待原始文件核对。
证据目录`/home/ajifang/zrq/backups/reward_v2_resume_20261116_v4TczG`，详见
[结构化回执](diagnostics/20261116_reward_v2_remote_receipt.json)和EXPERIMENT_LOG；原始远程文件未下载。
此前list-envs的任务计数退出码问题已由模块入口修复，补验完整结束；不用重装或重复跑这批软件测试。
用户已回传Viser录屏：坑底/前腿基本正常，后腿借关节低位爬台、出坑后机身和关节过低，建议修改奖励。
初版已由用户commit/push保留；原任务/11项奖励与旧模型继续可用，不改写为v2结果。
原model_1198的3/6/9/12/15/20 cm初筛各10/10，30 cm为8/10；尚非正式大样本/直线高台验收。
阶段0～2及阶段3原核验、首次短训/低阶续训已有用户回传，原始实验路径及参数归
[EXPERIMENT_LOG](EXPERIMENT_LOG.md)。阶段4未触发，不引入平地预训练或接触引导机制。

## 最新v2原始证据核对

实际下载`diagnostics/20261117_reward_v2_continue_remote/remote_evidence_tCcuUA.tar.gz`，
SHA256=`775e111e57cf2e9296578f46eadcd058868059b52bccf9c709065bd80b248f4f`，98成员、85文件。
归档含初始trial及续训报告/trace/stdout/配置/迁移恢复审计，不含模型。
[报告核对](diagnostics/20261117_v2_downloaded_reports_analysis.json)重算两轮四组比较：
输入报告hash一致，报告来源hash与归档冻结清单一致，实际reset在1e-6内配对，汇总完全一致。
这不是逐帧动力学/FK重建，也未取得权重独立rehash；来源清单不等于远程当前源码重新读取。

| 当前配对条件，各3回合 | 父2197通过 | v2模型2198通过 | v2低净空/有效帧 | v2出坑后非轮接触 | v2出坑后最低root-台面 |
| --- | --- | --- | --- | --- | --- |
| 3 cm，0.5 m/s | 3/3 | 3/3 | 72.03% | 0 | 0.324923 m |
| 30 cm，0.5 m/s | 1/3 | 3/3 | 72.11% | 0 | 0.325567 m |
| 30 cm，0.3 m/s | 2/3 | 3/3 | 0.71% | 0 | 0.338516 m |

当前v2共9/9、均完整出坑、failed=0；三回合/条件不是正式验收。
非轮接触0仅覆盖首次complete后的记录帧及>1 N判据，不能证明后腿上台全程无刮擦。
速度RMSE低台0.048481→0.042277，高台0.5为0.031983→0.036420，低速0.025294→0.029422。
低净空占比覆盖全回合有效扫描帧；高速净空仍略低，不能把低速0.71%推广到全部速度。
同父跨轮实际初态最大差0、报告来源hash一致，30 cm通过仍从1→2及2→1；复现波动根因待逐帧核查。

## 当前训练状态与回放

实际续训run：`logs/rsl_rl/lego2_blind_pit_reward_v2/2026-11-16_22-25-03_v2_continue_1024x1900_20261116_222451_3365307/model_2198.pt`。
SHA=`27c1bdf3a05fa3455a16932dd4d5493fef526680575caee01e31a5c0d169248c`，报告/归档文本一致，非本地重算权重。
stdout核对trial有300条更新、Total steps=3686400；续训有1900条更新、Total steps=46694400，
v2新奖励合计2200次、50380800样本由日志支持，非仅计划推算；初始actor继承父2197，仍非同起点消融。

此前用户要求看Viser，已给出进程内临时注册v2固定工厂：真实0.30 m、vx=0.50 m/s，
1环境、seed7、auto_reset=True、课程关闭，复用run_play，不写源码、不更新PPO。
[回放入口核验](diagnostics/20261117_v2_play_entry_validation.json)仅配置/API，无本地仿真或网页启动。
默认Viser端口8080，实际服务器端口以启动日志为准；本地SSH隧道后浏览http://localhost:8080。
给出改forward_speed为0.30的低速观察方式；人工回放不当作新增配对评估。
保持父2197、v2模型299/2198和旧结果，不因当前9次初筛立即改权重或追加训练。

## 最新录屏与姿态评估（建议，尚未实施）

用户提供20.224607 s的2026-11-17 17:21:06录屏。录像3～5 s可见起步后腿抬落，
12～16 s左右前腿伸展不同，18～20 s机身近水平但前后构型不同；画面无腿名/模型SHA标签。
结合此前2198/30 cm/.5指令理解来源，但不把录像独立当作模型hash证明。
20 s处界面约222控制步，即约4.44 s仿真；不能把20 s录像当20 s稳定行驶。
[关键帧](diagnostics/20261117_v2_video_keyframes.jpg)、
[视频/轨迹核查记录](diagnostics/20261117_v2_video_posture_review.json)保存SHA、方法与限制。

对归档18回合（9父/9候选）作运动学检查，非推进物理时间。比较首次complete再过1 s后的窗口：
候选0.5 m/s六回合（3/30 cm各3），RF轮相对自身HFE髋关节的机身前向偏移
比LF多0.07924～0.07946 m，持续3.44～3.58 s；不是单次台沿过渡或视角假象。
这不是录像同一episode；录像自身未标RF/LF，具体名称由带名轨迹核实。
这些窗口1059帧无非轮接触/电机饱和标记，实际腿关节距硬限位最小14.16°，
碰撞体底部相对延伸台面平面最小约3.75 cm；不等于真机安全认证或完整过沿几何距离。
重建用的XML hash匹配冻结清单，四轮反算台面一致性误差≤1.495e-7 m。
9个候选在首次front_up至首次complete的所取窗口非轮接触标记均0，仍不能排除采样间碰撞。

当前奖励管跟速/机身姿态/最低高度/接触和动作平滑，没有偏好某套腿构型或左右一致的显式目标。
前后默认关节角本来正负相反，不能直接强迫四腿原始角度相等；前后外观不同也不自动代表错误。
后段打印Mean reward的分段均值约39.92、40.78、40.42、40.30，无持续上升趋势，
但这是混合课程日志，不能据此声称策略已严格收敛。继续长训不保证去掉未被奖励区分的姿态偏置。

建议先将第2/3点转化为平地、直行、稳定窗口内的弱默认姿态回归候选，先只动一项，
保留过沿/转向时的俯仰与不对称自由；第1点先观察，不加全程四轮贴地/抬腿惩罚。
新候选需独立run、明确warmstart处理，并以2198为参照先作有限微调/配对核验，再安排长训。
若用户优先验证v2能力上限，可改选原奖励新增1000更新、+500/+1000检查的对照；不承诺自然消除偏置。
以上均为本次建议，未改src/奖励/PPO、未准备新的开训命令或启动训练，用户尚未选择方案。

## 已有目录与注释约定

远程执行.sh及配套清单/包传到`~/zrq/mjlab_lego2/docs/codex/artifacts/`，同名目标先备份。
脚本切目录前固定自身绝对路径；旧顶层上传物按已知清单快照后移至artifacts/_legacy_uploads/，
现有根目录版本不覆盖。整理实际完成情况未回传，不因续训成功推测全部旧文件已整理。
本地注释版仍不同步；旧包/旧清单保留，后续同步须另建来源证据。

## 本批阅读注释

- 按用户要求，27个重点文件在模块/函数开头说明用途、配置范围和调用关系；
  地形、v1/v2权重、网络/PPO、预算、指标口径等关键参数加行旁说明，较长内容分多行。
- 用户随后要求将完整偏好整理为长期规范；已归入
  [AGENTS链接目标的“中文代码注释规范”](../../CLAUDE.md#中文代码注释规范)。
  该处统一维护覆盖范围、开头用途/文件关系、参数行旁解释、指标口径、通俗表述及分行规则；
  交接文件只保留入口引用，不重复维护另一套条款。本次只更新规范和上下文索引。
- 区分当前中央坑与旧15 cm固定高台、通用默认值与任务覆盖值；
  核对并修正旧Rough“两列10×20米”及腿scale0.25等过时文字，实际代码为三类10×10米和scale0.45。
  这些是说明修正，不是更换地形或驱动参数。长期注释偏好已写入AGENTS链接目标CLAUDE.md，链接保留。
- [修改前源码归档](backups/20261116_training_comments_before.tar.gz)仅含27个.py，无Markdown备份。
  [核验记录](diagnostics/20261116_training_comments_validation.json)逐文件去除说明字符串比较AST，
  参数/执行结构全部一致；格式通过、新增行均≤88字符，没有运行仿真或训练。
  Ruff原13条、ty原5条、Pyright原5条问题与修改前完全相同，无新增问题；未声称全库检查通过。
- 注释改变文本hash，本地注释版不再匹配旧运行清单是预期现象；旧清单/安装包均未改写。
  已交付trial_v2包仍为b593ff44…024d3；未同步本批注释，用户原远程试验继续使用已冻结的版本。
  后续同步须新建清单/包，不把本地新注释混入当前远程结果的来源证明。

## 既有v1配对结果与解释

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
用户反馈和录像已收到，低位姿态与旧非轮接触证据相符；“水平惩罚导致后腿卡台”仍是因果假设。
已据明确缺口新增独立v2奖励候选；原稳定判据保持，未实施额外接触抬升机制或子步回放工具。
两模型均保留，各10组初态不是正式大样本/多训练seed/直线高台泛化验收。

## 本批奖励v2候选

- 新任务`Mjlab-Blind-Pit-Lego2-RewardV2`，日志名`lego2_blind_pit_reward_v2`；
  [独立工厂](../../src/mjlab/tasks/velocity/config/lego2/blind_pit_reward_v2_env_cfg.py)
  从v1工厂派生，只修改奖励及两个诊断指标，旧入口仍为v1。
- [奖励实现](../../src/mjlab/tasks/velocity/config/lego2/blind_pit_rewards.py)：
  用俯仰±25°死区替换全水平惩罚（权重-0.5，原侧倾分量保留）；
  非轮geom通道接触力范数>1 N时每通道-0.5；
  复用已有terrain_scan，在全部射线有效且局部高差≤0.03 m时，
  对root距局部地面不足0.34 m的归一化缺口平方施加-2.0权重，最多-2速率。
  高于下限不扣分、不提供抬高奖金；较高台沿跨层扫描时关闭该高度项。
- 共13项奖励，actor144/critic254、12维动作、传感器/课程/PPO/资产/原通过定义保持。
  `base_clearance_eligible`与`base_clearance_below_minimum`记录高度项覆盖和低净空帧占比。
  权重/阈值是候选起点，不是已验证最优；软净空代价不保证每个关节绝对安全距离。
- 新增[actor迁移审计](../../src/mjlab/tasks/velocity/config/lego2/blind_pit_warmstart.py)及
  [独立训练入口](../../scripts/tools/train_lego2_blind_pit_reward_v2.py)，完整继承actor参数/归一化/动作分布，
  critic（含归一化）、优化器、迭代、环境计数及课程新建，不走会恢复旧计数的runner.load。
  按新run保存initial_transfer.pt、warmstart.json及params；拒绝已有输出目录，旧run不覆盖。
- 计划512环境×rollout24×300新增更新=3686400样本，seed7、课程从3 cm起，原PPO配置不变；
  新优化器学习率从配置1e-3重新开始、adaptive沿用，非父模型末尾优化器状态续训。
- 评估工具新增可选reward_version（默认v1不变），记录局部有效净空、非轮接触和出坑后最低root高度。
  新[配对质量汇总](../../scripts/tools/compare_lego2_reward_v2.py)核对实际reset再比较；
  两模型均在v2配置下无PPO复评：3/30 cm的vx0.5及30 cm的vx0.3，各seeds11～13，合计18个episode。
  每条件仅3次是动作质量初筛；父模型未更新参照不是同预算v1再训练消融。

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

以上为已完成的v1配对工具；其11项权重和入口保持。v2奖励差异仅在上节独立任务。
actor144/critic254、12维动作、训练速度分布、课程、机器人资产及PPO保持。
评估工具不更新PPO；已有零动作/随机权重软件核验不当作真实模型策略能力。

## 本地核验

- 本批[迁移链路/旧功能测试](diagnostics/20261116_v2_warmstart_cpu_tests.txt)26 passed（12.41 s）；
  最终[迁移/单次合成CPU更新/质量汇总](diagnostics/20261116_v2_warmstart_verified_tests.txt)3 passed（4.13 s）。
  定向5个Python文件格式/Ruff/ty/Pyright通过；只有合成checkpoint的2环境×24步软件更新，不是真实2197效果。
- [新交付核验](diagnostics/20261116_v2_trial_validation.json)及
  [安装/脚本模拟](diagnostics/20261116_v2_trial_launcher_validation.json)通过。
  正式交付`lego2_reward_v2_trial_20261116_v2.tar.gz`，9项文件、50项来源校验；本地中间v1包不交付。
  改前评估器/changelog归档`backups/20261116_v2_warmstart_before.tar.gz`及同名JSON；无新Markdown备份。
- [v2测试](diagnostics/20261115_reward_v2_cpu_tests_passed.txt)：24 passed，10.28 s，
  含原19项及5项v2测试；4个既有ONNX弃用警告。真实两环境CPU步进、奖励dt汇总、
  推理后reset、原入口隔离、跨台沿/无效射线逐环境过滤通过；定向Ruff/格式/ty/Pyright通过。
  本机可用内存约21 GiB，GPU驱动不可用；远程GPU软件测试已回传通过，最新v2训练结果见上节。
  [配置/录像证据层级](diagnostics/20261115_reward_v2_validation.json)保留数值与限制。
- [v2增量安装核验](diagnostics/20261115_reward_v2_sync_validation.json)：
  合成目录中验证旧文件备份、重复安装、未知源码/同名文件/损坏包拒绝，旧模型/.venv/交接不变。
  新源码包8项、核对43项文件，不含上下文Markdown、日志或模型；改前源码归档在
  `backups/20261115_reward_v2_before.tar.gz`及同名JSON。
- 本次只读配置核对：普通注册play为3～30 cm十级、初始level0且课程关闭，出生在3 cm，非固定30 cm。
  已验证临时进程内注册30 cm/0.50 m/s/单环境/自动reset配置，复用标准run_play及model2197；
  不改源码或原注册任务。Viser本地依赖默认监听8080，实际服务器监听端口以启动日志为准。
  配置/调用连接已离线核对，未在本机启动GPU回放；用户之后已提供录屏反馈，见EXPERIMENT_LOG。
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

1. 用户选择先补平地稳定姿态目标，还是保留v2做有上限的续训对照；当前只完成评估，未实施。
2. 若选姿态候选，先核对适用条件/权重量级及加载边界，再独立小批验证，比较能力、安全余量和姿态，不全程强制对称。
3. 原2198继续保留，配对筛查扩至每条件约10回合后再决定长训/正式泛化；不把9/9当最终验收。
4. 父2197、v2模型299/2198及旧结果保留；正式大样本/直线高台/平地验收仍待后续。

## Git、同步与实验边界

**GitHub仅供用户自行版本管理；是否commit/push由用户决定。服务器固定采用SCP/SSH直传，
GitHub连通、CI、提交号或push均不作为机器人开发/同步/验证前置条件。**
规范归AGENTS链接目标及PROJECT_CONTEXT，DECISIONS已移除冲突的旧操作规则，只留失效历史说明。
用户已将初版奖励commit/push保留；v2开发前的63f293a及当时干净状态是历史核对，非本批干净声明。
当前工作区已有v2/注释/文档及工具改动，均保留；Codex未commit/push，不要求每批形成Git版本阶段。
恢复资料仍在`artifacts/git_history_before_20261115_jlp8m7cp/`，禁止自动清理。
服务器继续使用`~/zrq/mjlab_lego2`及其现有环境；Git候选目录/失败现场保留，不再接入或切换环境。
v1原始配对及v2两轮报告已下载核对；v2完整stdout及Total steps已核对，旧v1完整stdout仍缺。
归档检查：历史`lego2_phase3_20261031_v1.tar.gz`及`lego2_phase3_reset_fix_20261101_v1.tar.gz`
含旧HANDOFF/CURRENT_STATUS，仅作历史保留，不重放覆盖当前交接；新speed_eval包无上下文快照。
v2注册及新评估器分别改变__init__/工具内容hash，历史清单保持为旧证据，不盲跑旧入口。
本批trial包无上下文快照，安装后以v2_trial_sources_20261116.sha256核对新版本。
