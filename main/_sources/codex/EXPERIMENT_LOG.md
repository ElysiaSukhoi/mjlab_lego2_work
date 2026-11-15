# 训练实验记录

资料位置与关联见[统一资料索引](README.md)；当前交接入口见[HANDOFF.md](HANDOFF.md)。
原始日志数值保留；早期行为解释中的撤回/更正已紧邻标出，不能当当前成功证据。

实验结论仅按已确认的完成或中止状态记录。另行保留有来源的历史指令线索，明确标注执行状态待确认，不能当作完成训练。缺少可靠run、checkpoint或原始日志时不补猜。

资产版本说明：2026-10-27之前的已列实验基于圆柱惯量修复前MJCF；
后续全量run及回放所用资产的对应仍待确认，不能一概归入同一版本。
本地现已修正 hip/FOOT 圆柱惯量轴向和 FOOT 近似质心，详见
[惯量核查报告](INERTIA_AUDIT_20261027.md)。修复后的训练尚未确认完成，
不能将 CPU 编译/数值验证记为登顶训练实验；后续记录应附资产版本与实际 run 参数。

## 2026-08-25至2026-09-07：继承的早期实验档案

- 来源：[源码根目录原人工日志](../../日志.md)，按原文日期覆盖Flat与Rough的早期试验。
  这是继承的历史记录，不是本轮复现实验；原文和`log_viewer.html`均原样保留。
- Flat记录含skid转向、默认姿态/接触/平滑项消融；Rough记录含地形拼接、
  terrain/command/friction课程及轮速协同配置的演变，日期目录和具体参数保留在原日志。
- 原文多处训练、sim2sim或sim2real仍标“待测”；另有约model_8100、5450等中断记录。
  不将这些条目统一标为训练完成、部署通过或本项目已复现。
- 历史日期目录、checkpoint文件是否仍存在以及与当前资产的关系均待远程核查；
  此处只建立档案索引，不复制整份原始表格或改写其观察结论。

## 2026-09-22：zrq_first_run历史指令线索（执行结果待确认）

2026-10-31经用户批准，从两份旧说明迁入；本次没有启动训练或回放。

| 项目 | 原文记载及证据范围 |
| --- | --- |
| 来源 | [第一次训练指令原文](/home/asus/work/mjlab_doc_quarantine/20261031_approved_cleanup_b5b1f0c3/files/Jiche_Project/粗糙地面速度跟踪/第一次训练相关指令.txt)；旧路径`/home/asus/work/Jiche_Project/粗糙地面速度跟踪/第一次训练相关指令.txt` |
| 任务/run | `Mjlab-Velocity-Rough-Lego2`；`zrq_first_run` |
| 指令意图 | 4096环境、max_iterations=10000、TensorBoard；原指令未写resume参数，不能据此推断实际优化器状态 |
| 日期目录线索 | `logs/rsl_rl/lego2_velocity/rough/2026-09-22_20-49-03_zrq_first_run/` |
| checkpoint线索 | `logs/rsl_rl/lego2_velocity/rough/2026-09-22_20-49-03_zrq_first_run/model_9999.pt` |
| 完成/续训/资产对应 | **待确认**；原文称其为最终模型，本轮未连接服务器验证文件、实际迭代量、配置、NaN/OOM或结果 |

[旧核心代码解析原文](/home/asus/work/mjlab_doc_quarantine/20261031_approved_cleanup_b5b1f0c3/files/Jiche_Project/粗糙地面速度跟踪/核心代码解析.txt)还提及256环境、10轮短测及
61,440条累计step。只确认该历史文字记载，run目录、原始event及配置版本待确认，
不作为新增完成实验或速度跟踪/登顶成功证据。

原指令包含`\\//`式讲解注释，不能直接当作验证过的shell脚本执行。
通用控制/PPO解释已按当前源码整合至[PROJECT_CONTEXT](PROJECT_CONTEXT.md)；
保留的部署手册及当前SSH/screen约定不由旧示例覆盖。
上述原文的SHA-256、移出前路径和可恢复位置均在[恢复清单](/home/asus/work/mjlab_doc_quarantine/20261031_approved_cleanup_b5b1f0c3/manifest.json)中，
其余粗糙地面曲线分析和历史交接保持原样，未据其估读值补写本run结果。

## 2026-10-12：Rough V1 基线

- 任务：`Mjlab-Velocity-Rough-Lego2`
- 参数：4096 环境，10000 iterations，run `high_platform_300mm_v1`，TensorBoard logger。
- checkpoint：`logs/rsl_rl/lego2_velocity/rough/2026-10-12_20-28-29_high_platform_300mm_v1/model_9999.pt`
- 结果：有一定越障能力，但平地抬腿、高台卡边和偶发成功；`Episode_Reward/upward` 不是登顶成功指标。

## 2026-10-15 至 2026-10-20：Platform-Climb 基线与诊断

### 300 mm、短训与续训

- 任务：`Mjlab-Platform-Climb-Lego2`，128 环境，约 200 iterations；之后有一次不改代码的续训。
- 结果：无 NaN、无 CUDA OOM、无翻倒；`platform_approach_mean` 约 `0.6733`，续训约 `0.7507`；顶部奖励、顶部指标和成功终止均为 0。

### 300 mm、接触诊断

- `platform_front_contact_count_max=2`，`platform_contact_count_max=4`，`platform_root_z_max=0.1705`。
- 当时解释：前足接触台阶但机身未抬升。**后续更正**：接触计数不区分低地、墙面和台面，
  不能单独证明碰到台壁；根部高度应与对应回放/坐标一起判断。原始数值保留。

### 150 mm、几何阈值修正后的诊断

- `platform_root_x_max=-1.8044`，`platform_root_z_max=0.3197`，前足接触最大 2、总接触最大 4。
- `platform_edge_lift=0`，顶部到达/稳定/成功均为 0。
- 结论：降低高度后仍未自动形成抬升动作，下一步增加局部引导。

### 150 mm、前足抬升奖励

- 修改：`platform_edge_lift` 低根部参考下移 0.05 m；新增 `platform_front_lift`，权重 `0.75`，只在 `x∈[-2.1,-1.5]` 使用 `foot_height_scan` 前足高度。
- 结果：`platform_approach_mean=0.8790`；`platform_edge_lift_mean=0.0007`；`platform_front_lift_mean=0.3268`；前足奖励均值约 `0.2268`；顶部到达、稳定和成功仍为 0。
- 当时解释为“前足抬升被诱导”。**该解释已撤回**：轮心正常着地也可能获得正奖励，
  此组均值不能证明主动抬前腿；仍保留未登顶的观察及原始数值。

### 150 mm、前足抬升后的较大规模运行

- 结果：`platform_root_x_max=-1.6119`，`platform_root_z_max=0.3255`，前足接触最大 2，总接触最大约 3.1667，`platform_edge_lift_mean=0.1048`，`platform_front_lift_mean=0.5188`；顶部到达、稳定和成功均为 0。
- 当时据此提出接触耦合奖励。**后续更正**：这些批量均值不证明前足已抬升或台面支撑；
  root-x的统计距边缘约0.11m不代表每个机器人都停在该点。旧路线保留归档，不据此续加shaping。
- run 名称、精确迭代数、checkpoint 路径：**待确认**。

### 4096 环境 OOM 尝试

- 在 CUDA Graph 创建阶段出现 `Warp CUDA error 2: out of memory`。
- 结论：先用 2048，必要时 1024；这次没有证据表明是奖励或地形代码错误。
- 精确命令、显存采样和 checkpoint：**待确认**。

## 2026-10-21：前足接触后继续前移奖励（代码阶段，训练待验证）

- 目标：解决机器人前足已经抬升/接触平台，但 root x 停在边缘前约 `0.11 m` 的问题。
- 本地修改：在 `platform_climb_rewards.py` 新增 `platform_front_contact_advance`；在 `platform_climb_env_cfg.py` 注册为权重 `1.5`，x 门控范围 `(-2.1,-1.0)`，接触传感器为 `feet_ground_contact`，力阈值为 `1.0`。
- 信号定义：`edge_progress × front_contact_fraction`。只在前足接触时提供额外的边缘前移信号，并把门控范围延伸到边缘之后。
- 本地状态：编译、任务注册、配置加载、参数断言和合成接触数值测试通过。
- 远程同步状态：**待确认**。run 名称、环境数、迭代数、checkpoint、NaN/OOM 和顶部成功指标：**待确认**。
- 训练决策：奖励/配置发生变化，因此不从旧 checkpoint 续训；先进行新的 150 mm 短 run。

## 历史待验证项与当前研究边界

2026-10-28 用户指定回放已训练的 run `platform_climb_front_lift_full_150mm_2048_v1`，
暂不进行下一步训练。2026-10-30用户确认视频对应全量训练checkpoint编号9999；
实际日期目录、完整参数以及与该run和上述日志的精确对应仍待确认，不凭run名称推定环境数。

`platform_front_contact_advance`的远程同步/实验归属仍未确认，作为旧路线历史待验证项保留；
当前重构不再安排继续该shaping实验，不把它当作下一条训练指令。

## 2026-10-29：全量回放反馈与研究暂停点

- 用户确认URDF/资产纠偏完成且远程已同步；没有提供新增修改文件或逐文件版本。
- 用户查看全量训练回放，认为效果不理想，提供
  `/home/asus/视频/录屏/录屏 2026年10月28日 21时58分42秒.webm`（约10.1s）。
  抽帧未见完整稳定通过，不能由片段推算成功率。
- 2026-10-28已指定run `platform_climb_front_lift_full_150mm_2048_v1`，但本视频与
  该run、环境数/iterations、训练和回放资产的对应仍待确认；编号9999已于10月30日由用户确认。
- 更正旧结论：front_lift是轮心高度门控，普通着地也可能得分；接触计数不区分台面；
  contact_advance为位置×接触，不保证正在前移。旧日志均值不是主动抬腿或台面支撑的证明。
- 用户明确先研究盲走＋坑式＋初始高度课程＋通用速度奖励路线，本轮没有新训练实验。
  研究报告和源码来源记录在外层Jiche_Project；不把参考项目结果计为本项目实验结果。

## 记录规则

每次实验至少记录任务 ID、环境数、iterations、run 名称、是否续训、关键 checkpoint、NaN/OOM、顶部指标和下一步结论。日志只报告实际观察到的量，不把接近奖励或 `upward` 当作成功率。

## 2026-10-30：全量训练checkpoint 9999的重复失败反馈

- 证据来源：用户确认，非新增本地/远程训练。视频使用全量训练编号9999的checkpoint。
- 用户多次Reset均无法通过；每次碰台后前轮完全无法抬起，后腿反而抬起。
- 测试总次数未提供，不能计算精确成功率；记录为多次测试均失败的可重复行为模式。
- 原始项目actor不输入height_scan；用户后来注释掉删除语句，才使后续版本保留扫描。
- 更正旧解读：奖励均值增大不等于前腿已学会抬升；不把此轮失败归结为仅需延长相同配置。
- checkpoint完整目录、实际env/agent配置、训练和回放资产对应仍待核对。
- 后续候选：重构前先核对足端/动作映射、关节符号与限幅及前腿运动学可行性，
  配合既定盲走、坑式高度课程和通用奖励路线讨论。尚未证实映射错误或某项惩罚是主因。
- 本轮仅更新文档，没有改业务代码或启动训练。

## 2026-10-30：正式计划v2与接口就绪核查（非训练实验）

- 正式计划见[REFACTOR_PLAN.md](REFACTOR_PLAN.md)，接口检查见
  [就绪核查](IMPLEMENTATION_READINESS_20261030.md)；阶段0～5未实施，无新任务run/checkpoint。
- 已只读核对本地注册、历史、动作、地形/课程、reset和奖励接口，无需用户补传源码。
- 用现有uv环境CPU检查候选3/15/30cm坑几何/原点、历史buffer局部reset回填，均通过。
  这些不属于PPO训练、完整环境验收或越障能力结果，未运行GPU训练。
- 保留checkpoint9999多次失败反馈与版本待确认信息；起始奖励表和课程参数仅为未验证提案。

## 2026-10-31：指定旧run文件定位更正（非新增训练）

证据为用户回传的远程find输出，非本轮自行访问远程。此前“完整目录待确认”
已对以下指定run文件得到补充；不改写旧训练原始结果或推定未读取的参数。

- checkpoint：`logs/rsl_rl/lego2_platform_climb/2026-10-20_22-08-39_platform_climb_front_lift_full_150mm_2048_v1/model_9999.pt`。
- 同一run目录存在`params/env.yaml`及`params/agent.yaml`；内容尚未读取。
- 实际环境数/训练命令/资产版本、该文件与视频的逐一对应仍待确认，
  不因目录含2048、150mm或checkpoint编号而推定实际配置/训练完成量。
- 本轮没有开训、续训或回放。阶段0远程CPU流程通过的当前状态归CURRENT_STATUS维护，
  不把CPU诊断计为新的训练实验或越障成功。

## 2026-10-31：阶段3首次流程短训128环境/200 iterations

证据来源为用户回传训练末段日志、实际文件清单及脚本退出码，未自行访问远程或读取完整stdout。
任务`Mjlab-Blind-Pit-Lego2`；对应已校验启动脚本参数为128环境、rollout24、200 iterations、
seed7、TensorBoard、resume=False，首训即启用高度课程。实际params内容尚未回传核对。

- 实际run：`logs/rsl_rl/lego2_blind_pit/2026-10-31_21-58-03_phase3_smoke_128x200_v1_20261031_215751_2154252`。
- 已存在checkpoint：`model_0.pt`、`model_50.pt`、`model_100.pt`、`model_150.pt`、`model_199.pt`；
  同run的`params/env.yaml`与`params/agent.yaml`存在。下一次评估使用实际`model_199.pt`。
- 远程证据目录：`/home/ajifang/zrq/backups/phase3_smoke_20261031_215751_2154252`。
- 用户回传进程退出码0、ETA 0、elapsed 0:05:48，流程短训正常结束。
  按脚本规模推算采样量128×24×200=614400；完整stdout的Total timesteps尚未回传，未独立核验。
- 末段课程日志：mean=0.1540、max=1、demoted=0.1429；这是等级统计，不是稳定通过率。
- 末段指标：pit_front_up=0.1292、pit_complete=0.0241、pit_stable_pass=0；
  前两者是支撑阶段时间占比，不能当作12.92%/2.41%的episode成功率。
  stable_pass=0是本次回传日志窗口，不足以断言整个run从未出现成功。
- 末段终止：tile_boundary=0.8571、nan=0、fell_over=0；边界truncate不等于完整稳定出坑。
- 末段非足碰撞通道均值0.0586、yaw误差0.0803；腿力矩成本-0.1893、腿加速度-0.0986、
  腿动作变化-0.0918。它们是日志汇总，未读受阻事件轨迹，不据此直接调整奖励或限幅。
- 没有OOM/NaN中止的用户回报，退出正常；完整日志、实时显存峰值和checkpoint hash仍待核验。

结论：训练/保存流程已跑通，低阶稳定越障能力待固定高度评估；保留本run和全部旧结果。
下一步对`model_199.pt`在3/6/9 cm各10个完整episode做初筛，关闭课程、保留终止及0.5 s稳定定义，
记录真实n/通过/前后轮支撑/失败/停滞和轨迹后再决定低阶能力训练；此评估尚未执行。

## 2026-11-01：阶段3固定高度评估因指标缓存异常中止

- 证据：用户回传evaluate_lego2_blind_pit.py → wrapper.reset → MetricsManager.reset →
  PitProgress.reset堆栈，报“Inplace update to inference tensor outside InferenceMode”。
  用户称其为烟雾测试脚本；按堆栈确认失败发生在固定高度评估工具的reset路径。
- 此次评估已尝试但异常中止；失败评估目录、实际已完成episode数与完整report均未回传，标为待确认。
  不把残留trace片段当作3/6/9 cm完整评估或填入成功率，不推定失败发生在哪个episode/高度。
- 原128环境/200 iterations短训退出码0及model_199.pt存在的记录仍有效；本轮没有重训或删除旧结果。
- 本地复现并修复指标缓存生命周期；本地零动作/临时随机权重检查只作软件回归，
  不计为真实model_199.pt策略的越障评估。修复验收范围归CURRENT_STATUS维护。
- 下一步先部署修复并核验，再以同一model_199.pt和同一稳定定义，在新输出目录重新完成固定高度初筛。

## 2026-11-01：固定高度评估仍在推进的用户回传（未完成）

- 目录：`/home/ajifang/zrq/backups/phase3_eval_20261101_172748_2195875`。
- 进程命令确认使用本次model_199.pt、cuda:0、3/6/9 cm各10个episode、seed7；未指定max_steps。
- 用户快照：Python PID2195929运行25:56，CPU100%；同PID GPU显存924 MiB，GPU利用率13%。
- 相隔20秒，3 cm trace均5994行、6 cm均9297行，9 cm由8482增至8792行，总行数23773→24083。
  因此确认该观察窗口内评估持续推进，约15.5步/s；终端静默不能据此判为卡死。
- 已进入最后高度；行数不是成功episode数，也不提供通过率。最终退出码、summary/report仍待回传。
- 本轮没有自行连接远程、停止/重启评估或修改业务代码；继续等待该进程完成，保留所有原目录。

## 2026-11-01：model_199.pt固定高度初筛完成（用户回传）

证据为用户回传最终summary及“评估结束”行；原始report/trace未下载，脚本数字退出码未回传。
对应3/6/9 cm、每高度10个完整episode、seed7的同一评估，保留旧失败目录。

- checkpoint：`/home/ajifang/zrq/mjlab_lego2/logs/rsl_rl/lego2_blind_pit/2026-10-31_21-58-03_phase3_smoke_128x200_v1_20261031_215751_2154252/model_199.pt`。
- 用户回传SHA256：`901cc02246fcd47517a0791ceec4513cc6725a1d8d0786955ab3c945ddb1b2f4`。
- 证据目录：`/home/ajifang/zrq/backups/phase3_eval_20261101_172748_2195875`。
- [结构化回传](diagnostics/20261101_phase3_model199_user_eval_summary.json)仅转录summary，未伪装为原始report。

| 高度 | n | 稳定通过 | root离坑 | 前轮上沿 | 后轮上沿 | 完整出坑 | 翻倒/失败 | 停滞 | mean speed RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 cm | 10 | 5（50%） | 10 | 10 | 7 | 7 | 0/0 | 3 | 0.173761692315679 |
| 6 cm | 10 | 1（10%） | 10 | 10 | 7 | 7 | 0/0 | 3 | 0.312183466094233 |
| 9 cm | 10 | 0（0%） | 4 | 4 | 0 | 0 | 0/0 | 10 | 0.42347593239897846 |

本表前后轮/完整出坑是episode内曾达到该阶段的次数，区别于训练日志的时间占比。
6 cm完整出坑7次、稳定通过1次，说明连续0.5 s有效支撑及倾角条件尚难稳定满足；
究竟是接触保持、姿态或检测边界，需要事件轨迹确认，不能仅凭summary定因。
9 cm后轮上沿为0、全部停滞，尚未形成该高度完整稳定通过能力；不推定前腿机械不可达。
每高度仅10个样本、单seed，结果为初筛，不宣称收敛或具备300 mm能力。

下一步按阶段3做低阶能力验证：同任务/观测/动作/奖励/资产，从本checkpoint恢复权重和优化器，
新建run，1024环境新增1000 iterations；课程重新由3 cm起步，之后同条件固定高度评估。
该续训脚本仅准备并本地核对参数，尚未执行；不预填新run/checkpoint或新增采样完成量。

## 2026-11-01：1024环境低阶续训正常结束（用户回传）

- 实际run：`logs/rsl_rl/lego2_blind_pit/2026-11-01_20-17-22_phase3_low_1024x1000_v1_20261101_201712_2200851`。
- 用户回传model_200.pt至model_1150.pt间多个checkpoint、末尾model_1198.pt，
  同run的params/env.yaml及params/agent.yaml存在，启动脚本退出码0。
- 证据目录：`/home/ajifang/zrq/backups/phase3_low_20261101_201712_2200851`。
- 对应已校验脚本：从上一轮model_199.pt恢复，1024环境、rollout24、新增1000 iterations、seed7；
  保持观测/动作/奖励/资产和高度课程，新建run保留父模型。
  已知runner从恢复的编号起新增指定更新次数，因此1198的末尾编号符合脚本预期，非从零1198次。
- 计划新增采样量1024×24×1000=24576000；完整stdout/Total timesteps、实际params内容未读取，
  不把计划推算冒充独立核实的采样量。此次未回传末段奖励/课程/失败指标、elapsed或峰值显存。
- [结构化回执](diagnostics/20261101_phase3_low_user_receipt.json)只转录用户证据；新checkpoint SHA尚待评估时记录。
- 结论：续训/保存流程正常结束，尚不能说明越障能力已提升；下一步对model_1198.pt重做同条件
  3/6/9 cm各10个完整episode初筛，与model_199.pt的5/10、1/10、0/10比较。
  本轮没有自行访问远程、启动评估或新一轮训练；所有旧结果保留。

## 2026-11-01：model_1198.pt低高度复评完成（用户回传）

- checkpoint：`/home/ajifang/zrq/mjlab_lego2/logs/rsl_rl/lego2_blind_pit/2026-11-01_20-17-22_phase3_low_1024x1000_v1_20261101_201712_2200851/model_1198.pt`。
- 用户回传SHA256：`7780cf0d2557c86f51c0486f870e7d471b13703777e9cee5e1854427def05027`。
- 评估证据目录：`/home/ajifang/zrq/backups/phase3_eval1198_20261101_210150_2202384`。
- 同条件3/6/9 cm、每高度10个完整episode、seed7。证据为最终summary及“评估结束”行，
  原始report/trace未下载，数字退出码未单独回传；
  [结构化摘要](diagnostics/20261101_phase3_model1198_user_eval_summary.json)只转录用户输出。

| 高度 | 父模型model_199稳定通过 | model_1198稳定通过 | 父模型速度RMSE | model_1198速度RMSE |
| --- | --- | --- | --- | --- |
| 3 cm | 5/10 | 10/10 | 0.173761692315679 | 0.07241831920391109 |
| 6 cm | 1/10 | 10/10 | 0.312183466094233 | 0.06240884230790049 |
| 9 cm | 0/10 | 10/10 | 0.42347593239897846 | 0.053261255794071996 |

本次各高度root_out/front_up/rear_up/complete均10/10，fell_over/failed/stalled均0/10。
结果支持现有通用奖励和课程路线能形成低阶越障能力，暂不因低阶瓶颈引入阶段4或平地预训练。
每高度只有10次、单seed，并不证明真实成功率必为100%、全高度收敛或达到300 mm；
尚无独立直线高台/平地保留能力/多seed泛化结果，也没有本次策略真实回放轨迹的人工核查。

下一步先用同一model_1198.pt摸底12/15/20/30 cm，各10个完整episode、seed7，保留稳定0.5 s定义。
据能力边界决定继续课程训练还是事件诊断；该高高度评估仅准备脚本，尚未执行。
当前不改模型/观测/动作/奖励/资产，不直接启动更大训练；旧模型和全部评估结果保留。

## 2026-11-14：高高度初筛完成（用户回传）

- 证据目录：`/home/ajifang/zrq/backups/phase3_upper1198_20261114_214623_3270561`。
- 用户贴出四个高度最终summary及“评估结束”行；原始report/trace及数字退出码未回传。
- 本次粘贴未含checkpoint路径、SHA和seed；按已发
  [评估入口](artifacts/evaluate_phase3_upper_1198_20261101.sh)预期为同model_1198.pt、seed7，
  预期SHA为`7780cf0d2557c86f51c0486f870e7d471b13703777e9cee5e1854427def05027`，
  待原始report核对，不能冒充本次已回传或独立读取的hash。
- [结构化摘要](diagnostics/20261114_phase3_model1198_upper_user_summary.json)仅转录用户证据。

| 高度 | n | 稳定通过 | root离坑 | 前轮上沿 | 后轮上沿 | 完整出坑 | 翻倒 | failed | 停滞 | mean speed RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 12 cm | 10 | 10 | 10 | 10 | 10 | 10 | 0 | 0 | 0 | 0.04810981028233111 |
| 15 cm | 10 | 10 | 10 | 10 | 10 | 10 | 0 | 0 | 0 | 0.045089428042688943 |
| 20 cm | 10 | 10 | 10 | 10 | 10 | 10 | 0 | 0 | 0 | 0.044076157799581316 |
| 30 cm | 10 | 8 | 9 | 9 | 9 | 9 | 0 | 0 | 1 | 0.10854789592697618 |

结论：当前坑式评估已有30 cm稳定通过样本；低/中高度初筛支持继续现有通用奖励和课程。
30 cm完整出坑9次、稳定通过8次，说明单次达到支撑阶段与连续0.5 s稳定判据仍有差别；
无法从汇总计数确定未出坑、停滞、两个未通过样本的对应关系或确切原因。
不把8/10外推为真实80%成功率，不把无翻倒/failed当作全部成功，亦不据此宣布直线300 mm高台验收。

下一步先取回report、30 cm轨迹和checkpoint hash，检查样本、退出方向/坑角借力、
有效支撑/倾角持续时间及动作。随后推进计划中的每高度至少100次（3/5/10/15/20/30 cm）、
有条件多seed、独立直线高台及平地保持能力测试；是否追加训练由核查结果决定。
本轮未改奖励/任务/资产/模型、未新增训练或自行访问远程；旧结果保留，阶段4仍未触发。

## 2026-11-14：高高度原始证据及30 cm轨迹离线核查

用户下载[原始report](diagnostics/20261114_phase3_upper1198_remote/report.json)、同目录trace_0.300000.jsonl与checkpoint.sha256。
实读确认：model_1198.pt完整路径与上条预期相同，seed7、cuda:0、checkpoint SHA为
`7780cf0d2557c86f51c0486f870e7d471b13703777e9cee5e1854427def05027`，
与独立checkpoint.sha256文本一致；未下载模型本体，非本地重新散列模型权重。
report记录14项源码/资产hash均与本地匹配；四高度summary吻合，30 cm十次trace步数、
阶段事件、最终通过标记及重算speed RMSE均与report一致。

- [分析脚本](artifacts/analyze_phase3_upper1198_20261114.py)、
  [分析JSON](diagnostics/20261114_phase3_upper1198_analysis.json)及[图](diagnostics/20261114_phase3_upper1198_analysis.png)保存可复核过程。
- 匹配源码的控制dt=0.02 s；两行单列固定坑的level0原点为(-5,0,-0.3) m。
  仅用MuJoCo运动学重建轮几何，不推进仿真；轮底与trace的最大误差1.449e-7 m。
- 全10次四轮Y侧向包络距坑侧壁均至少0.6328 m，路径从+x正面出坑，未见轮子坑角借力。
  这不替代独立高台测试，也不全面排除trace未记录的机身/非轮接触。

下表使用1起编号（原始episode为0起）：

| 样本 | vx (m/s) | 表现 | 最大连续稳定支撑 | 最大倾角 | 结束 |
| --- | --- | --- | --- | --- | --- |
| 第2次（id=1） | 0.3195317388 | 无前轮/后轮/完整上沿，最终root相对坑中心x=0.9944 m | 0 s | 12.096° | 20 s超时、stalled |
| 第3次（id=2） | 0.3004603684 | 11.42 s首次完整出坑，最终root相对坑中心x=4.0022 m | 0.22 s | 15.504° | 17.16 s地块边界截断，非stalled |

第2次RF/LF轮底相对坑底最高0.0754/0.2355 m，均低于30 cm台面；低速前轮抬升/通过未形成。
第3次完整支撑共193个采样点；几何上四轮均出坑且轮底距台面<2.5 cm共288步，
其中95步至少一轮未满足有效支撑，RF/LF/RH/LH缺失接触通道分别27/50/22/13步（可同时发生）。
这些步均可由found=0且对应force=0解释；倾角未接近45°，主要不通过项是支撑连续性。
trace未记contact.pos，且仅控制帧采样，不能断言真实跳跃/卸载、传感器漏检或GPU接触求解具体根因。

十次记录采样点均无力矩饱和；目标越硬限均发生在膝关节，两未通过样本分别198/263步，
成功样本也有34～66步。目标越限是待跟踪现象，不能直接认定它或力矩上限是失败原因。
两未通过样本恰为最低vx，八次成功vx范围0.3545～0.5362 m/s，但速度与初态扰动混杂，
仅形成低速相关假设，不能证明因果。下一批建议30 cm固定vx=0.30/0.40/0.50 m/s、
配对初态/seed的小批对照和回放，奖励/课程/判据保持，入口尚未实现或远程执行。

本批本地Ruff/格式/ty/Pyright检查分析脚本通过，运行内置一致性断言通过；无仿真/PPO或远程操作。
阶段4未触发，不因本次8/10自动追加大训练；正式大样本/多seed条件/独立高台/平地验证仍待实施。

## 2026-11-15：model_1198的30 cm固定速度配对初筛完成（用户回传）

用户回传三组最终摘要、`PASS: 10 paired resets, tolerance=1e-6`及“评估完成/没有PPO更新”。
证据目录：`/home/ajifang/zrq/backups/phase3_speed_20261115_185511_3295996`。
[结构化回传](diagnostics/20261115_phase3_speed_user_summary.json)只转录用户输出，非下载的原始报告。
按入口预期为同model_1198.pt、内容SHA
`7780cf0d2557c86f51c0486f870e7d471b13703777e9cee5e1854427def05027`、reset seeds 7～16；
本次摘录未贴模型hash，原始report/trace/paired_summary及CPU、GPU短片段输出尚未下载。
入口在完整评估前检查源码/模型与6个两步GPU片段，但不把执行顺序推断冒充独立读取的原始回执。

| vx (m/s) | 完整episode n | 稳定通过 | 完整出坑 | 停滞 | 翻倒 | failed | mean speed RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.30 | 10 | 4 | 5 | 5 | 0 | 0 | 0.19099997204442862 |
| 0.40 | 10 | 10 | 10 | 0 | 0 | 0 | 0.08419473722306246 |
| 0.50 | 10 | 10 | 10 | 0 | 0 | 0 | 0.05930095785608082 |

相同10组初态下表现随速度改变，低速薄弱项已在本批配对初筛中确认；仍非多训练seed或正式大样本验收。
0.30 m/s有6次未稳定通过、5次停滞，至少1次曾完整出坑但未满足连续支撑；
汇总不能确定每个样本的停滞/未出坑对应关系，更不能断言冲量、控制幅度、接触求解或奖励的具体因果。
保持通用奖励和课程，不在本轮同时改速度分布/机制；阶段4和平地预训练仍未触发。

下一步准备从model_1198恢复权重/归一化/优化器和迭代状态，新建run，1024环境、rollout24、
新增1000 iterations、seed7；训练vx保持0.3～0.6 m/s，仿真/历史/课程重新reset，初始最低3 cm。
计划新增24576000采样量，不是已完成数值；训练尚未执行，不预填实际run或checkpoint路径。
[续训入口](artifacts/continue_phase3_baseline_20261115.sh)及
[本地参数/选择器核验](diagnostics/20261115_phase3_continue_launcher_validation.json)已准备。
结束后以同样10组配对初态复评0.30/0.40/0.50 m/s，检查低速改善及中高速保留，再决定后续预算。

## 2026-11-15：第二轮原基线续训完成，model_2197待复评（用户回传）

- 实际run：`logs/rsl_rl/lego2_blind_pit/2026-11-15_19-28-14_phase3_continue_1024x1000_v1_20261115_192803_3297676`。
- 用户回传model_1200.pt～model_2150.pt间多个checkpoint及末尾model_2197.pt，params/env.yaml、agent.yaml存在。
- 启动入口退出码0，elapsed 0:34:02，末次iteration time 1.72 s。
- 证据目录：`/home/ajifang/zrq/backups/phase3_continue_20261115_192803_3297676`。
- [结构化回执](diagnostics/20261115_phase3_continue_user_receipt.json)仅转录用户摘录；
  model_2197内容SHA、完整stdout/Total timesteps、实际params内容未下载，SHA留待复评记录。
- 对应已核对入口：从model_1198恢复，1024环境、rollout24、新增1000 updates、seed7，
  训练速度仍0.3～0.6 m/s、原奖励/课程/动作/资产/PPO；课程与仿真重新reset，保留父run。
  runner恢复编号后再执行新增次数，末尾2197符合1198+1000-1，不表示本轮新增2197次。
  24576000是入口的计划采样量，尚无Total timesteps行独立核实。

用户提供末段窗口：pit_complete=0.3408，twist/error_vel_xy=0.0199，time_out=0；
flat_orientation=-0.0006、joint_torques=-0.0198、wheel joint_acc=-0.0002、yaw tracking reward=0.7246。
pit_complete为阶段时间占比，不是episode稳定通过率；训练误差与固定速度RMSE的场景/口径不同，
不能由这些末段值宣称低速问题已解决、模型更强或高台正式验收通过。

下一步同model_2197做30 cm、vx=0.30/0.40/0.50 m/s各10次、reset seeds7～16配对复评，
使用原判据并保持原源码；先用6个两步GPU片段确认新模型推理/reset，再进入30个完整episode。
在新模型组内核对配对后，再核对与父model_1198实际初态，输出每速度新增通过/退步的seed。
[复评入口](artifacts/evaluate_phase3_model2197_20261115.sh)及
[本地CLI/合成报告逻辑核验](diagnostics/20261115_phase3_model2197_eval_launcher_validation.json)已准备。
合成JSON只用于验证汇总/拒绝错误输入逻辑，不是新模型成功率；远程复评尚未执行。
本轮未改src/模型/奖励或启动第三轮训练，全部旧结果保留，GitHub不参与服务器传输。

## 2026-11-15：model_2197配对复评完成，低速改善伴随高速稳定率退步（用户回传）

用户回传30 cm三速度各10次完整评估，组内及新/父模型初态核对均通过（容差1e-6），无PPO更新。
证据目录：`/home/ajifang/zrq/backups/phase3_eval2197_20261115_201606_3299957`；
父模型目录：`/home/ajifang/zrq/backups/phase3_speed_20261115_185511_3295996`。
[结构化摘要](diagnostics/20261115_phase3_model2197_user_eval_summary.json)只转录用户输出。
原始report/trace/comparison.json及model2197 SHA未下载；本次摘录未含启动时的checkpoint.sha256行。

| vx (m/s) | n | model1198稳定通过 | model2197稳定通过 | model2197完整出坑 | 新停滞/翻倒/failed | 父模型速度RMSE | 新模型速度RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.30 | 10 | 4 | 9 | 10 | 0/0/0 | 0.19099997204442862 | 0.02533347966582821 |
| 0.40 | 10 | 10 | 10 | 10 | 0/0/0 | 0.08419473722306246 | 0.02920764114724614 |
| 0.50 | 10 | 10 | 8 | 10 | 0/0/0 | 0.05930095785608082 | 0.031171845598559456 |

跨模型配对变动：
- 0.30 m/s：新增通过seed7/9/10/12/13/14，退步seed11，净增加5次；唯一未稳定通过为seed11。
- 0.40 m/s：无新增或退步，10次均保留稳定通过。
- 0.50 m/s：退步seed11/13，其他8次保留通过。

结论：追加原基线训练在本组配对样本上改善低速完整出坑/速度跟踪，但未均匀改善稳定通过。
新模型全部30次都曾完整出坑且无停滞/翻倒/failed；3个未稳定通过样本均涉及已达成出坑后，
仍未满足整个episode内连续0.5 s支撑/倾角/无失败的联合判据。
不能从summary判定是接触中断、姿态、有效支撑几何还是可用时间，也不能仅凭更低RMSE宣布全面更好。
不把不同速度合计27/30当作真实部署成功率，且每速度仅10组初态，不是多训练seed/正式大样本验收。

下一步只读取回新/父模型full报告与轨迹、comparison、模型SHA及训练配置，优先核查seed11/13，
必要时再对比低速新增通过seed；保留两模型，不立即追加训练或放宽稳定判据。
[本地SSH读取入口](artifacts/collect_phase3_comparison_20261115.sh)已准备，
[离线连线/中断保留核验](diagnostics/20261115_phase3_collection_validation.json)通过，实际下载待用户执行。
该入口只在服务器运行tar读取证据，压缩流写入本地新文件；不装环境、不读模型权重、不启动仿真或更新源码。

## 2026-11-15：两模型原始配对证据已下载，支撑连续性退步离线核验

用户回传下载完成，本地实际读取归档
`diagnostics/20261115_model2197_comparison_remote/remote_evidence_kO71wR.tar.gz`，
SHA256=`501d3b382997b001c98eb8e11a3a24d0670392c497697526852848eb5d4e8aaf`。
共47个成员，含两模型六组full报告/trace/配置、组内汇总、新模型comparison、两份checkpoint.sha256及新训练params。
原归档只读，不解压覆盖源码或改写历史报告。

- 父模型SHA：`7780cf0d2557c86f51c0486f870e7d471b13703777e9cee5e1854427def05027`。
- model2197 SHA：`b8bc52a8ee370eb94e359bc71d310f5f1cd0ec013cae9e5982b44fb13b8a54a4`。
- 各自三份report与SHA文本一致；权重文件未下载，不声称本地独立rehash权重。
- 六报告记录的14个源码hash均与本地一致，组内报告摘要hash和跨模型comparison重算匹配。
- 60个完整episode的实际reset状态/144维actor首帧配对最大差0，按原定义只排除vx输入。
- 33718帧按原XML仅做运动学重建，轮底与trace最大误差1.398631988e-7 m；
  root_out/front_up/rear_up/complete/stable_now、连续计时、通过锁存和速度RMSE全部复算匹配。
  控制dt=0.02 s，由保存配置0.005 s×decimation4核对；未启动仿真、推理或PPO。
- 保存训练params确认1024环境、rollout24、max_iterations1000、seed7、resume=true，
  load_run为父run、load_checkpoint为model_1198选择器；训练vx=0.3～0.6、11项奖励权重与六组评估一致。
  这些是保存配置证据，不替代完整训练stdout/Total timesteps，后者仍未下载。

[分析脚本](artifacts/analyze_phase3_paired_comparison_20261115.py)、
[派生JSON](diagnostics/20261115_model2197_comparison_analysis.json)、
[连续支撑对照图](diagnostics/20261115_model2197_comparison_analysis.png)已生成。
原始结果与上一条用户回传一致；本条将证据层级提升为原始报告/轨迹离线核验。

| 条件 | 父模型最大连续支撑 | model2197最大连续支撑 | 新模型首次完整出坑 | 新模型出坑后最大倾角 | 新模型出坑后缺完整支撑帧/总帧 |
| --- | --- | --- | --- | --- | --- |
| vx=0.30，seed11 | 0.52 s | 0.48 s | 6.72 s | 1.865° | 59/322 |
| vx=0.50，seed11 | 2.28 s | 0.34 s | 3.96 s | 3.160° | 35/195 |
| vx=0.50，seed13 | 2.82 s | 0.28 s | 4.32 s | 3.249° | 39/196 |

三个新模型未通过样本在首次complete后，四轮几何出坑/本地块内/轮底高度均保持合格，
轮底距台面最大绝对值分别1.608/1.568/1.605 mm；有接触时法向、力阈及接触点高度均合格。
全部59/35/39个中断帧都对应至少一轮found=0且force=0。
按RF/LF/RH/LH排列的缺失接触帧分别为24/8/27/5、17/0/14/8、15/7/14/3，允许同帧多轮缺失。
单轮最长连续缺失为1～2个控制帧（20～40 ms）；中断反复清零保持计时。
两高速父样本同一阶段仅3/2帧缺完整支撑；新模型右前/右后轮的接触间歇更频繁。
三个新样本从首次complete到结束约6.4/3.9/3.9 s，最终root相对x约4 m触发地块截断，
不是只剩不足0.5 s可用时间；无翻倒/failed，选中退步条件均无力矩饱和。
目标越硬限在父模型同条件也存在，不能由这些相关统计认定其为失败原因。

两模型均有RF/LF_thigh_collision2对地形的非轮接触：
出坑后力范数>1 N的任一非轮接触帧，新模型为138/322、66/195、47/196，
父模型为121/278、37/180、43/181。前大腿box最低点重建也接近台面，说明不能忽略非轮接触。
但它不是新模型独有；高速seed13占比接近，且多数轮接触中断并未与非轮接触出现在同一控制帧，
不能直接证明“大腿承重导致轮子卸载”。非轮传感器reduce=none、global_frame=false，
force在接触坐标系，只报告旋转不变范数，不把其第3分量误作世界竖直力或据此计算承重比例。

结论：低速爬出/跟踪改善已确认；三个退步样本的直接不通过项是四轮有效支撑连续性，
并非倾角超45°、几何未出坑或没有足够评估时间。真实动力学根因仍待确认：
20 ms控制帧中的found=0/force=0不能独立区分真实短暂卸载、传感/碰撞处理或接触求解现象。
保持两模型和原0.5 s判据；下一批建议两模型×上述三个条件小规模配对视频/5 ms子步接触回放，
核对姿态与非轮接触分担，再决定训练调整。回放入口尚未实现或执行，不记作已完成实验。
本批分析脚本Ruff/格式/ty/Pyright定向检查通过；不涉及src/奖励/资产修改或第三轮训练。
