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

## 2026-11-15：用户Viser录屏反馈与奖励v2本地软件核验（尚未训练）

用户提供[录屏](</home/asus/视频/录屏/录屏 2026年11月15日 22时17分45秒.webm>)，
时长23.242798 s，SHA256=`8162a9fe50aa7a17999c0d7eabc23a4ed4ad9619f90deddc2745fbd790e1a7d2`。
用户评价：坑底步态基本正常，前腿姿态基本正确；前轮上台后后腿借关节低位爬上，
出坑后机身/关节离地过低；建议修改奖励，并回传初版已commit/push至GitHub保留。
本批开始时本地HEAD和origin/main跟踪引用均为`63f293a`，工作区干净；未独立联网核实GitHub。

Codex抽取录屏0/2/4/…/22 s帧并查看16 s放大帧，可见低机身和后腿低位过台沿的动作。
录像本身不能量出接触力或证明“为了水平奖励才卡台”，也未独立显示checkpoint完整路径/SHA；
结合此前用户执行的model2197/30 cm/0.50 m/s指令及已核验的非轮接触轨迹作解释，证据层级分开。
原v1 flat_orientation_l2为倾斜惩罚，非额外水平正奖励；25°纯pitch时加权速率约-0.0893。
原11项确实没有非轮接触和最低机身净空成本，不能只降低水平权重就宣称两处问题已解决。

本批新增独立奖励候选任务`Mjlab-Blind-Pit-Lego2-RewardV2`，日志名`lego2_blind_pit_reward_v2`，
原v1工厂/11项奖励完整保留。v2共13项：姿态项改为±25°pitch死区并保留侧倾，
新增非轮接触通道计数成本(-0.5，force norm>1 N)和局部平整地面最低root净空成本
(-2.0，0.34 m下限，扫描高差≤3 cm且全部射线有效时启用)。
0.22 m净空时该高度项加权速率约-0.2491，单个非轮接触通道为-0.5；RewardManager仍按dt积分。
这些是函数尺度示例，不是新策略实测奖励，也不是已验证最优参数。
完整设计/理由归DECISIONS；原通过判据、actor144/critic254、动作、课程、传感器、资产和PPO不变。

- 本地设备：可用内存约21 GiB，NVIDIA驱动不可用；只运行小CPU场景。
- [CPU测试](diagnostics/20261115_reward_v2_cpu_tests_passed.txt)：24 passed，10.28 s，
  4个既有ONNX弃用警告。含原19项及v2五项：配置隔离、俯仰/侧倾、局部净空/跨层/无效射线、
  非轮力范数筛选、真实两环境4步的奖励管理器/dt汇总与inference后partial reset。
- 新/改Python文件定向Ruff/格式/ty/Pyright通过；未声称全库旧错误已修复。
- 首次本地运行遇到home Warp缓存只读及一项terrain lambda身份比较，已使用可写缓存并改为比较实际配置；
  初次失败输出保留为`diagnostics/20261115_reward_v2_cpu_tests.txt`，最终通过输出另存。
- [配置/证据记录](diagnostics/20261115_reward_v2_validation.json)；
  [增量安装模拟核验](diagnostics/20261115_reward_v2_sync_validation.json)通过，验证备份、重复安装、
  未知源码/同名文件/损坏包拒绝，并保护旧模型/.venv/当前交接。
- 改前__init__/changelog归档`backups/20261115_reward_v2_before.tar.gz`及同名JSON；不创建Markdown备份。
- 新源码包`artifacts/lego2_reward_v2_20261115_v1.tar.gz`含8项，SHA256为
  `a11777d6efe5105aa1e03177f502b47eb0bbf09e5a26c977cd2ce4459d7aa0d9`，不含上下文/模型/日志。
  [安装入口](artifacts/apply_phase3_reward_v2_20261115.sh)与
  [独立软件核验入口](artifacts/check_phase3_reward_v2_20261115.sh)已准备；尚未远程执行。

本批没有PPO、v2 checkpoint、v2成功率或改善结论。原策略评估器仍选择v1；
v2策略复评/训练入口及是否迁移model2197尚待下一批设计，不能直接跑旧hash入口或把旧策略回放算作v2效果。
先SSH同步并在screen zrq做CPU/GPU软件核验，回传后再安排独立短训/同条件复评；不自动commit/push。

## 2026-11-16：奖励v2远程核验部分回传，最终测试统计待确认

用户贴出部分源码/测试文件hash成功，以及含原Blind-Pit、RewardV2的17项任务列表。
内存输出MemAvailable=39907 MiB；片段截至list-envs，未含CPU/GPU pytest汇总、CUDA资源或最终证据目录。
只确认该片段，不把注册成功当成仿真/奖励管理器核验通过，不据此启动PPO或声称v2效果。
当前等待原命令结束后的最终回执；无新的本地源码修改、仿真、训练或Git操作。

## 2026-11-16：更正远程核验状态，list-envs退出码导致提前终止

用户明确原命令已结束，上一条“等待原命令结束”的状态不成立。
本地复现`uv run --no-sync list-envs`打印17个任务后退出17；
`uv run --no-sync python -m mjlab.scripts.list_envs`输出相同、退出0。
list_environments返回任务数，console入口将它传给sys.exit；原核验脚本set -euo pipefail因此停止，CPU/GPU未进入。
不是仿真/奖励测试失败；先前让用户继续等待的判断已纠正。

新增[补验脚本](artifacts/resume_phase3_reward_v2_checks_20261116.sh)及同名.sha256，
使用模块入口、显式阶段进度及ERR trap报告行号/退出码/证据目录。
不改共享list-envs源码、不覆盖旧包或历史hash清单；新入口仍核对原43项文件，在独立新目录补验。
[本地实际CPU运行](diagnostics/20261116_reward_v2_resume_cpu.txt)退出0：24 passed，29.08 s，4个既有ONNX警告。
[控制流核验](diagnostics/20261116_reward_v2_resume_validation.json)验证模拟GPU路径和CPU/GPU故意失败时保留退出码并停止；
模拟GPU不记为真实GPU通过。本机未执行GPU，远程补验待用户运行；无PPO或Git操作。

## 2026-11-16：奖励v2远程CPU/GPU补验完整通过（用户回传）

用户回传CPU `24 passed, 4 warnings in 40.63s`，GPU `1 passed, 4 deselected in 10.15s`，
以及补验结束行。4个警告来自既有TorchScript/ONNX导出弃用提示，不是测试失败。
证据目录：`/home/ajifang/zrq/backups/reward_v2_resume_20261116_v4TczG`。
[结构化回执](diagnostics/20261116_reward_v2_remote_receipt.json)仅转录用户输出；原始文件未下载。

2026-11-16 10:56:18截图为RTX4090、驱动550.144.03；资源检查报告空闲20956 MiB。
另有PID3324546（`/home/ajifang/yhy/mjlab/.venv/bin/python3`）占2846 MiB、截图GPU利用率68%；
不能把该任务认作本项目训练，也不终止它。后续短训前需重新检查资源，截图不保证届时余量。

结论为候选奖励/观测/推理reset的软件核验通过；没有PPO更新、v2模型或动作改善/成功率结论。
下一步准备独立迁移短训和同条件策略评估；拟仅载model2197 actor及归一化，critic/优化器/迭代/课程新建，
入口尚未实现/执行，不能直接运行旧v1全状态续训脚本。此次仅核对了相关runner加载实现，未启动训练。

## 2026-11-16：v2迁移短训与质量复评入口本地核验完成（远程尚未执行）

用户明确继续后，新增actor-only迁移审计、独立训练入口、v2评估选择及配对质量汇总；原v1默认评估保留。
实际迁移边界/预算归DECISIONS本日新条目；本批仅实现并本地核验，不预填远程run或新checkpoint路径。

- [本地集成回归](diagnostics/20261116_v2_warmstart_cpu_tests.txt)：26 passed，12.41 s，10个既有ONNX类警告。
- [最终专项](diagnostics/20261116_v2_warmstart_verified_tests.txt)：3 passed，4.13 s，6个ONNX类警告。
  涵盖actor/归一化/分布完整继承、旧critic/优化器/迭代/env计数不恢复、输出目录防覆盖、
  两CPU环境的单次PPO更新（48样本）、保存/新策略加载、v2 capped评估n=0及故意错配初态被拒绝。
  使用本地合成checkpoint，非真实model2197，不能记为已完成真实v2训练或策略效果。
- 本机约23 GiB可用内存、GPU驱动不可用；没有本机CUDA训练。5个新/改Python文件定向格式/Ruff/ty/Pyright通过。
- CLI预检--validate-only及多高度/配对seed参数真实解析通过。
- 变更后专项曾暴露已有输出目录检查晚于CUDA资源检查的问题；已前置目录拒绝，修复后3项通过，失败输出保留。
- [交付参数/限制](diagnostics/20261116_v2_trial_validation.json)及
  [安装/脚本模拟](diagnostics/20261116_v2_trial_launcher_validation.json)：旧评估器备份、旧模型/.venv/交接保留、
  重复安装、check-only/完整成功流程、CPU失败时停止均通过；模拟CUDA和训练不计为真实实验。
- 旧评估器/changelog归档`backups/20261116_v2_warmstart_before.tar.gz`及同名JSON。
- 正式增量包`artifacts/lego2_reward_v2_trial_20261116_v2.tar.gz`，9项文件、50项来源hash，SHA256为
  `b593ff441092af184e7b7a7961c4e1507ea844265f0aeef3577005e593e024d3`。
  本地trial_v1为中间开发包，未交付，正式使用trial_v2；均无上下文Markdown快照/模型/训练日志。

[运行入口](artifacts/run_phase3_reward_v2_trial_20261116.sh)计划：新工具CPU3/GPU2检查、
真实model2197只迁移预检、新run微调512×300、18个父/新策略配对episode及两个速度比较JSON。
计划真实父SHA为`b8bc52a8ee370eb94e359bc71d310f5f1cd0ec013cae9e5982b44fb13b8a54a4`；
expected model_299.pt仅是300次新迭代的计划选择器，真实文件待远程执行后回传。
本地未访问服务器、未commit/push、未覆盖任何旧训练结果；尚无v2动作改善结论。

## 2026-11-16：v2真实迁移短训与配对结束回传，低速30 cm初筛明显退步

来源为用户粘贴comparison_0.3片段及trial结束行；原始文件尚未下载，不冒充本地重算。
远程证据：`/home/ajifang/zrq/backups/reward_v2_trial_20261116_AuHs5A`。
模型：`logs/rsl_rl/lego2_blind_pit_reward_v2/20261116_121444_3330412_actor2197_512x300/model_299.pt`。
候选SHA=`cb06e04dd5528ada05b730b687444ab82a0f0a03ca26c9b5f2e94a0005ea742c`；
父2197 SHA=`b8bc52a8ee370eb94e359bc71d310f5f1cd0ec013cae9e5982b44fb13b8a54a4`。
当前只有回传文本中的hash，未下载权重独立rehash。

| 30 cm、0.3 m/s（各3回合） | 父2197 | v2候选299 |
| --- | --- | --- |
| 稳定通过 | 1/3 | 0/3 |
| 曾完整出坑 | 3/3 | 0/3 |
| failed | 0/3 | 0/3 |
| mean speed RMSE (m/s) | 0.025173065371836684 | 0.25235855734231155 |
| 净空有效帧/所有帧 | 0.5824337585868499 | 0.12533333333333332 |
| 低净空帧/有效帧 | 0.7076663858466723 | 0.0 |
| 首次complete后非轮接触帧占比 | 0.4045881126173097 | null |
| 首次complete后最低root-台面高度 (m) | 0.21527396142482758 | null |

[结构化回传](diagnostics/20261116_reward_v2_trial_remote_receipt.json)保存所贴数值及输入report哈希。
用户报告流程结束，模型299实际路径已回传；512×300/rollout24/seed7及具体更新量仍需保存params/stdout核对。
结束行对应脚本已完成两组比较，但用户仅贴0.3内容，不推测0.5下3/30 cm表现。
候选未出坑，post-complete的null是无适用数据，不是零接触；低净空0仅适用12.53%有效帧。
相比父模型，当前model299在此条件越障与速度保持较差，不能认定质量改善；
用户随后指出新奖励训练仅300次，本条不作为v2充分训练后退化的结论。
奖励组合、critic/优化器重置、学习率/预算/课程等共同变化，当前数据不能隔离单项原因。
本条各3回合且含挑选的困难seed，不同于此前10回合结果，亦非正式验收。

新增[本地只读SSH收集入口](artifacts/collect_reward_v2_trial_20261116.sh)：
白名单读取报告、trace、训练stdout、source清单、软件检查结果、迁移审计及params；
不读取模型或缓存，不写远程文件，下载中断保留partial，重跑使用新文件。
本地bash语法、模拟SSH重复下载/失败保留与上下文检查通过，
见[收集工具核验](diagnostics/20261116_reward_v2_collection_validation.json)，没有实际访问服务器。
下载待用户执行；下一步先核对完整证据与3 cm能力/30 cm停滞阶段，再决定是否调整。
本批不修改奖励/PPO，不追加训练，不删除或替换父/候选模型，不commit/push。

## 2026-11-16：用户要求延长v2训练，续训入口本地验证完成（计划，未远程执行）

用户希望先与父模型相近训练轮次再比较，已撤销此前“先不追加训练”的临时安排。
新增continue_reward_v2_2200_20261116.sh及同名.sha256，直接SCP至~/zrq/，无需同步本地注释源码。
从回传的model299及其SHA启动完整状态续训，新run再更新1900次，1024环境/rollout24/seed7；
预计标签model2198，但不是实际完成记录。计划v2累计2200次、50380800新奖励样本。
父原预算约49766400样本；两者不含义为同起点消融，原因与加载边界归DECISIONS本日条目。

[本地验证](diagnostics/20261116_v2_continue_validation.json)和
[CPU运行输出](diagnostics/20261116_v2_continue_cpu_resume.txt)：
真实Tyro解析1900/1024/resume/load selectors/upload false；合成2环境更新1次后保存，
新4环境完整恢复actor/critic/归一化/优化器/实际学习率/迭代/env计数，再更新2次通过。
测试中手动设迭代299仅核对加载/循环编号，不是实际model299或远程效果。
新环境课程为0，物理状态不恢复；本机可用内存约23 GiB、GPU驱动不可用。
模拟shell完整成功/预检仅检查/预检失败/训练失败通过，错误时不启动后续评估，旧run保留。
启动入口校验冻结源码清单及父/候选SHA、资源、非空优化器及7200控制步计数。
本批无真实远程训练或新checkpoint结果、无奖励/PPO实现修改，无commit/push。

## 2026-11-16：v2续训启动在自备份失败，未进入训练

用户回传：脚本.sha256检查成功，证据目录
`/home/ajifang/zrq/backups/reward_v2_continue_20261116_zNd3XT`，
line26的cp报找不到相对continue_reward_v2_2200_20261116.sh，exit1。
源码流程显示先cd项目再读取相对BASH_SOURCE，故失败发生在来源/资源校验和train命令之前。
没有新增PPO、checkpoint或训练效果；失败目录与model299均保留。
[回执](diagnostics/20261116_v2_continue_launch_failure_receipt.json)转录用户输出。
本地修为cd前解析绝对路径；[路径及整理回归](diagnostics/20261116_remote_upload_paths_validation.json)
覆盖3种启动路径、预览/重复/冲突/备份，不执行真实GPU或访问远程。
用户新目录约定归AGENTS/PROJECT_CONTEXT/DECISIONS；修正入口待同步重试，不记为续训完成。

## 2026-11-17：用户回传v2续训结束，30 cm低速三回合质量初筛改善

实际run时间在2026-11-16；本条为2026-11-17用户粘贴的comparison_0.3片段及完整流程结束行。
证据目录：`/home/ajifang/zrq/backups/reward_v2_continue_20261116_scKcS4`。
实际模型：`logs/rsl_rl/lego2_blind_pit_reward_v2/2026-11-16_22-25-03_v2_continue_1024x1900_20261116_222451_3365307/model_2198.pt`。
新模型SHA=`27c1bdf3a05fa3455a16932dd4d5493fef526680575caee01e31a5c0d169248c`；
父2197 SHA=`b8bc52a8ee370eb94e359bc71d310f5f1cd0ec013cae9e5982b44fb13b8a54a4`。
[结构化回执](diagnostics/20261117_reward_v2_continue_remote_receipt.json)记录报告hash及全部所贴数值。
原始报告、轨迹、训练stdout和params未下载，权重未本地独立rehash，不冒充离线已复算。

| 30 cm、0.3 m/s，各3回合 | 本轮父2197 | v2候选2198 |
| --- | --- | --- |
| 稳定通过 | 2/3 | 3/3 |
| 曾完整出坑 | 3/3 | 3/3 |
| failed | 0 | 0 |
| 平均速度RMSE (m/s) | 0.02529424194315438 | 0.029421547741987963 |
| 净空有效帧占比 | 0.5828824397442204 | 0.6022889842632332 |
| 低净空帧/有效帧 | 0.7071729957805907 | 0.007125890736342043 |
| 首次complete后非轮接触帧占比 | 0.3928944618599791 | 0.0 |
| 首次complete后最低root-台面高度 (m) | 0.21599851548671722 | 0.3385159373283386 |

本条件出坑后非轮接触与低净空方向改善明显，且高度判断覆盖没有减少；
速度误差略增约0.00413 m/s，不能称所有指标均提高。低净空比例是在全回合有效扫描帧内计算，
与首次complete后的接触/最低高度不是同一窗口；零接触仅指所记录帧且传感力范数>1 N的判据。
无法由出坑后零接触推断后腿爬台过程完全无刮擦，root高度也不是每个关节净空。
0.5 m/s两条件尚未贴出；三回合仍是初筛，不能据此宣告正式成功率或泛化验收。

同父模型SHA在前trial为1/3，本轮2/3，RMSE也略变；需核对两轮实际初态、来源和接触轨迹。
不把预设seed相同当动力学确定性证明，不先断言GPU非确定性或硬件/其他进程是唯一原因。
用户要求充分训练后再比较是合理的实验顺序；原model299短训较差不决定本次结果。
续训计划新增1900次、1024×24，v2新奖励累计2200次/50380800样本仍需stdout核对，
这不是同起点奖励消融。阶段4未触发，不新增奖励或PPO训练。

新增[本地只读收集入口](artifacts/collect_reward_v2_continue_20261117.sh)同时读取两轮证据，
不读模型/缓存、不上传脚本或写远程，不覆盖旧下载，中断保留partial。
[本地模拟核验](diagnostics/20261117_v2_continue_collection_validation.json)通过；实际下载待用户运行。
本批仅记录/解释回传并准备证据收集，无源码/奖励/资产修改、仿真、PPO或Git提交/推送。

## 2026-11-17：v2两轮证据已下载，报告/配对/训练量核对，准备Viser观察

归档`diagnostics/20261117_reward_v2_continue_remote/remote_evidence_tCcuUA.tar.gz`，
SHA256=`775e111e57cf2e9296578f46eadcd058868059b52bccf9c709065bd80b248f4f`；
98成员、85文件、143172244字节未压缩文件内容，不含模型权重。归档只读，不解压覆盖源码。
[离线报告核对](diagnostics/20261117_v2_downloaded_reports_analysis.json)：
两轮四份comparison的输入report hash核对、真实reset配对（1e-6）、汇总重算均通过。
报告记录的源码hash与对应归档冻结清单一致；不是对服务器当前源码重新读取，也没有权重独立rehash。

| 当前配对，各3回合 | 父2197通过 | v2模型2198通过 | v2低净空/有效帧 | v2首次complete后非轮接触帧占比 | v2首次complete后最低root-台面 |
| --- | --- | --- | --- | --- | --- |
| 3 cm、0.5 m/s | 3/3 | 3/3 | 0.7203166226912929 | 0 | 0.3249231278896332 m |
| 30 cm、0.5 m/s | 1/3 | 3/3 | 0.7210526315789474 | 0 | 0.3255670368671417 m |
| 30 cm、0.3 m/s | 2/3 | 3/3 | 0.007125890736342043 | 0 | 0.3385159373283386 m |

本轮v2共9/9，均complete、failed=0；小样本仍非正式验收。
高速低净空比例并未降到低速的0.71%，只是最低root高度从父约0.22 m提高到约0.325 m。
三条件速度RMSE分别父/新0.0484807339/0.0422774391、0.0319828093/0.0364196264、
0.0252942419/0.0294215477；不能概括为所有速度误差改善。
同父跨轮同高度/速度实际初态最大差0、来源hash相同，高台0.3通过1→2、0.5通过2→1；
说明当前重复结果有波动，不证明唯一硬件/求解根因，需后续逐帧核查。

stdout实际记录trial为300条Learning iteration（0～299）、Total steps=3686400，
续训1900条（299～2198）、Total steps=46694400；合计2200次/50380800样本由下载日志支持。
恢复预检记录父299学习率5.0625e-5、env控制步7200；初始继承v1 actor，仍非同起点奖励消融。
本批仅完成报告/日志核对，尚未逐帧重建新的trace动力学，也未本地加载真实策略。

用户要求Viser回放。已核对进程内临时任务：v2固定0.30 m、vx0.50 m/s、1环境、seed7、
auto_reset=True、课程关闭；PlayConfig加载实际model2198，viewer=viser，无PPO/录像写入。
[入口核验](diagnostics/20261117_v2_play_entry_validation.json)通过真实注册/配置/API；
默认8080，实际监听以远程启动日志为准。本批未本地启动仿真/网页，也未远程执行回放。
给用户SSH端口转发与临时Python指令；无.sh上传、无源码/模型修改，不新增训练。

## 2026-11-17：新录屏三处姿态反馈核查及后续路线建议（未修改奖励/未开训）

用户提供[20.224607 s录屏](</home/asus/视频/录屏/录屏 2026年11月17日 17时21分06秒.webm>)，
SHA256=`85f7ebfdd3f47ee5c3de8a47d41835943b614beb912335791efcd67d15d53543`，2618×1782、VP8；原文件只读保留。
反馈：起步右后腿抬起后回落、前腿上台后右前腿更前伸、出坑后前后腿构型不同但机身水平。
本地抽取0～20 s的17张概览帧及3个阶段放大帧，
[关键帧图](diagnostics/20261117_v2_video_keyframes.jpg)保留4/14/20 s的画面。
画面可确认后腿短暂抬落（约录像3～5 s）、前腿左右伸展不同（12～16 s）、末段前后构型不同（18～20 s）。
画面无腿名/模型SHA：来源结合前述2198/30 cm/.5命令理解，不能由像素独立确认RF/RH标签或策略hash。
20 s抽帧的界面约222步，约4.44 s仿真；不是20 s平地稳定运行，也没有充分的出坑后长窗口。

[结构化核查](diagnostics/20261117_v2_video_posture_review.json)记录原视频、归档及XML hash、
观测/方法/限制和18回合的派生指标。仅MuJoCo运动学，不推进动力学或加载策略。
XML hash与归档冻结清单一致，四轮反算台面一致性误差≤1.495e-7 m。
按名称匹配RF/LF/RH/LH、关节及几何，比较首次complete至少1 s后的窗口，而非台沿瞬时构型。

| 候选已保存轨迹 | RF与LF轮相对各自髋部的机身前向偏移差（每回合均值） | 迟后窗口长度 | 硬限余量最小 |
| --- | --- | --- | --- |
| 3 cm，0.5 m/s，3回合 | 7.927～7.946 cm，RF更前 | 3.44～3.54 s | 约14.16° |
| 30 cm，0.5 m/s，3回合 | 7.924～7.929 cm，RF更前 | 3.52～3.58 s | 约14.18° |
| 30 cm，0.3 m/s，3回合 | 5.982～7.018 cm，RF更前 | 6.18～7.08 s | 约15.99° |

0.5 m/s六回合迟后窗口共1059帧，非轮接触/电机饱和标记均0，
非轮碰撞几何底部相对延伸台面平面最低约3.75 cm，KFE关节中心最低约14.79 cm。
几何体与关节中心不是同一指标；延伸平面不是精确台沿距离查询，更不是硬件安全认证。
候选9回合首次front_up至首次complete的选取窗口非轮接触标记均0，仍可能漏掉20 ms之间的接触。
视频与归档是不同episode，不把这些数值冒充录像对应帧的精确量值。
轨迹支持前腿不对称持续存在，而不是相机角度或单次过沿的唯一现象。

源配置核对：前腿HFE/KFE默认+0.72/-1.44、后腿-0.72/+1.44，原始角度不能直接设为相等。
v2只管速度、机身姿态/最低高度/接触/平滑和代价，没有明确的腿构型或左右一致性目标。
打印Mean reward分段均值500～999约39.92、1000～1499约40.78、1500～1999约40.42、末199轮约40.30；
这是混合课程日志，既不等于配对通过率，也不能据此证明严格收敛。
但目前没有依据指望单纯增加轮次自动消除未被明确区分的姿态偏置。

建议先小幅补目标，再长训：起步抬腿暂不加惩罚；第2/3点优先考虑仅局部平地、直行、
稳定窗口启用的一项弱默认姿态回归，保留过沿/转向自由，不全程强制四腿同角或四轮贴地。
保留原v2/2198，新候选需独立run并设计加载边界，先有限微调/配对，确认通过/接触/净空不损失后再长训。
如果用户只优先验证v2能力上限，可选择原奖励新增1000、+500/+1000检查的受控对照；不承诺自然变对称。
以上是待用户选择的建议，未改src/奖励/PPO、未新增远程开训脚本、未启动训练，不把建议当已采纳决策。
