# 训练实验记录

本文件只记录完成或明确中止的实验结论。没有可靠的 run 名称、checkpoint 或原始日志时标记为“待确认”。

资产版本说明（2026-10-27）：以下已有训练结果均基于圆柱惯量修复前的 MJCF。
本地现已修正 hip/FOOT 圆柱惯量轴向和 FOOT 近似质心，详见
[惯量核查报告](INERTIA_AUDIT_20261027.md)。修复后的训练尚未确认完成，
不能将 CPU 编译/数值验证记为登顶训练实验；后续记录应附资产版本与实际 run 参数。

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
- 结论：前足能接触台阶，但机身没有抬升；因此不能只继续增加接近样本。

### 150 mm、几何阈值修正后的诊断

- `platform_root_x_max=-1.8044`，`platform_root_z_max=0.3197`，前足接触最大 2、总接触最大 4。
- `platform_edge_lift=0`，顶部到达/稳定/成功均为 0。
- 结论：降低高度后仍未自动形成抬升动作，下一步增加局部引导。

### 150 mm、前足抬升奖励

- 修改：`platform_edge_lift` 低根部参考下移 0.05 m；新增 `platform_front_lift`，权重 `0.75`，只在 `x∈[-2.1,-1.5]` 使用 `foot_height_scan` 前足高度。
- 结果：`platform_approach_mean=0.8790`；`platform_edge_lift_mean=0.0007`；`platform_front_lift_mean=0.3268`；前足奖励均值约 `0.2268`；顶部到达、稳定和成功仍为 0。
- 结论：前足抬升被诱导，但机身仍未越过边缘。

### 150 mm、前足抬升后的较大规模运行

- 结果：`platform_root_x_max=-1.6119`，`platform_root_z_max=0.3255`，前足接触最大 2，总接触最大约 3.1667，`platform_edge_lift_mean=0.1048`，`platform_front_lift_mean=0.5188`；顶部到达、稳定和成功均为 0。
- 结论：前足已经抬升/接触平台，机身仍在边缘前约 0.11 m；下一步改为接触耦合的继续前移奖励。
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
  该run、实际checkpoint编号、环境数/iterations、训练和回放资产的对应仍待确认。
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

- 正式计划在REFACTOR_PLAN中；阶段0～5未实施，新任务尚未注册，无新run/checkpoint。
- 已只读核对本地注册、历史、动作、地形/课程、reset和奖励接口，无需用户补传源码。
- 用现有uv环境CPU检查候选3/15/30cm坑几何/原点、历史buffer局部reset回填，均通过。
  这些不属于PPO训练、完整环境验收或越障能力结果，未运行GPU训练。
- 保留checkpoint9999多次失败反馈与版本待确认信息；起始奖励表和课程参数仅为未验证提案。
