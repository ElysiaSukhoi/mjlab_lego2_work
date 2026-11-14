# 当前状态

资料位置与关联见[统一资料索引](README.md)；当前交接入口见[HANDOFF.md](HANDOFF.md)。
截至本轮结束：正式计划完成、阶段0待实施；无新任务注册和新训练。

更新时间：2026-10-30（Asia/Shanghai）

## 状态结论

2026-10-30已按用户要求完成本地接口核查和正式修改计划v2。**只更新文档，未实施训练代码、
未注册新任务、未启动训练；不存在必须先补交源码的阻塞项。** 下一次实施从阶段0诊断开始。
用户确认 URDF/资产纠偏已完成且远程已同步。2026-10-30进一步确认：视频来自全量训练
编号9999的checkpoint；多次重置均未通过，每次碰台后前轮无法抬起、后腿反而抬起。
这是用户重复回放确认的失败模式，不再仅记为短视频效果不理想。重置总次数、checkpoint
完整路径、训练/回放资产版本仍待确认，不据此给出精确成功率或单一失败原因。

2026-10-29已完成对本地Dreamwaq-highplatform和三个GitHub轮足项目的只读研究。
研究见[交叉对比报告](/home/asus/work/Jiche_Project/高台训练/2026-10-29_盲走轮足高台开源对比报告.md)，
下载来源与提交见[源码来源清单](/home/asus/work/Jiche_Project/OpenSource/SOURCES_20261029.md)。
旧中文项目根目录已迁移到 `/home/asus/work/Jiche_Project`，用户输入 Jicha_Project 是拼写差异。

当前建议方向（尚未实施）：中央单级坑、从开始启用低高度课程、速度跟踪主导、撤销固定
地图位置 shaping；先明确盲走 actor 与本体历史，critic/奖励可使用训练特权信息。
正式计划见 [REFACTOR_PLAN.md](REFACTOR_PLAN.md)：0动作/接触前置诊断→1独立盲走任务与
原生历史→2坑/出生/指令/初始高度课程→3通用奖励与评估/首次短训→4条件性接触反应→
5完整训练与独立高台评估。阶段4不是必做。阶段1、2逐批验收，阶段3的配置/功能检查
通过后在阶段3开始首轮短训；无需等阶段4/5。首轮训练就带课程。
本轮接口证据见 [IMPLEMENTATION_READINESS_20261030.md](IMPLEMENTATION_READINESS_20261030.md)。

## 研究纠正与重要证据

- 用户已澄清：原始Lego2配置不向actor提供height_scan；后来由用户注释掉删除语句，
  才在该修改版本中保留扫描。不能把当前状态写成原始项目默认开启扫描。当前本地actor
  有height_scan、无历史MLP；未来盲走配置应恢复actor不读扫描的约束。
- platform_front_lift 读轮心到地面高度，正常平地接触约0.115m也可产生正值；此前不能据其
  上升断言主动抬腿。contact_advance 是位置值乘接触比例，静止也可能得分。
- Dreamwaq 默认M20确为本体历史/VAE盲走，中央4×4m坑、40级0.20～0.98m、初始化随机
  0～20级，距离地块中心>4m升级；速度跟踪主导，但有高台yaw专项项和特定命令/终止。
- Dreamwaq没有正z/抬腿奖励，其lin_vel_z仍是-2；接触触发上抬是候选设计，不是该源码已有项。
- 新下载 LocoWheeledLegged、官方 DeepRobotics_rl_training、robot_lab 是通用轮足对照，
  未将其当成已证实80cm单台阶项目。论文CTBC/MUJICA未定位完整训练代码，只作论文参考。

2026-10-28曾指定回放run `platform_climb_front_lift_full_150mm_2048_v1`。用户现已确认
视频来自全量训练checkpoint编号9999；其与上述run的完整日期目录、实际参数和资产版本
仍需文件核对，不通过run名称猜测环境数、源码版本或训练样本量。

## 机器人资产状态（用户已确认完成远程同步）

- 已读取 URDF、训练实际加载的 MJCF 和 STL。修复前两种资产与只读远程快照一致。
- 已在本地 URDF/MJCF 同步校正 4 个轮子、4 个髋部的圆柱近似惯量轴向 Z→Y；
  轮子质心按既有圆柱几何移到 RF/RH `(0,-0.062,0)`、LF/LH `(0,0.062,0)`。
- 本次是沿用原均匀圆柱近似使坐标和几何自洽，不是实物 CAD 标定。
- 大腿/小腿质心均在关节原点、四腿惯性积完全复制；是否正确需要 CAD 质量属性。
  未凭外观将腿部张量对角化、强制轴对称或猜测其质心。
- URDF 雷达 `2 kg`/零惯量，与 MJCF 质量汇总相差 `2 kg`，是否被 base 吸收待确认。
- 总质量 `49.176 kg`、关节、动作和几何未改，但共享资产修改影响 Rough/Flat/Platform-Climb。
- CPU 验证工具、Entity 编译测试和 20 个配置质量矩阵检查通过；没有 GPU 长训。
  远程资产同步由用户确认完成；动态接触核查和修复后训练版本对应关系仍待确认。

详细证据、备份和公式见 [INERTIA_AUDIT_20261027.md](INERTIA_AUDIT_20261027.md)。

## 本地工作副本

路径：`/home/asus/work/mjlab_lego2_work`

当前本地代码包含：

1. 独立的 `Mjlab-Platform-Climb-Lego2` 注册和 `lego2_platform_climb` 日志实验。
2. 固定单平台地形、确定性低地出生和正向速度指令。
3. `platform_approach`、`platform_edge_lift`、`platform_front_lift`。
4. 最新增加的 `platform_front_contact_advance`：权重 `1.5`，x 门控 `(-2.1,-1.0)`，按前足接触比例耦合继续前移。
5. 顶部几何/接触保持奖励与终止，以及 root x/z 和前足/总接触诊断指标。
6. Platform-Climb 的基础随机化关闭；共同接触容量为 `300/500/99`。
7. 当前固定平台高度为 `0.15 m`，低地根部参考约 `0.32 m`，顶部根部参考约 `0.47 m`。

本地已做过：指定文件 `compileall`、`list-envs`、配置加载、奖励/指标参数检查，以及合成位置/接触数值测试。GPU 可用性和 MuJoCo/Warp 导入也已确认。未在本地运行大规模长训练。

## 远程已确认

远程项目：`~/zrq/mjlab_lego2`。远程已经验证过任务注册、编译、平台指标/奖励/终止配置，并完成过短训练和诊断训练。最近一组 150 mm、加入前足抬升奖励的结果为：

```text
platform_root_x_max             = -1.6119
platform_root_z_max             =  0.3255
platform_front_contact_count_max=  2
platform_contact_count_max      =  3.1667
platform_edge_lift_mean         =  0.1048
platform_front_lift_mean        =  0.5188
platform_top_reached             = 0
platform_top_stable              = 0
platform_success                 = 0
```

解释边界：这些数值是训练日志中的聚合量，root x 最大值的统计距离边缘约 0.11 m，
不代表每个机器人都停在同一点。接触计数包含地面/平台相关接触，不能区分底面、
侧壁和台面；前足抬升信号也不能单独证明已经在台面支撑。此前的接触/动作判断是
工作假设，仍需回放和接触位置验证。上述结果均来自惯量修复前资产。

最新本地 `platform_front_contact_advance` 是否已同步远程、对应远程 run 名称和 checkpoint 路径：**待确认**。

## 旧高台奖励最近一次调整（非本轮重构；其后另有资产纠偏）

状态：**本地已验证，远程同步待确认，训练待验证**。

涉及文件：

- `src/mjlab/tasks/velocity/config/lego2/platform_climb_rewards.py`
- `src/mjlab/tasks/velocity/config/lego2/platform_climb_env_cfg.py`

新增 `platform_front_contact_advance` 奖励。它把 root x 在 `(-2.1,-1.0)` 区间内的前移进度与 RF/LF 前足接触比例相乘，权重为 `1.5`，用于解决原 `platform_edge_lift` 在 `x=-1.5` 处饱和后无法继续鼓励机身越过边缘的问题。该调整没有改变平台高度、顶部成功判定、全局姿态惩罚或接触容量配置。

本地验证已完成：指定文件编译、任务注册、配置加载、奖励参数检查和合成接触数值测试均通过。远程是否已经接收这两个文件尚未确认；新的 run 名称、迭代数、checkpoint 和训练指标也均待确认。

## 已确认的历史训练状态

- Rough V1：4096 环境、10000 iterations，checkpoint `logs/rsl_rl/lego2_velocity/rough/2026-10-12_20-28-29_high_platform_300mm_v1/model_9999.pt`；该训练不是独立 Platform-Climb 任务。
- 300 mm Platform-Climb 的 128 环境短训和续训均无 NaN/OOM，但没有顶部成功。
- 诊断数据显示 300 mm 时前足接触最多 2、root z 最大约 `0.1705`；因此先降到 150 mm 验证动作能力。
- 4096 环境的某次全量尝试在 CUDA Graph 创建阶段 OOM；该次具体显存快照和最终 checkpoint：**待确认**。后续应先用 2048，必要时 1024。

## 下一步

1. 用户要求开始实施后，先做阶段0：按名称核对足端/动作、关节符号与目标/力矩饱和，
   用前腿运动学和必要的单环境PD诊断排除基础问题；不得凭失败现象直接交换前后腿。
2. 阶段1恢复新任务actor盲走，使用原生5帧本体历史，动作/命令仅当前值；预计144维，
   不将此维数写成已运行的新任务事实。旧任务观测不变。
3. 阶段2接入10×10m地块/3×3m坑，3～30cm/10级、初始等级0、3m晋级及支撑/失败过滤；
   正向速度0.3～0.6m/s与heading反馈、20s episode、分地块边界一并核查。
4. 阶段3冻结通用速度奖励v1与固定高度评估后才首训，首训已启用课程；先小规模再扩大。
5. 本轮没有新训练。完整checkpoint路径/旧参数/资产对应/启动时显存为非阻塞待确认项。
   每批仍由Codex本地uv校验，用户执行远程同步/校验，训练使用screen zrq。

## 未解决问题

- 150 mm 到 300 mm 的可行动作策略尚未形成。
- 平台真实物理高度、机器人轮足的最大有效越障高度和最优接触力阈值：待通过源码/测量确认。
- 最新远程源码与本地工作副本的逐文件同步状态：待下一次远程校验确认。

## 新发现：EPA horizon warning

训练过程中观察到 `EPA horizon = 24 isn't large enough`。该信息来自 MuJoCo Warp
凸体碰撞检测中 EPA horizon 的边数容量上限，不是 EPA 迭代次数，也不是 PPO 的
`num_steps_per_env=24`（此前聊天把它称为迭代上限不准确）。warning 可能导致该次
碰撞未生成接触；出现频率、对应接触对、与训练失败及惯量问题的关系均待确认。

处理原则：若没有 NaN、CUDA OOM、机器人穿透地面/平台或明显接触爆炸，先记录并继续短训；若出现上述接触异常，再针对初始姿态、平台边缘接触和 MuJoCo-Warp 版本/上限进行单独诊断，不直接修改 PPO rollout 长度、`ccd_iterations` 或 `nconmax`。

## 正式计划准备阶段的已核查事实

- 本地框架自带term级历史和partial-reset回填，不需先重写历史框架或PPO。
- CPU射线核对候选3/15/30cm单级坑及坑底origin通过；5帧历史局部reset检查通过。
- 控制周期0.02s；动作scale为0.45/5，当前action与RL clip均未配置。
- curriculum先于metrics初始化，reset时又先于各manager清空；新增课程按此生命周期设计。
- out_of_terrain_bounds仅检查整网格；新任务应复用局部terrain_edge_reached能力防串地块。
- 上述是规划准备检查，不是阶段0～3验收完成。所有阶段仍未实施；v1奖励/课程参数待实验。
