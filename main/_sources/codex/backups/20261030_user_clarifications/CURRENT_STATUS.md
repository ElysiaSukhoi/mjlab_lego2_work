# 当前状态

更新时间：2026-10-29（Asia/Shanghai）

## 状态结论

2026-10-29 用户已要求制定大体修改规划；**规划已完成，未实施任何训练代码改动，也未启动训练**。
用户确认 URDF/资产纠偏已完成且远程已同步。已查看全量训练回放视频，用户认为效果不理想。
回放对应的实际 checkpoint、训练/回放资产版本与定量成功率尚待确认，不能将新资产回放旧策略
的现象单独归因于奖励。

本轮已完成对本地 Dreamwaq-highplatform 和新增三个 GitHub 轮足项目的只读研究。
报告位于 `/home/asus/work/Jiche_Project/高台训练/2026-10-29_盲走轮足高台开源对比报告.md`，
下载来源清单位于 `/home/asus/work/Jiche_Project/OpenSource/SOURCES_20261029.md`。
旧中文项目根目录已迁移到 `/home/asus/work/Jiche_Project`，用户输入 Jicha_Project 是拼写差异。

当前建议方向（尚未实施）：中央单级坑、从开始启用低高度课程、速度跟踪主导、撤销固定
地图位置 shaping；先明确盲走 actor 与本体历史，critic/奖励可使用训练特权信息。
本轮形成五阶段规划，见 [REFACTOR_PLAN.md](REFACTOR_PLAN.md)：独立盲走任务与历史→中央坑及从头高度课程→精简奖励/评估与首轮短训→可选通用接触反应→完整训练与独立高台评估。第4阶段不是必做；当前只是规划，不按旧交接直接同步奖励或开新训练。

## 研究纠正与重要证据

- 当前本地 actor 实际仍有 height_scan、无历史 MLP，与盲走目标不符；不是已完成的盲走模型。
- platform_front_lift 读轮心到地面高度，正常平地接触约0.115m也可产生正值；此前不能据其
  上升断言主动抬腿。contact_advance 是位置值乘接触比例，静止也可能得分。
- Dreamwaq 默认M20确为本体历史/VAE盲走，中央4×4m坑、40级0.20～0.98m、初始化随机
  0～20级，距离地块中心>4m升级；速度跟踪主导，但有高台yaw专项项和特定命令/终止。
- Dreamwaq没有正z/抬腿奖励，其lin_vel_z仍是-2；接触触发上抬是候选设计，不是该源码已有项。
- 新下载 LocoWheeledLegged、官方 DeepRobotics_rl_training、robot_lab 是通用轮足对照，
  未将其当成已证实80cm单台阶项目。论文CTBC/MUJICA未定位完整训练代码，只作论文参考。

2026-10-28 曾指定回放 run `platform_climb_front_lift_full_150mm_2048_v1`；本次视频是否
精确对应该run、日期目录和model编号仍待确认。不通过run名称猜测参数。

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

## 最新本地代码调整

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

1. 按 `REFACTOR_PLAN.md` 的五阶段框架继续讨论；用户要求实施时从阶段1开始。
2. 前三阶段形成可训练的最小基线，首轮正式短训即启用高度课程；不等可选接触反应做完才训。
3. 起始坑深3cm、上限30cm、10级、最低级初始化、约5帧历史等为候选参数，未实施或验证。
4. 本轮仅改文档。未来阶段仍由Codex本地uv校验，用户执行远程同步与校验，训练使用screen zrq。
5. 视频对应checkpoint/实际训练参数/资产版本仍待确认，不阻止制定新方案，但不猜测旧实验归属。

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
