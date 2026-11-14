# 当前状态

更新时间：2026-10-28（Asia/Shanghai）

## 状态结论

2026-10-28 用户要求暂不进行下一步训练，先通过浏览器 Viser 查看已有 checkpoint
的策略表现。当前阶段仅做策略推理和回放；用户尚未提供实际回放路径、远程资产版本
或观察结果。不要自行启动新的训练。

本轮明确指定的目标 run 为 `platform_climb_front_lift_full_150mm_2048_v1`。
回放时只创建 1 个环境，使用 `Mjlab-Platform-Climb-Lego2 --viewer viser`，
从目标 run 实际存在的 `model_*.pt` 中选择最大编号；具体日期目录、checkpoint
编号、与此前日志的对应关系、远程惯量是否已替换及回放观察结果仍待确认。

高台任务配置闭环已建立，但尚无已确认的稳定登顶结果。2026-10-27 发现共享机器人资产的圆柱惯量轴向与几何不一致，当前优先级改为资产核查与小规模动力学验证，暂不继续堆叠奖励或扩大训练。前足接触后前移奖励的远程结果仍待用户提供。

## 最新优先事项：机器人资产惯量

- 已读取 URDF、训练实际加载的 MJCF 和 STL。修复前两种资产与只读远程快照一致。
- 已在本地 URDF/MJCF 同步校正 4 个轮子、4 个髋部的圆柱近似惯量轴向 Z→Y；
  轮子质心按既有圆柱几何移到 RF/RH `(0,-0.062,0)`、LF/LH `(0,0.062,0)`。
- 本次是沿用原均匀圆柱近似使坐标和几何自洽，不是实物 CAD 标定。
- 大腿/小腿质心均在关节原点、四腿惯性积完全复制；是否正确需要 CAD 质量属性。
  未凭外观将腿部张量对角化、强制轴对称或猜测其质心。
- URDF 雷达 `2 kg`/零惯量，与 MJCF 质量汇总相差 `2 kg`，是否被 base 吸收待确认。
- 总质量 `49.176 kg`、关节、动作和几何未改，但共享资产修改影响 Rough/Flat/Platform-Climb。
- CPU 验证工具、Entity 编译测试和 20 个配置质量矩阵检查通过；没有 GPU 长训。
  远程同步/动态接触/修复后训练：**待确认**。

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

当前先执行用户要求的浏览器策略回放，后续训练安排暂缓。下面的资产/训练路线
属于回放后待讨论事项，不是本轮启动指令。

1. 先在远程备份 URDF/MJCF，再同步两个资产和 CPU 惯量校验工具，执行校验。
2. 用户补充腿部/轮足组件的 CAD 质量属性，核对真实质心、惯量参考系和雷达质量。
3. 修复后的共享资产先做单环境回放/小规模新 run；保持当前奖励不变、保留旧 checkpoint，
   使用 screen `zrq`，不将修复前后结果混为同一实验。
4. 收到最新训练结果后记录实际 run/参数/资产版本；再决定腿部修复、课程或奖励路线。

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
