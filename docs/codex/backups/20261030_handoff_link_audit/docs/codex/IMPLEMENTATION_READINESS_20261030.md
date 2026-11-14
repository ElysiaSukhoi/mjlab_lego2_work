# 本地 mjlab 接口核查与计划就绪记录

日期：2026-10-30。本轮仅只读源码、构造内存配置、CPU几何/缓冲检查和文档更新；
没有新建训练任务、修改业务代码、运行GPU仿真或启动训练。
源码根目录：`/home/asus/work/mjlab_lego2_work`。

## 1. 是否还缺少必须补交的源码

不缺。当前本地已包含任务注册、机器人资产、动作处理、观测与历史、地形生成、
等级更新、reset、指令、奖励、训练和回放入口，足够制定正式实施计划。
不再让用户重复上传这些文件。

这不等于已确定前轮不抬的原因，或已验证新策略有效：实际动作跟踪与接触、
前腿动力学能力、旧checkpoint完整参数和本地/远程版本对应，仍需阶段0或后续运行核查。
未知项均不阻塞制定计划。

## 2. 已核实接口与含义

| 核查点 | 本地证据（相对源码根目录） | 结论 |
| --- | --- | --- |
| 任务注册/独立日志 | `src/mjlab/tasks/velocity/config/lego2/__init__.py`、`rl_cfg.py` | 可追加独立任务，原PPO工厂支持experiment_name参数 |
| 实际观测配置 | `config/lego2/env_cfgs.py:281`、`tasks/velocity/velocity_env_cfg.py` | 当前修改版本保留扫描；原始无扫描由用户确认，不反推原始默认 |
| 原生历史 | `src/mjlab/managers/observation_manager.py` | term/group均可设置history；group设置会覆盖term；flatten为term-major |
| 历史reset | `src/mjlab/utils/buffers/circular_buffer.py`、环境reset/compute路径 | 旧行清除后首帧回填；局部reset不推进其他环境历史 |
| 动作目标 | `src/mjlab/envs/mdp/actions/actions.py` | raw×scale＋offset，只有显式cfg.clip才裁剪 |
| RL动作裁剪 | `src/mjlab/rl/config.py`、`vecenv_wrapper.py` | 默认clip_actions=None，不能把0.45缩放说成最大角偏移 |
| 动作/传感器顺序 | `config/lego2/env_cfgs.py:114`、`sensor/contact_sensor.py` | 动作LF/RF/LH/RH，传感器声明RF/LF/RH/LH；应按实际名称映射，不能断言已接反 |
| 轮高语义 | `src/mjlab/sensor/terrain_height_sensor.py`、轮心sites | frame_z-hit_z；轮心离地不等于轮底净空 |
| 单级坑 | `src/mjlab/terrains/primitive_terrains.py:276` | 倒金字塔在阶数0时返回中央单级坑，origin在坑底 |
| 高度级别 | `src/mjlab/terrains/terrain_generator.py:265` | 难度按row/(rows-1)，10级覆盖两个端点，不照抄Dreamwaq的row/rows |
| 类型列数 | `terrain_generator.py` | curriculum模式列数等于sub_terrains数量，num_cols不控制实际列数 |
| 初始级别 | `src/mjlab/terrains/terrain_entity.py:325` | max_init_terrain_level=0可全部分配最低级 |
| 升降级 | `tasks/velocity/mdp/curriculums.py`、`terrain_entity.py` | 现有距离阈值硬编码半地块；可新增参数化课程复用origin更新，不改旧任务默认 |
| 初次reset | `terrain_levels_vel` | 已有common_step_counter==0保护，新增函数须保留这一语义 |
| reset顺序 | `src/mjlab/envs/manager_based_rl_env.py:_reset_idx` | curriculum在状态/统计清空前计算，再reset sim/scene/events及各manager |
| manager初始化 | `manager_based_rl_env.py:336` | curriculum先于metrics创建；课程构造器不能假设metrics已经存在 |
| 越界 | `tasks/velocity/mdp/terminations.py` | out_of_terrain_bounds是整网格；terrain_edge_reached可用于局部地块边界 |
| 奖励时间尺度 | `src/mjlab/managers/reward_manager.py` | 默认乘dt；Episode_Reward记录加权速率，不能把权重直接当一次性成功奖金 |
| 跟踪与z代价 | `tasks/velocity/mdp/rewards.py:513,569` | 跟踪使用机体系xy/yaw，z为机体系速度平方且受upright scale影响 |
| 控制/驱动 | `lego2_constants.py`、环境配置 | dt=0.005×4=0.02s；8腿位置＋4轮速度，scale=0.45/5，腿PD=100/5，effort=150/65 |

上表出现的 `config/lego2/`、`tasks/velocity/` 是 `src/mjlab/tasks/velocity/` 等上下文缩写；
正式计划文件清单给出完整项目相对路径。实现时仍用实际导入和签名，不照抄其他框架API。

## 3. 本轮已运行的轻量CPU检查

使用现有隔离环境 `uv run --no-sync`，只读取配置或创建内存对象：

1. 加载rough配置，确认控制dt为0.02s；打印当前actor项及位置/轮速动作配置：
   action clip=None，位置scale=0.45，轮速scale=5，明确名称顺序。
2. 现有 `BoxInvertedPyramidStairsTerrainCfg` 在10×10m、border=3.5、platform=3、
   step_width=2、候选高度0.03～0.30时，CPU编译并射线检查：

   | 候选行 | origin（子地块局部） | 坑底z / 坑沿z |
   | --- | --- | --- |
   | 0 | (5,5,-0.03) | -0.03 / 0 |
   | 4 | (5,5,-0.15) | -0.15 / 0 |
   | 9 | (5,5,-0.30) | -0.30 / 0 |

3. 5帧CPU CircularBuffer执行局部reset和backfill：重置行全部为新首帧，
   未重置行内容保持原值，断言通过。

这些检查确认可复用接口，不等于完整新任务的观测维数、真实接触、课程或PPO训练已验证。
未使用显卡，没有给用户新增本地手工验证步骤。

## 4. 正式计划中的非事实参数

新任务ID、144维actor、5帧本体历史、3～30cm/10级、3m课程阈值、候选奖励权重等
为尚待实施的v1设计，不写成当前已有功能。详见 [REFACTOR_PLAN.md](REFACTOR_PLAN.md)。
0～5阶段的验收与本地/远程同步按该计划执行；本轮只完成计划和记录。
