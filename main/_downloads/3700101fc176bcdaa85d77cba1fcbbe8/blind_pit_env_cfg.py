"""当前中央坑任务的总配置，保存11项奖励的v1基线。

从env_cfgs.py借用机器人、驱动和基础传感器，再重新设置坑、出生位置、
速度命令、盲走观测、奖励和结束条件；并非直接沿用Rough的全部设置。
blind_pit_curriculums.py决定难度升降，blind_pit_metrics.py记录越障阶段。
网络/PPO参数看rl_cfg.py；v2在blind_pit_reward_v2_env_cfg.py中派生。
普通play从3厘米等级出生；固定高度回放应使用下方eval工厂。
"""

import math
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import (
  CurriculumTermCfg,
  EventTermCfg,
  MetricsTermCfg,
  RewardTermCfg,
  SceneEntityCfg,
  TerminationTermCfg,
)
from mjlab.sensor import ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import BoxInvertedPyramidStairsTerrainCfg, TerrainGeneratorCfg

from .blind_pit_curriculums import (
  PitTerrainLevels,
  PitTileBoundary,
  nonfoot_contact_count,
)
from .blind_pit_metrics import PitProgress
from .env_cfgs import lego2_rough_env_cfg

# 阶段1的本体输入：角速度3维、重力投影3维、腿位置偏差8维、关节速度12维。
# 轮子的累计转角不参与位置观测；速度观测包含8个腿关节和4个轮关节。
_PROPRIOCEPTION = ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel")
# 白名单还包含上一控制步已执行的12维raw动作，以及当前vx/wz两维命令。
# 这里列出的名字也决定最终拼接顺序，不依赖基础配置恰好含有哪些额外观测。
_ACTOR_TERMS = (*_PROPRIOCEPTION, "actions", "command")


def lego2_blind_pit_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """组装中央坑v1的训练或普通回放配置。

  机器人从坑底出生，坑深由3厘米逐级到30厘米；命令要求向前走并保持朝向。
  actor只看本体近5帧、上次动作和当前速度命令，critic保留训练用地形信息。
  play关闭观测噪声和课程，但普通play仍从最低等级开始。
  """
  # 阶段1：从Rough训练配置派生独立对象，修改新任务不会改到旧任务。
  # 不采用Rough的play分支，避免一并继承超长episode等回放专用覆盖。
  cfg = deepcopy(lego2_rough_env_cfg(play=False))
  # deepcopy整份cfg仍会保留其内部原有的共享关系；基础配置的两组共享部分term。
  # 因此actor/critic必须各自再复制，防止actor的历史设置影响critic。
  actor = deepcopy(cfg.observations["actor"])
  critic = deepcopy(cfg.observations["critic"])
  # actor只取白名单：不输入扫描、真实线速度、接触或世界位置；critic保留特权量。
  actor.terms = {name: actor.terms[name] for name in _ACTOR_TERMS}
  # None表示不在group层统一覆盖历史长度，下面才能对本体量和动作/命令分别设置。
  actor.history_length = None
  actor.flatten_history_dim = True
  actor.concatenate_terms = True
  # 训练保留观测噪声；play关闭噪声，便于回放和核对输入。
  actor.enable_corruption = not play
  for name, term in actor.terms.items():
    # 四类本体量各保留5帧（含当前帧）；动作和命令的0表示不堆叠，并非观测置零。
    term.history_length = 5 if name in _PROPRIOCEPTION else 0
    # 原生buffer按“每个term的旧帧→新帧”展开，再按白名单顺序拼接各term。
    # 因而actor维数=(3+3+8+12)*5+12+2=144，而非把六类输入整帧统一堆叠。
    term.flatten_history_dim = True
  # 历史更新和局部reset由原生ObservationManager负责：重置行用新首帧回填5帧。
  # 这里仅定义输入契约，不新增一套历史buffer，也不改变PPO。
  cfg.observations = {"actor": actor, "critic": critic}

  # 以下为阶段2接入的地形/出生/指令/课程；阶段1的观测与历史配置到此结束。
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "generator"
  cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
    curriculum=True,  # 地形按等级生成，课程模块决定每个环境使用哪一级。
    size=(10.0, 10.0),  # 每块地形长宽，单位米；不是坑底的尺寸。
    num_rows=10,  # 10个坑深等级，包含3和30厘米两个端点。
    num_cols=1,  # 只有中央坑这一种地形。
    merge_curriculum_rows=False,  # 各等级地块独立，避免合并后混淆出生原点。
    sub_terrains={
      "pit": BoxInvertedPyramidStairsTerrainCfg(
        proportion=1.0,  # 所有地块都用此类型。
        step_height_range=(0.03, 0.30),  # 坑底到周围台面的高差，单位米。
        step_width=2.0,  # 台阶宽度参数；结合下方坑底/边框尺寸形成单级坑。
        platform_width=3.0,  # 中央坑底为3×3米。
        border_width=3.5,  # 周围保留3.5米宽区域，构成坑外台面。
      )
    },
  )
  cfg.scene.terrain.max_init_terrain_level = 0  # 新环境从最低3厘米开始。
  cfg.episode_length_s = 20.0  # 一回合最长20秒，提前失败/越界也会结束。
  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  command.resampling_time_range = (20.0, 20.0)  # 每回合前进速度基本固定。
  command.ranges.lin_vel_x = (0.3, 0.6)  # 前进命令，单位米/秒。
  command.ranges.lin_vel_y = (0.0, 0.0)  # 不要求横移。
  command.ranges.ang_vel_z = (-0.5, 0.5)  # 转向角速度范围，单位弧度/秒。
  command.ranges.heading = (0.0, 0.0)  # 目标朝向固定为0弧度。
  command.heading_command = True  # 根据朝向偏差生成转向命令。
  command.heading_control_stiffness = 0.5  # 朝向偏差转换成转向命令的力度。
  command.rel_heading_envs = 1.0
  command.rel_standing_envs = command.rel_spin_envs = 0.0
  command.rel_world_envs = command.rel_forward_envs = 0.0
  command.init_velocity_prob = 0.0
  # 出生原点已经在坑底，不能再减一次坑深；root默认高度0.47 m来自机器人配置。
  cfg.events = {
    "reset_base": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.15), "yaw": (-0.05, 0.05)},
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": deepcopy(cfg.events["reset_robot_joints"]),
  }
  cfg.events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)
  cfg.events["reset_robot_joints"].params["velocity_range"] = (0.0, 0.0)
  # 轮碰撞几何已有固定滑动摩擦0.4；此任务不另加随机摩擦。
  contact = next(s for s in cfg.scene.sensors if s.name == "feet_ground_contact")
  assert isinstance(contact, ContactSensorCfg)
  rim_contact = deepcopy(contact)
  rim_contact.name = "pit_rim_contact"
  rim_contact.fields = ("found", "force", "normal", "tangent", "pos")
  rim_contact.reduce = "maxforce"
  rim_contact.global_frame = True
  rim_contact.track_air_time = False
  cfg.scene.sensors = (*cfg.scene.sensors, rim_contact)
  cfg.curriculum = (
    {}
    if play
    else {
      "pit_levels": CurriculumTermCfg(
        func=PitTerrainLevels,
        params={"promote_distance": 3.0},  # 晋级还要求走出3米。
      )
    }
  )
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=envs_mdp.bad_orientation,
      params={"limit_angle": math.radians(70)},  # 倾斜超过70°结束；成功保持另用45°。
    ),
    "nan": TerminationTermCfg(func=envs_mdp.nan_detection),
    "tile_boundary": TerminationTermCfg(func=PitTileBoundary, time_out=True),
  }
  # v1只记录非轮碰撞，不因它直接扣分/结束；v2在派生配置中另外加入接触代价。
  cfg.metrics["pit_nonfoot_contact_count"] = MetricsTermCfg(
    func=nonfoot_contact_count, reduce="mean"
  )
  # 阶段3：显式白名单重建11项奖励，不保留Rough的姿态/高度/贴地等额外约束。
  # 腿力矩选择actuator，腿/轮加速度选择joint；两类索引不可互换。
  legs = SceneEntityCfg("robot", joint_names=(".*_HFE", ".*_KFE"))
  wheels = SceneEntityCfg("robot", joint_names=(".*_Wheel",))
  leg_motors = SceneEntityCfg("robot", actuator_names=[".*_HFE", ".*_KFE"])
  cfg.rewards = {
    "track_lin_vel_xy_exp": RewardTermCfg(
      func=mdp.track_lin_vel_xy_exp,
      weight=3.0,  # 前进/横向跟速奖励的权重。
      params={"command_name": "twist", "std": 0.5},  # 速度误差容许尺度，米/秒。
    ),
    "track_ang_vel_z_exp": RewardTermCfg(
      func=mdp.track_ang_vel_z_exp,
      weight=1.5,  # 转向跟速奖励，帮助保持目标朝向。
      params={"command_name": "twist", "std": 0.5},  # 误差容许尺度，弧度/秒。
    ),
    # 两项速度平方沿用原函数的机体系语义和直立缩放，不提供向上运动奖金。
    "lin_vel_z_l2": RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-0.3),  # 少上下抖动。
    "ang_vel_xy_l2": RewardTermCfg(
      func=mdp.ang_vel_xy_l2, weight=-0.05
    ),  # 少剧烈摆动。
    "flat_orientation_l2": RewardTermCfg(
      func=mdp.flat_orientation_l2,
      weight=-0.5,  # v1倾斜代价；v2替换此项。
    ),
    "joint_torques_l2": RewardTermCfg(
      func=mdp.joint_torques_l2,
      weight=-2.5e-5,  # 限制8个腿电机的力矩平方和，不含轮电机。
      params={"asset_cfg": leg_motors},
    ),
    "joint_acc_l2": RewardTermCfg(
      func=mdp.joint_acc_l2,
      weight=-2.5e-7,  # 腿关节加速度代价，减轻急动。
      params={"asset_cfg": legs},
    ),
    "joint_acc_wheel_l2": RewardTermCfg(
      func=mdp.joint_acc_l2,
      weight=-2.5e-9,  # 轮关节加速度单独计分，使用较小权重。
      params={"asset_cfg": wheels},
    ),
    # 按动作term分开统计raw动作一阶差分，不对缩放后的角度/轮速混算。
    "action_rate_l2": RewardTermCfg(
      func=mdp.action_rate_l2,
      weight=-0.01,  # 腿动作相邻两步变化的代价。
      params={"action_name": "joint_pos"},
    ),
    "action_rate_wheel_l2": RewardTermCfg(
      func=mdp.action_rate_l2,
      weight=-0.01,  # 轮动作变化代价；与腿分开统计raw动作。
      params={"action_name": "joint_vel"},
    ),
    "joint_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-5.0,  # 实际腿关节越过软限位时扣分，不是动作硬裁剪。
      params={"asset_cfg": legs},
    ),
  }
  # Episode_Metrics/pit_*：普通阶段记录满足条件的帧占比，而非完整回合成功率。
  # pit_stable_pass用last输出本回合是否曾通过，非轮计数则是每帧平均通道数。
  # Metrics/twist/error_vel_xy来自命令模块，表示速度跟踪误差，课程等级看Curriculum日志。
  # Episode_Reward是加权速率的回合累计/最大时长，不能直接与这些比例指标混比。
  # 稳定通过必须连续支撑0.5 s，课程仍使用阶段2纯判据，不依赖metrics初始化。
  for field in ("root_out", "front_up", "rear_up", "complete", "stable_pass"):
    cfg.metrics[f"pit_{field}"] = MetricsTermCfg(
      func=PitProgress,
      params={"field": field},
      reduce="last" if field == "stable_pass" else "mean",
    )
  return cfg


def lego2_blind_pit_eval_cfg(
  height: float, forward_speed: float | None = None
) -> ManagerBasedRlEnvCfg:
  """生成指定坑深的评估配置，可选固定前进速度。

  height单位为米，forward_speed单位为米/秒；传入0.30才是固定30厘米。
  保留终止帧，调用者结算后显式reset，用于统计完整回合和配对复评。
  """
  if not math.isfinite(height) or not 0 < height <= 0.30:
    raise ValueError("Pit evaluation height must be in (0, 0.30] metres")
  if forward_speed is not None and (
    not math.isfinite(forward_speed) or not 0.3 <= forward_speed <= 0.6
  ):
    raise ValueError("Fixed forward speed must be in [0.3, 0.6] m/s")
  cfg = lego2_blind_pit_env_cfg(play=True)
  if forward_speed is not None:
    # 只收窄评估vx采样范围，单位m/s；不改训练分布、heading反馈或actor输入。
    # 仍走原命令采样流程，便于同一reset seed在不同速度下生成相同出生状态。
    command = cfg.commands["twist"]
    assert isinstance(command, UniformVelocityCommandCfg)
    command.ranges.lin_vel_x = (forward_speed, forward_speed)
  assert cfg.scene.terrain is not None
  gen = cfg.scene.terrain.terrain_generator
  assert gen is not None
  # 关闭升降级本身不固定地形；必须同时把生成器的深度上下限设为同一真实高度。
  # 保留两行避免生成器row/(rows-1)除零；两行几何相同，仍从第0行出生。
  gen.num_rows = 2
  pit = gen.sub_terrains["pit"]
  assert isinstance(pit, BoxInvertedPyramidStairsTerrainCfg)
  pit.step_height_range = (height, height)
  cfg.auto_reset = False
  # 显式reset前保留终止帧和指标，评估器只结算已结束episode，人工reset不进入样本。
  return cfg
