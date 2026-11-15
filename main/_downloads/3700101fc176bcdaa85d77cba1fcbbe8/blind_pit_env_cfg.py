"""Blind Lego2 pit with explicit generic rewards and privileged diagnostics."""

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
  """Use five proprioceptive frames and current action/command, term-major.

  Train/play share geometry and observations. Play disables actor corruption
  and curriculum; it does not inherit Rough's play-only overrides.
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
    curriculum=True,
    size=(10.0, 10.0),
    num_rows=10,
    num_cols=1,
    merge_curriculum_rows=False,
    sub_terrains={
      "pit": BoxInvertedPyramidStairsTerrainCfg(
        proportion=1.0,
        step_height_range=(0.03, 0.30),
        step_width=2.0,
        platform_width=3.0,
        border_width=3.5,
      )
    },
  )
  cfg.scene.terrain.max_init_terrain_level = 0
  cfg.episode_length_s = 20.0
  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  command.resampling_time_range = (20.0, 20.0)
  command.ranges.lin_vel_x = (0.3, 0.6)
  command.ranges.lin_vel_y = (0.0, 0.0)
  command.ranges.ang_vel_z = (-0.5, 0.5)
  command.ranges.heading = (0.0, 0.0)
  command.heading_command = True
  command.heading_control_stiffness = 0.5
  command.rel_heading_envs = 1.0
  command.rel_standing_envs = command.rel_spin_envs = 0.0
  command.rel_world_envs = command.rel_forward_envs = 0.0
  command.init_velocity_prob = 0.0
  # Origin is already on the pit floor: do not subtract pit depth again.
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
  # Contact geometry already has fixed slide friction 0.4; no DR is applied.
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
        func=PitTerrainLevels, params={"promote_distance": 3.0}
      )
    }
  )
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=envs_mdp.bad_orientation, params={"limit_angle": math.radians(70)}
    ),
    "nan": TerminationTermCfg(func=envs_mdp.nan_detection),
    "tile_boundary": TerminationTermCfg(func=PitTileBoundary, time_out=True),
  }
  # Log non-foot collisions without a reward weight or a contact termination.
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
      weight=3.0,
      params={"command_name": "twist", "std": 0.5},
    ),
    "track_ang_vel_z_exp": RewardTermCfg(
      func=mdp.track_ang_vel_z_exp,
      weight=1.5,
      params={"command_name": "twist", "std": 0.5},
    ),
    # 两项速度平方沿用原函数的机体系语义和直立缩放，不提供向上运动奖金。
    "lin_vel_z_l2": RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-0.3),
    "ang_vel_xy_l2": RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.05),
    "flat_orientation_l2": RewardTermCfg(func=mdp.flat_orientation_l2, weight=-0.5),
    "joint_torques_l2": RewardTermCfg(
      func=mdp.joint_torques_l2,
      weight=-2.5e-5,
      params={"asset_cfg": leg_motors},
    ),
    "joint_acc_l2": RewardTermCfg(
      func=mdp.joint_acc_l2,
      weight=-2.5e-7,
      params={"asset_cfg": legs},
    ),
    "joint_acc_wheel_l2": RewardTermCfg(
      func=mdp.joint_acc_l2,
      weight=-2.5e-9,
      params={"asset_cfg": wheels},
    ),
    # 按动作term分开统计raw动作一阶差分，不对缩放后的角度/轮速混算。
    "action_rate_l2": RewardTermCfg(
      func=mdp.action_rate_l2,
      weight=-0.01,
      params={"action_name": "joint_pos"},
    ),
    "action_rate_wheel_l2": RewardTermCfg(
      func=mdp.action_rate_l2,
      weight=-0.01,
      params={"action_name": "joint_vel"},
    ),
    "joint_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-5.0,
      params={"asset_cfg": legs},
    ),
  }
  # 每项独立状态由MetricsManager按env_ids重置；普通阶段记录时间占比。
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
  """Build a real fixed-depth pit with the same actor and actions as training."""
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
