"""计算v2新增的姿态和机身离地高度代价。

blind_pit_reward_v2_env_cfg.py决定哪些函数启用，以及各自权重和阈值。
本文件只返回每个环境的代价值；reward_manager.py乘权重和控制周期后汇总。
地面扫描只供训练评分和评估使用，机器人选动作时仍使用原144维盲走输入。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.sensor import RayCastSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def pitch_relaxed_orientation_l2(
  env: ManagerBasedRlEnv, free_pitch_deg: float = 25.0
) -> torch.Tensor:
  """允许适度抬头/低头，仍对侧倾和过大俯仰扣分。

  free_pitch_deg单位为度；默认±25°内不增加俯仰代价，超出后逐渐增加。
  """
  robot: Entity = env.scene["robot"]
  gravity = robot.data.projected_gravity_b
  # gravity为(E,3)机体系单位重力投影。|gx|=|sin(pitch)|，gy随roll变化。
  # 保留原gy²侧倾代价；俯仰在±25°内不扣分，超出后连续增加，不规定抬腿顺序。
  # 此项不替代70°翻倒终止，亦不在actor中加入接触或地形信息。
  excess_pitch = (
    gravity[:, 0].abs() - math.sin(math.radians(free_pitch_deg))
  ).clamp_min(0)
  return gravity[:, 1].square() + excess_pitch.square()


def _level_ground_clearance(
  env: ManagerBasedRlEnv, sensor_name: str, max_ground_relief: float
) -> tuple[torch.Tensor, torch.Tensor]:
  """估计机身离局部地面的高度，并标记这次估计能否用于评分。

  扫描全部有效且地面高差较小时才可用；跨较高台沿或漏检时不套最低高度限制。
  返回每个环境的米制净空，以及是否可用的布尔值。
  """
  robot: Entity = env.scene["robot"]
  sensor: RayCastSensor = env.scene[sensor_name]
  scan = sensor.data
  # 复用critic已有扫描，既不新增射线也不改变144/254维观测。
  # 各环境独立检查全部射线；一行漏检/NaN不能关闭其他环境的高度代价。
  z = scan.hit_pos_w[..., 2].reshape(env.num_envs, -1)
  distances = scan.distances.reshape_as(z)
  normal_z = scan.normals_w[..., 2].reshape_as(z)
  valid = torch.isfinite(z) & torch.isfinite(distances) & (distances >= 0)
  valid &= torch.isfinite(normal_z) & (normal_z > 0.5)
  safe_z = torch.where(valid, z, 0.0)
  relief = safe_z.amax(dim=-1) - safe_z.amin(dim=-1)
  eligible = valid.all(dim=-1) & (relief <= max_ground_relief)
  # 扫描高差超过阈值（例如跨较高台沿）时关闭高度项，让机器人自由俯仰。
  # 只在局部地面平整且扫描有效时，用root世界Z减局部地面Z，单位m。
  clearance = (robot.data.root_link_pos_w[:, 2] - safe_z.mean(dim=-1)).clamp_min(0)
  return clearance, eligible


def minimum_base_clearance_l2(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  minimum_height: float = 0.34,
  max_ground_relief: float = 0.03,
) -> torch.Tensor:
  """对平整地面上的过低机身姿态扣分，超过最低高度则不扣。

  minimum_height是机身root到局部地面的距离，不是膝关节的安全间隙。
  归一化后的缺口平方在0到1之间，最后由环境配置乘负权重。
  """
  clearance, eligible = _level_ground_clearance(env, sensor_name, max_ground_relief)
  # 仅罚低于下限，不把0.34 m设成必须贴合的目标，也不给无限抬高的正奖励。
  # 净空已截为非负，归一化缺口平方落在[0,1]；权重-2时最大加权速率为-2。
  deficit = (minimum_height - clearance).clamp_min(0) / minimum_height
  return torch.where(eligible, deficit.square(), 0.0)


def base_clearance_eligible(
  env: ManagerBasedRlEnv, sensor_name: str, max_ground_relief: float = 0.03
) -> torch.Tensor:
  """记录有多少帧可使用局部净空判断；用来解释高度代价的覆盖范围。"""
  _, eligible = _level_ground_clearance(env, sensor_name, max_ground_relief)
  return eligible.float()


def base_clearance_below_minimum(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  minimum_height: float = 0.34,
  max_ground_relief: float = 0.03,
) -> torch.Tensor:
  """记录既能判断净空、又低于下限的帧。

  训练日志是这类帧占整个回合的比例；配对汇总还会计算它占有效扫描帧的比例。
  """
  clearance, eligible = _level_ground_clearance(env, sensor_name, max_ground_relief)
  # 与eligible指标一起读；这是全episode帧占比，不是条件均值或成功率。
  return (eligible & (clearance < minimum_height)).float()
