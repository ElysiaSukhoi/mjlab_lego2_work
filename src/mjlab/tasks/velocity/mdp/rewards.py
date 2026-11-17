"""速度类任务可复用的奖励计算，提供跟速、姿态、能耗等原始分数。

Lego2的env_cfg文件选择函数并指定权重，reward_manager.py负责乘权重和dt。
当前Blind-Pit只选其中一部分；还有关节限位等通用项来自envs/mdp/rewards.py。
v2新增的俯仰容许区间/最低净空计算在lego2/blind_pit_rewards.py，不在这里。
多数跟速和运动代价还乘直立程度系数，解释日志时需结合所选函数。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.tasks.velocity.mdp.terrain_utils import terrain_normal_from_sensors
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + z_error
  return torch.exp(-lin_vel_error / std**2)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  ang_vel_error = z_error + xy_error
  return torch.exp(-ang_vel_error / std**2)


class upright:
  """Reward for keeping the base upright.

  Without ``terrain_sensor_names``, penalizes tilt relative to world up (correct for
  flat ground).

  With ``terrain_sensor_names``, penalizes tilt relative to the terrain surface normal.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self._terrain_sensor_names: tuple[str, ...] | None = cfg.params.get(
      "terrain_sensor_names"
    )
    self._debug_vis_enabled = True
    self._env = env
    self._asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    terrain_sensor_names: tuple[str, ...] | None = None,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]

    if asset_cfg.body_ids:
      body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
      body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    else:
      body_quat_w = asset.data.root_link_quat_w  # [B, 4]

    if terrain_sensor_names is not None:
      terrain_normal = terrain_normal_from_sensors(env, terrain_sensor_names)  # [B, 3]
      # Project terrain normal into body frame. When aligned with the terrain surface
      # this should be (0, 0, 1); XY measures tilt.
      target_b = quat_apply_inverse(body_quat_w, terrain_normal)  # [B, 3]
      xy_squared = torch.sum(torch.square(target_b[:, :2]), dim=1)
    else:
      gravity_w = asset.data.gravity_vec_w  # [3]
      projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
      xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)

    return torch.exp(-xy_squared / std**2)

  def reset(self, env_ids: torch.Tensor) -> None:
    del env_ids  # Unused.

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    if not self._debug_vis_enabled or self._terrain_sensor_names is None:
      return

    env = self._env
    asset: Entity = env.scene[self._asset_cfg.name]

    env_indices = list(visualizer.get_env_indices(env.num_envs))
    if not env_indices:
      return

    terrain_normal = terrain_normal_from_sensors(env, self._terrain_sensor_names)
    if self._asset_cfg.body_ids:
      body_quat_w = asset.data.body_link_quat_w[:, self._asset_cfg.body_ids, :].squeeze(
        1
      )
    else:
      body_quat_w = asset.data.root_link_quat_w
    up_local = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand_as(
      body_quat_w[:, :3]
    )
    body_up_w = quat_apply(body_quat_w, up_local)

    positions = asset.data.root_link_pos_w.cpu().numpy()
    offset = np.array([0.0, 0.3, 0.0])
    terrain_normal_np = terrain_normal.cpu().numpy()
    body_up_np = body_up_w.cpu().numpy()
    scale = 0.25

    for i in env_indices:
      origin = positions[i] + offset
      # Terrain normal (magenta).
      visualizer.add_arrow(
        start=origin,
        end=origin + terrain_normal_np[i] * scale,
        color=(0.8, 0.2, 0.8, 0.8),
        width=0.01,
      )
      # Body up (orange).
      visualizer.add_arrow(
        start=origin,
        end=origin + body_up_np[i] * scale,
        color=(1.0, 0.5, 0.0, 0.8),
        width=0.01,
      )


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.sum(dim=-1).float()


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize whole-body angular momentum to encourage natural arm swing."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return angmom_magnitude_sq


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold_min: float = 0.05,
  threshold_max: float = 0.5,
  command_name: str | None = None,
  command_threshold: float = 0.5,
) -> torch.Tensor:
  """Reward feet air time."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
  reward = torch.sum(in_range.float(), dim=1)
  in_air = current_air_time > 0
  num_in_air = torch.sum(in_air.float())
  mean_air_time = torch.sum(current_air_time * in_air.float()) / torch.clamp(
    num_in_air, min=1
  )
  env.extras["log"]["Metrics/air_time_mean"] = mean_air_time
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command > command_threshold).float()
      reward *= scale
  return reward


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  height_sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target clearance height, weighted by foot velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  height_sensor = env.scene[height_sensor_name]
  assert isinstance(height_sensor, TerrainHeightSensor), (
    f"feet_clearance requires a TerrainHeightSensor, got {type(height_sensor).__name__}"
  )
  foot_height = height_sensor.data.heights  # [B, F]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, F, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, F]
  delta = torch.abs(foot_height - target_height)  # [B, F]
  cost = torch.sum(delta * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class feet_swing_height:
  """Penalize deviation from target swing height, evaluated at landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    height_sensor = env.scene[cfg.params["height_sensor_name"]]
    assert isinstance(height_sensor, TerrainHeightSensor), (
      f"feet_swing_height requires a TerrainHeightSensor, got {type(height_sensor).__name__}"
    )
    num_feet = height_sensor.num_frames
    self.peak_heights = torch.zeros(
      (env.num_envs, num_feet), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    height_sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
  ) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    height_sensor: TerrainHeightSensor = env.scene[height_sensor_name]
    foot_heights = height_sensor.data.heights
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def feet_slide(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  force_threshold: float = 1.0,
) -> torch.Tensor:
  """Penalize foot sliding while in contact (robot_lab ``feet_slide``).

  Computes lateral foot velocity in the base body frame, relative to the base
  linear velocity, and gates by contact force magnitude.
  """
  contact_sensor: ContactSensor = env.scene[sensor_name]
  asset: Entity = env.scene[asset_cfg.name]

  assert contact_sensor.data.force is not None
  contacts = torch.linalg.norm(contact_sensor.data.force, dim=-1) > force_threshold

  foot_vel_w = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]
  rel_vel_w = foot_vel_w - asset.data.root_link_lin_vel_w.unsqueeze(1)

  root_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(
    -1, len(asset_cfg.body_ids), -1
  )
  rel_vel_b = quat_apply_inverse(root_quat, rel_vel_w)
  foot_lateral_vel = torch.linalg.norm(rel_vel_b[..., :2], dim=-1)

  reward = torch.sum(foot_lateral_vel * contacts.float(), dim=1)
  return reward * _upright_scale(env)


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class variable_posture:
  """Penalize deviation from default pose with speed-dependent tolerance.

  Uses per-joint standard deviations to control how much each joint can deviate
  from default pose. Smaller std = stricter (less deviation allowed), larger
  std = more forgiving. The reward is: exp(-mean(error² / std²))

  Three speed regimes (based on linear + angular command velocity):
    - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
    - std_walking (walking_threshold <= speed < running_threshold): Moderate.
    - std_running (speed >= running_threshold): Loose tolerance for large motion.

  Tune std values per joint based on how much motion that joint needs at each
  speed. Map joint name patterns to std values, e.g. {".*knee.*": 0.35}.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))


##
# Unitree B2W / robot_lab aligned rewards.
##


def _upright_scale(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Scale used by robot_lab wheeled rewards when the base is upright."""
  asset: Entity = env.scene["robot"]
  return torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7


def lin_vel_z_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """计算机身自身上下方向的速度平方，配负权重可减少竖直抖动。"""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_lin_vel_b[:, 2]) * _upright_scale(env)


def ang_vel_xy_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """计算机身侧倾/俯仰转动速度的平方和，配负权重限制剧烈摆动。"""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(
    torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1
  ) * _upright_scale(env)


def upward(
  env: ManagerBasedRlEnv,
  std: float = math.sqrt(0.5),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward upright base (robot_lab ``upward``): exp kernel on gravity-z error."""
  asset: Entity = env.scene[asset_cfg.name]
  upward_error = torch.square(asset.data.projected_gravity_b[:, 2] - (-1.0))
  return torch.exp(-upward_error / std**2)


def base_height_l2(
  env: ManagerBasedRlEnv,
  target_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  sensor_name: str | None = None,
) -> torch.Tensor:
  """计算机身高度与目标的双向平方差，是旧Rough等任务可选项。

  可按扫描均值调整目标；当前Blind-Pit v1未启用，v2最低净空使用另一个单向函数。
  """
  asset: Entity = env.scene[asset_cfg.name]
  if sensor_name is not None:
    sensor = env.scene[sensor_name]
    ray_hits = sensor.data.hit_pos_w[..., 2]
    if (
      torch.isnan(ray_hits).any()
      or torch.isinf(ray_hits).any()
      or torch.max(torch.abs(ray_hits)) > 1e6
    ):
      adjusted_target_height = asset.data.root_link_pos_w[:, 2]
    else:
      adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
  else:
    adjusted_target_height = target_height
  reward = torch.square(asset.data.root_link_pos_w[:, 2] - adjusted_target_height)
  return reward * _upright_scale(env)


def track_lin_vel_xy_exp(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """前进/横向速度越接近命令，分数越接近1；std控制偏差容许程度。"""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  lin_vel_error = torch.sum(
    torch.square(command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]), dim=1
  )
  return torch.exp(-lin_vel_error / std**2) * _upright_scale(env)


def track_ang_vel_z_exp(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """转向速度越接近命令，分数越高；最后还乘原有直立程度系数。"""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  ang_vel_error = torch.square(command[:, 2] - asset.data.root_link_ang_vel_b[:, 2])
  return torch.exp(-ang_vel_error / std**2) * _upright_scale(env)


def joint_deviation_l1(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  return torch.sum(
    torch.abs(
      asset.data.joint_pos[:, asset_cfg.joint_ids]
      - default_joint_pos[:, asset_cfg.joint_ids]
    ),
    dim=1,
  )


def stand_still(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None
  reward = joint_deviation_l1(env, asset_cfg)
  reward *= torch.linalg.norm(command, dim=1) < command_threshold
  return reward * _upright_scale(env)


def joint_pos_penalty(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  stand_still_scale: float = 5.0,
  velocity_threshold: float = 0.5,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  cmd = torch.linalg.norm(command, dim=1)
  body_vel = torch.linalg.norm(asset.data.root_link_lin_vel_b[:, :2], dim=1)
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  running_reward = torch.linalg.norm(
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - default_joint_pos[:, asset_cfg.joint_ids],
    dim=1,
  )
  reward = torch.where(
    torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
    running_reward,
    stand_still_scale * running_reward,
  )
  return reward * _upright_scale(env)


def joint_pos_penalty_std(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  stand_still_scale: float = 5.0,
  velocity_threshold: float = 0.5,
  command_threshold: float = 0.1,
  std: float = 1.0,
) -> torch.Tensor:
  """Reward joint proximity to default using exp(-error^2 / std^2), in (0, 1]."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  cmd = torch.linalg.norm(command, dim=1)
  body_vel = torch.linalg.norm(asset.data.root_link_lin_vel_b[:, :2], dim=1)
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  error_squared = torch.sum(
    torch.square(
      asset.data.joint_pos[:, asset_cfg.joint_ids]
      - default_joint_pos[:, asset_cfg.joint_ids]
    ),
    dim=1,
  )
  running_reward = torch.exp(-error_squared / std**2)
  reward = torch.where(
    torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
    running_reward,
    stand_still_scale * running_reward,
  )
  return reward * _upright_scale(env)


def joint_power(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(
    torch.abs(
      asset.data.joint_vel[:, asset_cfg.joint_ids]
      * asset.data.qfrc_actuator[:, asset_cfg.joint_ids]
    ),
    dim=1,
  )


class joint_mirror:
  """Penalize asymmetry between mirrored leg joint pairs."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    mirror_joints: list[list[str]] = cfg.params["mirror_joints"]
    self._pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for joint_pair in mirror_joints:
      ids_a, _ = asset.find_joints(joint_pair[0])
      ids_b, _ = asset.find_joints(joint_pair[1])
      self._pairs.append(
        (
          torch.tensor(ids_a, device=env.device, dtype=torch.long),
          torch.tensor(ids_b, device=env.device, dtype=torch.long),
        )
      )
    self._num_pairs = max(len(self._pairs), 1)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    mirror_joints: list[list[str]],
  ) -> torch.Tensor:
    del asset_cfg, mirror_joints
    asset: Entity = env.scene["robot"]
    reward = torch.zeros(env.num_envs, device=env.device)
    for ids_a, ids_b in self._pairs:
      reward += torch.sum(
        torch.square(asset.data.joint_pos[:, ids_a] - asset.data.joint_pos[:, ids_b]),
        dim=-1,
      )
    reward *= 1.0 / self._num_pairs
    return reward * _upright_scale(env)


def _feet_in_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 1.0,
) -> torch.Tensor:
  """Return per-foot contact flags ``[B, N]`` from a contact sensor."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.found is not None
  in_contact = contact_sensor.data.found > 0
  if contact_sensor.data.force is not None:
    force_mag = torch.linalg.norm(contact_sensor.data.force, dim=-1)
    force_mag = torch.nan_to_num(force_mag, nan=0.0, posinf=0.0, neginf=0.0)
    in_contact = torch.logical_and(in_contact, force_mag > force_threshold)
  return in_contact


def _feet_contact_ratio(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  expect_contact_num: int,
  force_threshold: float,
  power: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return per-env contact reward in [0, 1] and raw contact ratio."""
  in_contact = _feet_in_contact(env, sensor_name, force_threshold)
  contact_ratio = in_contact.float().sum(dim=-1) / expect_contact_num
  reward = torch.pow(contact_ratio.clamp(0.0, 1.0), power)
  return reward, contact_ratio


def feet_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  expect_contact_num: int = 4,
  force_threshold: float = 1.0,
  command_name: str | None = None,
  command_threshold: float = 0.1,
  power: float = 2.0,
) -> torch.Tensor:
  """Reward keeping wheels/feet in ground contact while moving.

  Returns ``(contact_count / expect_contact_num) ** power`` in ``[0, 1]``.
  ``power > 1`` (default 2) strongly favors all feet touching over three-of-four.
  """
  reward, contact_ratio = _feet_contact_ratio(
    env, sensor_name, expect_contact_num, force_threshold, power
  )
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    assert command is not None
    reward *= (torch.linalg.norm(command, dim=1) > command_threshold).float()
  env.extras["log"]["Metrics/feet_contact_ratio_mean"] = contact_ratio.mean()
  return reward * _upright_scale(env)


def feet_contact_without_cmd(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  expect_contact_num: int = 4,
  force_threshold: float = 1.0,
  command_threshold: float = 0.1,
  power: float = 2.0,
) -> torch.Tensor:
  """Reward keeping wheels/feet in ground contact while command is near zero.

  Same contact shaping as :func:`feet_contact`, but active only when
  ``‖command‖ < command_threshold`` (standing / no-motion commands).
  """
  reward, contact_ratio = _feet_contact_ratio(
    env, sensor_name, expect_contact_num, force_threshold, power
  )
  command = env.command_manager.get_command(command_name)
  assert command is not None
  reward *= (torch.linalg.norm(command, dim=1) < command_threshold).float()
  env.extras["log"]["Metrics/feet_contact_without_cmd_ratio_mean"] = (
    contact_ratio.mean()
  )
  return reward * _upright_scale(env)


def wheel_skid_steer_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  wheel_cfg: SceneEntityCfg,
  foot_body_cfg: SceneEntityCfg | None = None,
  wheel_radius: float = 0.115,
  std: float = 8.0,
  command_threshold: float = 0.05,
  joint_sign: float = 1.0,
  side_signs: tuple[float, ...] | list[float] | None = (1.0, -1.0, 1.0, -1.0),
  half_track: float = 0.275,
) -> torch.Tensor:
  """Reward matching wheel speeds to an explicit L/R skid-steer model.

  Wheel order must match ``side_signs`` (default LF, RF, LH, RH)::

      side = +1 left,  -1 right
      v_x,i = v_cmd - side_i * omega_cmd * half_track
      omega_wheel,i = joint_sign * v_x,i / wheel_radius

  With this convention, ``omega_cmd > 0`` (CCW / left yaw) drives **left
  wheels backward** and **right wheels forward**; ``omega_cmd < 0`` flips both.

  ``joint_sign=+1`` matches Lego2 open-loop mapping (``+qvel`` → base ``+x``).
  ``half_track`` is the nominal |y| of each wheel in the base frame (~0.187 m
  for Lego2 default pose). Prefer fixed ``side_signs``+``half_track`` over
  measured body ``y`` so leg swing does not scramble the L/R targets.

  If ``side_signs`` is ``None``, falls back to measured base-frame ``y_i`` from
  ``foot_body_cfg`` (must be provided). MSE is always averaged over all wheels
  (no contact mask), so lifting wheels cannot hide speed error.
  """
  asset: Entity = env.scene[wheel_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None

  v_cmd = command[:, 0]
  omega_cmd = command[:, 2]
  active = (torch.abs(v_cmd) + torch.abs(omega_cmd) > command_threshold).float()

  n_wheels = len(wheel_cfg.joint_ids)
  if side_signs is not None:
    if len(side_signs) != n_wheels:
      raise ValueError(
        f"side_signs length {len(side_signs)} != number of wheels {n_wheels}"
      )
    side = torch.as_tensor(side_signs, device=env.device, dtype=torch.float32)
    # left (+1): v_x = v - ω·L  → ω>0 backward; right (−1): v_x = v + ω·L → ω>0 forward
    v_x = v_cmd.unsqueeze(1) - omega_cmd.unsqueeze(1) * side * half_track
  else:
    if foot_body_cfg is None:
      raise ValueError("foot_body_cfg is required when side_signs is None")
    foot_pos_w = asset.data.body_link_pos_w[:, foot_body_cfg.body_ids]
    rel_w = foot_pos_w - asset.data.root_link_pos_w.unsqueeze(1)
    root_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(
      -1, len(foot_body_cfg.body_ids), -1
    )
    y_i = quat_apply_inverse(root_quat, rel_w)[..., 1]
    v_x = v_cmd.unsqueeze(1) - omega_cmd.unsqueeze(1) * y_i

  # joint_sign: Lego2 maps +qvel → base +x (see test_lego2_skid_steer_reward).
  expected_wheel_vel = joint_sign * v_x / wheel_radius
  actual_wheel_vel = asset.data.joint_vel[:, wheel_cfg.joint_ids]
  mse = torch.square(actual_wheel_vel - expected_wheel_vel).mean(dim=1)

  # Debug: mean target qvel on left vs right under the current batch.
  if side_signs is not None and n_wheels >= 2:
    side_t = torch.as_tensor(side_signs, device=env.device, dtype=torch.float32)
    left_m = side_t > 0
    right_m = side_t < 0
    if torch.any(left_m) and torch.any(right_m):
      env.extras["log"]["Metrics/wheel_skid_target_left_mean"] = expected_wheel_vel[
        :, left_m
      ].mean()
      env.extras["log"]["Metrics/wheel_skid_target_right_mean"] = expected_wheel_vel[
        :, right_m
      ].mean()

  env.extras["log"]["Metrics/wheel_skid_steer_mse"] = mse.mean()
  return torch.exp(-mse / std**2) * active * _upright_scale(env)


def wheel_side_sync_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  wheel_cfg: SceneEntityCfg,
  std: float = 2.0,
  command_threshold: float = 0.05,
  pure_threshold: float = 0.05,
  side_signs: tuple[float, ...] | list[float] = (1.0, -1.0, 1.0, -1.0),
) -> torch.Tensor:
  """Soft skid prior: same-side sync always, L/R pairing on pure v or ω.

  Wheel order must match ``side_signs`` (default LF, RF, LH, RH with
  ``+1`` left / ``-1`` right)::

      always:   var(left wheels) + var(right wheels) small
                (LF≈LH and RF≈RH for skid-steer)
      pure v:   mean(left) ≈ mean(right)   (all wheels same speed)
      pure ω:   mean(left) ≈ -mean(right)  (opposite equal magnitude)

  Pure-axis gates use ``command_threshold`` on the active axis and
  ``pure_threshold`` on the idle axis. Mixed ``(v, ω)`` commands only get
  the same-side term (no absolute ``half_track`` / ``wheel_radius``).
  Inactive when both ``|v|`` and ``|ω|`` are below ``command_threshold``.
  """
  asset: Entity = env.scene[wheel_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None

  v_cmd = command[:, 0]
  omega_cmd = command[:, 2]
  abs_v = torch.abs(v_cmd)
  abs_w = torch.abs(omega_cmd)
  active = (abs_v + abs_w > command_threshold).float()
  pure_lin = ((abs_v > command_threshold) & (abs_w < pure_threshold)).float()
  pure_ang = ((abs_w > command_threshold) & (abs_v < pure_threshold)).float()

  n_wheels = len(wheel_cfg.joint_ids)
  if len(side_signs) != n_wheels:
    raise ValueError(
      f"side_signs length {len(side_signs)} != number of wheels {n_wheels}"
    )
  side = torch.as_tensor(side_signs, device=env.device, dtype=torch.float32)
  left_m = side > 0
  right_m = side < 0
  if not torch.any(left_m) or not torch.any(right_m):
    raise ValueError("side_signs must include both left (+1) and right (-1) wheels")

  wheel_vel = asset.data.joint_vel[:, wheel_cfg.joint_ids]
  left = wheel_vel[:, left_m]
  right = wheel_vel[:, right_m]
  # Unbiased=False so a single wheel per side does not NaN.
  err_side = left.var(dim=1, unbiased=False) + right.var(dim=1, unbiased=False)
  w_left = left.mean(dim=1)
  w_right = right.mean(dim=1)
  err_lr = torch.square(w_left - w_right)
  err_opp = torch.square(w_left + w_right)

  mse = err_side + pure_lin * err_lr + pure_ang * err_opp

  env.extras["log"]["Metrics/wheel_side_sync_mse"] = mse.mean()
  env.extras["log"]["Metrics/wheel_side_sync_err_side"] = err_side.mean()
  env.extras["log"]["Metrics/wheel_side_sync_pure_lin_frac"] = pure_lin.mean()
  env.extras["log"]["Metrics/wheel_side_sync_pure_ang_frac"] = pure_ang.mean()
  return torch.exp(-mse / std**2) * active * _upright_scale(env)


def contact_forces(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float = 100.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  forces = torch.linalg.norm(sensor.data.force, dim=-1)
  return torch.sum((forces - threshold).clip(min=0.0), dim=1)


def undesired_contacts(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float = 1.0,
) -> torch.Tensor:
  """按非轮接触通道数计算代价，并乘直立程度系数。

  当前Blind-Pit使用curriculums中的原始接触计数，避免翻倒后计数被压低。
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  is_contact = sensor.data.found > 0
  if sensor.data.force is not None:
    force_mag = torch.linalg.norm(sensor.data.force, dim=-1)
    is_contact = torch.logical_and(is_contact, force_mag > threshold)
  reward = torch.sum(is_contact, dim=-1).float()
  return reward * _upright_scale(env)
