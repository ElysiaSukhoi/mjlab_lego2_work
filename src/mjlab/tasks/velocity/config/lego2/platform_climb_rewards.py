"""旧固定高台任务的专项奖励计算，不用于当前Blind-Pit。

platform_climb_env_cfg.py指定权重、台沿位置和高度，再调用下面的函数。
这些函数会利用固定台面区域鼓励接近、抬升或进入台面；它们不是当前v2的通用代价。
返回值由奖励管理器乘权重/时间步长；是否结束回合由terminations模块另行判断。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def platform_approach(
  env: ManagerBasedRlEnv,
  start_x: float,
  platform_edge_x: float,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """按靠近固定台沿的进度给分，鼓励走到台前；不代表已经上台。"""
  asset: Entity = env.scene[asset_cfg.name]
  relative_x = asset.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
  progress = (relative_x - start_x) / (platform_edge_x - start_x)
  progress = progress.clamp(0.0, 1.0)
  env.extras["log"]["Metrics/platform_approach_mean"] = progress.mean()
  return progress


def platform_top_bonus(
  env: ManagerBasedRlEnv,
  platform_height: float,
  top_x_range: tuple[float, float],
  top_y_half_width: float,
  root_z_margin: float,
  sensor_name: str,
  min_contacts: int,
  force_threshold: float,
  root_z_threshold: float | None = None,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """当前位置和接触数满足设定台面条件时返回1，每个满足的控制步都可给分。

  连续保持多久才算成功，由platform_success另外判断；这里不记忆首次到达事件。
  """
  asset: Entity = env.scene[asset_cfg.name]
  relative_pos = asset.data.root_link_pos_w - env.scene.env_origins
  inside_x = (relative_pos[:, 0] >= top_x_range[0]) & (
    relative_pos[:, 0] <= top_x_range[1]
  )
  inside_y = torch.abs(relative_pos[:, 1]) <= top_y_half_width
  threshold = (
    platform_height + root_z_margin if root_z_threshold is None else root_z_threshold
  )
  above_top = relative_pos[:, 2] >= threshold
  in_top_region = inside_x & inside_y & above_top

  sensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None
  in_contact = found > 0
  if sensor.data.force is not None:
    force_mag = torch.linalg.norm(sensor.data.force, dim=-1)
    in_contact &= force_mag > force_threshold
  enough_contacts = in_contact.sum(dim=1) >= min_contacts
  return (in_top_region & enough_contacts).float()


def platform_edge_lift(
  env: ManagerBasedRlEnv,
  edge_gate_x_range: tuple[float, float],
  low_root_z: float,
  top_root_z: float,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """进入固定台沿附近后，按机身抬升程度给分，是旧任务的专项引导。"""
  asset: Entity = env.scene[asset_cfg.name]
  relative_pos = asset.data.root_link_pos_w - env.scene.env_origins
  edge_gate = (
    (relative_pos[:, 0] - edge_gate_x_range[0])
    / (edge_gate_x_range[1] - edge_gate_x_range[0])
  ).clamp(0.0, 1.0)
  lift = ((relative_pos[:, 2] - low_root_z) / (top_root_z - low_root_z)).clamp(0.0, 1.0)
  value = edge_gate * lift
  env.extras["log"]["Metrics/platform_edge_lift_mean"] = value.mean()
  return value


def platform_front_lift(
  env: ManagerBasedRlEnv,
  edge_gate_x_range: tuple[float, float],
  height_sensor_name: str,
  target_height: float,
  top_height: float,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """接近固定台沿时，按前轮附近的离地高度给分。

  取RF/LF两个传感器高度的均值；不能仅凭此分数认定两只前轮已真正上台。
  """
  asset: Entity = env.scene[asset_cfg.name]
  relative_x = asset.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
  edge_gate = (
    (relative_x - edge_gate_x_range[0]) / (edge_gate_x_range[1] - edge_gate_x_range[0])
  ).clamp(0.0, 1.0)

  sensor = env.scene[height_sensor_name]
  heights = sensor.data.heights
  assert heights is not None
  # Site order is RF, LF, RH, LH in the Lego2 configuration.
  front_height = heights[:, :2].mean(dim=1)
  clearance = ((front_height - target_height) / (top_height - target_height)).clamp(
    0.0, 1.0
  )
  value = edge_gate * clearance
  env.extras["log"]["Metrics/platform_front_lift_mean"] = value.mean()
  return value


def platform_front_contact_advance(
  env: ManagerBasedRlEnv,
  edge_gate_x_range: tuple[float, float],
  sensor_name: str,
  force_threshold: float = 1.0,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """前轮有接触时，按机身越过固定台沿区域的进度给分。

  虽然函数名含advance，计算使用的是相对X位置进度，并未直接检查向前速度。
  """
  asset: Entity = env.scene[asset_cfg.name]
  relative_x = asset.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
  edge_progress = (
    (relative_x - edge_gate_x_range[0]) / (edge_gate_x_range[1] - edge_gate_x_range[0])
  ).clamp(0.0, 1.0)

  sensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None
  in_contact = found > 0
  if sensor.data.force is not None:
    force_mag = torch.linalg.norm(sensor.data.force, dim=-1)
    in_contact &= force_mag > force_threshold
  front_contact_fraction = in_contact[:, :2].sum(dim=1).float() / 2.0

  value = edge_progress * front_contact_fraction
  env.extras["log"]["Metrics/platform_front_contact_advance_mean"] = value.mean()
  return value
