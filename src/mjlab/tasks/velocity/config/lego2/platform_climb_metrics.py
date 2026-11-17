"""观察旧固定高台任务的位置、接触和台面保持情况。

platform_climb_env_cfg.py把这些函数登记为日志指标；本文件本身不加分。
位置相对旧任务的台面中心，长度单位为米；接触数按传感器通道统计。
当前Blind-Pit的四轮几何支撑与0.5秒判据位于blind_pit_metrics.py。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def platform_relative_root_x(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """记录机身相对旧台面中心的前后位置，单位米。"""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]


def platform_relative_root_z(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """记录机身相对旧台面原点的高度，单位米。"""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]


def platform_contact_count(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 1.0,
) -> torch.Tensor:
  """记录四轮中有多少个满足有效接触条件。"""
  sensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None
  in_contact = found > 0
  if sensor.data.force is not None:
    force_mag = torch.linalg.norm(sensor.data.force, dim=-1)
    in_contact &= force_mag > force_threshold
  return in_contact.sum(dim=1).float()


def platform_front_contact_count(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 1.0,
) -> torch.Tensor:
  """记录RF/LF两个前轮的有效接触数，便于判断是否卡在台沿。"""
  sensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None
  in_contact = found > 0
  if sensor.data.force is not None:
    force_mag = torch.linalg.norm(sensor.data.force, dim=-1)
    in_contact &= force_mag > force_threshold
  return in_contact[:, :2].sum(dim=1).float()


def platform_top_reached(
  env: ManagerBasedRlEnv,
  platform_height: float,
  top_x_range: tuple[float, float],
  top_y_half_width: float,
  root_z_margin: float = 0.02,
  root_z_threshold: float | None = None,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """仅按机身位置/高度判断是否进入旧台面区域，不保证四轮都上台。"""
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
  return (inside_x & inside_y & above_top).float()


class platform_top_stable:
  """记录旧任务是否在台面区域维持了指定接触数和连续步数。"""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self._hold_steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      self._hold_steps.zero_()
    else:
      self._hold_steps[env_ids] = 0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    platform_height: float,
    top_x_range: tuple[float, float],
    top_y_half_width: float,
    root_z_margin: float,
    sensor_name: str,
    min_contacts: int,
    force_threshold: float,
    hold_steps: int,
    root_z_threshold: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  ) -> torch.Tensor:
    """满足区域和接触条件就累加保持步数，中断时清零；仅输出指标。"""
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
    stable_now = in_top_region & enough_contacts

    self._hold_steps = torch.where(
      stable_now,
      self._hold_steps + 1,
      torch.zeros_like(self._hold_steps),
    )
    return (self._hold_steps >= hold_steps).float()
