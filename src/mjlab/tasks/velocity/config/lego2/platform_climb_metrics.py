"""Metrics used by the incremental Lego2 platform-climb task."""

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
  """Return root x relative to the platform center."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]


def platform_relative_root_z(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Return root z relative to the platform-top environment origin."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]


def platform_contact_count(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 1.0,
) -> torch.Tensor:
  """Return the number of feet with valid terrain contact."""
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
  """Return valid contact count for the front RF/LF feet."""
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
  """Return one when the base is inside the platform-top region.

  This is deliberately a metric only.  It does not provide reward and does
  not terminate an episode.  A later step can add contact and hold-time
  requirements after this geometric detector has been checked in play mode.
  """
  asset: Entity = env.scene[asset_cfg.name]
  relative_pos = asset.data.root_link_pos_w - env.scene.env_origins
  inside_x = (relative_pos[:, 0] >= top_x_range[0]) & (
    relative_pos[:, 0] <= top_x_range[1]
  )
  inside_y = torch.abs(relative_pos[:, 1]) <= top_y_half_width
  threshold = (
    platform_height + root_z_margin
    if root_z_threshold is None
    else root_z_threshold
  )
  above_top = relative_pos[:, 2] >= threshold
  return (inside_x & inside_y & above_top).float()


class platform_top_stable:
  """Metric for holding the top region with multiple supported feet."""

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
    asset: Entity = env.scene[asset_cfg.name]
    relative_pos = asset.data.root_link_pos_w - env.scene.env_origins
    inside_x = (relative_pos[:, 0] >= top_x_range[0]) & (
      relative_pos[:, 0] <= top_x_range[1]
    )
    inside_y = torch.abs(relative_pos[:, 1]) <= top_y_half_width
    threshold = (
      platform_height + root_z_margin
      if root_z_threshold is None
      else root_z_threshold
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
