"""Rewards used by the incremental Lego2 platform-climb task."""

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
  """Reward normalized forward progress toward the platform's near edge.

  The platform center is the terrain environment origin for the fixed
  single-platform task.  This term supplies a small dense signal based on the
  robot's relative x position; it does not decide whether the platform has
  been climbed successfully.
  """
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
  """Return a one-step bonus signal for supported entry onto the top."""
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
  return (in_top_region & enough_contacts).float()


def platform_edge_lift(
  env: ManagerBasedRlEnv,
  edge_gate_x_range: tuple[float, float],
  low_root_z: float,
  top_root_z: float,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Reward lifting the body after it reaches the platform approach edge."""
  asset: Entity = env.scene[asset_cfg.name]
  relative_pos = asset.data.root_link_pos_w - env.scene.env_origins
  edge_gate = (
    (relative_pos[:, 0] - edge_gate_x_range[0])
    / (edge_gate_x_range[1] - edge_gate_x_range[0])
  ).clamp(0.0, 1.0)
  lift = ((relative_pos[:, 2] - low_root_z) / (top_root_z - low_root_z)).clamp(
    0.0, 1.0
  )
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
  """Reward front-foot clearance while the robot is at the platform edge."""
  asset: Entity = env.scene[asset_cfg.name]
  relative_x = asset.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
  edge_gate = (
    (relative_x - edge_gate_x_range[0])
    / (edge_gate_x_range[1] - edge_gate_x_range[0])
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
  """Reward forward motion while the front feet are braced at the edge.

  ``platform_front_lift`` encourages the front feet to clear the vertical
  face, but it does not tell the policy to keep driving the body forward once
  that clearance has happened.  This term supplies that missing coupling: the
  reward increases across an x-window that extends past the edge and is gated
  by the number of contacting front feet.  A robot that merely drives into an
  empty area therefore receives no extra signal, while a braced front-foot
  configuration is encouraged to translate over the edge.
  """
  asset: Entity = env.scene[asset_cfg.name]
  relative_x = asset.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
  edge_progress = (
    (relative_x - edge_gate_x_range[0])
    / (edge_gate_x_range[1] - edge_gate_x_range[0])
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
