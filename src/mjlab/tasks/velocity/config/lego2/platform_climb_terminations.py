"""旧固定高台任务的成功结束条件。

platform_climb_env_cfg.py传入台面区域、有效接触数和连续保持步数。
与platform_climb_metrics.py的台面保持检查相似，但这里返回是否结束回合。
这是旧任务的定义，不能代替Blind-Pit的四轮稳定通过标准。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class platform_success:
  """按旧任务的台面区域、接触数和保持步数决定是否成功结束。"""

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
    """连续满足配置要求达到hold_steps后，返回结束回合的标记。"""
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
    return self._hold_steps >= hold_steps
