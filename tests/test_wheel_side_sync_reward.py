"""Unit tests for ``wheel_side_sync_exp`` (soft same-side + pure v/ω pairing)."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp.rewards import wheel_side_sync_exp

_SIDE = (1.0, -1.0, 1.0, -1.0)  # LF, RF, LH, RH


def _make_env(wheel_vel: torch.Tensor, command: torch.Tensor) -> MagicMock:
  """Mock env with upright robot and fixed wheel joint velocities.

  ``wheel_vel``: [B, 4] in LF, RF, LH, RH order.
  ``command``: [B, 3] twist (lin_x, lin_y, ang_z).
  """
  B = wheel_vel.shape[0]
  asset = MagicMock()
  # Full joint_vel buffer; reward indexes via joint_ids [0,1,2,3].
  asset.data.joint_vel = wheel_vel
  # Fully upright → _upright_scale == 1.
  asset.data.projected_gravity_b = torch.zeros(B, 3)
  asset.data.projected_gravity_b[:, 2] = -1.0

  env = MagicMock()
  env.device = wheel_vel.device
  env.scene.__getitem__ = MagicMock(return_value=asset)
  env.command_manager.get_command = MagicMock(return_value=command)
  env.extras = {"log": {}}
  return env


def _reward(wheel_vel: torch.Tensor, command: torch.Tensor, **kwargs) -> torch.Tensor:
  env = _make_env(wheel_vel, command)
  wheel_cfg = SceneEntityCfg("robot")
  wheel_cfg.joint_ids = [0, 1, 2, 3]
  return wheel_side_sync_exp(
    env,
    command_name="twist",
    wheel_cfg=wheel_cfg,
    side_signs=_SIDE,
    **kwargs,
  )


def test_inactive_when_command_near_zero():
  vel = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
  cmd = torch.tensor([[0.01, 0.0, 0.01]])
  r = _reward(vel, cmd, command_threshold=0.05)
  assert torch.allclose(r, torch.zeros(1))


def test_perfect_same_side_mixed_command_near_one():
  # Mixed v+ω: only same-side term; L≠R is OK.
  vel = torch.tensor([[2.0, -1.0, 2.0, -1.0]])
  cmd = torch.tensor([[1.0, 0.0, 0.5]])
  r = _reward(vel, cmd)
  assert r.item() > 0.99


def test_pure_lin_penalizes_left_right_mismatch():
  cmd = torch.tensor([[1.0, 0.0, 0.0]])  # pure v
  good = _reward(torch.tensor([[2.0, 2.0, 2.0, 2.0]]), cmd)
  bad = _reward(torch.tensor([[2.0, -2.0, 2.0, -2.0]]), cmd)
  assert good.item() > 0.99
  assert bad.item() < good.item()


def test_pure_ang_penalizes_same_sign_sides():
  cmd = torch.tensor([[0.0, 0.0, 1.0]])  # pure ω
  good = _reward(torch.tensor([[-2.0, 2.0, -2.0, 2.0]]), cmd)
  bad = _reward(torch.tensor([[2.0, 2.0, 2.0, 2.0]]), cmd)
  assert good.item() > 0.99
  assert bad.item() < good.item()


def test_same_side_mismatch_hurts_even_on_mixed():
  cmd = torch.tensor([[1.0, 0.0, 0.5]])
  synced = _reward(torch.tensor([[2.0, -1.0, 2.0, -1.0]]), cmd)
  twisted = _reward(torch.tensor([[2.0, -1.0, -2.0, 1.0]]), cmd)
  assert synced.item() > twisted.item()


def test_lego2_env_wires_wheel_side_sync():
  from mjlab.tasks.velocity.config.lego2.env_cfgs import lego2_rough_env_cfg

  cfg = lego2_rough_env_cfg()
  assert "wheel_side_sync_exp" in cfg.rewards
  term = cfg.rewards["wheel_side_sync_exp"]
  assert term.weight == 0.5
  assert term.params["side_signs"] == (1.0, -1.0, 1.0, -1.0)
  assert term.params["wheel_cfg"].preserve_order is True
