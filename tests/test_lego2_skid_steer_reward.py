"""Smoke tests: Lego2 skid-steer reward formula vs platform kinematics.

Applies the ``wheel_skid_steer_exp`` targets in the training env and checks that
the resulting base twist matches the commanded (v_x, omega_z) convention:

- ``omega > 0`` → left wheels backward, right wheels forward → ``ang_vel_z > 0``
- ``joint_sign`` must map body-frame ``v_x`` to joint velocity the same way the
  plant does (Lego2: ``+qvel`` → base ``+x``).
"""

from __future__ import annotations

import warnings

import pytest
import torch

from conftest import get_test_device
from mjlab.asset_zoo.robots.lego2.lego2_constants import LEGO2_WHEEL_ACTION_SCALE
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.config.lego2.env_cfgs import lego2_flat_env_cfg
from mjlab.tasks.velocity.mdp.rewards import wheel_skid_steer_exp

pytestmark = pytest.mark.slow

_WHEEL_ORDER = ("LF_Wheel", "RF_Wheel", "LH_Wheel", "RH_Wheel")
_SIDE_SIGNS = (1.0, -1.0, 1.0, -1.0)  # LF, RF, LH, RH


def _strip_for_kinematics(cfg) -> None:
  """Keep a minimal flat play env for open-loop wheel kinematics."""
  cfg.scene.num_envs = 1
  cfg.episode_length_s = 1e9
  cfg.terminations.clear()
  cfg.curriculum = {}
  # Drop DR / pushes; keep joint + base reset only.
  for name in list(cfg.events):
    if name not in ("reset_base", "reset_robot_joints"):
      cfg.events.pop(name, None)


def _skid_params_from_cfg(cfg) -> dict:
  term = cfg.rewards["wheel_skid_steer_exp"]
  return {
    "side_signs": tuple(term.params["side_signs"]),
    "half_track": float(term.params["half_track"]),
    "wheel_radius": float(term.params["wheel_radius"]),
    "joint_sign": float(term.params["joint_sign"]),
  }


def _expected_wheel_qvel(
  v_cmd: float,
  omega_cmd: float,
  *,
  side_signs: tuple[float, ...],
  half_track: float,
  wheel_radius: float,
  joint_sign: float,
  device: str,
) -> torch.Tensor:
  side = torch.as_tensor(side_signs, device=device, dtype=torch.float32)
  v_x = v_cmd - side * omega_cmd * half_track
  return joint_sign * v_x / wheel_radius


def _make_action(env: ManagerBasedRlEnv, wheel_qvel: torch.Tensor) -> torch.Tensor:
  """Leg hold (zeros) + wheel velocity actions for the formula targets."""
  action = torch.zeros(
    env.num_envs, env.action_manager.total_action_dim, device=env.device
  )
  n_leg = env.action_manager._terms["joint_pos"].action_dim
  scale = float(LEGO2_WHEEL_ACTION_SCALE)
  action[:, n_leg : n_leg + wheel_qvel.numel()] = wheel_qvel.unsqueeze(0) / scale
  return action


def _settle_twist(
  env: ManagerBasedRlEnv,
  action: torch.Tensor,
  *,
  n_warm: int = 80,
  n_meas: int = 40,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Step open-loop; return mean body lin/ang vel and measured wheel qvel."""
  lins: list[torch.Tensor] = []
  angs: list[torch.Tensor] = []
  for i in range(n_warm + n_meas):
    env.step(action)
    if i < n_warm:
      continue
    robot = env.scene["robot"]
    lins.append(robot.data.root_link_lin_vel_b[0].detach().clone())
    angs.append(robot.data.root_link_ang_vel_b[0].detach().clone())
  wheel_ids = env.action_manager._terms["joint_vel"]._target_ids
  jv = env.scene["robot"].data.joint_vel[0, wheel_ids].detach().clone()
  return torch.stack(lins).mean(0), torch.stack(angs).mean(0), jv


@pytest.fixture(scope="module")
def device() -> str:
  return get_test_device()


@pytest.fixture(scope="module")
def skid_env(device: str):
  """Training-shaped Lego2 flat env stripped for open-loop smoke."""
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    cfg = lego2_flat_env_cfg(play=False)
    _strip_for_kinematics(cfg)
    # Keep skid reward params for assertions; clear other rewards for speed.
    skid = cfg.rewards["wheel_skid_steer_exp"]
    cfg.rewards = {"wheel_skid_steer_exp": skid}
    env = ManagerBasedRlEnv(cfg, device=device)
  yield env, _skid_params_from_cfg(cfg)
  env.close()


def test_skid_formula_left_right_signs_for_positive_yaw():
  """ω>0 must command left backward and right forward in body v_x."""
  side = torch.tensor(_SIDE_SIGNS)
  half_track = 0.187
  omega = 0.5
  v_x = 0.0 - side * omega * half_track
  # Left (LF, LH): negative v_x (backward); right (RF, RH): positive (forward).
  assert torch.all(v_x[[0, 2]] < 0)
  assert torch.all(v_x[[1, 3]] > 0)


def test_skid_formula_matches_forward_platform_motion(skid_env):
  """Applying formula wheel targets for +v_x must drive the base forward."""
  env, params = skid_env
  assert params["joint_sign"] > 0, (
    "Lego2 plant maps +qvel → +base_x; joint_sign must be +1 "
    f"(got {params['joint_sign']})"
  )

  env.reset()
  v_cmd, omega_cmd = 0.5, 0.0
  qvel = _expected_wheel_qvel(v_cmd, omega_cmd, device=env.device, **params)
  lin, ang, jv = _settle_twist(env, _make_action(env, qvel))

  assert torch.allclose(jv, qvel, atol=0.35), (
    f"wheels failed to track targets: got {jv.tolist()} want {qvel.tolist()}"
  )
  assert lin[0].item() > 0.35, f"expected forward motion, lin_x={lin[0].item():.3f}"
  assert abs(lin[0].item() - v_cmd) < 0.15, (
    f"lin_x={lin[0].item():.3f} not near v_cmd={v_cmd}"
  )
  assert abs(ang[2].item()) < 0.08, f"unexpected yaw during pure forward: {ang[2]}"


def test_skid_formula_matches_backward_platform_motion(skid_env):
  env, params = skid_env
  env.reset()
  v_cmd, omega_cmd = -0.5, 0.0
  qvel = _expected_wheel_qvel(v_cmd, omega_cmd, device=env.device, **params)
  lin, ang, _ = _settle_twist(env, _make_action(env, qvel))

  assert lin[0].item() < -0.35, f"expected backward motion, lin_x={lin[0].item():.3f}"
  assert abs(lin[0].item() - v_cmd) < 0.15
  assert abs(ang[2].item()) < 0.08


def test_wrong_joint_sign_drives_opposite_forward(skid_env):
  """Sanity: flipping joint_sign inverts plant response vs command."""
  env, params = skid_env
  bad = dict(params)
  bad["joint_sign"] = -float(params["joint_sign"])

  env.reset()
  qvel = _expected_wheel_qvel(0.5, 0.0, device=env.device, **bad)
  lin, _, _ = _settle_twist(env, _make_action(env, qvel))
  assert lin[0].item() < -0.35, (
    "negated joint_sign should drive base opposite to +v_cmd"
  )


def test_positive_yaw_targets_are_left_back_right_forward(skid_env):
  """Formula targets for ω>0: left qvel more negative than right (joint_sign>0)."""
  _, params = skid_env
  qvel = _expected_wheel_qvel(0.0, 0.8, device="cpu", **params)
  left = qvel[[0, 2]]
  right = qvel[[1, 3]]
  assert torch.all(left < 0) and torch.all(right > 0), (
    f"ω>0 should be left back / right forward, got qvel={qvel.tolist()}"
  )


def test_open_loop_differential_yaw_sign(skid_env):
  """Open-loop L-back/R-fwd must yaw the same sign as +omega command.

  Pure spin magnitude is weak under default foot friction, so we only assert
  the yaw *sign* relative to the opposite differential.
  """
  env, params = skid_env
  assert params["joint_sign"] > 0

  # Strong equal-magnitude differential matching ω>0 targets (left-, right+).
  q_pos = torch.tensor([-5.0, 5.0, -5.0, 5.0], device=env.device)
  q_neg = -q_pos

  env.reset()
  _, ang_pos, _ = _settle_twist(
    env, _make_action(env, q_pos), n_warm=100, n_meas=50
  )
  env.reset()
  _, ang_neg, _ = _settle_twist(
    env, _make_action(env, q_neg), n_warm=100, n_meas=50
  )

  # Left-back/right-fwd should be more positive yaw than the opposite pattern.
  assert ang_pos[2].item() > ang_neg[2].item(), (
    f"skid pattern for +ω should yaw more positive than opposite: "
    f"ang_z(+pattern)={ang_pos[2].item():.4f} "
    f"ang_z(-pattern)={ang_neg[2].item():.4f}"
  )


def test_skid_reward_high_when_wheels_match_formula(skid_env):
  """After tracking formula targets, ``wheel_skid_steer_exp`` should be near 1."""
  env, params = skid_env
  env.reset()

  twist = env.command_manager.get_term("twist")
  assert twist is not None
  v_cmd, omega_cmd = 0.5, 0.0
  with torch.no_grad():
    twist.vel_command_b[:, 0] = v_cmd
    twist.vel_command_b[:, 1] = 0.0
    twist.vel_command_b[:, 2] = omega_cmd

  qvel = _expected_wheel_qvel(v_cmd, omega_cmd, device=env.device, **params)
  _settle_twist(env, _make_action(env, qvel), n_warm=60, n_meas=20)

  with torch.no_grad():
    twist.vel_command_b[:, 0] = v_cmd
    twist.vel_command_b[:, 1] = 0.0
    twist.vel_command_b[:, 2] = omega_cmd

  wheel_cfg = SceneEntityCfg(
    "robot", joint_names=_WHEEL_ORDER, preserve_order=True
  )
  wheel_cfg.resolve(env.scene)

  reward = wheel_skid_steer_exp(
    env,
    command_name="twist",
    wheel_cfg=wheel_cfg,
    side_signs=params["side_signs"],
    half_track=params["half_track"],
    wheel_radius=params["wheel_radius"],
    joint_sign=params["joint_sign"],
    std=8.0,
    command_threshold=0.05,
  )
  assert reward.shape == (env.num_envs,)
  assert reward.mean().item() > 0.85, (
    f"expected high skid reward when tracking formula, got {reward.mean().item():.3f}"
  )


def test_skid_reward_low_when_wheel_signs_flipped(skid_env):
  env, params = skid_env
  env.reset()

  twist = env.command_manager.get_term("twist")
  assert twist is not None
  v_cmd, omega_cmd = 0.5, 0.0
  with torch.no_grad():
    twist.vel_command_b[:, 0] = v_cmd
    twist.vel_command_b[:, 1] = 0.0
    twist.vel_command_b[:, 2] = omega_cmd

  # Drive with negated targets relative to the reward formula.
  qvel = -_expected_wheel_qvel(v_cmd, omega_cmd, device=env.device, **params)
  _settle_twist(env, _make_action(env, qvel), n_warm=60, n_meas=20)

  with torch.no_grad():
    twist.vel_command_b[:, 0] = v_cmd
    twist.vel_command_b[:, 1] = 0.0
    twist.vel_command_b[:, 2] = omega_cmd

  wheel_cfg = SceneEntityCfg(
    "robot", joint_names=_WHEEL_ORDER, preserve_order=True
  )
  wheel_cfg.resolve(env.scene)
  reward = wheel_skid_steer_exp(
    env,
    command_name="twist",
    wheel_cfg=wheel_cfg,
    side_signs=params["side_signs"],
    half_track=params["half_track"],
    wheel_radius=params["wheel_radius"],
    joint_sign=params["joint_sign"],
    std=8.0,
    command_threshold=0.05,
  )
  assert reward.mean().item() < 0.4, (
    f"flipped wheel targets should score low, got {reward.mean().item():.3f}"
  )
