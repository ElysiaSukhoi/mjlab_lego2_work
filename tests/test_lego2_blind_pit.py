"""Stage 1 observation contract and two-environment reset integration checks."""

from copy import deepcopy

import pytest
import torch
from conftest import get_test_device

from mjlab.envs import ManagerBasedRlEnv, mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.config.lego2.blind_pit_env_cfg import (
  lego2_blind_pit_env_cfg,
)
from mjlab.tasks.velocity.config.lego2.env_cfgs import lego2_rough_env_cfg
from mjlab.tasks.velocity.config.lego2.rl_cfg import lego2_ppo_runner_cfg

TERMS = {
  "base_ang_vel": (mdp.builtin_sensor, 3, 5),
  "projected_gravity": (mdp.projected_gravity, 3, 5),
  "joint_pos": (mdp.joint_pos_rel, 8, 5),
  "joint_vel": (mdp.joint_vel_rel, 12, 5),
  "actions": (mdp.last_action, 12, 0),
  "command": (mdp.generated_commands_xz, 2, 0),
}


@pytest.mark.parametrize("play", [False, True])
def test_actor_whitelist_and_independent_critic(play):
  cfg = lego2_blind_pit_env_cfg(play)
  actor, critic = cfg.observations["actor"], cfg.observations["critic"]
  assert tuple(actor.terms) == tuple(TERMS)
  assert actor.history_length is None
  assert actor.enable_corruption is not play
  for name, (func, _, history) in TERMS.items():
    term = actor.terms[name]
    assert term.func is func
    assert term.history_length == history
    assert term.flatten_history_dim
    assert term is not critic.terms[name]
    assert term.params is not critic.terms[name].params
  assert actor.terms["base_ang_vel"].params == {"sensor_name": "robot/imu_ang_vel"}
  assert "height_scan" in critic.terms and "base_lin_vel" in critic.terms
  assert all(t.history_length == 0 for t in critic.terms.values())
  actor.terms["joint_pos"].params["asset_cfg"].joint_names = ("RF_Wheel",)
  assert critic.terms["joint_pos"].params["asset_cfg"].joint_names != ("RF_Wheel",)
  assert "height_scan" in lego2_rough_env_cfg().observations["actor"].terms


def test_registration_and_shared_train_play_contract():
  train = load_env_cfg("Mjlab-Blind-Pit-Lego2")
  play = load_env_cfg("Mjlab-Blind-Pit-Lego2", play=True)
  assert train.actions == play.actions
  assert train.episode_length_s == play.episode_length_s
  for name in TERMS:
    a, b = (
      train.observations["actor"].terms[name],
      play.observations["actor"].terms[name],
    )
    assert a.func is b.func and a.history_length == b.history_length
    assert a.params == b.params
  agent = load_rl_cfg("Mjlab-Blind-Pit-Lego2")
  assert agent == lego2_ppo_runner_cfg(experiment_name="lego2_blind_pit")


@pytest.fixture
def observation_env():
  # Diagnostic-only plane/reset overrides avoid building Rough's entire grid.
  # Keep real action, command, actor/critic, sensor and environment reset paths.
  cfg = deepcopy(lego2_blind_pit_env_cfg(play=True))
  cfg.scene.num_envs = 2
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.events = {"reset": EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")}
  cfg.curriculum = {}
  cfg.rewards = {}
  cfg.metrics = {}
  cfg.terminations = {}
  cfg.seed = 7
  for group in cfg.observations.values():
    group.nan_policy = "error"
  env = ManagerBasedRlEnv(cfg, device=get_test_device())
  yield env
  env.close()


def test_real_observations_term_major_and_partial_reset(observation_env):
  env = observation_env
  obs, _ = env.reset(seed=7)
  assert obs["actor"].shape == (2, 144)
  manager = env.observation_manager
  assert manager.group_obs_term_dim["actor"] == [
    (15,),
    (15,),
    (40,),
    (60,),
    (12,),
    (2,),
  ]
  assert obs["critic"].shape[1] != 144
  action = torch.arange(24, device=env.device).reshape(2, 12).float() * 0.005
  resolved = dict(zip(TERMS, manager._group_obs_term_cfgs["actor"], strict=True))

  def current_value(name):
    term = resolved[name]
    value = term.func(env, **term.params).clone()
    return value if term.scale is None else value * term.scale

  frames = {name: [current_value(name)] * 5 for name in list(TERMS)[:4]}
  for _ in range(3):
    obs, _, _, _, _ = env.step(action)
    for name in frames:
      frames[name].append(current_value(name))
  before = obs["actor"].clone()
  assert torch.isfinite(before).all()
  assert torch.equal(before[:, 130:142], action)
  offset = 0
  for name, (_, dim, history) in TERMS.items():
    if history:
      expected = torch.stack(frames[name][-5:], dim=1).reshape(2, -1)
      assert torch.allclose(before[:, offset : offset + dim * history], expected)
      offset += dim * history
  # Different per-episode state ensures reset cannot silently reuse old rows.
  env.reset(env_ids=torch.tensor([0], device=env.device))
  after = env.obs_buf["actor"]
  assert torch.equal(after[1], before[1])
  offset = 0
  robot = env.scene["robot"]
  for name, (_, dim, history) in TERMS.items():
    if history:
      block = after[0, offset : offset + dim * history].reshape(history, dim)
      assert torch.equal(block, block[0].expand_as(block))
      assert torch.allclose(block[0], current_value(name)[0])
      offset += dim * history
  assert robot.data.joint_pos.shape[1] == 12
  assert torch.equal(after[0, 130:142], torch.zeros(12, device=env.device))
  # Same initial state in all five slots; no zero-padding or old-episode angles.
  assert not torch.equal(after[0, :130], before[0, :130])
