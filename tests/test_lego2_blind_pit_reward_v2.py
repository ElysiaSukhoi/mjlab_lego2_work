"""Reward-v2 isolation, climb freedom and actual reward-manager integration."""

import math
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from conftest import get_test_device

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.config.lego2.blind_pit_curriculums import (
  nonfoot_contact_count,
)
from mjlab.tasks.velocity.config.lego2.blind_pit_env_cfg import lego2_blind_pit_env_cfg
from mjlab.tasks.velocity.config.lego2.blind_pit_reward_v2_env_cfg import (
  lego2_blind_pit_reward_v2_env_cfg,
  lego2_blind_pit_reward_v2_eval_cfg,
)
from mjlab.tasks.velocity.config.lego2.blind_pit_rewards import (
  base_clearance_below_minimum,
  base_clearance_eligible,
  minimum_base_clearance_l2,
  pitch_relaxed_orientation_l2,
)
from mjlab.terrains import BoxInvertedPyramidStairsTerrainCfg


def test_v2_isolation_and_shared_train_play_contract():
  baseline = lego2_blind_pit_env_cfg()
  candidate = load_env_cfg("Mjlab-Blind-Pit-Lego2-RewardV2")
  play = lego2_blind_pit_reward_v2_env_cfg(play=True)
  assert len(baseline.rewards) == 11 and len(candidate.rewards) == 13
  assert baseline.rewards["flat_orientation_l2"].weight == -0.5
  assert "flat_orientation_l2" not in candidate.rewards
  assert candidate.rewards == play.rewards
  assert candidate.actions == baseline.actions
  assert candidate.observations == baseline.observations
  assert candidate.scene.entities == baseline.scene.entities
  assert candidate.scene.sensors == baseline.scene.sensors
  assert candidate.scene.terrain is not None and baseline.scene.terrain is not None
  assert (
    candidate.scene.terrain.terrain_generator
    == baseline.scene.terrain.terrain_generator
  )
  assert (
    candidate.scene.terrain.max_init_terrain_level
    == baseline.scene.terrain.max_init_terrain_level
  )
  assert candidate.terminations == baseline.terminations
  assert candidate.curriculum == baseline.curriculum
  for name, term in baseline.rewards.items():
    if name != "flat_orientation_l2":
      assert candidate.rewards[name] == term
  assert load_env_cfg("Mjlab-Blind-Pit-Lego2").rewards == baseline.rewards
  old_agent = load_rl_cfg("Mjlab-Blind-Pit-Lego2")
  new_agent = load_rl_cfg("Mjlab-Blind-Pit-Lego2-RewardV2")
  assert isinstance(old_agent, RslRlOnPolicyRunnerCfg)
  assert isinstance(new_agent, RslRlOnPolicyRunnerCfg)
  assert (
    new_agent.actor == old_agent.actor and new_agent.algorithm == old_agent.algorithm
  )
  assert new_agent.experiment_name == "lego2_blind_pit_reward_v2"
  assert new_agent.experiment_name != old_agent.experiment_name
  fixed = lego2_blind_pit_reward_v2_eval_cfg(0.3, 0.5)
  assert fixed.scene.terrain is not None
  gen = fixed.scene.terrain.terrain_generator
  assert gen is not None
  pit = gen.sub_terrains["pit"]
  assert isinstance(pit, BoxInvertedPyramidStairsTerrainCfg)
  assert pit.step_height_range == (0.3, 0.3)
  assert fixed.rewards == candidate.rewards and not fixed.auto_reset


def test_pitch_deadband_keeps_roll_cost_and_penalizes_excess_pitch():
  pitches = torch.deg2rad(torch.tensor([0.0, 20.0, -20.0, 35.0, 0.0]))
  gravity = torch.stack((pitches.sin(), torch.zeros(5), -pitches.cos()), dim=1)
  gravity[4] = torch.tensor(
    [0.0, math.sin(math.radians(15)), -math.cos(math.radians(15))]
  )
  env = cast(
    ManagerBasedRlEnv,
    SimpleNamespace(
      scene={
        "robot": SimpleNamespace(data=SimpleNamespace(projected_gravity_b=gravity))
      }
    ),
  )
  costs = pitch_relaxed_orientation_l2(env)
  assert costs[:3].tolist() == [0, 0, 0]
  assert costs[3] == pytest.approx(
    (math.sin(math.radians(35)) - math.sin(math.radians(25))) ** 2
  )
  assert costs[4] == pytest.approx(math.sin(math.radians(15)) ** 2)


def test_clearance_is_local_one_sided_and_disabled_at_edges_per_environment():
  z = torch.tensor(
    [[0.0, 0.0], [0.0, 0.0], [0.3, 0.3], [0.0, 0.3], [float("nan"), 0.0], [0.0, 0.0]]
  )
  hits = torch.zeros(6, 2, 3)
  hits[..., 2] = z
  normals = torch.zeros_like(hits)
  normals[..., 2] = 1
  distances = torch.ones_like(z)
  distances[5, 0] = -1  # Missed ray affects only this row.
  root = torch.zeros(6, 3)
  root[:, 2] = torch.tensor([0.45, 0.22, 0.52, 0.22, 0.22, 0.22])
  scene = {
    "robot": SimpleNamespace(data=SimpleNamespace(root_link_pos_w=root)),
    "scan": SimpleNamespace(
      data=SimpleNamespace(hit_pos_w=hits, normals_w=normals, distances=distances)
    ),
  }
  env = cast(ManagerBasedRlEnv, SimpleNamespace(scene=scene, num_envs=6))
  costs = minimum_base_clearance_l2(env, "scan")
  expected = ((0.34 - 0.22) / 0.34) ** 2
  assert costs.tolist() == pytest.approx([0, expected, expected, 0, 0, 0])
  assert torch.isfinite(costs).all()
  assert base_clearance_eligible(env, "scan").tolist() == [1, 1, 1, 0, 0, 0]
  assert base_clearance_below_minimum(env, "scan").tolist() == [0, 1, 1, 0, 0, 0]


def test_nonfoot_cost_uses_force_norm_and_keeps_tipped_contacts():
  contact = SimpleNamespace(
    found=torch.tensor([[1, 1, 0], [1, 1, 0]]),
    force=torch.tensor(
      [
        [[0.0, 0.0, 2.0], [0.0, 0.5, 0.0], [9.0, 0.0, 0.0]],
        [[2.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 9.0]],
      ]
    ),
  )
  env = cast(
    ManagerBasedRlEnv,
    SimpleNamespace(scene={"undesired_ground_contact": SimpleNamespace(data=contact)}),
  )
  assert nonfoot_contact_count(env).tolist() == [1, 1]


def test_real_v2_rewards_observations_and_inference_reset():
  cfg = lego2_blind_pit_reward_v2_eval_cfg(0.03, 0.5)
  cfg.scene.num_envs = 2
  cfg.seed = 7
  env = ManagerBasedRlEnv(cfg, device=get_test_device())
  try:
    obs, _ = env.reset()
    assert isinstance(obs["actor"], torch.Tensor)
    assert isinstance(obs["critic"], torch.Tensor)
    assert obs["actor"].shape == (2, 144) and obs["critic"].shape == (2, 254)
    assert len(env.reward_manager.active_terms) == 13
    eligible = base_clearance_eligible(env, "terrain_scan")
    assert eligible.tolist() == [1, 1]
    assert minimum_base_clearance_l2(env, "terrain_scan").tolist() == [0, 0]
    action = torch.zeros(2, 12, device=env.device)
    for _ in range(4):
      with torch.inference_mode():
        _, reward, terminated, _, _ = env.step(action)
      assert torch.isfinite(reward).all() and not terminated.any()
      rates = [
        torch.tensor(v, device=env.device)
        for _, v in env.reward_manager.get_active_iterable_terms(0)
      ]
      assert float(reward[0]) == pytest.approx(
        float(sum(rates)) * env.step_dt, abs=1e-6
      )
    env.reset(env_ids=torch.tensor([0], device=env.device))
    assert obs["actor"].shape == (2, 144)
  finally:
    env.close()
