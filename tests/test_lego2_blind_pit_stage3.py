"""Reward scale, stable-pass state and real fixed-height evaluation contracts."""

from dataclasses import asdict
from types import SimpleNamespace
from typing import cast

import mujoco
import numpy as np
import pytest
import torch
from conftest import get_test_device

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.config.lego2 import blind_pit_metrics as module
from mjlab.tasks.velocity.config.lego2.blind_pit_env_cfg import (
  lego2_blind_pit_env_cfg,
  lego2_blind_pit_eval_cfg,
)
from mjlab.tasks.velocity.config.lego2.blind_pit_metrics import PitProgress, pit_stages
from mjlab.terrains import BoxInvertedPyramidStairsTerrainCfg


def test_reward_baseline_and_eval_factory():
  train = lego2_blind_pit_env_cfg()
  expected = dict(
    zip(
      (
        "track_lin_vel_xy_exp",
        "track_ang_vel_z_exp",
        "lin_vel_z_l2",
        "ang_vel_xy_l2",
        "flat_orientation_l2",
        "joint_torques_l2",
        "joint_acc_l2",
        "joint_acc_wheel_l2",
        "action_rate_l2",
        "action_rate_wheel_l2",
        "joint_pos_limits",
      ),
      (3, 1.5, -0.3, -0.05, -0.5, -2.5e-5, -2.5e-7, -2.5e-9, -0.01, -0.01, -5),
      strict=True,
    )
  )
  assert {k: v.weight for k, v in train.rewards.items()} == expected
  for height in (0.03, 0.05, 0.10, 0.15, 0.20, 0.30):
    cfg = lego2_blind_pit_eval_cfg(height)
    assert cfg.curriculum == {} and not cfg.auto_reset
    assert cfg.episode_length_s == 20
    assert cfg.actions == train.actions
    assert set(cfg.terminations) == set(train.terminations)
    assert cfg.scene.terrain is not None
    gen = cfg.scene.terrain.terrain_generator
    assert gen is not None
    assert isinstance(gen.sub_terrains["pit"], BoxInvertedPyramidStairsTerrainCfg)
    assert gen.sub_terrains["pit"].step_height_range == (height, height)
    assert gen.num_rows == 2
  for height in (0, -0.1, 0.31, float("nan")):
    with pytest.raises(ValueError):
      lego2_blind_pit_eval_cfg(height)


def test_named_stages_reject_partial_wall_and_tilt():
  centers = torch.tensor(
    [[[3.0, 0.0, 0.115], [3.0, 0.0, 0.115], [2.3, 0.0, 0.115], [2.3, 0.0, 0.115]]]
  ).repeat(4, 1, 1)
  extents = torch.tensor([0.115, 0.026, 0.115]).expand_as(centers)
  contact = SimpleNamespace(
    found=torch.ones(4, 4),
    force=torch.zeros(4, 4, 3),
    normal=torch.zeros(4, 4, 3),
    pos=torch.zeros(4, 4, 3),
  )
  contact.force[..., 2] = -100
  contact.normal[..., 2] = -1
  centers[1, 0, 0] = 0
  contact.normal[2, 2] = torch.tensor([1.0, 0.0, 0.0])
  gravity = torch.tensor([[0.0, 0.0, -1.0]]).repeat(4, 1)
  gravity[3] = torch.tensor([1.0, 0.0, 0.0])
  # Deliberately interleaved front/rear order.
  stages = pit_stages(
    centers,
    extents,
    torch.zeros(4, 3),
    torch.zeros(4),
    contact,
    torch.tensor([[3.0, 0.0, 0.47]]).repeat(4, 1),
    gravity,
    [0, 2],
    [1, 3],
  )
  assert stages["front_up"].tolist() == [True, False, False, True]
  assert stages["rear_up"].all()
  assert stages["stable_now"].tolist() == [True, False, False, False]


@pytest.fixture
def stage3_env():
  cfg = lego2_blind_pit_eval_cfg(0.05)
  cfg.scene.num_envs = 2
  cfg.seed = 7
  env = ManagerBasedRlEnv(cfg, device=get_test_device())
  yield env
  env.close()


def test_real_height_rewards_state_and_partial_reset(stage3_env):
  env = stage3_env
  obs, extras = env.reset()
  assert obs["actor"].shape == (2, 144) and obs["critic"].shape == (2, 254)
  assert all(torch.isfinite(v) for k, v in extras["log"].items() if "pit_" in k)
  terrain = env.scene.terrain
  assert torch.allclose(
    terrain.terrain_origins[..., 2],
    torch.full_like(terrain.terrain_origins[..., 2], -0.05),
  )
  model = env.sim.mj_model
  native = mujoco.MjData(model)
  mujoco.mj_forward(model, native)
  for origin in terrain.terrain_origins[:, 0].cpu().numpy():
    for dx, expected in ((0, -0.05), (2, 0)):
      ray = origin.copy()
      ray[0] += dx
      ray[2] = 2
      dist = mujoco.mj_ray(
        model,
        native,
        ray,
        np.array([0.0, 0.0, -1.0]),
        np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8),
        True,
        -1,
        np.array([-1], dtype=np.int32),
      )
      assert 2 - dist == pytest.approx(expected, abs=1e-6)
  robot = env.scene["robot"]
  pose = robot.data.root_link_pose_w.clone()
  pose[:, 0] = env.scene.env_origins[:, 0] + 3.0
  pose[:, 1] = env.scene.env_origins[:, 1]
  pose[:, 2] = 0.47
  pose[:, 3:] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
  robot.write_root_link_pose_to_sim(pose)
  action = torch.zeros(2, 12, device=env.device)
  progress = PitProgress(env.cfg.metrics["pit_stable_pass"], env)
  reward = torch.zeros(2, device=env.device)
  for _ in range(65):
    _, reward, terminated, _, _ = env.step(action)
    assert not terminated.any()
    assert torch.isfinite(reward).all()
    progress(env)
  assert progress.passed.all()
  hold = progress.hold.clone()
  progress(env)
  progress(env)
  assert torch.equal(progress.hold, hold)
  progress.reset(torch.tensor([0], device=env.device))
  assert progress.hold[0] == 0 and progress.hold[1] == hold[1]
  assert not progress.passed[0] and progress.passed[1]
  # Resolve actuator vs joint selectors through the real manager.
  torque = env.reward_manager.get_term_cfg("joint_torques_l2")
  selected = torque.params["asset_cfg"].actuator_ids
  assert len(selected) == 8
  raw = torque.func(env, **torque.params)
  assert torch.allclose(raw, robot.data.actuator_force[:, selected].square().sum(-1))
  rates = [v[0] for _, v in env.reward_manager.get_active_iterable_terms(0)]
  assert float(reward[0]) == pytest.approx(sum(rates) * env.step_dt, abs=1e-6)
  # Losing contact interrupts continuous hold, even at rim height.
  pose = robot.data.root_link_pose_w.clone()
  pose[:, 2] = 1.5
  robot.write_root_link_pose_to_sim(pose)
  for _ in range(3):
    env.step(action)
    progress(env)
  assert torch.count_nonzero(progress.hold) == 0
  managed = env.metrics_manager.cfg["pit_stable_pass"].func
  assert managed.passed.all()
  env.reset(env_ids=torch.tensor([0], device=env.device))
  assert managed.passed.tolist() == [False, True]


@pytest.mark.parametrize("inference", [False, True])
def test_hold_updates_once_and_failure_cannot_finish(monkeypatch, inference):
  stable = torch.ones(2, dtype=torch.bool)
  monkeypatch.setattr(module, "pit_stages", lambda *args: {"stable_now": stable})
  metric = object.__new__(PitProgress)
  metric.front, metric.rear = [0, 1], [2, 3]
  metric.depths = torch.tensor([0.03])
  metric.hold = torch.zeros(2)
  metric.passed = torch.zeros(2, dtype=torch.bool)
  metric.last_step = torch.full((2,), -1, dtype=torch.long)
  metric.values = {}
  metric.wheels = lambda: (torch.zeros(2, 4, 3), torch.zeros(2, 4, 3))
  monkeypatch.setattr(metric, "contact", SimpleNamespace(data=None), raising=False)
  monkeypatch.setattr(
    metric,
    "robot",
    SimpleNamespace(
      data=SimpleNamespace(root_link_pos_w=None, projected_gravity_b=None)
    ),
    raising=False,
  )
  env = SimpleNamespace(
    scene=SimpleNamespace(
      terrain=SimpleNamespace(terrain_levels=torch.zeros(2, dtype=torch.long)),
      env_origins=torch.zeros(2, 3),
    ),
    step_dt=0.02,
    termination_manager=SimpleNamespace(terminated=torch.zeros(2, dtype=torch.bool)),
  )
  fake_env = cast(ManagerBasedRlEnv, env)
  for step in range(24):
    env.common_step_counter = step
    with torch.inference_mode(inference):
      metric(fake_env)
      metric(fake_env)
    assert all(not v.is_inference() for v in metric.values.values())
  assert not metric.passed.any()
  env.termination_manager.terminated[1] = True
  env.common_step_counter = 24
  metric(fake_env)
  assert metric.passed.tolist() == [True, False]
  assert metric.hold[1] == 0
  metric.reset(torch.tensor([1]))
  assert metric.passed[0] and metric.hold[0] > 0
  assert metric.values["stable_pass"].tolist() == [1.0, 0.0]
  buffers = {name: id(value) for name, value in metric.values.items()}
  with torch.inference_mode(inference):
    metric(fake_env)
  assert buffers == {name: id(value) for name, value in metric.values.items()}
  assert metric.hold[0] == pytest.approx(0.5)
  metric.reset()
  assert all(torch.count_nonzero(v) == 0 for v in metric.values.values())


@pytest.mark.parametrize("auto_reset", [False, True])
def test_inference_checkpoint_contract(stage3_env, tmp_path, auto_reset):
  env = stage3_env
  env.cfg.auto_reset = auto_reset
  agent = load_rl_cfg("Mjlab-Blind-Pit-Lego2")
  wrapper = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner_cls = load_runner_cls("Mjlab-Blind-Pit-Lego2")
  assert runner_cls is not None
  runner = runner_cls(wrapper, asdict(agent), device=env.device)
  # Temporary random weights only: exercise save/load/inference without PPO.
  checkpoint = tmp_path / "untrained.pt"
  runner.save(str(checkpoint))
  runner.load(
    str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=env.device
  )
  for _ in range(2):
    obs, _ = wrapper.reset()
    if auto_reset:
      env.episode_length_buf[0] = env.max_episode_length - 1
    with torch.inference_mode():
      action = runner.get_inference_policy(device=env.device)(obs)
      assert action.shape == (2, 12) and torch.isfinite(action).all()
      wrapper.step(action)
    # Evaluator resets between episodes outside the inference context.
    obs, _ = wrapper.reset()
    assert obs["actor"].shape == (2, 144)
