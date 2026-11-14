"""Stage 2 geometry, curriculum decisions and real pit reset/contact checks."""

import math

import mujoco
import numpy as np
import pytest
import torch
from conftest import get_test_device

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.velocity.config.lego2.blind_pit_curriculums import (
  PitTerrainLevels,
  cylinder_extents,
  level_changes,
  rim_supported,
)
from mjlab.tasks.velocity.config.lego2.blind_pit_env_cfg import lego2_blind_pit_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg


def test_train_play_stage2_contract():
  train, play = lego2_blind_pit_env_cfg(), lego2_blind_pit_env_cfg(play=True)
  assert train.episode_length_s == play.episode_length_s == 20
  assert set(train.events) == {"reset_base", "reset_robot_joints"}
  assert set(play.events) == set(train.events)
  assert set(train.curriculum) == {"pit_levels"} and play.curriculum == {}
  assert set(train.terminations) == {"time_out", "fell_over", "nan", "tile_boundary"}
  assert train.terminations["tile_boundary"].time_out
  assert train.metrics["pit_nonfoot_contact_count"].reduce == "mean"
  assert train.terminations["fell_over"].params["limit_angle"] == math.radians(70)
  for cfg in (train, play):
    command = cfg.commands["twist"]
    assert isinstance(command, UniformVelocityCommandCfg)
    assert command.resampling_time_range == (20, 20)
    assert command.ranges.lin_vel_x == (0.3, 0.6)
    assert command.ranges.lin_vel_y == (0, 0)
    assert command.ranges.heading == (0, 0)
    assert command.heading_command and command.rel_heading_envs == 1
    assert command.heading_control_stiffness == 0.5
    assert command.rel_standing_envs == command.rel_spin_envs == 0
    assert command.rel_forward_envs == command.rel_world_envs == 0
    assert command.init_velocity_prob == 0


def test_support_rejects_wall_air_pit_and_neighbor_contacts():
  centers = torch.tensor(
    [[[3.0, -0.25, 0.115], [3.0, 0.25, 0.115], [2.3, -0.25, 0.115], [2.3, 0.25, 0.115]]]
  ).repeat(8, 1, 1)
  extents = torch.tensor([0.115, 0.026, 0.115]).expand_as(centers).clone()
  found = torch.ones(8, 4)
  force = torch.zeros_like(centers)
  force[..., 2] = -100
  normal = torch.zeros_like(centers)
  normal[..., 2] = -1
  pos = centers.clone()
  pos[..., 2] = 0
  # One invalid wheel is sufficient to reject complete traversal.
  normal[1, 0] = torch.tensor([-1.0, 0, 0])  # Side wall.
  found[2, 0] = 0  # In air, even though geometry looks valid.
  centers[3, 0, 0] = 0  # Pit bottom.
  centers[4, 0, 0] = 1.60  # Wheel partly over pit.
  centers[5, 0, 0] = 4.9  # Neighboring tile.
  force[6, 0] = torch.tensor([100.0, 0, 0])  # Pure horizontal obstruction.
  pos[7, 0, 2] = -0.15  # Contact below rim.
  supported = rim_supported(
    centers, extents, torch.zeros(8, 2), torch.zeros(8), found, force, normal, pos
  )
  assert supported.tolist() == [True, False, False, False, False, False, False, False]
  axes = torch.tensor([[[0.0, 1, 0], [0.0, 0, 1]]])
  sizes = torch.tensor([[0.115, 0.026], [0.115, 0.026]])
  assert torch.allclose(
    cylinder_extents(axes, sizes)[0, :, 2], torch.tensor([0.115, 0.026])
  )


def test_level_decisions_failure_shortfall_and_manual_reset():
  distance = torch.tensor([3.5, 3.5, 3.5, 0.2, 2.5, 3.5, float("nan")])
  expected = torch.full((7,), 4.0)
  supported = torch.tensor([True, True, False, False, False, True, True])
  failed = torch.tensor([False, True, False, False, False, False, False])
  eligible = torch.tensor([True, True, True, True, True, False, True])
  up, down = level_changes(distance, expected, supported, failed, eligible)
  assert up.tolist() == [True, False, False, False, False, False, False]
  assert down.tolist() == [False, True, False, True, False, False, True]


@pytest.fixture
def pit_env():
  cfg = lego2_blind_pit_env_cfg()
  cfg.scene.num_envs = 2
  cfg.seed = 7
  # Rewards are a separate stage; keep full terrain/reset/command/curriculum.
  cfg.rewards = {}
  cfg.metrics = {"pit_nonfoot_contact_count": cfg.metrics["pit_nonfoot_contact_count"]}
  env = ManagerBasedRlEnv(cfg, device=get_test_device())
  yield env
  env.close()


def test_all_ten_heights_and_repeated_pit_spawns(pit_env):
  env = pit_env
  terrain = env.scene.terrain
  model = env.sim.mj_model
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  origins = terrain.terrain_origins[:, 0].cpu().numpy()
  assert origins.shape == (10, 3)
  np.testing.assert_allclose(origins[:, 2], -np.arange(1, 11) * 0.03, atol=1e-7)
  for origin in origins:
    for dx, target in ((0, origin[2]), (2.0, 0)):
      p = origin.copy()
      p[0] += dx
      p[2] = 2
      distance = mujoco.mj_ray(
        model,
        data,
        p,
        np.array([0.0, 0, -1]),
        np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8),
        True,
        -1,
        np.array([-1], dtype=np.int32),
      )
      assert 2 - distance == pytest.approx(target, abs=1e-6)
  assert terrain.terrain_levels.tolist() == [0, 0]
  levels = env.curriculum_manager.get_term_cfg("pit_levels").func
  wheel_ids = levels.robot.indexing.geom_ids[levels.geom_ids].cpu().numpy()
  np.testing.assert_allclose(model.geom_friction[wheel_ids, 0], 0.4)
  for row in range(10):
    ids = torch.arange(2, device=env.device)
    terrain.terrain_levels[:] = row
    terrain.update_env_origins(
      ids,
      torch.zeros(2, dtype=torch.bool, device=env.device),
      torch.zeros(2, dtype=torch.bool, device=env.device),
    )
    obs, _ = env.reset()
    robot = env.scene["robot"]
    offset = robot.data.root_link_pos_w - env.scene.env_origins
    assert (offset[:, :2].abs() <= 0.15 + 1e-6).all()
    assert torch.allclose(offset[:, 2], torch.full((2,), 0.47, device=env.device))
    assert torch.allclose(robot.data.joint_pos, robot.data.default_joint_pos)
    assert torch.count_nonzero(robot.data.root_link_vel_w) == 0
    assert torch.count_nonzero(robot.data.joint_vel) == 0
    assert (robot.data.heading_w.abs() <= 0.05 + 1e-6).all()
    command = env.command_manager.get_command("twist")
    assert ((command[:, 0] >= 0.3) & (command[:, 0] <= 0.6)).all()
    assert torch.count_nonzero(command[:, 1]) == 0
    assert torch.allclose(command[:, 2], -0.5 * robot.data.heading_w, atol=1e-6)
    assert terrain.terrain_levels.tolist() == [row, row]
    centers, extent = levels.wheels()
    assert ((centers[..., 2] - extent[..., 2]) > env.scene.env_origins[:, 2:3]).all()
    assert (
      (centers[..., :2] - env.scene.env_origins[:, None, :2]).abs() + extent[..., :2]
      < 1.5
    ).all()
    assert obs["actor"].shape == (2, 144)


def test_real_rim_support_curriculum_and_partial_auto_reset(pit_env):
  env = pit_env
  _, extras = env.reset(seed=7)
  assert torch.isfinite(extras["log"]["Episode_Metrics/pit_nonfoot_contact_count"])
  assert "pit_nonfoot_contact_count" in env.metrics_manager.active_terms
  terrain = env.scene.terrain
  robot = env.scene["robot"]
  ids = torch.tensor([0], device=env.device)
  origin_other = env.scene.env_origins[1].clone()
  pose = robot.data.root_link_pose_w[ids].clone()
  pose[:, 0] = env.scene.env_origins[ids, 0] + 3.3
  pose[:, 1] = env.scene.env_origins[ids, 1]
  pose[:, 2] = 0.47  # Above actual rim z=0, not relative to pit bottom.
  pose[:, 3:] = torch.tensor([1.0, 0, 0, 0], device=env.device)
  robot.write_root_link_pose_to_sim(pose, env_ids=ids)
  robot.write_root_link_velocity_to_sim(
    torch.zeros(1, 6, device=env.device), env_ids=ids
  )
  action = torch.zeros(2, 12, device=env.device)
  initial_vx = env.command_manager.get_command("twist")[:, 0].clone()
  for _ in range(40):
    env.step(action)
  assert torch.equal(initial_vx, env.command_manager.get_command("twist")[:, 0])
  levels = env.curriculum_manager.get_term_cfg("pit_levels").func
  assert isinstance(levels, PitTerrainLevels)
  center, extent = levels.wheels()
  c = env.scene["pit_rim_contact"].data
  supported = rim_supported(
    center,
    extent,
    env.scene.env_origins[:, :2],
    torch.zeros(2, device=env.device),
    c.found,
    c.force,
    c.normal,
    c.pos,
  )
  assert supported.tolist() == [True, False]
  # Force a timeout only for env 0; normal framework reset must read support
  # before clearing sensors, move the tile, then spawn on the new floor.
  env.episode_length_buf[0] = env.max_episode_length - 1
  env.step(action)
  assert terrain.terrain_levels.tolist() == [1, 0]
  assert torch.equal(env.scene.env_origins[1], origin_other)
  assert torch.allclose(
    robot.data.root_link_pos_w[0, 2], env.scene.env_origins[0, 2] + 0.47
  )
  # Mid-episode manual reset must not demote; repeated reset remains frozen.
  for _ in range(3):
    env.step(action)
  env.episode_length_buf[0] = 100
  for _ in range(2):
    env.reset(env_ids=ids)
    assert terrain.terrain_levels.tolist() == [1, 0]
  # Existing TerrainEntity enforces the lowest floor and random top revisit.
  terrain.terrain_levels[0] = 0
  terrain.update_env_origins(
    ids,
    torch.tensor([False], device=env.device),
    torch.tensor([True], device=env.device),
  )
  assert terrain.terrain_levels[0] == 0
  terrain.terrain_levels[0] = 9
  terrain.update_env_origins(
    ids,
    torch.tensor([True], device=env.device),
    torch.tensor([False], device=env.device),
  )
  assert 0 <= terrain.terrain_levels[0] < 10
  assert torch.equal(
    env.scene.env_origins[0], terrain.terrain_origins[terrain.terrain_levels[0], 0]
  )


def test_fall_demotes_and_local_boundary_truncates(pit_env):
  env = pit_env
  env.reset()
  ids = torch.tensor([0], device=env.device)
  terrain = env.scene.terrain
  robot = env.scene["robot"]
  terrain.terrain_levels[0] = 2
  terrain.update_env_origins(
    ids,
    torch.tensor([False], device=env.device),
    torch.tensor([False], device=env.device),
  )
  env.reset(env_ids=ids)
  pose = robot.data.root_link_pose_w[ids].clone()
  pose[:, 0] = env.scene.env_origins[ids, 0] + 3.3
  pose[:, 2] = 0.47
  pose[:, 3:] = torch.tensor([0.0, 1, 0, 0], device=env.device)
  robot.write_root_link_pose_to_sim(pose, env_ids=ids)
  env.sim.forward()
  env.scene.update(0.0)
  env.sim.sense()
  env.common_step_counter = 10
  env.episode_length_buf[0] = 10
  env.termination_manager.compute()
  assert env.termination_manager.terminated[0]
  env.curriculum_manager.compute(env_ids=ids)
  assert terrain.terrain_levels.tolist() == [1, 0]
  # Local edge detection must work inside the overall 100m terrain footprint.
  pose[:, 0] = env.scene.env_origins[ids, 0] + 4.2
  pose[:, 3:] = torch.tensor([1.0, 0, 0, 0], device=env.device)
  robot.write_root_link_pose_to_sim(pose, env_ids=ids)
  env.sim.forward()
  env.scene.update(0.0)
  env.sim.sense()
  env.termination_manager.compute()
  assert env.termination_manager.get_term("tile_boundary")[0]
  assert env.termination_manager.time_outs[0]
  assert not env.termination_manager.terminated[0]
