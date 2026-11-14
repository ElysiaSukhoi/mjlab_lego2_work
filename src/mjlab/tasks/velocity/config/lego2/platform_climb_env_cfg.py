"""Lego2 platform-climb task configuration, built in incremental steps."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import TYPE_CHECKING

import torch

import mjlab.terrains as terrain_gen
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul

from .env_cfgs import lego2_rough_env_cfg
from .platform_climb_metrics import (
  platform_contact_count,
  platform_front_contact_count,
  platform_relative_root_x,
  platform_relative_root_z,
  platform_top_reached,
  platform_top_stable,
)
from .platform_climb_rewards import (
  platform_approach,
  platform_edge_lift,
  platform_front_contact_advance,
  platform_front_lift,
  platform_top_bonus,
)
from .platform_climb_terminations import platform_success

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_PLATFORM_HEIGHT = 0.15
_DEFAULT_ROOT_Z = 0.47
_LOW_ROOT_Z = _DEFAULT_ROOT_Z - _PLATFORM_HEIGHT
_TOP_ROOT_Z = _DEFAULT_ROOT_Z
_TOP_ROOT_Z_THRESHOLD = _LOW_ROOT_Z + 0.5 * _PLATFORM_HEIGHT
_LIFT_REWARD_LOW_ROOT_Z = _LOW_ROOT_Z - 0.05


PLATFORM_CLIMB_TERRAIN_CFG = TerrainGeneratorCfg(
  curriculum=True,
  size=(10.0, 10.0),
  border_width=0.0,
  num_rows=1,
  num_cols=1,
  difficulty_range=(1.0, 1.0),
  merge_curriculum_rows=False,
  sub_terrains={
    "platform": terrain_gen.BoxPyramidStairsTerrainCfg(
      proportion=1.0,
      step_height_range=(_PLATFORM_HEIGHT, _PLATFORM_HEIGHT),
      step_width=2.0,
      platform_width=3.0,
      border_width=3.5,
    ),
  },
)


def reset_platform_climb_root_state(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  platform_height: float = _PLATFORM_HEIGHT,
  spawn_x_range: tuple[float, float] = (-3.2, -2.8),
  spawn_y_range: tuple[float, float] = (-0.15, 0.15),
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
  """Reset the robot on low ground in front of the platform.

  The terrain origin for ``BoxPyramidStairsTerrainCfg`` is the top platform
  center.  The negative z offset therefore places the robot's default root
  height back on the surrounding low ground.  The x offset approaches the
  platform from its negative-x side and the yaw offset remains zero.
  """
  env_ids = envs_mdp.resolve_env_ids(env, env_ids)
  asset: Entity = env.scene[asset_cfg.name]
  default_root_state = asset.data.default_root_state
  assert default_root_state is not None
  root_states = default_root_state[env_ids].clone()

  pose_samples = torch.zeros((len(env_ids), 6), device=env.device)
  pose_samples[:, 0].uniform_(*spawn_x_range)
  pose_samples[:, 1].uniform_(*spawn_y_range)
  pose_samples[:, 2] = -platform_height

  positions = (
    root_states[:, 0:3]
    + pose_samples[:, 0:3]
    + env.scene.env_origins[env_ids]
  )
  orientations_delta = quat_from_euler_xyz(
    pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
  )
  orientations = quat_mul(root_states[:, 3:7], orientations_delta)

  asset.write_root_link_pose_to_sim(
    torch.cat([positions, orientations], dim=-1), env_ids=env_ids
  )
  asset.write_root_link_velocity_to_sim(root_states[:, 7:13], env_ids=env_ids)


def lego2_platform_climb_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the deterministic single-platform climbing configuration."""
  cfg = deepcopy(lego2_rough_env_cfg(play=play))

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = deepcopy(PLATFORM_CLIMB_TERRAIN_CFG)
  cfg.scene.terrain.max_init_terrain_level = 0

  cfg.events["reset_base"] = EventTermCfg(
    func=reset_platform_climb_root_state,
    mode="reset",
    params={
      "platform_height": _PLATFORM_HEIGHT,
      "spawn_x_range": (-3.2, -2.8),
      "spawn_y_range": (-0.15, 0.15),
    },
  )

  # The single-row/single-column terrain is already deterministic. Remove the
  # play-only terrain randomizer so this remains explicit if the base config
  # changes later.
  if play:
    cfg.events.pop("randomize_terrain", None)

  # Step 3: train the approach phase with forward-only commands.  This keeps
  # the initial climbing task from spending samples on reversing, turning, or
  # standing behaviors that are unrelated to reaching the platform edge.
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.heading_command = False
  twist_cmd.ranges.lin_vel_x = (0.30, 0.80)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
  twist_cmd.ranges.heading = None
  twist_cmd.rel_standing_envs = 0.0
  twist_cmd.rel_heading_envs = 0.0
  twist_cmd.rel_world_envs = 0.0
  twist_cmd.rel_forward_envs = 1.0
  twist_cmd.rel_spin_envs = 0.0
  twist_cmd.init_velocity_prob = 0.0

  # Step 4: provide a small dense signal for reaching the platform's near
  # edge.  The reward is intentionally separate from the later top-platform
  # success reward and termination logic.
  cfg.rewards["platform_approach"] = RewardTermCfg(
    func=platform_approach,
    weight=0.5,
    params={
      "start_x": -3.0,
      "platform_edge_x": -1.5,
    },
  )
  cfg.rewards["platform_edge_lift"] = RewardTermCfg(
    func=platform_edge_lift,
    weight=1.0,
    params={
      "edge_gate_x_range": (-2.1, -1.5),
      "low_root_z": _LIFT_REWARD_LOW_ROOT_Z,
      "top_root_z": _TOP_ROOT_Z,
    },
  )
  cfg.rewards["platform_front_lift"] = RewardTermCfg(
    func=platform_front_lift,
    weight=0.75,
    params={
      "edge_gate_x_range": (-2.1, -1.5),
      "height_sensor_name": "foot_height_scan",
      "target_height": 0.05,
      "top_height": 0.20,
    },
  )
  cfg.rewards["platform_front_contact_advance"] = RewardTermCfg(
    func=platform_front_contact_advance,
    weight=1.5,
    params={
      # Extend beyond the platform edge so contact continues to reward body
      # translation instead of saturating at x=-1.5.
      "edge_gate_x_range": (-2.1, -1.0),
      "sensor_name": "feet_ground_contact",
      "force_threshold": 1.0,
    },
  )

  # Step 5: observe geometric entry into the platform-top region.  This is
  # logged as an episode maximum first; it is not yet a reward or termination.
  cfg.metrics["platform_top_reached"] = MetricsTermCfg(
    func=platform_top_reached,
    reduce="max",
    params={
      "platform_height": _PLATFORM_HEIGHT,
      "top_x_range": (-1.0, 1.0),
      "top_y_half_width": 1.0,
      "root_z_margin": 0.02,
      "root_z_threshold": _TOP_ROOT_Z_THRESHOLD,
    },
  )

  # Step 6: require a short, multi-foot support hold after entering the top
  # region.  This remains a metric until contact behavior is validated.
  cfg.metrics["platform_top_stable"] = MetricsTermCfg(
    func=platform_top_stable,
    reduce="max",
    params={
      "platform_height": _PLATFORM_HEIGHT,
      "top_x_range": (-1.0, 1.0),
      "top_y_half_width": 1.0,
      "root_z_margin": 0.02,
      "root_z_threshold": _TOP_ROOT_Z_THRESHOLD,
      "sensor_name": "feet_ground_contact",
      "min_contacts": 2,
      "force_threshold": 1.0,
      "hold_steps": 10,
    },
  )

  # Diagnostics: collect maxima without changing reward or termination logic.
  cfg.metrics["platform_root_x_max"] = MetricsTermCfg(
    func=platform_relative_root_x,
    reduce="max",
  )
  cfg.metrics["platform_root_z_max"] = MetricsTermCfg(
    func=platform_relative_root_z,
    reduce="max",
  )
  cfg.metrics["platform_contact_count_max"] = MetricsTermCfg(
    func=platform_contact_count,
    reduce="max",
    params={"sensor_name": "feet_ground_contact", "force_threshold": 1.0},
  )
  cfg.metrics["platform_front_contact_count_max"] = MetricsTermCfg(
    func=platform_front_contact_count,
    reduce="max",
    params={"sensor_name": "feet_ground_contact", "force_threshold": 1.0},
  )

  # Step 7: convert the validated top-support condition into a sparse bonus
  # and a successful episode termination after the same 10-step hold.
  top_support_params = {
    "platform_height": _PLATFORM_HEIGHT,
    "top_x_range": (-1.0, 1.0),
    "top_y_half_width": 1.0,
    "root_z_margin": 0.02,
    "root_z_threshold": _TOP_ROOT_Z_THRESHOLD,
    "sensor_name": "feet_ground_contact",
    "min_contacts": 2,
    "force_threshold": 1.0,
  }
  cfg.rewards["platform_top_bonus"] = RewardTermCfg(
    func=platform_top_bonus,
    weight=8.0,
    params=top_support_params,
  )
  cfg.terminations["platform_success"] = TerminationTermCfg(
    func=platform_success,
    time_out=True,
    params={**top_support_params, "hold_steps": 10},
  )

  # Step 8: establish a repeatable baseline before the first short training
  # run.  Keep the fixed 0.4 slide-friction event, but remove its curriculum
  # and all other reset/startup randomizers that would hide climbing failures.
  for event_name in (
    "push_robot",
    "foot_friction_spin",
    "foot_friction_roll",
    "pd_gains",
    "base_mass",
    "base_inertia",
    "link_inertia",
    "base_com",
    "encoder_bias",
  ):
    cfg.events.pop(event_name, None)
  cfg.curriculum.pop("terrain_levels", None)
  cfg.curriculum.pop("foot_friction_slide", None)

  # The rough task intentionally removed this termination for locomotion, but
  # the dedicated climbing baseline should reset promptly after a fall.
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=envs_mdp.bad_orientation,
    params={"limit_angle": math.radians(70.0)},
  )

  return cfg
