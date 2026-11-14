"""Lego2 wheel-legged velocity environment configurations.

Aligned with robot_lab LegoWheel2 velocity rewards and actuator kp/kd.
Command velocity curriculum and terrain-level curriculum are both enabled.
Terrain generator uses curriculum mode (one column per sub-terrain; rows =
difficulty). Two columns (flat|rough) with size=(10, 20): length 100 m,
width 20 m/col (40 m total). ``random_rough`` uses ``scale_with_difficulty``
so amplitude ramps from flat (row 0) to full ``noise_range`` (last row).
Actuator gains match LEGOWHEEL2_CFG. Command is 2D [lin_x, ang_z].

Rough terrain is inspired by sim2sim ``scene_random_rough`` but uses a
sparser/taller Lego2-friendly heightfield (``horizontal_scale=0.2`` to stay
under MuJoCo's hfield mjMAXCONPAIR=50; ``downsampled_scale=1.5`` for longer
wavelength bumps).
"""

import math
from dataclasses import replace

import mjlab.terrains as terrain_gen
from mjlab.asset_zoo.robots import (
  LEGO2_LEG_ACTION_SCALE,
  LEGO2_WHEEL_ACTION_SCALE,
  get_lego2_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.terrains import FlatPatchSamplingCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

# Spawn on locally flat heightfield patches; z offset clears wheels above surface.
_LEGO2_SPAWN_PATCH_CFG = FlatPatchSamplingCfg(
  num_patches=64,
  patch_radius=0.35,
  max_height_diff=0.08,
)

# Lego2 rough terrain: sparser/taller random_rough + terrain curriculum grid.
# curriculum=True → one column per sub-terrain; num_rows = difficulty levels.
# 2 columns (flat|rough), size=(10, 20): length = 10*10 = 100 m, width/col = 20 m.
# scale_with_difficulty → noise amplitude 0→full across rows.
def _lego2_flat_terrain(*, proportion: float) -> terrain_gen.BoxFlatTerrainCfg:
  return terrain_gen.BoxFlatTerrainCfg(
    proportion=proportion,
    flat_patch_sampling={"spawn": _LEGO2_SPAWN_PATCH_CFG},
  )


def _lego2_rough_terrain(*, proportion: float) -> terrain_gen.HfRandomUniformTerrainCfg:
  return terrain_gen.HfRandomUniformTerrainCfg(
    proportion=proportion,
    # Taller bumps at difficulty=1; step keeps a few discrete levels.
    noise_range=(-0.20, 0.50),
    noise_step=0.04,
    # Coarse sample spacing (was 0.8): longer wavelength → sparser hills.
    downsampled_scale=1.2,
    # 0.1 overflows mjMAXCONPAIR with Lego2 wheels/boxes.
    horizontal_scale=0.2,
    vertical_scale=0.005,
    # Flat rim at tile edges (>= horizontal_scale); eases cross-tile seams.
    border_width=0.5,
    scale_with_difficulty=True,
    flat_patch_sampling={"spawn": _LEGO2_SPAWN_PATCH_CFG},
  )


ROUGH_LEGO2_TERRAIN_CFG = TerrainGeneratorCfg(
  curriculum=True,
  size=(10.0, 10.0),
  border_width=0.0,
  num_rows=10,
  num_cols=20,  # ignored when curriculum=True (cols = len(sub_terrains))
  difficulty_range=(0.0, 1.0),
  merge_curriculum_rows=False,
  sub_terrains={
    "flat": _lego2_flat_terrain(proportion=0.2),
    "high_platform_up": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
      proportion=0.5,
      step_height_range=(0.02, 0.30),
      step_width=2.0,
      platform_width=3.0,
      border_width=3.5,
    ),
    "high_platform_down": terrain_gen.BoxPyramidStairsTerrainCfg(
      proportion=0.3,
      step_height_range=(0.02, 0.30),
      step_width=2.0,
      platform_width=3.0,
      border_width=3.5,
    ),
  },
)

# Regex patterns for rewards/events (order-independent).
_LEG_JOINT_NAMES = (".*_HFE", ".*_KFE")
_WHEEL_JOINT_NAMES = (".*_Wheel",)

# Explicit order for policy obs/actions: LF, RF, LH, RH.
_LEG_JOINT_ORDER = (
  "LF_HFE",
  "LF_KFE",
  "RF_HFE",
  "RF_KFE",
  "LH_HFE",
  "LH_KFE",
  "RH_HFE",
  "RH_KFE",
)
_WHEEL_JOINT_ORDER = ("LF_Wheel", "RF_Wheel", "LH_Wheel", "RH_Wheel")
_JOINT_VEL_OBS_ORDER = _LEG_JOINT_ORDER + _WHEEL_JOINT_ORDER
_FOOT_NAMES = ("RF", "LF", "RH", "LH")
_SITE_NAMES = _FOOT_NAMES
_FOOT_GEOM_NAMES = tuple(f"{name}_foot_collision" for name in _FOOT_NAMES)
_FOOT_BODY_NAMES = tuple(f"{name}_FOOT" for name in _FOOT_NAMES)
# Match _WHEEL_JOINT_ORDER: LF, RF, LH, RH.
_FOOT_BODY_WHEEL_ORDER = tuple(f"{leg}_FOOT" for leg in ("LF", "RF", "LH", "RH"))


def lego2_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Lego2 rough terrain velocity configuration (LegoWheel2-aligned rewards)."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 300
  cfg.sim.mujoco.impratio = 10
  cfg.sim.mujoco.cone = "elliptic"
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 99

  cfg.scene.entities = {"robot": get_lego2_robot_cfg()}

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "base"

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in _SITE_NAMES
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.08, num_samples=4)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=_FOOT_GEOM_NAMES, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  thigh_geom_names = tuple(
    f"{leg}_thigh_collision{i}" for leg in _FOOT_NAMES for i in (1, 2)
  )
  calf_geom_names = tuple(
    f"{leg}_calf_collision{i}" for leg in _FOOT_NAMES for i in (1, 2)
  )
  # Non-foot contacts kept for sensors/debug; reward weight is 0 in LegoWheel2.
  undesired_ground_cfg = ContactSensorCfg(
    name="undesired_ground_contact",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=(
        *thigh_geom_names,
        *calf_geom_names,
        "base_collision",
        "lidar_collision",
        *(f"{leg}_hip_collision" for leg in _FOOT_NAMES),
      ),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    undesired_ground_cfg,
  )

  # Curriculum grid: col0=flat, col1=random_rough; rows = difficulty (100×20 m/col).
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = replace(ROUGH_LEGO2_TERRAIN_CFG)
  cfg.scene.terrain.max_init_terrain_level = 5

  # Flat-patch spawn: patch z + default root height + z offset (not tile-center z).
  cfg.events["reset_base"] = EventTermCfg(
    func=envs_mdp.reset_root_state_from_flat_patches,
    mode="reset",
    params={
      "patch_name": "spawn",
      "pose_range": {
        "x": (-0.15, 0.15),
        "y": (-0.15, 0.15),
        "z": (0.05, 0.12),
        "yaw": (-3.14, 3.14),
      },
      "velocity_range": {},
    },
  )

  # Terrain levels only. No lin/ang command curriculum: fixed ranges below.
  # Friction climb is paced so each μ band gets ~1k iters before the next promote.
  cfg.curriculum = {
    "terrain_levels": CurriculumTermCfg(
      func=mdp.terrain_levels_vel,
      params={"command_name": "twist"},
    ),
    # Friction curriculum (v6): thresholds scaled for track_ang.weight=3
    # (≈85% / 70% of max Episode_Reward). Gated on spin-in-place commands
    # (rel_spin_envs) rather than any nonzero yaw (driving+turning excluded).
    "foot_friction_slide": CurriculumTermCfg(
      func=mdp.foot_friction_slide_curriculum,
      params={
        "event_name": "foot_friction_slide",
        "command_name": "twist",
        "reward_name": "track_ang_vel_z_exp",
        "hard_ratio": 0.6,
        "track_ang_threshold": 2.55,
        "demote_threshold": 2.1,
        "start_level": 0,
        "cmd_gate": 0.2,
        "require_spin_inplace": True,
        "lin_cmd_gate": 0.1,
        "friction_levels": [0.4, 0.7, 1.0, 1.3, 1.6, 2.0],
        "min_promote_steps": 1500 * 24,
        "min_demote_steps": 1500 * 24,
        "ema_alpha": 0.05,
        "min_gated_envs": 64,
        "require_both_yaw": True,
        "min_side_envs": 16,
        "promote_ok_needed": 50,
      },
    ),
  }

  cfg.metrics["ang_vel_z_rel_error"] = MetricsTermCfg(
    func=mdp.ang_vel_z_rel_error,
    params={"command_name": "twist", "cmd_threshold": 0.3},
  )

  # Dual action space (LegoWheel2 scales: leg 0.25, wheel 5.0).
  cfg.actions["joint_pos"] = JointPositionActionCfg(
    entity_name="robot",
    actuator_names=_LEG_JOINT_ORDER,
    scale=0.45,
    use_default_offset=True,
    preserve_order=True,
  )
  cfg.actions["joint_vel"] = JointVelocityActionCfg(
    entity_name="robot",
    actuator_names=_WHEEL_JOINT_ORDER,
    scale=LEGO2_WHEEL_ACTION_SCALE,
    use_default_offset=True,
    preserve_order=True,
  )

  # --- Observations (LegoWheel2 policy: no lin_vel / height_scan; joint_pos without wheels) ---
  del cfg.observations["actor"].terms["base_lin_vel"]
  # if "height_scan" in cfg.observations["actor"].terms:
  #   del cfg.observations["actor"].terms["height_scan"]

  for group in ("actor", "critic"):
    cfg.observations[group].terms["joint_pos"].params["asset_cfg"] = SceneEntityCfg(
      "robot", joint_names=_LEG_JOINT_ORDER, preserve_order=True
    )
    cfg.observations[group].terms["joint_vel"].params["asset_cfg"] = SceneEntityCfg(
      "robot", joint_names=_JOINT_VEL_OBS_ORDER, preserve_order=True
    )
    cfg.observations[group].terms["command"] = ObservationTermCfg(
      func=envs_mdp.generated_commands_xz,
      params={"command_name": "twist"},
    )

  # LegoWheel2 observation scales.
  cfg.observations["actor"].terms["base_ang_vel"].scale = 0.25
  cfg.observations["actor"].terms["joint_pos"].scale = 1.0
  cfg.observations["actor"].terms["joint_vel"].scale = 0.05
  cfg.observations["actor"].terms["base_ang_vel"].noise = Unoise(n_min=-0.2, n_max=0.2)
  cfg.observations["actor"].terms["joint_pos"].noise = Unoise(n_min=-0.01, n_max=0.01)
  cfg.observations["actor"].terms["joint_vel"].noise = Unoise(n_min=-1.5, n_max=1.5)

  # Fixed command ranges for the whole run (no command_vel curriculum).
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.rel_spin_envs = 0.2
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.lin_vel_x = (-1.0, 1.0)
  twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)

  cfg.viewer.body_name = "base"
  cfg.viewer.distance = 2.0
  cfg.viewer.elevation = -10.0
  cfg.viewer.reward_bar_max_terms = 32

  # NaN localization: print offending obs term names; dump physics on sim NaN.
  cfg.observations["actor"].nan_policy = "warn"
  cfg.observations["actor"].nan_check_per_term = True
  cfg.observations["critic"].nan_policy = "warn"
  cfg.observations["critic"].nan_check_per_term = True
  cfg.sim.nan_guard.enabled = True

  # Per-axis friction DR for condim-6 wheel contacts.
  # Slide friction uses easy/hard mixture driven by foot_friction_slide curriculum.
  del cfg.events["foot_friction"]
  cfg.events["foot_friction_slide"] = EventTermCfg(
    mode="reset",
    func=envs_mdp.dr.geom_friction_easy_hard,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=_FOOT_GEOM_NAMES),
      "axes": [0],
      # Matches start_level=0 until curriculum promotes.
      "easy_values": [0.4],
      "hard_values": [0.4],
      "hard_ratio": 0.6,
      "shared_random": True,
    },
  )
  cfg.events["foot_friction_spin"] = EventTermCfg(
    mode="reset",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=_FOOT_GEOM_NAMES),
      "operation": "abs",
      "distribution": "log_uniform",
      "axes": [1],
      "ranges": (1e-4, 2e-2),
      "shared_random": True,
    },
  )
  cfg.events["foot_friction_roll"] = EventTermCfg(
    mode="reset",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=_FOOT_GEOM_NAMES),
      "operation": "abs",
      "distribution": "log_uniform",
      "axes": [2],
      "ranges": (1e-5, 5e-3),
      "shared_random": True,
    },
  )

  # Match robot_lab: actuator PD gains scale on reset; base mass/COM on startup.
  cfg.events["pd_gains"] = EventTermCfg(
    mode="reset",
    func=envs_mdp.dr.pd_gains,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "kp_range": (0.8, 1.2),
      "kd_range": (0.8, 1.2),
      "operation": "scale",
      "distribution": "uniform",
    },
  )
  # Payload-style mass add at base COM (robot_lab randomize_rigid_body_mass).
  cfg.events["base_mass"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("base",)),
      "operation": "add",
      "ranges": (-3.0, 3.0),
    },
  )
  # Inertia DR split by mass scale: base ~15.7 kg vs links ~1.25–2.9 kg.
  cfg.events["base_inertia"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.body_inertia,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("base",)),
      "operation": "scale",
      "ranges": (0.5, 1.5),
    },
  )
  cfg.events["link_inertia"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.body_inertia,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot",
        body_names=(".*_hip", ".*_thigh", ".*_calf", ".*_FOOT"),
      ),
      "operation": "scale",
      "ranges": (0.8, 1.2),
    },
  )
  # robot_lab randomize_com_positions: add U(-0.1, 0.1) on base.
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base",)
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.1, 0.1),
    1: (-0.1, 0.1),
    2: (-0.1, 0.1),
  }

  # --- Rewards: robot_lab LegoWheel2 non-zero set (command ranges unchanged) ---
  cfg.rewards.clear()
  leg_cfg = SceneEntityCfg("robot", joint_names=_LEG_JOINT_NAMES)
  wheel_cfg = SceneEntityCfg("robot", joint_names=_WHEEL_JOINT_NAMES)
  # Ordered LF/RF/LH/RH for side-sync (+1 left, -1 right).
  wheel_order_cfg = SceneEntityCfg(
    "robot", joint_names=_WHEEL_JOINT_ORDER, preserve_order=True
  )
  leg_act_cfg = SceneEntityCfg("robot", actuator_names=_LEG_JOINT_NAMES)

  cfg.rewards = {
    "track_lin_vel_xy_exp": RewardTermCfg(
      func=mdp.track_lin_vel_xy_exp,
      weight=3.0,
      params={"command_name": "twist", "std": math.sqrt(0.5)},
    ),
    "track_ang_vel_z_exp": RewardTermCfg(
      func=mdp.track_ang_vel_z_exp,
      # v6: heavier + tighter so half-tracking is no longer cheap.
      weight=3.0,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "upward": RewardTermCfg(
      func=mdp.upward,
      weight=1.0,
      params={"std": math.sqrt(0.5)},
    ),
    "lin_vel_z_l2": RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-0.3),
    "ang_vel_xy_l2": RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-1.0),
    "flat_orientation_l2": RewardTermCfg(
      func=mdp.flat_orientation_l2,
      weight=-2.5,
    ),
    "base_height_l2": RewardTermCfg(
      func=mdp.base_height_l2,
      weight=-2.5,
      params={
        "target_height": 0.45,
        "asset_cfg": SceneEntityCfg("robot", body_names=("base",)),
        "sensor_name": "terrain_scan",
      },
    ),
    "joint_torques_l2": RewardTermCfg(
      func=mdp.joint_torques_l2,
      weight=-2.5e-9,
      params={"asset_cfg": leg_act_cfg},
    ),
    "joint_acc_l2": RewardTermCfg(
      func=mdp.joint_acc_l2,
      weight=-2.5e-7,
      params={"asset_cfg": leg_cfg},
    ),
    "joint_acc_wheel_l2": RewardTermCfg(
      func=mdp.joint_acc_l2,
      weight=-2.5e-9,
      params={"asset_cfg": wheel_cfg},
    ),
    "joint_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-5.0,
      params={"asset_cfg": leg_cfg},
    ),
    "joint_power": RewardTermCfg(
      func=mdp.joint_power,
      weight=-1.0e-5,
      params={"asset_cfg": leg_cfg},
    ),
    "stand_still": RewardTermCfg(
      func=mdp.stand_still,
      weight=-2.0,
      params={
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": leg_cfg,
      },
    ),
    "joint_pos_penalty": RewardTermCfg(
      func=mdp.joint_pos_penalty,
      weight=0.0,
      params={
        "command_name": "twist",
        "asset_cfg": leg_cfg,
        "stand_still_scale": 5.0,
        "velocity_threshold": 100.0,
        "command_threshold": 0.1,
      },
    ),
    "joint_pos_penalty_std": RewardTermCfg(
      func=mdp.joint_pos_penalty_std,
      weight= 0.5,
      params={
        "command_name": "twist",
        "asset_cfg": leg_cfg,
        "stand_still_scale": 5.0,
        "velocity_threshold": 100.0,
        "command_threshold": 0.1,
        "std": 0.5,
      },
    ),
    "action_rate_l2": RewardTermCfg(
      func=mdp.action_rate_l2,
      weight=-0.1,
      params={"action_name": "joint_pos"},
    ),
    "action_rate_wheel_l2": RewardTermCfg(
      func=mdp.action_rate_l2,
      weight=-0.01,
      params={"action_name": "joint_vel"},
    ),
    "action_smooth_l2": RewardTermCfg(
      func=mdp.action_smooth_l2,
      weight=0.0,
      params={"action_name": "joint_pos"},
    ),
    "action_smooth_wheel_l2": RewardTermCfg(
      func=mdp.action_smooth_l2,
      weight=0.0,
      params={"action_name": "joint_vel"},
    ),
    # Soft skid prior (no half_track / radius): always same-side sync;
    # pure v → L≈R; pure ω → L≈−R. Light weight vs track_*.
    "wheel_side_sync_exp": RewardTermCfg(
      func=mdp.wheel_side_sync_exp,
      weight=0.0,
      params={
        "command_name": "twist",
        "wheel_cfg": wheel_order_cfg,
        "std": 2.0,
        "command_threshold": 0.05,
        "pure_threshold": 0.05,
        "side_signs": (1.0, -1.0, 1.0, -1.0),
      },
    ),
    "feet_slide": RewardTermCfg(
      func=mdp.feet_slide,
      weight=-0.1,
      params={
        "sensor_name": "feet_ground_contact",
        "asset_cfg": SceneEntityCfg("robot", body_names=_FOOT_BODY_NAMES),
      },
    ),
    "feet_contact": RewardTermCfg(
      func=mdp.feet_contact,
      weight=0.0,
      params={
        "sensor_name": "feet_ground_contact",
        "expect_contact_num": 4,
        "force_threshold": 1.0,
        "command_name": "twist",
        "command_threshold": 0.1,
        "power": 2.0,
      },
    ),
    # Complements feet_contact: active when ||cmd|| < threshold (no overlap at
    # threshold). Does not write sim state; uses the same contact sensor read
    # as feet_slide (penalizes slip) without logical conflict.
    "feet_contact_without_cmd": RewardTermCfg(
      func=mdp.feet_contact_without_cmd,
      weight=0.1,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "expect_contact_num": 4,
        "force_threshold": 1.0,
        "command_threshold": 0.1,
        "power": 2.0,
      },
    ),
  }

  # LegoWheel2: no illegal_contact termination.
  cfg.terminations.pop("fell_over", None)
  cfg.terminations.pop("illegal_contact", None)

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    # Play uses fewer envs but taller rough can need more contact slots.
    cfg.sim.nconmax = 128
    cfg.events.pop("push_robot", None)
    # Keep foot_friction_slide; play fixes it to 0.4 below.
    cfg.events.pop("foot_friction_spin", None)
    cfg.events.pop("foot_friction_roll", None)
    cfg.events.pop("link_inertia", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    # Play: use final curriculum command ranges.
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-2.0, 3.0)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )
    # Play: random row/col on the same merged training grid (no 5×5 remesh).
    cfg.curriculum.pop("terrain_levels", None)
    cfg.curriculum.pop("foot_friction_slide", None)
    # Play: fixed easy slide friction (no curriculum mixture).
    slide = cfg.events.get("foot_friction_slide")
    if slide is not None:
      slide.params["easy_values"] = [1.0]
      slide.params["hard_values"] = [1.0]
      slide.params["hard_ratio"] = 0.0

  return cfg


def lego2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Lego2 flat terrain velocity configuration (LegoWheel2-aligned rewards)."""
  cfg = lego2_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Keep feet + undesired contact sensors; drop terrain height scans.
  remove_sensors = {"terrain_scan", "foot_height_scan"}
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name not in remove_sensors
  )
  cfg.observations["critic"].terms.pop("height_scan", None)
  for key in (
    "foot_height",
    "foot_air_time",
    "foot_contact",
    "foot_contact_forces",
  ):
    cfg.observations["critic"].terms.pop(key, None)

  # Flat: absolute height target (robot_lab flat sets sensor_cfg=None).
  cfg.rewards["base_height_l2"].params["sensor_name"] = None

  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation,
    params={"limit_angle": math.radians(70.0)},
  )

  # Flat plane: no terrain curriculum (command ranges already fixed in rough cfg).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-2.0, 3.0)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)

  return cfg
