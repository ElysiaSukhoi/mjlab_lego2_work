"""Lego2 wheel-legged robot constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg, BuiltinVelocityActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

LEGO2_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "lego2" / "xmls" / "lego2.xml"
)
assert LEGO2_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(LEGO2_XML))


##
# Actuator config.
##

# Match robot_lab LEGOWHEEL2_CFG actuator gains.
# Legs: stiffness=100, damping=5, effort=150; wheels: velocity damping=1, effort=65.
LEGO2_LEG_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_HFE", ".*_KFE"),
  stiffness=100.0,
  damping=5.0,
  effort_limit=150.0,
)

LEGO2_WHEEL_ACTUATOR_CFG = BuiltinVelocityActuatorCfg(
  target_names_expr=(".*_Wheel",),
  damping=1.0,
  effort_limit=65.0,
)

##
# Keyframes.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.47),
  joint_pos={
    "RF_HFE": 0.72,
    "LF_HFE": 0.72,
    "RH_HFE": -0.72,
    "LH_HFE": -0.72,
    "RF_KFE": -1.44,
    "LF_KFE": -1.44,
    "RH_KFE": 1.44,
    "LH_KFE": 1.44,
    ".*_Wheel": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = "^(RF|LF|RH|LH)_foot_collision$"
_collision_regex = r".*_collision\d*"

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(_foot_regex,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(1.5,),
)

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(_collision_regex, _foot_regex, "base_collision", "lidar_collision"),
  contype=1,
  conaffinity=1,
  solref=(0.01, 1),
  condim={_foot_regex: 6, ".*": 1},
  priority={_foot_regex: 1, ".*": 0},
  friction={_foot_regex: (0.4, 5e-3, 5e-4)},
)

##
# Final config.
##

LEGO2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    LEGO2_LEG_ACTUATOR_CFG,
    LEGO2_WHEEL_ACTUATOR_CFG,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_lego2_robot_cfg() -> EntityCfg:
  """Get a fresh Lego2 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=LEGO2_ARTICULATION,
  )


# Match LegoWheel2 / B2W action scales (leg position 0.25, wheel velocity 5.0).
LEGO2_LEG_ACTION_SCALE: float = 0.25
LEGO2_WHEEL_ACTION_SCALE: float = 5.0

LEGO2_ACTION_SCALE: dict[str, float] = {
  ".*_HFE": LEGO2_LEG_ACTION_SCALE,
  ".*_KFE": LEGO2_LEG_ACTION_SCALE,
  ".*_Wheel": LEGO2_WHEEL_ACTION_SCALE,
}


if __name__ == "__main__":
  # Prefer the lightweight Viser script (no X11 / no mujoco_warp import):
  #   uv run python src/mjlab/asset_zoo/robots/lego2/view_viser.py
  from mjlab.asset_zoo.robots.lego2.view_viser import main

  main()
