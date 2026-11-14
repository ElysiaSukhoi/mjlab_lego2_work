"""Inspect Lego2 action routing, wheel kinematics and native CPU PD response.

This is a diagnostic scene with a flat floor, one free robot, and the current
Platform-Climb actions, sensors, actuators and simulation options. It does not
load a policy, run PPO, or edit task/asset files. Native MuJoCo contact results
do not validate MuJoCo-Warp contact behavior or climbing success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import mujoco
import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions.actions import BaseAction, JointPositionAction
from mjlab.managers.action_manager import ActionManager
from mjlab.scene import Scene
from mjlab.sensor import ContactSensor, ObjRef, TerrainHeightSensor
from mjlab.sim import Simulation
from mjlab.tasks.velocity.config.lego2.platform_climb_env_cfg import (
  lego2_platform_climb_env_cfg,
)
from mjlab.tasks.velocity.config.lego2.rl_cfg import lego2_ppo_runner_cfg

ROOT = Path(__file__).resolve().parents[2]
ASSET = ROOT / "src/mjlab/asset_zoo/robots/lego2"


@dataclass
class Inspection:
  cfg: ManagerBasedRlEnvCfg
  scene: Scene
  sim: Simulation
  robot: Entity
  actions: ActionManager

  @property
  def model(self) -> mujoco.MjModel:
    return self.sim.mj_model

  def control(self, raw: list[float]) -> np.ndarray:
    """Use the real action manager and built-in actuator write path."""
    value = torch.tensor([raw], dtype=torch.float32)
    self.actions.process_action(value)
    self.actions.apply_action()
    self.robot.write_data_to_sim()
    return self.sim.data.ctrl[0].numpy().copy()

  def initial_data(self) -> mujoco.MjData:
    data = mujoco.MjData(self.model)
    mujoco.mj_resetDataKeyframe(self.model, data, 0)
    data.ctrl[:] = self.control([0.0] * self.actions.total_action_dim)
    mujoco.mj_forward(self.model, data)
    return data


def build_inspection() -> Inspection:
  cfg = lego2_platform_climb_env_cfg()
  scene_cfg = deepcopy(cfg.scene)
  scene_cfg.num_envs = 1
  assert scene_cfg.terrain is not None
  scene_cfg.terrain.terrain_type = "plane"
  scene_cfg.terrain.terrain_generator = None
  scene_cfg.sensors = tuple(
    s
    for s in scene_cfg.sensors
    if s.name in ("feet_ground_contact", "foot_height_scan")
  )
  scene = Scene(scene_cfg, "cpu")
  sim = Simulation(num_envs=1, cfg=cfg.sim, model=scene.compile(), device="cpu")
  robot: Entity = scene["robot"]
  robot.initialize(sim.mj_model, sim.model, sim.data, "cpu")
  # ActionManager only needs these three environment attributes. Keep the real
  # scene/entity/action implementations without constructing RL managers.
  env = cast(ManagerBasedRlEnv, SimpleNamespace(scene=scene, device="cpu", num_envs=1))
  actions = ActionManager(cfg.actions, env)
  return Inspection(cfg, scene, sim, robot, actions)


def _number_at(value: float | torch.Tensor, index: int) -> float:
  return float(value[0, index]) if isinstance(value, torch.Tensor) else value


def action_mapping(inspect: Inspection) -> list[dict[str, Any]]:
  """Resolve policy slices through actual local joint and global actuator IDs."""
  model = inspect.model
  urdf = ET.parse(ASSET / "urdf/urdf_legowheel2.urdf").getroot()
  actuators = {
    name: int(ctrl_id)
    for actuator in inspect.robot.actuators
    for name, ctrl_id in zip(
      actuator.target_names, actuator.global_ctrl_ids, strict=True
    )
  }
  rows = []
  for term_name in inspect.actions.active_terms:
    term = inspect.actions.get_term(term_name)
    if not isinstance(term, BaseAction):
      raise TypeError(f"Unsupported diagnostic action: {type(term)}")
    for index, (name, local_id) in enumerate(
      zip(term.target_names, term.target_ids.tolist(), strict=True)
    ):
      jid = model.joint(f"robot/{name}").id
      aid = actuators[name]
      if model.actuator_trnid[aid, 0] != jid:
        raise RuntimeError(f"Actuator/joint transmission mismatch: {name}")
      if inspect.robot.joint_names[local_id] != name:
        raise RuntimeError(f"Action/local joint mismatch: {name}")
      is_position = isinstance(term, JointPositionAction)
      limit = urdf.find(f"joint[@name='{name}']/limit")
      assert limit is not None
      rows.append(
        {
          "action_index": len(rows),
          "term": term_name,
          "joint": name,
          "target_id": local_id,
          "joint_id": jid,
          "qpos_adr": int(model.jnt_qposadr[jid]),
          "dof_adr": int(model.jnt_dofadr[jid]),
          "actuator_id": aid,
          "actuator": model.actuator(aid).name,
          "mode": "position" if is_position else "velocity",
          "axis_local": model.jnt_axis[jid].tolist(),
          "default_position_rad": float(
            inspect.robot.data.default_joint_pos[0, local_id]
          ),
          "scale": _number_at(term.scale, index),
          "offset": _number_at(term.offset, index),
          "action_clip": term.cfg.clip,
          "hard_range_rad": model.jnt_range[jid].tolist()
          if model.jnt_limited[jid]
          else None,
          "soft_range_rad": inspect.robot.data.soft_joint_pos_limits[
            0, local_id
          ].tolist()
          if model.jnt_limited[jid]
          else None,
          "ctrl_range": model.actuator_ctrlrange[aid].tolist()
          if model.actuator_ctrllimited[aid]
          else None,
          "actuator_force_range_Nm": model.actuator_forcerange[aid].tolist()
          if model.actuator_forcelimited[aid]
          else None,
          "joint_force_range_Nm": model.jnt_actfrcrange[jid].tolist()
          if model.jnt_actfrclimited[jid]
          else None,
          "kp": float(model.actuator_gainprm[aid, 0]) if is_position else 0.0,
          "kd": float(-model.actuator_biasprm[aid, 2]),
          "urdf_velocity_rad_s": float(limit.attrib["velocity"]),
          "implemented_velocity_limit_rad_s": None,
        }
      )
  return rows


def sensor_mapping(inspect: Inspection) -> dict[str, Any]:
  contact = inspect.scene["feet_ground_contact"]
  height = inspect.scene["foot_height_scan"]
  assert isinstance(contact, ContactSensor)
  assert isinstance(height, TerrainHeightSensor)
  frames = height.cfg.frame
  if isinstance(frames, ObjRef):
    frames = (frames,)
  rows = []
  for index, frame in enumerate(frames):
    if frame.type != "site":
      raise ValueError("Lego2 wheel-height frames must be sites")
    site = inspect.model.site(frame.prefixed_name())
    geom_name = f"{frame.name}_foot_collision"
    geom = inspect.model.geom(f"robot/{geom_name}")
    if int(inspect.model.site_bodyid[site.id]) != int(
      inspect.model.geom_bodyid[geom.id]
    ):
      raise RuntimeError(f"Wheel site/geom body mismatch: {frame.name}")
    rows.append(
      {
        "frame_index": index,
        "leg": frame.name,
        "site": site.name,
        "site_id": site.id,
        "geom": geom.name,
        "geom_id": geom.id,
        "contact_primary_index": contact.primary_names.index(geom_name),
        "radius_m": float(inspect.model.geom_size[geom.id, 0]),
      }
    )
  return {"primary_names": contact.primary_names, "frames": rows}


def cylinder_extent(axis: np.ndarray, radius: float, half_width: float) -> np.ndarray:
  """Cylinder support distance along world X/Y/Z, including a tilted axle."""
  return radius * np.sqrt(np.maximum(0.0, 1.0 - axis**2)) + half_width * abs(axis)


def wheel_states(
  inspect: Inspection, data: mujoco.MjData, sensors: dict[str, Any]
) -> dict[str, Any]:
  result = {}
  for frame in sensors["frames"]:
    gid = frame["geom_id"]
    center = data.geom_xpos[gid]
    axis = data.geom_xmat[gid].reshape(3, 3)[:, 2]
    radius, half_width = inspect.model.geom_size[gid, :2]
    extent = cylinder_extent(axis, float(radius), float(half_width))
    result[frame["leg"]] = {
      "center_m": center.tolist(),
      "bottom_m": float(center[2] - extent[2]),
      "min_x_m": float(center[0] - extent[0]),
      "max_x_m": float(center[0] + extent[0]),
      "site_center_error_m": float(
        np.linalg.norm(data.site_xpos[frame["site_id"]] - center)
      ),
    }
  return result


def target_probes(
  inspect: Inspection, mapping: list[dict[str, Any]]
) -> list[dict[str, Any]]:
  results = []
  for row in mapping:
    for raw_value in (-3.0, -1.0, 0.0, 1.0, 3.0):
      raw = [0.0] * len(mapping)
      raw[row["action_index"]] = raw_value
      target = float(inspect.control(raw)[row["actuator_id"]])
      limits = row["hard_range_rad"] if row["mode"] == "position" else None
      soft = row["soft_range_rad"] if row["mode"] == "position" else None
      demand = row["kp"] * (target - row["default_position_rad"])
      if row["mode"] == "velocity":
        demand = row["kd"] * target
      results.append(
        {
          "joint": row["joint"],
          "synthetic_raw": raw_value,
          "written_ctrl_target": target,
          "outside_hard_range": limits is not None
          and not limits[0] <= target <= limits[1],
          "outside_soft_range": soft is not None and not soft[0] <= target <= soft[1],
          "zero_speed_initial_effort_demand_Nm": demand,
        }
      )
  return results


def perturb_joints(
  inspect: Inspection, mapping: list[dict[str, Any]], sensors: dict[str, Any]
) -> list[dict[str, Any]]:
  baseline = inspect.initial_data()
  initial = wheel_states(inspect, baseline, sensors)
  results = []
  for row in mapping:
    if row["mode"] != "position":
      continue
    for delta in (-0.1, 0.1):
      data = inspect.initial_data()
      data.qpos[row["qpos_adr"]] += delta
      mujoco.mj_kinematics(inspect.model, data)
      state = wheel_states(inspect, data, sensors)
      displacement = {
        leg: (np.array(wheel["center_m"]) - np.array(initial[leg]["center_m"])).tolist()
        for leg, wheel in state.items()
      }
      moved = [
        leg for leg, value in displacement.items() if np.linalg.norm(value) > 1e-8
      ]
      if moved != [row["joint"].split("_")[0]]:
        raise RuntimeError(f"Unexpected wheel response: {row['joint']}: {moved}")
      results.append(
        {
          "joint": row["joint"],
          "delta_rad": delta,
          "moving_frames": moved,
          "wheel_delta_xyz_m": displacement,
        }
      )
  return results


def scan_front_legs(
  inspect: Inspection,
  mapping: list[dict[str, Any]],
  sensors: dict[str, Any],
  points: int,
  edge_x: float,
) -> list[dict[str, Any]]:
  """Fixed-root FK samples, not collision-free paths or dynamic feasibility."""
  by_name = {row["joint"]: row for row in mapping}
  results = []
  for leg in ("RF", "LF"):
    hip, knee = (by_name[f"{leg}_{suffix}"] for suffix in ("HFE", "KFE"))
    for window in ("raw_minus1_plus1", "soft", "hard"):
      ranges = []
      for row in (hip, knee):
        bounds = row["hard_range_rad"]
        if window == "soft":
          bounds = row["soft_range_rad"]
        elif window == "raw_minus1_plus1":
          bounds = (
            max(bounds[0], row["offset"] - row["scale"]),
            min(bounds[1], row["offset"] + row["scale"]),
          )
        ranges.append(np.linspace(float(bounds[0]), float(bounds[1]), points))
      best = None
      over_edge = None
      counts = {str(height): 0 for height in (0.03, 0.15, 0.30)}
      closest = dict.fromkeys(counts)
      max_abs_x = 0.0
      data = inspect.initial_data()
      for hfe in ranges[0]:
        for kfe in ranges[1]:
          data.qpos[hip["qpos_adr"]] = hfe
          data.qpos[knee["qpos_adr"]] = kfe
          mujoco.mj_kinematics(inspect.model, data)
          wheel = wheel_states(inspect, data, sensors)[leg]
          pose = {
            **wheel,
            "hfe_rad": float(hfe),
            "kfe_rad": float(kfe),
            "raw_hfe": float((hfe - hip["offset"]) / hip["scale"]),
            "raw_kfe": float((kfe - knee["offset"]) / knee["scale"]),
          }
          max_abs_x = max(max_abs_x, abs(wheel["min_x_m"]), abs(wheel["max_x_m"]))
          if best is None or pose["bottom_m"] > best["bottom_m"]:
            best = pose
          # The entire wheel must be beyond the hypothetical edge, with 1 cm
          # vertical clearance. Other links and the approach path are unchecked.
          if wheel["min_x_m"] >= edge_x:
            if over_edge is None or pose["bottom_m"] > over_edge["bottom_m"]:
              over_edge = pose
            for height in counts:
              if wheel["bottom_m"] >= float(height) + 0.01:
                counts[height] += 1
                previous = closest[height]
                if previous is None or math.hypot(
                  pose["raw_hfe"], pose["raw_kfe"]
                ) < math.hypot(previous["raw_hfe"], previous["raw_kfe"]):
                  closest[height] = pose
      results.append(
        {
          "leg": leg,
          "window": window,
          "points_per_joint": points,
          "max_bottom_pose": best,
          "max_bottom_over_edge_pose": over_edge,
          "poses_clearing_height_plus_1cm": counts,
          "smallest_raw_norm_clearance_pose": closest,
          "sampled_max_abs_wheel_x_m": max_abs_x,
        }
      )
  return results


def wheel_contact_forces(
  inspect: Inspection, data: mujoco.MjData, sensors: dict[str, Any]
) -> dict[str, Any]:
  forces = {frame["leg"]: np.zeros(3) for frame in sensors["frames"]}
  geom_to_leg = {frame["geom_id"]: frame["leg"] for frame in sensors["frames"]}
  terrain_body = inspect.model.body("terrain").id
  counts = dict.fromkeys(forces, 0)
  for index in range(data.ncon):
    contact = data.contact[index]
    g1, g2 = int(contact.geom1), int(contact.geom2)
    for wheel_gid, other_gid, sign in ((g1, g2, -1), (g2, g1, 1)):
      if wheel_gid not in geom_to_leg:
        continue
      if inspect.model.geom_bodyid[other_gid] != terrain_body:
        continue
      local_force = np.zeros(6)
      mujoco.mj_contactForce(inspect.model, data, index, local_force)
      leg = geom_to_leg[wheel_gid]
      forces[leg] += sign * contact.frame.reshape(3, 3).T @ local_force[:3]
      counts[leg] += 1
  return {
    leg: {
      "contacts": counts[leg],
      "horizontal_force_N": float(np.linalg.norm(force[:2])),
      "vertical_force_abs_N": float(abs(force[2])),
    }
    for leg, force in forces.items()
  }


def check_static_candidates(
  inspect: Inspection, scans: list[dict[str, Any]], edge_x: float
) -> list[dict[str, Any]]:
  """Check selected soft-limit FK poses against a temporary 3 m wide step."""
  checks = []
  for height in (0.03, 0.15, 0.30):
    spec = inspect.scene.spec.copy()
    spec.worldbody.add_geom(
      name="diagnostic_step",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=[1.5, 1.5, height / 2],
      pos=[edge_x + 1.5, 0, height / 2],
      contype=1,
      conaffinity=1,
    )
    model = spec.compile()
    inspect.cfg.sim.mujoco.apply(model)
    for scan in scans:
      if scan["window"] != "soft":
        continue
      pose = scan["smallest_raw_norm_clearance_pose"][str(height)]
      if pose is None:
        continue
      data = mujoco.MjData(model)
      mujoco.mj_resetDataKeyframe(model, data, 0)
      for suffix, key in (("HFE", "hfe_rad"), ("KFE", "kfe_rad")):
        jid = model.joint(f"robot/{scan['leg']}_{suffix}").id
        data.qpos[model.jnt_qposadr[jid]] = pose[key]
      mujoco.mj_forward(model, data)
      checks.append(
        {
          "leg": scan["leg"],
          "height_m": height,
          "pose": pose,
          "penetrations_beyond_0p1mm": [
            {
              "geom1": model.geom(int(c.geom1)).name,
              "geom2": model.geom(int(c.geom2)).name,
              "distance_m": float(c.dist),
            }
            for c in data.contact
            if c.dist < -1e-4
          ],
        }
      )
  return checks


def pd_trials(
  inspect: Inspection,
  mapping: list[dict[str, Any]],
  sensors: dict[str, Any],
  seconds: float,
) -> list[dict[str, Any]]:
  """Independent flat-floor trials; commanded pulses are diagnostic inputs."""
  model = inspect.model
  trials = []
  root_id = model.body("robot/base").id
  ctrl_ids = [row["actuator_id"] for row in mapping]
  qpos_ids = [row["qpos_adr"] for row in mapping]
  dof_ids = [row["dof_adr"] for row in mapping]
  limits = np.array(
    [
      min(
        max(abs(v) for v in row["actuator_force_range_Nm"]),
        max(abs(v) for v in row["joint_force_range_Nm"]),
      )
      for row in mapping
    ]
  )
  for trial, legs in (
    ("neutral", ()),
    ("front_tuck", ("RF", "LF")),
    ("rear_tuck", ("RH", "LH")),
  ):
    data = inspect.initial_data()
    # Start every trial from the same settled free-base state.
    for _ in range(math.ceil(0.5 / model.opt.timestep)):
      mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    baseline = wheel_states(inspect, data, sensors)
    raw = [0.0] * len(mapping)
    for row in mapping:
      if row["joint"].split("_")[0] in legs and row["mode"] == "position":
        # Increase flexion with mirrored front/hind defaults. This pulse is
        # deliberately not a proposed policy, reward, or robot controller.
        direction = 1.0 if row["offset"] > 0 else -1.0
        raw[row["action_index"]] = direction * (
          0.5 if row["joint"].endswith("HFE") else 1.0
        )
    data.ctrl[:] = inspect.control(raw)
    targets = data.ctrl[ctrl_ids].copy()
    saturation = np.zeros(len(mapping))
    max_effort = np.zeros(len(mapping))
    max_speed = np.zeros(len(mapping))
    max_error = np.zeros(len(mapping))
    trace = []
    steps = math.ceil(seconds / model.opt.timestep)
    for step in range(steps):
      mujoco.mj_step(model, data)
      mujoco.mj_forward(model, data)
      if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise RuntimeError(f"Non-finite native PD state: {trial}")
      actual = np.array(
        [
          data.qpos[row["qpos_adr"]]
          if row["mode"] == "position"
          else data.qvel[row["dof_adr"]]
          for row in mapping
        ]
      )
      effort = np.abs(data.actuator_force[ctrl_ids])
      saturation += effort >= limits * 0.99
      max_effort = np.maximum(max_effort, effort)
      max_speed = np.maximum(max_speed, abs(data.qvel[dof_ids]))
      max_error = np.maximum(max_error, abs(targets - actual))
      if step % inspect.cfg.decimation == 0 or step == steps - 1:
        trace.append(
          {
            "time_s": float(data.time),
            "raw_actions": raw,
            "targets": targets.tolist(),
            "actual_position_or_velocity": actual.tolist(),
            "joint_positions_rad": data.qpos[qpos_ids].tolist(),
            "joint_velocities_rad_s": data.qvel[dof_ids].tolist(),
            "actuator_effort_Nm": data.actuator_force[ctrl_ids].tolist(),
            "root_xyz_m": data.xpos[root_id].tolist(),
            "wheels": wheel_states(inspect, data, sensors),
            "wheel_terrain_contact": wheel_contact_forces(inspect, data, sensors),
          }
        )
    trials.append(
      {
        "trial": trial,
        "backend": "native_mujoco_cpu",
        "settle_seconds": 0.5,
        "pulse_seconds": seconds,
        "baseline_wheels": baseline,
        "max_abs_error_rad_or_rad_s": max_error.tolist(),
        "final_abs_error_rad_or_rad_s": abs(
          targets - np.array(trace[-1]["actual_position_or_velocity"])
        ).tolist(),
        "max_abs_joint_velocity_rad_s": max_speed.tolist(),
        "max_abs_effort_Nm": max_effort.tolist(),
        "effort_saturation_fraction": (saturation / steps).tolist(),
        "trace": trace,
      }
    )
  return trials


def inspect_actions(
  points: int = 41, edge_x: float = 0.5, pd_seconds: float = 0.4
) -> dict[str, Any]:
  if points < 3 or not math.isfinite(edge_x):
    raise ValueError("Use at least 3 scan points and a finite edge X")
  if not math.isfinite(pd_seconds) or not 0 <= pd_seconds <= 2:
    raise ValueError("PD duration must be finite and between 0 and 2 seconds")
  inspect = build_inspection()
  mapping = action_mapping(inspect)
  sensors = sensor_mapping(inspect)
  initial = inspect.initial_data()
  scans = scan_front_legs(inspect, mapping, sensors, points, edge_x)
  sources = [
    Path(__file__),
    ASSET / "xmls/lego2.xml",
    ASSET / "urdf/urdf_legowheel2.urdf",
    ASSET / "lego2_constants.py",
    ROOT / "src/mjlab/tasks/velocity/config/lego2/env_cfgs.py",
    ROOT / "src/mjlab/tasks/velocity/config/lego2/platform_climb_env_cfg.py",
    ROOT / "src/mjlab/tasks/velocity/config/lego2/rl_cfg.py",
    ROOT / "src/mjlab/envs/mdp/actions/actions.py",
  ]
  return {
    "schema_version": 1,
    "task_action_source": "Mjlab-Platform-Climb-Lego2 (training config)",
    "scene": "one free robot, plane z=0, no policy/events/randomization",
    "mujoco_version": mujoco.__version__,
    "source_sha256": {
      str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
      for p in sources
    },
    "physics_dt_s": inspect.model.opt.timestep,
    "control_dt_s": inspect.model.opt.timestep * inspect.cfg.decimation,
    "rl_clip_actions": lego2_ppo_runner_cfg().clip_actions,
    "action_mapping": mapping,
    "sensor_mapping": sensors,
    "initial_root_qpos": initial.qpos[:7].tolist(),
    "initial_wheels": wheel_states(inspect, initial, sensors),
    "synthetic_target_probes": target_probes(inspect, mapping),
    "joint_perturbations": perturb_joints(inspect, mapping, sensors),
    "scan_edge_x_m": edge_x,
    "front_leg_scans": scans,
    "static_candidate_checks": check_static_candidates(inspect, scans, edge_x),
    "pd_trials": pd_trials(inspect, mapping, sensors, pd_seconds)
    if pd_seconds > 0
    else [],
    "limits_of_evidence": [
      "Synthetic actions, not checkpoint 9999 policy trajectories.",
      "FK grids do not filter collisions; only selected soft-limit poses are checked.",
      "Static poses do not validate an approach path or sufficient support/torque.",
      "Raw [-1,1] is a comparison window, not an implemented action clamp.",
      "URDF velocity values are declarations; current built-ins have no speed cap.",
      "Native CPU flat-floor PD does not validate Warp contacts or climbing.",
    ],
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--scan-points", type=int, default=41)
  parser.add_argument(
    "--edge-x",
    type=float,
    default=0.5,
    help="Hypothetical edge in root-default coordinates (m)",
  )
  parser.add_argument(
    "--pd-seconds",
    type=float,
    default=0.4,
    help="Native CPU pulse duration; 0 skips PD (max 2 s)",
  )
  args = parser.parse_args()
  if args.output.exists():
    parser.error("Output already exists; choose a new path to preserve evidence")
  report = inspect_actions(args.scan_points, args.edge_x, args.pd_seconds)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  with args.output.open("x") as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
    stream.write("\n")
  for row in report["action_mapping"]:
    print(
      f"action[{row['action_index']:2}] -> {row['joint']:8} "
      f"local={row['target_id']:2} ctrl={row['actuator_id']:2} "
      f"target=raw*{row['scale']}+{row['offset']:.3f}"
    )
  print("Contact primary_names:", report["sensor_mapping"]["primary_names"])
  print("Wheel frames:", [f["leg"] for f in report["sensor_mapping"]["frames"]])
  print(f"Report: {args.output.resolve()}")
  print("No policy/training or task/asset edits; see limits_of_evidence in report.")


if __name__ == "__main__":
  main()
