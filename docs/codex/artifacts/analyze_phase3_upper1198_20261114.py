"""Audit downloaded report/trace using kinematics only, without simulation."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import mujoco
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mjlab.tasks.velocity.config.lego2.blind_pit_env_cfg import lego2_blind_pit_eval_cfg


def sha(path):
  return hashlib.sha256(path.read_bytes()).hexdigest()


def longest(mask):
  count = best = 0
  for value in mask:
    count = count + 1 if value else 0
    best = max(best, count)
  return best


def run():
  root = Path(__file__).resolve().parents[3]
  evidence = root / "docs/codex/diagnostics/20261114_phase3_upper1198_remote"
  out = root / "docs/codex/diagnostics/20261114_phase3_upper1198_analysis.json"
  figure = out.with_suffix(".png")
  # 只重算本脚本的派生JSON/图；下载的原始证据始终只读。
  report = json.loads((evidence / "report.json").read_text())
  expected = "7780cf0d2557c86f51c0486f870e7d471b13703777e9cee5e1854427def05027"
  assert report["checkpoint_sha256"] == expected
  assert (evidence / "checkpoint.sha256").read_text().split()[0] == expected
  assert report["seed"] == 7 and report["mode"] == "policy"
  matches = {k: sha(root / k) == v for k, v in report["source_sha256"].items()}
  assert all(matches.values())
  samples = next(h for h in report["heights"] if h["height_m"] == 0.3)["samples"]
  episodes = defaultdict(list)
  for line in (evidence / "trace_0.300000.jsonl").open():
    row = json.loads(line)
    episodes[row["episode"]].append(row)

  cfg = lego2_blind_pit_eval_cfg(0.3)
  dt = cfg.sim.mujoco.timestep * cfg.decimation
  assert cfg.scene.terrain is not None
  gen = cfg.scene.terrain.terrain_generator
  assert gen is not None
  # 两行单列地块居中排列、初始level=0；原点在首行坑底，非出生随机位置。
  origin = np.array([(1 - gen.num_rows) * gen.size[0] / 2, 0, -0.3])
  assert gen.num_cols == 1 and cfg.scene.terrain.max_init_terrain_level == 0
  model = mujoco.MjModel.from_xml_path(
    str(root / "src/mjlab/asset_zoo/robots/lego2/xmls/lego2.xml")
  )
  data = mujoco.MjData(model)
  result: dict[str, Any] = {
    "method": "Read downloaded evidence; XML forward kinematics, no simulation/PPO",
    "checkpoint": report["checkpoint"],
    "checkpoint_sha256": expected,
    "checkpoint_file_downloaded": False,
    "seed": report["seed"],
    "source_matches": matches,
    "input_sha256": {p.name: sha(p) for p in sorted(evidence.iterdir()) if p.is_file()},
    "analysis_script_sha256": sha(Path(__file__)),
    "dt_s": dt,
    "origin_world_m": origin.tolist(),
    "limits": [
      "Actual GPU runtime model/config was not downloaded; reconstruct from matching sources",
      "Trace omits contact positions, cannot fully reconstruct contact_on_top",
      "Missing contact or weak force cannot distinguish real unloading from sensor/solver behavior",
      "Joint target violations and saturation are correlations, not proof of a cause",
      "Kinematics establishes wheel clearance from side walls, not absence of body-wall contacts",
    ],
    "episodes": [],
  }
  fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
  path_ax = axes[0, 0]
  path_ax.plot([-1.5, 1.5, 1.5, -1.5, -1.5], [-1.5, -1.5, 1.5, 1.5, -1.5], "k--")
  maximum_fk_error = 0.0
  for sample in samples:
    episode = sample["episode"]
    rows = episodes[episode]
    n = len(rows)
    assert n == sample["steps"]
    assert [r["step"] for r in rows] == list(range(1, n + 1))
    assert rows[-1]["stable_pass"] == sample["passed"]
    for key in ("root_out", "front_up", "rear_up", "complete"):
      assert any(r["stages"][key] for r in rows) == sample[key]
    error = np.array([r["speed_error_xy"] for r in rows])
    assert abs(np.sqrt(np.mean(np.sum(error**2, axis=1))) - sample["speed_rmse"]) < 1e-8
    wheel_ids = [model.geom(name).id for name in rows[0]["wheel_names"]]
    joint_ids = [model.joint(name).id for name in rows[0]["joint_names"]]
    centers, extents = [], []
    for row in rows:
      data.qpos[:7] = row["root"]
      data.qpos[model.jnt_qposadr[joint_ids]] = row["joint_pos"]
      # 仅由记录姿态/关节角重建轮碰撞圆柱的世界坐标，不推进物理时间。
      mujoco.mj_kinematics(model, data)
      axis = data.geom_xmat[wheel_ids].reshape(-1, 3, 3)[:, :, 2]
      size = model.geom_size[wheel_ids]
      extent = size[:, :1] * np.sqrt(np.maximum(0, 1 - axis**2))
      extent += size[:, 1:2] * abs(axis)
      centers.append(data.geom_xpos[wheel_ids].copy())
      extents.append(extent)
    centers, extents = np.array(centers), np.array(extents)
    bottom = centers[:, :, 2] - extents[:, :, 2]
    traced_bottom = np.array([r["wheel_bottom_above_rim"] for r in rows])
    fk_error = float(np.max(abs(bottom - traced_bottom)))
    maximum_fk_error = max(maximum_fk_error, fk_error)
    assert fk_error < 1e-5
    local = centers[:, :, :2] - origin[:2]
    outside = ((abs(local) - extents[:, :, :2]) > 1.52).any(axis=-1)
    top = abs(bottom) < 0.025
    found = np.array([r["contact_found"] for r in rows])
    normal = np.array([r["contact_normal"] for r in rows])
    force = np.array([r["contact_force"] for r in rows])
    support = (found > 0) & (normal[:, :, 2] < -0.5) & (abs(force[:, :, 2]) > 1)
    stable = np.array([r["stages"]["stable_now"] for r in rows])
    complete = np.array([r["stages"]["complete"] for r in rows])
    root_xy = np.array([r["root"][:2] for r in rows]) - origin[:2]
    tilt = np.degrees(
      np.arccos(np.clip(-np.array([r["projected_gravity"][2] for r in rows]), -1, 1))
    )
    after_outside = outside.all(axis=1)
    after_top = after_outside & top.all(axis=1)
    target_bad = np.array([r["target_over_hard_limit"] for r in rows])
    saturation = np.array([r["torque_saturated"] for r in rows])
    # 轮子Y外缘与侧壁±1.5的距离；正值表示轮子仍在侧壁之间。
    side_clearance = 1.5 - (abs(local[:, :, 1]) + extents[:, :, 1])
    first = {
      k: next((r["step"] * dt for r in rows if r["stages"][k]), None)
      for k in rows[0]["stages"]
    }
    summary = {
      "episode_zero_based": episode,
      "episode_display": episode + 1,
      "passed": sample["passed"],
      "steps": n,
      "duration_s": n * dt,
      "command_vx_m_s": rows[0]["command"][0],
      "first_stage_time_s": first,
      "max_stable_run_s": longest(stable) * dt,
      "complete_steps": int(complete.sum()),
      "max_tilt_deg": float(tilt.max()),
      "root_end_local_xy_m": root_xy[-1].tolist(),
      "max_abs_root_y_m": float(abs(root_xy[:, 1]).max()),
      "minimum_wheel_side_clearance_m": float(side_clearance.min()),
      "maximum_wheel_bottom_above_floor_m": (bottom + 0.3).max(axis=0).tolist(),
      "outside_and_top_steps": int(after_top.sum()),
      "outside_top_without_all_support_steps": int(
        (after_top & ~support.all(axis=1)).sum()
      ),
      "outside_top_missing_contact_steps_by_wheel": ((found == 0) & after_top[:, None])
      .sum(axis=0)
      .tolist(),
      "outside_top_weak_vertical_force_steps_by_wheel": (
        (abs(force[:, :, 2]) <= 1) & after_top[:, None]
      )
      .sum(axis=0)
      .tolist(),
      "target_over_hard_limit_steps_by_joint": dict(
        zip(
          rows[0]["position_target_names"], target_bad.sum(axis=0).tolist(), strict=True
        )
      ),
      "any_target_over_hard_limit_steps": int(target_bad.any(axis=1).sum()),
      "any_saturation_steps": int(saturation.any(axis=1).sum()),
      "peak_abs_actuator_force_nm": dict(
        zip(
          rows[0]["actuator_names"],
          abs(np.array([r["actuator_force"] for r in rows])).max(axis=0).tolist(),
          strict=True,
        )
      ),
    }
    result["episodes"].append(summary)
    path_ax.plot(
      root_xy[:, 0],
      root_xy[:, 1],
      label=f"Ep {episode + 1}" if not sample["passed"] else None,
      color="tab:red" if not sample["passed"] else "tab:green",
      alpha=0.8 if not sample["passed"] else 0.3,
    )
    if episode in (1, 2):
      ax = axes[1, episode - 1]
      ax.plot(np.arange(1, n + 1) * dt, bottom, label=rows[0]["wheel_names"])
      ax.axhline(0, color="k", linestyle="--")
      ax.set(
        title=f"Ep {episode + 1}: wheel bottom relative to rim",
        xlabel="Time (s)",
        ylabel="Height (m)",
      )
  path_ax.set(
    title="30 cm pit: root paths (green=pass)",
    xlabel="Local X (m)",
    ylabel="Local Y (m)",
    aspect="equal",
  )
  path_ax.legend()
  for ep in (0, 1, 2):
    rs = episodes[ep]
    ax = axes[0, 1]
    values = [r["stages"]["stable_now"] for r in rs]
    ax.step(
      np.arange(1, len(rs) + 1) * dt,
      np.array(values) * 0.7 + (2 - ep),
      where="post",
      label=f"Ep {ep + 1}",
    )
  ax.set(
    title="Continuous eligible support (high=all 4 wheels)",
    xlabel="Time (s)",
    yticks=[0, 1, 2],
    yticklabels=["Ep 3", "Ep 2", "Ep 1"],
  )
  axes[1, 0].legend(fontsize=7)
  result["maximum_wheel_bottom_fk_error_m"] = maximum_fk_error
  fig.savefig(figure, dpi=160)
  out.write_text(json.dumps(result, indent=2) + "\n")
  print(
    json.dumps(
      {"output": str(out), "figure": str(figure), "fk_error_m": maximum_fk_error}
    )
  )
  for ep in result["episodes"]:
    print(json.dumps(ep))


if __name__ == "__main__":
  run()
