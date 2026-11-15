"""Audit paired GPU traces offline; kinematics only, no policy or simulation."""

import hashlib
import json
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import mujoco
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / (
  "docs/codex/diagnostics/20261115_model2197_comparison_remote/"
  "remote_evidence_kO71wR.tar.gz"
)
OUTPUT = ROOT / "docs/codex/diagnostics/20261115_model2197_comparison_analysis.json"
PREFIXES = {
  "1198": "phase3_speed_20261115_185511_3295996",
  "2197": "phase3_eval2197_20261115_201606_3299957",
}
HASHES = {
  "1198": "7780cf0d2557c86f51c0486f870e7d471b13703777e9cee5e1854427def05027",
  "2197": "b8bc52a8ee370eb94e359bc71d310f5f1cd0ec013cae9e5982b44fb13b8a54a4",
}
DT = 0.02


def sha(path: Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      h.update(block)
  return h.hexdigest()


def longest(mask: Any) -> int:
  count = best = 0
  for value in mask:
    count = count + 1 if value else 0
    best = max(best, count)
  return best


def analyze(model: Any, rows: list[dict], sample: dict) -> tuple[dict, dict]:
  n = len(rows)
  assert n == sample["steps"] and sample["completed"]
  assert [r["step"] for r in rows] == list(range(1, n + 1))
  array = lambda key: np.array([r[key] for r in rows])  # noqa: E731
  wheel_ids = [model.geom(name).id for name in rows[0]["wheel_names"]]
  joint_ids = [model.joint(name).id for name in rows[0]["joint_names"]]
  thigh_names = ["RF_thigh_collision2", "LF_thigh_collision2"]
  thigh_ids = [model.geom(name).id for name in thigh_names]
  data = mujoco.MjData(model)
  centers, extents, thigh_bottom = [], [], []
  for row in rows:
    data.qpos[:7] = row["root"]
    data.qpos[model.jnt_qposadr[joint_ids]] = row["joint_pos"]
    # 仅做FK；圆柱世界包络和box最低点都从原XML重建，不推进物理时间。
    mujoco.mj_kinematics(model, data)
    rotation = data.geom_xmat[wheel_ids].reshape(-1, 3, 3)
    axis = rotation[:, :, 2]
    size = model.geom_size[wheel_ids]
    extent = size[:, :1] * np.sqrt(np.maximum(0, 1 - axis**2))
    extent += size[:, 1:2] * abs(axis)
    centers.append(data.geom_xpos[wheel_ids].copy())
    extents.append(extent)
    box_rotation = data.geom_xmat[thigh_ids].reshape(-1, 3, 3)
    box_extent = np.einsum("wij,wj->wi", abs(box_rotation), model.geom_size[thigh_ids])
    thigh_bottom.append(data.geom_xpos[thigh_ids, 2] - box_extent[:, 2])
  centers, extents = np.array(centers), np.array(extents)
  bottom = centers[:, :, 2] - extents[:, :, 2]
  fk_error = float(abs(bottom - array("wheel_bottom_above_rim")).max())
  assert fk_error < 1e-5
  xy = abs(centers[:, :, :2] - [-5, 0])
  outside = (xy - extents[:, :, :2] > 1.52).any(-1)
  inside = (xy + extents[:, :, :2] < 4.95).all(-1)
  top = abs(bottom) < 0.025
  found, force, normal = (
    array(k) for k in ("contact_found", "contact_force", "contact_normal")
  )
  contact_top = abs(array("contact_pos")[:, :, 2]) < 0.02
  present = found > 0
  normal_ok = normal[:, :, 2] < -0.5
  force_ok = abs(force[:, :, 2]) > 1
  supported = outside & inside & top & contact_top & present & normal_ok & force_ok
  front = [
    i for i, name in enumerate(rows[0]["wheel_names"]) if name[:2] in ("RF", "LF")
  ]
  rear = [i for i in range(4) if i not in front]
  relative = array("root")[:, :2] - [-5, 0]
  np.testing.assert_allclose(relative, array("root_relative_xy"), atol=1e-6, rtol=0)
  gravity = array("projected_gravity")
  stages = {
    "root_out": (abs(relative) > 1.5).any(-1) & (abs(relative) < 4).all(-1),
    "front_up": supported[:, front].all(-1),
    "rear_up": supported[:, rear].all(-1),
    "complete": supported.all(-1),
    "stable_now": supported.all(-1) & (-gravity[:, 2] > np.cos(np.pi / 4)),
  }
  for key, values in stages.items():
    np.testing.assert_array_equal(values, [r["stages"][key] for r in rows])
    if key != "stable_now":
      assert bool(values.any()) == sample[key]
  # 本批所有episode无terminated；按原0.5秒判据重建连续计时及锁存。
  assert not sample["failed"] and not sample["fell_over"]
  hold = np.zeros(n)
  for i in range(n):
    hold[i] = ((hold[i - 1] if i else 0) + DT) if stages["stable_now"][i] else 0
  np.testing.assert_allclose(hold, array("continuous_support_s"), atol=1e-5, rtol=0)
  passed = np.maximum.accumulate(hold >= 0.5 - 1e-6)
  np.testing.assert_array_equal(passed, array("stable_pass"))
  assert bool(passed[-1]) == sample["passed"]
  rmse = float(np.sqrt(np.mean(np.sum(array("speed_error_xy") ** 2, axis=1))))
  assert abs(rmse - sample["speed_rmse"]) < 1e-8
  complete_ids = np.flatnonzero(stages["complete"])
  first = int(complete_ids[0]) if len(complete_ids) else n
  tail = slice(first, n)
  tilt = np.degrees(np.arccos(np.clip(-gravity[:, 2], -1, 1)))
  nf_force = array("nonfoot_contact_force")
  # 此传感器global_frame=False，force在接触坐标系；只比较旋转不变的范数。
  nf_norm = np.linalg.norm(nf_force, axis=-1)
  nf_active = (array("nonfoot_contact_found") > 0) & (nf_norm > 1)
  incomplete = ~stages["complete"][tail]
  contact_loss = ~present[tail]
  summary: dict[str, Any] = {
    "seed": sample["reset_seed"],
    "passed": sample["passed"],
    "steps": n,
    "speed_rmse": rmse,
    "first_complete_s": (first + 1) * DT if first < n else None,
    "max_continuous_support_s": float(max(hold)),
    "max_wheel_bottom_fk_error_m": fk_error,
    "any_torque_saturated_steps": int(array("torque_saturated").any(-1).sum()),
    "target_over_hard_limit_steps": int(array("target_over_hard_limit").any(-1).sum()),
    "end_root_relative_xy_m": relative[-1].tolist(),
    "after_first_complete": None,
  }
  if first < n:
    # 只统计首次complete到原episode结束，不外推结束后的行为。
    summary["after_first_complete"] = {
      "steps": n - first,
      "incomplete_steps": int(incomplete.sum()),
      "missing_contact_steps_by_wheel": contact_loss.sum(0).tolist(),
      "max_missing_run_s_by_wheel": [longest(v) * DT for v in contact_loss.T],
      "incomplete_explained_by_missing_contact_steps": int(
        (incomplete & contact_loss.any(-1)).sum()
      ),
      "present_but_bad_normal_steps": int((present[tail] & ~normal_ok[tail]).sum()),
      "present_but_weak_force_steps": int((present[tail] & ~force_ok[tail]).sum()),
      "geometry_ineligible_steps": int((~(outside & inside & top)[tail].all(-1)).sum()),
      "present_but_bad_contact_z_steps": int(
        (present[tail] & ~contact_top[tail]).sum()
      ),
      "max_tilt_deg": float(tilt[tail].max()),
      "max_abs_wheel_bottom_rim_m": float(abs(bottom[tail]).max()),
      "mean_root_height_m": float(array("root")[tail, 2].mean()),
      "nonfoot_active_steps": int(nf_active[tail].any(-1).sum()),
      "nonfoot_active_and_incomplete_steps": int(
        (nf_active[tail].any(-1) & incomplete).sum()
      ),
      "nonfoot_channels": {
        name: {
          "steps_above_1N": int(nf_active[tail, i].sum()),
          "peak_force_norm_N": float(nf_norm[tail, i].max()),
        }
        for i, name in enumerate(rows[0]["nonfoot_names"])
        if nf_active[tail, i].any()
      },
      "front_thigh_box_bottom_range_m": {
        name: [float(v.min()), float(v.max())]
        for name, v in zip(thigh_names, np.array(thigh_bottom)[tail].T, strict=True)
      },
      "mean_joint_pos_rad": dict(
        zip(
          rows[0]["joint_names"], array("joint_pos")[tail].mean(0).tolist(), strict=True
        )
      ),
      "raw_action_change_rms": float(
        np.sqrt(np.mean(np.diff(array("raw_action")[tail], axis=0) ** 2))
      ),
    }
  plot_data = {
    "time_s": np.arange(1, n + 1) * DT,
    "hold": hold,
    "present": present,
    "first_s": (first + 1) * DT if first < n else 0,
    "nonfoot": nf_active.any(-1),
  }
  return summary, plot_data


def run() -> None:
  model = mujoco.MjModel.from_xml_path(
    str(ROOT / "src/mjlab/asset_zoo/robots/lego2/xmls/lego2.xml")
  )
  metadata: dict[str, bytes] = {}
  traces: dict[tuple[str, float], dict[int, list[dict]]] = {}
  input_hashes = {}
  # 不解压到源码目录；逐成员读取原归档，拒绝重复路径。
  with tarfile.open(ARCHIVE, "r|gz") as archive:
    for member in archive:
      if not member.isfile():
        continue
      assert member.name not in input_hashes
      stream = archive.extractfile(member)
      assert stream is not None
      raw = stream.read()
      input_hashes[member.name] = hashlib.sha256(raw).hexdigest()
      if member.name.endswith(".jsonl"):
        key = next(
          k for k, prefix in PREFIXES.items() if member.name.startswith(prefix)
        )
        speed = float(member.name.split("/vx_")[1].split("/")[0])
        groups = defaultdict(list)
        for line in raw.splitlines():
          row = json.loads(line)
          groups[row["episode"]].append(row)
        traces[key, speed] = dict(groups)
      else:
        metadata[member.name] = raw
  reports = {}
  result: dict[str, Any] = {
    "method": "Offline raw report/trace audit and XML kinematics; no simulation/PPO",
    "archive": str(ARCHIVE.relative_to(ROOT)),
    "archive_sha256": sha(ARCHIVE),
    "member_sha256": input_hashes,
    "analysis_script_sha256": sha(Path(__file__)),
    "checkpoint_sha256": HASHES,
    "weights_downloaded": False,
    "dt_s": DT,
    "origin_world_m": [-5, 0, -0.3],
    "limits": [
      "Weights not included: model SHA verified across reports and SHA records only",
      "20ms control-frame snapshots do not resolve 5ms physics-substep contacts",
      "Zero found/force does not establish real flight or a contact-solver defect",
      "Nonfoot forces are in contact coordinates; only norms are compared",
      "Body contacts and posture are correlations, not causal intervention evidence",
      "10 paired resets per speed are screening, not generalization acceptance",
    ],
    "episodes": [],
  }
  plots = {}
  # BaseLoader只读标量/序列/字典；不执行YAML内的Python对象构造标签。
  train_env = yaml.load(
    metadata[f"{PREFIXES['2197']}/train_env.yaml"], Loader=yaml.BaseLoader
  )
  train_agent = yaml.load(
    metadata[f"{PREFIXES['2197']}/train_agent.yaml"], Loader=yaml.BaseLoader
  )
  train_weights = {k: float(v["weight"]) for k, v in train_env["rewards"].items()}
  assert len(train_weights) == 11
  assert int(train_env["scene"]["num_envs"]) == 1024
  assert [float(v) for v in train_env["commands"]["twist"]["ranges"]["lin_vel_x"]] == [
    0.3,
    0.6,
  ]
  assert train_agent["resume"] == "true"
  result["saved_training_config"] = {
    "num_envs": int(train_env["scene"]["num_envs"]),
    "rollout": int(train_agent["num_steps_per_env"]),
    "max_iterations": int(train_agent["max_iterations"]),
    "seed": int(train_agent["seed"]),
    "load_run": train_agent["load_run"],
    "load_checkpoint": train_agent["load_checkpoint"],
    "reward_weights": train_weights,
    "total_timesteps_stdout_downloaded": False,
  }
  for key, prefix in PREFIXES.items():
    assert metadata[f"{prefix}/checkpoint.sha256"].decode().split()[0] == HASHES[key]
    paired = json.loads(metadata[f"{prefix}/full/paired_summary.json"])
    assert paired["pairing_verified"] and paired["checkpoint_sha256"] == HASHES[key]
    for speed in (0.3, 0.4, 0.5):
      path = f"{prefix}/full/vx_{speed:.2f}/report.json"
      report = json.loads(metadata[path])
      env_cfg = yaml.load(
        metadata[f"{prefix}/full/vx_{speed:.2f}/env_0.300000.yaml"],
        Loader=yaml.BaseLoader,
      )
      assert (
        float(env_cfg["sim"]["mujoco"]["timestep"]) * int(env_cfg["decimation"]) == DT
      )
      assert {
        k: float(v["weight"]) for k, v in env_cfg["rewards"].items()
      } == train_weights
      reports[key, speed] = report
      assert report["checkpoint_sha256"] == HASHES[key]
      assert report["seed"] == report["paired_seed_base"] == 7
      assert report["fixed_forward_speed_m_s"] == speed
      assert report["mode"] == "policy"
      assert (
        input_hashes[path]
        == paired["input_report_sha256"][f"vx_{speed:.2f}/report.json"]
      )
      for source, expected in report["source_sha256"].items():
        assert sha(ROOT / source) == expected, source
      height = report["heights"][0]
      assert height["height_m"] == 0.3 and height["n"] == height["requested_n"] == 10
      assert height["passed"] == sum(s["passed"] for s in height["samples"])
      item = next(item for item in paired["results"] if item["speed_m_s"] == speed)
      assert item["n"] == 10 and item["passed"] == height["passed"]
      assert item["pass_rate"] == height["passed"] / 10 == height["pass_rate"]
      for field in ("complete", "stalled", "fell_over", "failed"):
        assert item[field] == sum(s[field] for s in height["samples"])
      assert (
        abs(
          item["mean_speed_rmse"] - sum(s["speed_rmse"] for s in height["samples"]) / 10
        )
        < 1e-12
      )
      for sample in height["samples"]:
        episode = sample["episode"]
        assert sample["reset_seed"] == episode + 7
        summary, plot_data = analyze(model, traces[key, speed][episode], sample)
        result["episodes"].append({"model": key, "speed_m_s": speed, **summary})
        if (speed, sample["reset_seed"]) in ((0.3, 11), (0.5, 11), (0.5, 13)):
          plots[key, speed, sample["reset_seed"]] = plot_data
  # 用真实reset数值复核组内/跨模型配对；actor仅排除最后两项中的vx。
  max_pair_error = 0.0
  base_samples = reports["1198", 0.3]["heights"][0]["samples"]
  for report in reports.values():
    for base, sample in zip(base_samples, report["heights"][0]["samples"], strict=True):
      for field, values in base["initial_state"].items():
        left, right = np.array(values), np.array(sample["initial_state"][field])
        if field == "actor_observation":
          assert len(left) == len(right) == 144
          left, right = np.delete(left, 142), np.delete(right, 142)
        error = float(abs(left - right).max())
        max_pair_error = max(max_pair_error, error)
        assert error <= 1e-6
  comparison = json.loads(metadata[f"{PREFIXES['2197']}/comparison.json"])
  assert comparison["baseline_checkpoint_sha256"] == HASHES["1198"]
  assert comparison["new_checkpoint_sha256"] == HASHES["2197"]
  for entry in comparison["comparisons"]:
    old = reports["1198", entry["speed_m_s"]]["heights"][0]["samples"]
    new = reports["2197", entry["speed_m_s"]]["heights"][0]["samples"]
    assert entry["old_passed"] == sum(s["passed"] for s in old)
    assert entry["new_passed"] == sum(s["passed"] for s in new)
    for field, before, after in (
      ("improved_seeds", False, True),
      ("regressed_seeds", True, False),
    ):
      seeds = [
        a["reset_seed"]
        for a, b in zip(old, new, strict=True)
        if a["passed"] == before and b["passed"] == after
      ]
      assert seeds == entry[field]
  result["comparison"] = comparison
  result["max_initial_state_pair_error"] = max_pair_error
  result["verified"] = {
    "reports": 6,
    "episodes": 60,
    "trace_steps": sum(e["steps"] for e in result["episodes"]),
    "source_hashes_match": True,
    "all_stage_hold_pass_rmse_reconstructions_match": True,
  }
  fig, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
  for row_i, (speed, seed) in enumerate(((0.3, 11), (0.5, 11), (0.5, 13))):
    for col_i, key in enumerate(("1198", "2197")):
      p = plots[key, speed, seed]
      ax = axes[row_i, col_i]
      ax.plot(p["time_s"], p["hold"], label="Continuous support (s)")
      ax.axhline(0.5, color="tab:red", linestyle="--", label="Pass threshold")
      ax.axvline(p["first_s"], color="gray", linestyle=":", label="First complete")
      ax.fill_between(
        p["time_s"],
        0,
        0.05,
        where=p["nonfoot"],
        color="tab:orange",
        alpha=0.7,
        label="Nonfoot contact >1 N",
      )
      ax.set(
        title=f"Model {key}, vx={speed:.2f} m/s, seed={seed}",
        xlabel="Episode time (s)",
        ylabel="Hold (s)",
        ylim=(0, 3),
        xlim=(p["first_s"] - 0.2, p["time_s"][-1]),
      )
  axes[0, 0].legend(fontsize=8, loc="upper left")
  fig.savefig(OUTPUT.with_suffix(".png"), dpi=150)
  plt.close(fig)
  OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
  print(json.dumps(result["verified"]))
  print(OUTPUT)


if __name__ == "__main__":
  run()
