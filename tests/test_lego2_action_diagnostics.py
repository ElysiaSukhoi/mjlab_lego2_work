"""End-to-end checks of the diagnostic's real Lego2 routing and response."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/tools/inspect_lego2_actions.py"


@pytest.fixture(scope="module")
def diagnostic_report(tmp_path_factory):
  output = tmp_path_factory.mktemp("lego2_diagnostic") / "report.json"
  command = [
    "uv",
    "run",
    "--no-sync",
    "python",
    str(SCRIPT),
    "--scan-points",
    "17",
    "--output",
    str(output),
  ]
  subprocess.run(
    command, cwd=ROOT, env=os.environ.copy(), check=True, capture_output=True, text=True
  )
  return json.loads(output.read_text())


def test_resolved_action_and_sensor_names(diagnostic_report):
  mapping = diagnostic_report["action_mapping"]
  expected = [
    f"{leg}_{joint}" for leg in ("LF", "RF", "LH", "RH") for joint in ("HFE", "KFE")
  ] + [f"{leg}_Wheel" for leg in ("LF", "RF", "LH", "RH")]
  assert [row["joint"] for row in mapping] == expected
  assert len({row["actuator_id"] for row in mapping}) == 12
  sensors = diagnostic_report["sensor_mapping"]
  assert [f["leg"] for f in sensors["frames"]] == ["RF", "LF", "RH", "LH"]
  assert [f["contact_primary_index"] for f in sensors["frames"]] == [0, 1, 2, 3]
  assert all(
    w["site_center_error_m"] < 1e-10
    for w in diagnostic_report["initial_wheels"].values()
  )


def test_raw_actions_reach_real_ctrl_without_implicit_clipping(diagnostic_report):
  mapping = {row["joint"]: row for row in diagnostic_report["action_mapping"]}
  assert diagnostic_report["rl_clip_actions"] is None
  for probe in diagnostic_report["synthetic_target_probes"]:
    row = mapping[probe["joint"]]
    assert row["action_clip"] is None
    assert probe["written_ctrl_target"] == pytest.approx(
      probe["synthetic_raw"] * row["scale"] + row["offset"], abs=1e-6
    )
  assert any(
    p["outside_hard_range"] for p in diagnostic_report["synthetic_target_probes"]
  )
  assert all(
    row["implemented_velocity_limit_rad_s"] is None for row in mapping.values()
  )


def test_named_joint_perturbations_and_front_reach(diagnostic_report):
  for probe in diagnostic_report["joint_perturbations"]:
    leg, joint = probe["joint"].split("_")
    assert probe["moving_frames"] == [leg]
    if joint == "KFE":
      dz = probe["wheel_delta_xyz_m"][leg][2]
      front_sign = -1 if leg in ("RF", "LF") else 1
      assert dz * probe["delta_rad"] * front_sign > 0
  for scan in diagnostic_report["front_leg_scans"]:
    if scan["window"] == "soft":
      pose = scan["smallest_raw_norm_clearance_pose"]["0.15"]
      assert pose is not None
      assert pose["bottom_m"] >= 0.16
      assert pose["min_x_m"] >= diagnostic_report["scan_edge_x_m"]
  static = diagnostic_report["static_candidate_checks"]
  assert {(c["leg"], c["height_m"]) for c in static} == {
    (leg, height) for leg in ("RF", "LF") for height in (0.03, 0.15, 0.30)
  }
  assert all(not c["penetrations_beyond_0p1mm"] for c in static)


def test_native_pd_tracks_the_named_leg_pair_and_respects_effort_limits(
  diagnostic_report,
):
  mapping = diagnostic_report["action_mapping"]
  trials = {t["trial"]: t for t in diagnostic_report["pd_trials"]}
  neutral = np.array(trials["neutral"]["trace"][-1]["actual_position_or_velocity"])
  for name, legs in (("front_tuck", ("RF", "LF")), ("rear_tuck", ("RH", "LH"))):
    trial = trials[name]
    actual = trial["trace"][-1]["actual_position_or_velocity"]
    for index, row in enumerate(mapping):
      assert trial["max_abs_effort_Nm"][index] <= (
        max(abs(v) for v in row["actuator_force_range_Nm"]) + 1e-6
      )
      if row["joint"].split("_")[0] in legs and row["mode"] == "position":
        direction = np.sign(row["offset"])
        assert (actual[index] - neutral[index]) * direction > 0.05
    assert np.isfinite(trial["final_abs_error_rad_or_rad_s"]).all()


def test_cli_preserves_existing_report(tmp_path):
  output = tmp_path / "existing.json"
  output.write_text("old evidence\n")
  result = subprocess.run(
    ["uv", "run", "--no-sync", "python", str(SCRIPT), "--output", str(output)],
    cwd=ROOT,
    capture_output=True,
    text=True,
  )
  assert result.returncode != 0
  assert "Output already exists" in result.stderr
  assert output.read_text() == "old evidence\n"
