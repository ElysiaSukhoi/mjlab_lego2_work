"""Fixed-speed evaluation and real reset pairing across unequal rollout lengths."""

import json
import runpy
from pathlib import Path

import pytest

from mjlab.tasks.velocity.config.lego2.blind_pit_env_cfg import (
  lego2_blind_pit_env_cfg,
  lego2_blind_pit_eval_cfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg


def test_speed_factory_preserves_training_and_legacy_eval():
  train = lego2_blind_pit_env_cfg()
  legacy = lego2_blind_pit_eval_cfg(0.30)
  for cfg in (train, legacy):
    command = cfg.commands["twist"]
    assert isinstance(command, UniformVelocityCommandCfg)
    assert command.ranges.lin_vel_x == (0.3, 0.6)
  for speed in (0.3, 0.4, 0.5):
    cfg = lego2_blind_pit_eval_cfg(0.30, speed)
    command = cfg.commands["twist"]
    assert isinstance(command, UniformVelocityCommandCfg)
    assert command.ranges.lin_vel_x == (speed, speed)
    assert command.heading_command and command.heading_control_stiffness == 0.5
    assert cfg.actions == legacy.actions and cfg.rewards == legacy.rewards
    assert cfg.terminations == legacy.terminations and not cfg.auto_reset
    assert cfg.curriculum == {} and cfg.episode_length_s == 20
  for speed in (0, 0.29, 0.61, float("nan")):
    with pytest.raises(ValueError, match="forward speed"):
      lego2_blind_pit_eval_cfg(0.30, speed)


def test_real_paired_resets_and_capped_accounting(tmp_path):
  tool = runpy.run_path(
    str(
      Path(__file__).resolve().parents[1] / "scripts/tools/evaluate_lego2_blind_pit.py"
    )
  )
  reports = []
  for steps, speed in enumerate((0.3, 0.4, 0.5), start=2):
    output = tmp_path / f"vx_{speed:.2f}"
    report = tool["run"](
      tool["EvalConfig"](
        output=output,
        heights=(0.30,),
        forward_speed=speed,
        paired_seed_base=7,
        episodes=2,
        device="cpu",
        max_steps=steps,
      )
    )
    assert report["mode"] == "zero_action_diagnostic"
    assert report["fixed_forward_speed_m_s"] == speed
    height = report["heights"][0]
    assert height["n"] == 0 and height["pass_rate"] is None
    assert all(s["steps"] == steps and not s["completed"] for s in height["samples"])
    trace = [json.loads(line) for line in (output / "trace_0.300000.jsonl").open()]
    assert len(trace) == 2 * steps
    for row in trace:
      assert row["command"][0] == pytest.approx(speed)
      assert row["continuous_support_s"] == 0
      assert row["contact_pos"] is not None
      assert len(row["nonfoot_names"]) == len(row["nonfoot_contact_found"])
    reports.append(report)
  for episode in (0, 1):
    reference = reports[0]["heights"][0]["samples"][episode]
    assert reference["reset_seed"] == 7 + episode
    for report in reports[1:]:
      sample = report["heights"][0]["samples"][episode]
      assert sample["reset_seed"] == reference["reset_seed"]
      for field, values in reference["initial_state"].items():
        actual = sample["initial_state"][field]
        # 最后两维是vx/wz命令；vx正是本次对照变量，其他历史输入须相同。
        if field == "actor_observation":
          assert len(actual) == len(values) == 144
          actual, values = actual[:-2], values[:-2]
        assert actual == pytest.approx(values, abs=1e-6)
  summary_tool = runpy.run_path(
    str(
      Path(__file__).resolve().parents[1]
      / "scripts/tools/summarize_lego2_speed_eval.py"
    )
  )
  config = summary_tool["SummaryConfig"](directory=tmp_path, episodes=2, mode="smoke")
  summary = summary_tool["summarize"](config)
  assert summary["pairing_verified"]
  changed = tmp_path / "vx_0.40/report.json"
  report = json.loads(changed.read_text())
  report["heights"][0]["samples"][0]["initial_state"]["root_pose_w"][0] += 0.01
  changed.write_text(json.dumps(report))
  with pytest.raises(ValueError, match="Pairing mismatch"):
    summary_tool["summarize"](config)
