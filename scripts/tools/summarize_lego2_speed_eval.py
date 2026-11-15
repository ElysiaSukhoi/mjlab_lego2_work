"""Verify reset pairing before summarizing a three-speed Lego2 comparison."""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import tyro


@dataclass
class SummaryConfig:
  directory: Path
  episodes: int = 10
  mode: Literal["smoke", "full"] = "full"


def summarize(cfg: SummaryConfig) -> dict[str, Any]:
  reports = []
  paths = [cfg.directory / f"vx_{speed:.2f}/report.json" for speed in (0.3, 0.4, 0.5)]
  for speed, path in zip((0.3, 0.4, 0.5), paths, strict=True):
    report = json.loads(path.read_text())
    assert report["fixed_forward_speed_m_s"] == speed
    assert report["paired_seed_base"] == 7
    assert len(report["heights"]) == 1
    height = report["heights"][0]
    assert height["height_m"] == 0.3 and height["requested_n"] == cfg.episodes
    assert len(height["samples"]) == cfg.episodes
    if cfg.mode == "full":
      assert report["mode"] == "policy" and report["checkpoint_sha256"]
      assert height["n"] == cfg.episodes
      assert all(s["completed"] for s in height["samples"])
    else:
      assert height["n"] == 0 and not any(s["completed"] for s in height["samples"])
    for episode, sample in enumerate(height["samples"]):
      assert sample["episode"] == episode and sample["reset_seed"] == 7 + episode
      obs = sample["initial_state"]["actor_observation"]
      assert len(obs) == 144 and math.isclose(obs[-2], speed, abs_tol=1e-6)
    reports.append(report)

  reference = reports[0]
  for report in reports[1:]:
    assert report["checkpoint_sha256"] == reference["checkpoint_sha256"]
    assert report["source_sha256"] == reference["source_sha256"]
    assert report["seed"] == reference["seed"]
    for episode in range(cfg.episodes):
      left = reference["heights"][0]["samples"][episode]["initial_state"]
      right = report["heights"][0]["samples"][episode]["initial_state"]
      assert left.keys() == right.keys()
      for field, values in left.items():
        other = right[field]
        # actor最后两维是vx/wz，只排除对照变量vx，heading反馈wz仍需匹配。
        if field == "actor_observation":
          values, other = values[:-2] + values[-1:], other[:-2] + other[-1:]
        if len(values) != len(other) or not all(
          math.isclose(a, b, rel_tol=0, abs_tol=1e-6)
          for a, b in zip(values, other, strict=True)
        ):
          raise ValueError(
            f"Pairing mismatch: speed={report['fixed_forward_speed_m_s']} "
            f"episode={episode + 1} field={field}"
          )

  results = []
  for report in reports:
    height = report["heights"][0]
    samples = height["samples"]
    item = {
      "speed_m_s": report["fixed_forward_speed_m_s"],
      "n": height["n"],
      "passed": height["passed"],
      "pass_rate": height["pass_rate"],
      "complete": sum(s["complete"] for s in samples if s["completed"]),
      "stalled": sum(s["stalled"] for s in samples if s["completed"]),
      "fell_over": sum(s["fell_over"] for s in samples if s["completed"]),
      "failed": sum(s["failed"] for s in samples if s["completed"]),
      "mean_speed_rmse": sum(s["speed_rmse"] for s in samples) / cfg.episodes,
    }
    results.append(item)
    print(json.dumps(item), flush=True)
  summary = {
    "mode": cfg.mode,
    "pairing_verified": True,
    "checkpoint": reference["checkpoint"],
    "checkpoint_sha256": reference["checkpoint_sha256"],
    "input_report_sha256": {
      str(p.relative_to(cfg.directory)): hashlib.sha256(p.read_bytes()).hexdigest()
      for p in paths
    },
    "results": results,
  }
  with (cfg.directory / "paired_summary.json").open("x") as stream:
    json.dump(summary, stream, indent=2)
    stream.write("\n")
  print(f"PASS: {cfg.episodes} paired resets, tolerance=1e-6", flush=True)
  return summary


if __name__ == "__main__":
  summarize(tyro.cli(SummaryConfig))
