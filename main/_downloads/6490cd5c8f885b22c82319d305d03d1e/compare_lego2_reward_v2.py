"""比较父策略和v2策略的固定高度评估报告。

输入来自evaluate_lego2_blind_pit.py；先核对速度、源码和真实出生状态，再汇总表现。
重点看稳定通过、速度误差、出坑后非轮接触，以及有效扫描时机身过低的比例。
这里的比例按记录帧数计算，小样本结果供观察动作质量，不代表正式验收。
"""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import tyro


@dataclass
class CompareConfig:
  """父模型报告、新模型报告与新输出文件的位置。"""

  reference: Path
  candidate: Path
  output: Path


def compare(cfg: CompareConfig) -> dict:
  """先核对两份报告是否同条件、同真实初态，再比较通过率和动作质量。"""
  old, new = (json.loads(p.read_text()) for p in (cfg.reference, cfg.candidate))
  assert old["mode"] == new["mode"] == "policy"
  for key in (
    "reward_version",
    "fixed_forward_speed_m_s",
    "paired_seed_base",
    "source_sha256",
  ):
    assert old[key] == new[key], key
  assert old["reward_version"] == "v2"
  results = []
  for a, b in zip(old["heights"], new["heights"], strict=True):
    assert a["height_m"] == b["height_m"]
    assert a["n"] == a["requested_n"] == b["n"] == b["requested_n"]
    assert len(a["samples"]) == len(b["samples"]) == a["n"]
    for index, (x, y) in enumerate(zip(a["samples"], b["samples"], strict=True)):
      assert x["completed"] and y["completed"]
      assert x["reset_seed"] == y["reset_seed"] == old["paired_seed_base"] + index
      assert x["initial_state"].keys() == y["initial_state"].keys()
      for field, left in x["initial_state"].items():
        right = y["initial_state"][field]
        if len(left) != len(right) or not all(
          math.isclose(u, v, rel_tol=0, abs_tol=1e-6)
          for u, v in zip(left, right, strict=True)
        ):
          raise ValueError(
            f"Reset pairing mismatch: seed={x['reset_seed']} field={field}"
          )

    def summary(height):
      """把完整回合汇总；净空/接触比例按帧计，无适用数据时返回空值。"""
      samples = height["samples"]
      totals = {
        k: sum(s["quality"][k] for s in samples)
        for k in samples[0]["quality"]
        if k != "post_complete_min_root_rim_m"
      }
      values = [
        s["quality"]["post_complete_min_root_rim_m"]
        for s in samples
        if s["quality"]["post_complete_min_root_rim_m"] is not None
      ]
      return {
        "n": height["n"],
        "passed": height["passed"],
        "complete": sum(s["complete"] for s in samples),
        "failed": sum(s["failed"] for s in samples),
        "mean_speed_rmse": sum(s["speed_rmse"] for s in samples) / len(samples),
        "clearance_eligible_fraction": totals["eligible_frames"]
        / sum(s["steps"] for s in samples),
        "low_fraction_of_eligible_frames": totals["low_clearance_frames"]
        / totals["eligible_frames"]
        if totals["eligible_frames"]
        else None,
        "nonfoot_fraction_after_first_complete": totals["post_complete_nonfoot_frames"]
        / totals["post_complete_frames"]
        if totals["post_complete_frames"]
        else None,
        "min_root_rim_m_after_first_complete": min(values) if values else None,
      }

    results.append(
      {
        "height_m": a["height_m"],
        "speed_m_s": old["fixed_forward_speed_m_s"],
        "reference": summary(a),
        "candidate": summary(b),
      }
    )
  result = {
    "paired_initial_states_verified": True,
    "limits": "Small paired screen; frame-weighted quality fractions, not formal acceptance",
    "input_report_sha256": {
      str(p): hashlib.sha256(p.read_bytes()).hexdigest()
      for p in (cfg.reference, cfg.candidate)
    },
    "reference_checkpoint_sha256": old["checkpoint_sha256"],
    "candidate_checkpoint_sha256": new["checkpoint_sha256"],
    "results": results,
  }
  with cfg.output.open("x") as stream:
    json.dump(result, stream, indent=2)
    stream.write("\n")
  print(json.dumps(result, indent=2), flush=True)
  return result


if __name__ == "__main__":
  compare(tyro.cli(CompareConfig))
