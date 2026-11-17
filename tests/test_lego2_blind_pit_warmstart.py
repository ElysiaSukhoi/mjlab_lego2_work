"""Real runner transfer boundaries and one-update synthetic training smoke."""

import hashlib
import json
import runpy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import torch
from conftest import get_test_device

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg
from mjlab.tasks.velocity.config.lego2.blind_pit_reward_v2_env_cfg import (
  lego2_blind_pit_reward_v2_eval_cfg,
)
from mjlab.tasks.velocity.config.lego2.blind_pit_warmstart import (
  state_digest,
  transfer_actor,
)


@pytest.fixture
def transfer_case(tmp_path):
  cfg = lego2_blind_pit_reward_v2_eval_cfg(0.03)
  cfg.scene.num_envs = 2
  cfg.seed = 7
  env = ManagerBasedRlEnv(cfg, device=get_test_device())
  try:
    agent = load_rl_cfg("Mjlab-Blind-Pit-Lego2-RewardV2")
    runner = MjlabOnPolicyRunner(
      RslRlVecEnvWrapper(env), asdict(agent), device=env.device
    )
    actor = {
      k: v.detach().clone() for k, v in runner.alg.save()["actor_state_dict"].items()
    }
    actor["mlp.0.weight"] += 0.02
    for key in actor:
      if key.startswith("obs_normalizer.") and "mean" in key:
        actor[key] += 0.1
    path = tmp_path / "synthetic_parent.pt"
    torch.save(
      {
        "actor_state_dict": actor,
        "critic_state_dict": {"do_not_load": torch.ones(1)},
        "optimizer_state_dict": {"do_not_load": True},
        "iter": 2197,
        "infos": {"env_state": {"common_step_counter": 999}},
      },
      path,
    )
    yield runner, path, hashlib.sha256(path.read_bytes()).hexdigest(), actor
  finally:
    env.close()


def test_actor_buffers_transfer_but_critic_optimizer_and_counters_do_not(transfer_case):
  runner, path, digest, actor = transfer_case
  critic = state_digest(runner.alg.save()["critic_state_dict"])
  with pytest.raises(ValueError, match="SHA256"):
    transfer_actor(runner, path, "wrong")
  audit = transfer_actor(runner, path, digest)
  assert state_digest(runner.alg.save()["actor_state_dict"]) == state_digest(actor)
  assert state_digest(runner.alg.save()["critic_state_dict"]) == critic
  assert not runner.alg.save()["optimizer_state_dict"]["state"]
  assert runner.current_learning_iteration == 0
  assert runner.env.unwrapped.common_step_counter == 0
  assert audit["parent_iteration"] == 2197


def test_synthetic_one_update_and_v2_policy_evaluation(transfer_case, tmp_path):
  _, parent, digest, actor = transfer_case
  root = Path(__file__).resolve().parents[1]
  tool = runpy.run_path(str(root / "scripts/tools/train_lego2_blind_pit_reward_v2.py"))
  output = tmp_path / "new_run"
  tool["run"](
    tool["FineTuneConfig"](
      checkpoint=parent,
      output=output,
      expected_sha256=digest,
      num_envs=2,
      iterations=1,
      device=get_test_device(),
    )
  )
  assert (output / "model_0.pt").is_file()
  initial = torch.load(
    output / "initial_transfer.pt", map_location="cpu", weights_only=False
  )
  assert state_digest(initial["actor_state_dict"]) == state_digest(actor)
  assert (
    initial["iter"] == 0 and initial["infos"]["env_state"]["common_step_counter"] == 0
  )
  assert not initial["optimizer_state_dict"]["state"]
  audit = json.loads((output / "warmstart.json").read_text())
  assert audit["terrain_initial_level"] == 0 and audit["planned_new_samples"] == 48
  with pytest.raises(FileExistsError):
    tool["run"](tool["FineTuneConfig"](checkpoint=parent, output=output))
  evaluation = runpy.run_path(str(root / "scripts/tools/evaluate_lego2_blind_pit.py"))
  report = evaluation["run"](
    evaluation["EvalConfig"](
      output=tmp_path / "eval",
      checkpoint=output / "model_0.pt",
      heights=(0.03,),
      episodes=1,
      max_steps=2,
      device=get_test_device(),
      reward_version="v2",
      forward_speed=0.5,
      paired_seed_base=11,
    )
  )
  assert report["reward_version"] == "v2"
  assert report["heights"][0]["n"] == 0
  sample = report["heights"][0]["samples"][0]
  assert sample["quality"]["eligible_frames"] == 2
  assert sample["reset_seed"] == 11


def test_quality_comparison_rejects_unpaired_resets(tmp_path):
  root = Path(__file__).resolve().parents[1]
  tool = runpy.run_path(str(root / "scripts/tools/compare_lego2_reward_v2.py"))
  sample = {
    "completed": True,
    "reset_seed": 11,
    "initial_state": {"root_pose_w": [0, 0, 0.47], "actor_observation": [0] * 144},
    "complete": True,
    "failed": False,
    "steps": 100,
    "speed_rmse": 0.1,
    "quality": {
      "eligible_frames": 80,
      "low_clearance_frames": 40,
      "nonfoot_frames": 20,
      "post_complete_frames": 50,
      "post_complete_nonfoot_frames": 10,
      "post_complete_low_clearance_frames": 20,
      "post_complete_min_root_rim_m": 0.23,
    },
  }
  report: dict[str, Any] = {
    "mode": "policy",
    "reward_version": "v2",
    "fixed_forward_speed_m_s": 0.5,
    "paired_seed_base": 11,
    "source_sha256": {"a": "same"},
    "checkpoint_sha256": "synthetic",
    "heights": [
      {"height_m": 0.3, "n": 1, "requested_n": 1, "passed": 1, "samples": [sample]}
    ],
  }
  old, new = tmp_path / "old.json", tmp_path / "new.json"
  for p in (old, new):
    p.write_text(json.dumps(report))
  cfg = tool["CompareConfig"](
    reference=old, candidate=new, output=tmp_path / "comparison.json"
  )
  result = tool["compare"](cfg)
  assert (
    result["results"][0]["candidate"]["nonfoot_fraction_after_first_complete"] == 0.2
  )
  assert result["results"][0]["candidate"]["low_fraction_of_eligible_frames"] == 0.5
  report["heights"][0]["samples"][0]["initial_state"]["root_pose_w"][0] += 0.01
  new.write_text(json.dumps(report))
  with pytest.raises(ValueError, match="pairing mismatch"):
    tool["compare"](cfg)
