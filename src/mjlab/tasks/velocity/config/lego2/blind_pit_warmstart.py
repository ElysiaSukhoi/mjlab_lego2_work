"""让v2从父模型已有动作能力开始，同时核对迁移范围。

train_lego2_blind_pit_reward_v2.py先建立全新的环境和训练器，再调用这里。
只复制actor的网络、输入归一化和探索分布，保留新critic与新优化器。
迭代、环境计数和课程也保持新状态；这与恢复旧训练全部状态不同。
"""

import hashlib
from pathlib import Path

import torch

from mjlab.rl import MjlabOnPolicyRunner


def state_digest(state: dict[str, torch.Tensor]) -> str:
  """给整份网络状态计算校验值，包含参数及归一化等随模型保存的数据。"""
  # 同时核对参数和buffer（包括观测归一化、动作分布），不能只比较MLP权重。
  digest = hashlib.sha256()
  for name, value in sorted(state.items()):
    tensor = value.detach().cpu().contiguous()
    digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
    digest.update(tensor.numpy().tobytes())
  return digest.hexdigest()


def transfer_actor(
  runner: MjlabOnPolicyRunner, checkpoint: Path, expected_sha256: str
) -> dict:
  """把父actor装入全新训练器，并证明其余训练状态没有被恢复。

  先核对父文件内容，再对比加载前后的actor、critic和优化器；返回审计记录供保存。
  """
  raw = checkpoint.read_bytes()
  actual_sha256 = hashlib.sha256(raw).hexdigest()
  if actual_sha256 != expected_sha256:
    raise ValueError("Parent checkpoint SHA256 mismatch")
  # 只允许向全新的runner迁移，防止把已训练的优化器/计数误当成新实验。
  assert runner.current_learning_iteration == 0
  assert runner.env.unwrapped.common_step_counter == 0
  before = runner.alg.save()
  assert not before["optimizer_state_dict"]["state"]
  critic_digest = state_digest(before["critic_state_dict"])
  parent = torch.load(checkpoint, map_location=runner.device, weights_only=False)
  actor = parent["actor_state_dict"]
  assert actor["mlp.0.weight"].shape[1] == 144
  assert any(k.startswith("obs_normalizer.") for k in actor)
  # 直接使用算法的选择性加载，避免共享runner.load额外恢复旧env计数。
  # critic/优化器/迭代/RND均不载入；课程和仿真状态来自新环境，仍从3 cm开始。
  restore_iteration = runner.alg.load(
    {"actor_state_dict": actor},
    {
      "actor": True,
      "critic": False,
      "optimizer": False,
      "iteration": False,
      "rnd": False,
    },
    strict=True,
  )
  after = runner.alg.save()
  assert not restore_iteration
  assert state_digest(after["actor_state_dict"]) == state_digest(actor)
  assert state_digest(after["critic_state_dict"]) == critic_digest
  assert after["optimizer_state_dict"] == before["optimizer_state_dict"]
  assert runner.current_learning_iteration == 0
  assert runner.env.unwrapped.common_step_counter == 0
  return {
    "parent_checkpoint": str(checkpoint.resolve()),
    "parent_sha256": actual_sha256,
    "parent_iteration": parent.get("iter"),
    "actor_including_normalizer_and_distribution_sha256": state_digest(actor),
    "critic_fresh_unchanged_sha256": critic_digest,
    "optimizer_fresh": True,
    "iteration": 0,
    "env_step_counter": 0,
    "transfer_type": "actor-only reward-v2 fine-tuning, not baseline resume",
  }
