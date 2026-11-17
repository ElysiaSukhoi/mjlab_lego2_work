"""v2独立微调入口：接续已有动作能力，重新学习新奖励下的回报。

读取RewardV2任务的环境/网络配置，用blind_pit_warmstart.py只迁移父模型actor。
保存新run的params、迁移起点和审计信息，再调用VelocityOnPolicyRunner学习。
这里设置试训预算；奖励本身在blind_pit_reward_v2_env_cfg.py中。
远程run_phase3_reward_v2_trial_20261116.sh负责资源检查和调用本脚本。
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.config.lego2.blind_pit_warmstart import transfer_actor
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends


@dataclass
class FineTuneConfig:
  """这一批微调的命令行参数；仅控制加载文件、输出目录、预算和设备。"""

  checkpoint: Path  # 只读父模型文件，提供已有动作能力。
  output: Path  # 必须是新目录；保存本次params、审计和checkpoint。
  expected_sha256: str = (
    "b8bc52a8ee370eb94e359bc71d310f5f1cd0ec013cae9e5982b44fb13b8a54a4"
  )
  num_envs: int = 512  # 同时采样的环境数，影响每批样本量和显存占用。
  iterations: int = 300  # 新增更新次数，编号从0开始；不是父模型累计迭代号。
  seed: int = 7  # 控制本次随机初始化和出生抽样，方便重复实验。
  device: str = "cuda:0"  # 使用的仿真和网络设备。
  validate_only: bool = False  # True只核对迁移并保存起点，不学习。


def run(cfg: FineTuneConfig) -> dict:
  """建立新v2环境和训练器，审计actor迁移，然后按预算学习。

  validate_only只保存/核对迁移起点；正式运行从新计数开始，旧checkpoint只读。
  """
  if cfg.num_envs < 2 or cfg.iterations < 1:
    raise ValueError("num_envs>=2 and iterations>=1 required")
  if not cfg.checkpoint.is_file():
    raise FileNotFoundError(cfg.checkpoint)
  if cfg.output.exists():
    raise FileExistsError(cfg.output)
  if cfg.device.startswith("cuda"):
    assert torch.cuda.is_available()
    free, _ = torch.cuda.mem_get_info(cfg.device)
    assert free >= (12 if cfg.num_envs >= 512 else 3) * 2**30, (
      "Insufficient free GPU memory"
    )
  cfg.output.mkdir(parents=True, exist_ok=False)
  configure_torch_backends()
  task = "Mjlab-Blind-Pit-Lego2-RewardV2"
  env_cfg, agent_cfg = load_env_cfg(task), load_rl_cfg(task)
  assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = agent_cfg.seed = cfg.seed
  agent_cfg.max_iterations = cfg.iterations
  agent_cfg.logger = "tensorboard"
  agent_cfg.upload_model = False  # 本次只在新run本地保存模型。
  agent_cfg.resume = False  # 旧训练状态不整体恢复，actor由专用函数单独迁移。
  agent_cfg.run_name = cfg.output.name
  dump_yaml(cfg.output / "params/env.yaml", asdict(env_cfg))
  dump_yaml(cfg.output / "params/agent.yaml", asdict(agent_cfg))
  (cfg.output / "invocation.json").write_text(
    json.dumps(asdict(cfg), default=str, indent=2) + "\n"
  )
  env = ManagerBasedRlEnv(env_cfg, device=cfg.device)
  try:
    wrapper = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = VelocityOnPolicyRunner(
      wrapper, asdict(agent_cfg), str(cfg.output), cfg.device
    )
    audit = transfer_actor(runner, cfg.checkpoint, cfg.expected_sha256)
    assert env.scene.terrain is not None
    assert not env.scene.terrain.terrain_levels.any()
    audit["terrain_initial_level"] = 0
    audit["planned_new_samples"] = (
      cfg.num_envs * agent_cfg.num_steps_per_env * cfg.iterations
    )
    (cfg.output / "warmstart.json").write_text(json.dumps(audit, indent=2) + "\n")
    # 保存迁移起点，用于核对父策略动作；它不是一次PPO更新或训练效果。
    runner.save(str(cfg.output / "initial_transfer.pt"), infos={"warmstart": audit})
    print(json.dumps(audit, indent=2), flush=True)
    if not cfg.validate_only:
      runner.learn(cfg.iterations, init_at_random_ep_len=True)
    return audit
  finally:
    env.close()
    if cfg.device.startswith("cuda"):
      torch.cuda.empty_cache()


if __name__ == "__main__":
  run(tyro.cli(FineTuneConfig))
