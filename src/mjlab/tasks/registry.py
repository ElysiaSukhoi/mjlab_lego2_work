"""保存任务名到配置的对应关系，供train/play等入口查找。

机器人目录的__init__.py调用register_mjlab_task登记train/play、网络配置和runner。
load_env_cfg等函数返回副本，后续改一个实验的配置不会直接改登记模板。
本文件不构建地形、不定义奖励公式，也不启动训练。
"""

from copy import deepcopy
from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlBaseRunnerCfg


@dataclass
class _TaskCfg:
  env_cfg: ManagerBasedRlEnvCfg
  play_env_cfg: ManagerBasedRlEnvCfg
  rl_cfg: RslRlBaseRunnerCfg
  runner_cls: type | None


# Private module-level registry: task_id -> task config.
_REGISTRY: dict[str, _TaskCfg] = {}


def register_mjlab_task(
  task_id: str,
  env_cfg: ManagerBasedRlEnvCfg,
  play_env_cfg: ManagerBasedRlEnvCfg,
  rl_cfg: RslRlBaseRunnerCfg,
  runner_cls: type | None = None,
) -> None:
  """登记任务名与train/play环境、学习配置和训练器，已有同名任务会拒绝重复登记。"""
  if task_id in _REGISTRY:
    raise ValueError(f"Task '{task_id}' is already registered")
  _REGISTRY[task_id] = _TaskCfg(env_cfg, play_env_cfg, rl_cfg, runner_cls)


def list_tasks() -> list[str]:
  """List all registered task IDs."""
  return sorted(_REGISTRY.keys())


def load_env_cfg(task_name: str, play: bool = False) -> ManagerBasedRlEnvCfg:
  """返回环境配置的独立副本；play=True时选择回放模板。"""
  return deepcopy(
    _REGISTRY[task_name].env_cfg if not play else _REGISTRY[task_name].play_env_cfg
  )


def load_rl_cfg(task_name: str) -> RslRlBaseRunnerCfg:
  """返回任务网络/PPO配置的独立副本，供训练/回放入口使用。"""
  return deepcopy(_REGISTRY[task_name].rl_cfg)


def load_runner_cls(task_name: str) -> type | None:
  """返回该任务使用的训练器类，没有专用类时由调用者选择通用实现。"""
  return _REGISTRY[task_name].runner_cls
