"""Lego2任务入口：把任务名称和所需配置连起来。

train/play先在tasks/registry.py中按名字找到这里登记的配置。
blind_pit_env_cfg.py负责当前中央坑v1，blind_pit_reward_v2_env_cfg.py负责v2；
两者共用rl_cfg.py中的网络/PPO设置，使用独立日志目录保存结果。
platform_climb_env_cfg.py是旧固定高台方案，与当前Blind-Pit分开保留。
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .blind_pit_env_cfg import lego2_blind_pit_env_cfg
from .blind_pit_reward_v2_env_cfg import lego2_blind_pit_reward_v2_env_cfg
from .env_cfgs import (
  lego2_flat_env_cfg,
  lego2_rough_env_cfg,
)
from .platform_climb_env_cfg import lego2_platform_climb_env_cfg
from .rl_cfg import lego2_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Lego2",
  env_cfg=lego2_rough_env_cfg(),
  play_env_cfg=lego2_rough_env_cfg(play=True),
  rl_cfg=lego2_ppo_runner_cfg(experiment_name="lego2_velocity/rough"),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Lego2",
  env_cfg=lego2_flat_env_cfg(),
  play_env_cfg=lego2_flat_env_cfg(play=True),
  rl_cfg=lego2_ppo_runner_cfg(experiment_name="lego2_velocity/flat"),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Platform-Climb-Lego2",
  env_cfg=lego2_platform_climb_env_cfg(),
  play_env_cfg=lego2_platform_climb_env_cfg(play=True),
  rl_cfg=lego2_ppo_runner_cfg(experiment_name="lego2_platform_climb"),
  runner_cls=VelocityOnPolicyRunner,
)

# 阶段1新增独立入口，保留上方Rough/Flat/Platform-Climb三个旧任务的注册与配置。
# 注册使list-envs/train/play能按任务名查找配置，不代表后续阶段已经具备开训条件。
register_mjlab_task(
  task_id="Mjlab-Blind-Pit-Lego2",
  # 工厂分别创建train/play两份对象，避免两种模式共用可变配置。
  env_cfg=lego2_blind_pit_env_cfg(),
  play_env_cfg=lego2_blind_pit_env_cfg(play=True),
  # 复用原PPO参数，仅用独立实验名隔离新日志；144维新输入不直接续训旧模型。
  rl_cfg=lego2_ppo_runner_cfg(experiment_name="lego2_blind_pit"),
  runner_cls=VelocityOnPolicyRunner,
)

# v2奖励候选使用独立任务/日志目录，原v1入口继续供旧模型复评和对照。
# 仅注册配置，不自动启动训练；修改奖励后需新run，不能把旧模型回放当作v2效果。
register_mjlab_task(
  task_id="Mjlab-Blind-Pit-Lego2-RewardV2",
  env_cfg=lego2_blind_pit_reward_v2_env_cfg(),
  play_env_cfg=lego2_blind_pit_reward_v2_env_cfg(play=True),
  rl_cfg=lego2_ppo_runner_cfg(experiment_name="lego2_blind_pit_reward_v2"),
  runner_cls=VelocityOnPolicyRunner,
)
