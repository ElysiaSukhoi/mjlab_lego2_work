"""Lego2各任务共用的网络结构和PPO学习参数。

任务注册文件__init__.py调用此工厂，并用experiment_name区分日志目录。
actor负责选动作，critic在训练时估计后续回报；这里设置网络宽度和学习方法。
观测内容、奖励权重和地形在各env_cfg文件中，不在这里。
以下是默认值，训练入口可覆盖环境数、迭代预算、日志等；实际以run的params为准。
"""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def lego2_ppo_runner_cfg(
  *,
  experiment_name: str = "lego2_velocity",
) -> RslRlOnPolicyRunnerCfg:
  """创建Lego2共用的actor、critic和PPO配置。

  actor把观测变成动作；critic估计未来回报以帮助训练。两者各用三层网络。
  experiment_name只区分日志目录；环境数、短训轮数等可在训练入口覆盖。
  """
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),  # 三层隐藏层宽度；Blind-Pit输入另由观测决定为144维。
      activation="elu",  # 让网络能表达更复杂的动作关系。
      obs_normalization=True,  # 用运行中统计的均值/方差调整不同输入的量级。
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,  # 新模型初始探索幅度；迁移时继承父actor中已学习的值。
        "std_type": "scalar",  # 用标准差本身表示探索大小。
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),  # critic也用三层；Blind-Pit的输入为254维。
      activation="elu",
      obs_normalization=True,  # critic单独维护输入统计，不与actor混用。
    ),
    # 学习日志：Loss/value看回报预测误差，Loss/surrogate是动作更新的优化量。
    # Policy/mean_std看探索幅度，Loss/learning_rate看当前实际学习率。
    # 它们不能代替动作质量，要同时看非轮接触、低净空和稳定通过。
    # 一次学习迭代先采样，再按下方5轮×4份数据更新；并非只改一次网络参数。
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,  # critic预测误差在训练损失中的权重。
      use_clipped_value_loss=True,  # 抑制critic每次更新变化过大。
      clip_param=0.2,  # PPO裁剪阈值，减少策略一次大幅改变的倾向。
      entropy_coef=0.01,  # 鼓励保留一定探索，避免过早只尝试一种动作。
      num_learning_epochs=5,  # 每批采到的数据重复学习5轮。
      num_mini_batches=4,  # 每轮把这批数据分成4份更新。
      learning_rate=1.0e-3,  # 新优化器的初始步幅；不是父训练末尾的动态值。
      schedule="adaptive",  # 根据策略变化程度调整学习率。
      gamma=0.99,  # 未来奖励的折扣，越接近1越重视较远的后果。
      lam=0.95,  # 综合长短期回报估计，平衡估计抖动与偏差。
      desired_kl=0.01,  # 策略变化程度的参考值，用于调学习率。
      max_grad_norm=1.0,  # 限制过大的梯度，减轻单次异常更新。
    ),
    experiment_name=experiment_name,  # 日志一级目录，隔离v1/v2等实验。
    save_interval=50,  # 每50次更新保存一次；训练结束还会保存最终模型。
    num_steps_per_env=24,  # 每个环境先走24个控制步，再进行一批学习。
    max_iterations=10_000,  # 通用预算；本批独立v2入口默认覆盖为300。
  )
