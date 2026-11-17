"""把v1任务改造成独立的奖励v2候选，不改旧入口。

地形、机器人、观测和动作来自blind_pit_env_cfg.py，只替换/新增奖励及指标。
具体代价的计算在blind_pit_rewards.py，非轮接触计数复用curriculums模块。
这里集中设置权重和阈值；数值仍需训练验证，不能据配置直接断言动作已改善。
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import MetricsTermCfg, RewardTermCfg

from .blind_pit_curriculums import nonfoot_contact_count
from .blind_pit_env_cfg import lego2_blind_pit_env_cfg, lego2_blind_pit_eval_cfg
from .blind_pit_rewards import (
  base_clearance_below_minimum,
  base_clearance_eligible,
  minimum_base_clearance_l2,
  pitch_relaxed_orientation_l2,
)


def _with_reward_v2(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """在一份独立环境配置上换入v2的姿态、接触和最低净空代价。

  其余10项v1奖励继续保留；新增两个净空日志，帮助判断高度项是否真正启用。
  """
  # 只修改新工厂返回的独立配置；旧任务11项基线和旧模型回放保持可用。
  # v2是一组待训练验证的姿态/净空候选，不把用户因果假设记成已证实结论。
  cfg.rewards.pop("flat_orientation_l2")
  cfg.rewards["pitch_relaxed_orientation_l2"] = RewardTermCfg(
    func=pitch_relaxed_orientation_l2,
    weight=-0.5,  # 姿态代价权重；只在超出俯仰容许区间或出现侧倾时扣分。
    params={"free_pitch_deg": 25.0},  # 前后俯仰±25°内不增加俯仰代价。
  )
  # 计数的是非轮碰撞geom通道，接触力范数>1 N；不因倾倒而缩小惩罚。
  # 允许轮子离地/摆动，不加入强制四轮贴地或指定前后腿动作的奖励。
  cfg.rewards["nonfoot_contact"] = RewardTermCfg(
    func=nonfoot_contact_count,
    weight=-0.5,  # 每个有效非轮接触通道扣0.5速率。
  )
  clearance_params = {
    "sensor_name": "terrain_scan",  # 复用训练中已有扫描，不加到actor输入。
    "minimum_height": 0.34,  # 平整地面上期望的最低机身root净空，米。
    "max_ground_relief": 0.03,  # 扫描高差超过3厘米时关闭高度项，便于跨台沿。
  }
  cfg.rewards["minimum_base_clearance_l2"] = RewardTermCfg(
    func=minimum_base_clearance_l2,
    weight=-2.0,  # 原始缺口平方已归一化，最大扣2速率，再由管理器乘dt。
    params=clearance_params.copy(),
  )
  # 同时记录高度项可用率与低净空帧占比，避免把跨台沿时关闭高度项误读为改善。
  cfg.metrics["base_clearance_eligible"] = MetricsTermCfg(
    func=base_clearance_eligible,
    params={"sensor_name": "terrain_scan", "max_ground_relief": 0.03},
  )
  cfg.metrics["base_clearance_below_minimum"] = MetricsTermCfg(
    func=base_clearance_below_minimum, params=clearance_params.copy()
  )
  return cfg


def lego2_blind_pit_reward_v2_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """创建v2训练/普通回放配置，地形与观测仍复用v1。"""
  # 保留原10级高度课程、出生/命令、传感器、观测、动作及终止/通过判据。
  return _with_reward_v2(lego2_blind_pit_env_cfg(play=play))


def lego2_blind_pit_reward_v2_eval_cfg(
  height: float, forward_speed: float | None = None
) -> ManagerBasedRlEnvCfg:
  """创建真实固定坑深的v2评估配置，供父/新策略在同条件下比较。"""
  # 固定高度工厂仍生成真实指定坑深；不把关闭课程等同于固定30 cm。
  return _with_reward_v2(lego2_blind_pit_eval_cfg(height, forward_speed))
