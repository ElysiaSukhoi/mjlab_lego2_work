"""记录出坑过程，帮助区分没有爬上与爬上后不稳定。

借用blind_pit_curriculums.py的四轮支撑检查，由环境配置登记到指标管理器。
训练日志root_out/front_up/rear_up/complete是满足条件的帧占比，
stable_pass记录一回合是否曾连续稳定支撑0.5秒，不是同一种统计。
达到后即使后来翻倒，成功标记也保留；还需同时查看fell_over/failed。
评估脚本也用PitProgress记录阶段；这里的通过判断不会直接增加奖励。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.terrains import BoxInvertedPyramidStairsTerrainCfg

from .blind_pit_curriculums import _PitWheels, rim_supported

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers import MetricsTermCfg


def pit_stages(
  centers: torch.Tensor,
  extents: torch.Tensor,
  origins: torch.Tensor,
  rim_z: torch.Tensor,
  contact,
  root: torch.Tensor,
  gravity: torch.Tensor,
  front: list[int],
  rear: list[int],
) -> dict[str, torch.Tensor]:
  """把当前位置/支撑分成机身出坑、前轮上台、后轮上台和四轮完成几个阶段。

  complete要求四轮有效支撑；stable_now还要求机身倾角小于45°。
  """
  # E个环境、W=4个轮；坐标均为世界坐标，距离单位m。
  # 前后轮按传感器名称解析，不假定传感器和动作使用相同排列。
  args = (origins[:, :2], rim_z)

  def supported(ids: list[int]) -> torch.Tensor:
    return rim_supported(
      centers[:, ids],
      extents[:, ids],
      *args,
      contact.found[:, ids],
      contact.force[:, ids],
      contact.normal[:, ids],
      contact.pos[:, ids],
    )

  relative = (root[:, :2] - origins[:, :2]).abs()
  # root离开坑的XY投影只算阶段，不代表已获得坑沿支撑。
  root_out = (relative > 1.5).any(-1) & (relative < 4.0).all(-1)
  front_up, rear_up = supported(front), supported(rear)
  complete = front_up & rear_up
  # projected_gravity_b的-z是机身竖直夹角余弦；严格小于45°才累计保持时长。
  upright = -gravity[:, 2] > math.cos(math.pi / 4)
  return {
    "root_out": root_out,
    "front_up": front_up,
    "rear_up": rear_up,
    "complete": complete,
    "stable_now": complete & upright,
  }


class PitProgress(_PitWheels):
  """为每个环境累计连续稳定支撑时间，并记住本回合是否曾通过。

  达到0.5秒后保持成功标记到reset；中间掉一次支撑就重新计时。
  """

  def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
    del cfg
    super().__init__(env)
    names = self.contact.primary_names
    self.front = [i for i, n in enumerate(names) if n.startswith(("LF_", "RF_"))]
    self.rear = [i for i, n in enumerate(names) if n.startswith(("LH_", "RH_"))]
    terrain = env.scene.terrain
    assert terrain is not None and terrain.cfg.terrain_generator is not None
    gen = terrain.cfg.terrain_generator
    pit = gen.sub_terrains["pit"]
    assert isinstance(pit, BoxInvertedPyramidStairsTerrainCfg)
    self.depths = torch.linspace(
      *pit.step_height_range, gen.num_rows, device=env.device
    )
    self.hold = torch.zeros(env.num_envs, device=env.device)
    self.passed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.last_step = torch.full_like(self.hold, -1, dtype=torch.long)
    self.values: dict[str, torch.Tensor] = {}

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """清除指定环境的保持时间和成功标记，其他环境继续保留自己的进度。"""
    ids = slice(None) if env_ids is None else env_ids
    # 只清被重置的行；其他环境的连续支撑时长和已达成事件必须保留。
    self.hold[ids] = 0
    self.passed[ids] = False
    self.last_step[ids] = -1
    for value in self.values.values():
      value[ids] = 0

  def __call__(
    self, env: ManagerBasedRlEnv, field: str = "stable_pass"
  ) -> torch.Tensor:
    """每控制步最多更新一次阶段/保持时间，重复读取不会重复加时间。"""
    # 同一控制步可被logger/评估器多次读取；不能每读一次就多累计一次dt。
    fresh = self.last_step != env.common_step_counter
    if fresh.any():
      terrain = env.scene.terrain
      assert terrain is not None
      centers, extents = self.wheels()
      origins = env.scene.env_origins
      rim_z = origins[:, 2] + self.depths[terrain.terrain_levels]
      robot = self.robot.data
      stages = pit_stages(
        centers,
        extents,
        origins,
        rim_z,
        self.contact.data,
        robot.root_link_pos_w,
        robot.projected_gravity_b,
        self.front,
        self.rear,
      )
      # 失败帧不能补齐最后一帧保持时间；中断支撑立即清零，已成功标志按episode锁存。
      stable = stages["stable_now"] & ~env.termination_manager.terminated
      next_hold = torch.where(stable, self.hold + env.step_dt, 0.0)
      self.hold[fresh] = next_hold[fresh]
      self.passed[fresh] |= self.hold[fresh] >= 0.5 - 1e-6
      stages["stable_pass"] = self.passed.clone()
      # step可能在inference_mode内执行，reset却在外部；跨步缓存不能直接保存推理张量。
      # 首次分配时临时关闭推理模式，之后仅写入普通buffer，确保局部/完整reset都能修改。
      for name, value in stages.items():
        if name not in self.values:
          with torch.inference_mode(False):
            self.values[name] = torch.zeros_like(self.hold)
        self.values[name][fresh] = value[fresh].float()
      self.last_step[fresh] = env.common_step_counter
    return self.values[field]
