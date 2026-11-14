"""Pit traversal curriculum and tile boundary checks, privileged training only."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.entity import Entity
from mjlab.sensor import ContactSensor
from mjlab.terrains import BoxInvertedPyramidStairsTerrainCfg
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers import CurriculumTermCfg, TerminationTermCfg


# 本模块使用仿真中的几何/接触真值，供课程、截断和诊断使用，不送入actor。
# 下文E表示环境数，W表示轮子数（本任务为4）；长度单位m，力单位N。
def cylinder_extents(axis: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
  """World XYZ support distances for cylinders, including tilted wheel axles."""
  # axis形状(E,W,3)，是圆柱轴在世界坐标中的单位方向；sizes为(W,2)：半径r、半宽h。
  # 某坐标轴上的最大伸出距离为r*sqrt(1-axis_i²)+h*|axis_i|，倾斜时也适用。
  # clamp_min只防止浮点误差使根号内略小于0，避免由此产生NaN。
  return sizes[:, 0:1] * (1 - axis.square()).clamp_min(0).sqrt() + (
    sizes[:, 1:2] * axis.abs()
  )


def rim_supported(
  centers: torch.Tensor,
  extents: torch.Tensor,
  origin_xy: torch.Tensor,
  rim_z: torch.Tensor,
  found: torch.Tensor,
  force: torch.Tensor,
  normal: torch.Tensor,
  contact_pos: torch.Tensor,
  pit_half_width: float = 1.5,
  tile_half_width: float = 5.0,
) -> torch.Tensor:
  """All wheels clear the pit and support on the assigned tile's upper rim.

  Sensor normals point primary wheel -> secondary terrain, hence negative Z
  for support from below. Wall contact or unsupported geometric height fails.
  """
  # centers/extents均为(E,W,3)，使用轮碰撞圆柱中心，而非关节原点或机身位置。
  # 先转成相对本地块中心的XY距离，避免把相邻难度地块上的支撑也当成本轮出坑。
  xy = (centers[..., :2] - origin_xy[:, None]).abs()
  # 方形坑外：至少沿X或Y一轴，整个轮子已跨出台沿，另留2 cm余量。
  # .any(-1)归约的是两个水平坐标轴，此时仍为每环境、每轮一个判定。
  outside_pit = (xy - extents[..., :2] > pit_half_width + 0.02).any(dim=-1)
  # 本地块内：轮子的两个水平外缘均不能越界，距地块边缘保留5 cm。
  inside_tile = (xy + extents[..., :2] < tile_half_width - 0.05).all(dim=-1)
  # 轮心高度还包含轮子本身的尺寸；减去世界Z方向伸出距离才得到真实轮底。
  bottom = centers[..., 2] - extents[..., 2]
  # 轮底和实际接触点都应靠近坑沿高度，分别允许2.5 cm、2 cm接触/数值容差。
  on_top = (bottom - rim_z[:, None]).abs() < 0.025
  contact_on_top = (contact_pos[..., 2] - rim_z[:, None]).abs() < 0.02
  # 法向按“轮子→地形”定义，地面在轮下方时z为负；还必须有>1 N的竖直力幅值。
  # 因此单纯侧壁阻挡、悬空或仅轮心升高都不会被判为坑沿支撑。
  support = (found > 0) & (normal[..., 2] < -0.5) & (force[..., 2].abs() > 1.0)
  # 最后一维归约4个轮子，返回(E,)；必须四轮同时满足所有条件。
  return (outside_pit & inside_tile & on_top & contact_on_top & support).all(dim=1)


def level_changes(
  distance: torch.Tensor,
  expected_distance: torch.Tensor,
  supported: torch.Tensor,
  failed: torch.Tensor,
  eligible: torch.Tensor,
  promote_distance: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Per-episode decisions, never rewards or batch-mean success estimates."""
  # 所有输入按环境独立排列；eligible表示该行确有一个可结算的完整episode。
  finite = torch.isfinite(distance) & torch.isfinite(expected_distance)
  # 晋级要求距离、四轮支撑、正常状态同时成立，不能只凭走远或到达边界判成功。
  up = eligible & finite & ~failed & supported & (distance > promote_distance)
  # 先判晋级再判降级：失败/数值异常或低于半期望行程时降级，且两种操作互斥。
  down = eligible & ~up & (failed | ~finite | (distance < expected_distance * 0.5))
  return up, down


def nonfoot_contact_count(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Count terrain-contacting non-foot geom channels, even when tipped over."""
  sensor: ContactSensor = env.scene["undesired_ground_contact"]
  data = sensor.data
  assert data.found is not None and data.force is not None
  # 每个非足部碰撞几何通道计0/1，并非统计独立link数或所有物理接触点数量。
  # 不乘直立姿态系数，保证翻倒后仍能记录碰撞；配置中只把它作为episode均值指标。
  contact = (data.found > 0) & (torch.linalg.vector_norm(data.force, dim=-1) > 1.0)
  return contact.sum(dim=-1).float()


class _PitWheels:
  def __init__(self, env: ManagerBasedRlEnv):
    self.robot: Entity = env.scene["robot"]
    self.contact: ContactSensor = env.scene["pit_rim_contact"]
    names = self.contact.primary_names
    # 验证恰好是四个轮子，再按传感器实际顺序查几何，不能猜前后腿的数组下标。
    if set(names) != {f"{leg}_foot_collision" for leg in ("RF", "LF", "RH", "LH")}:
      raise ValueError(f"Unexpected pit wheel contact names: {names}")
    ids, _ = self.robot.find_geoms(tuple(names), preserve_order=True)
    # geom_ids是robot内部索引，用于读取EntityData；读取MjModel则要转换为全局索引。
    self.geom_ids = ids
    global_ids = self.robot.indexing.geom_ids[ids].cpu().numpy()
    model = env.sim.mj_model
    if not (model.geom_type[global_ids] == mujoco.mjtGeom.mjGEOM_CYLINDER).all():
      raise ValueError("Pit wheel clearance assumes cylindrical collision geoms")
    # MuJoCo圆柱geom_size的前两项是半径、半宽，不是直径、全宽。
    self.sizes = torch.tensor(
      model.geom_size[global_ids, :2], dtype=torch.float32, device=env.device
    )

  def wheels(self) -> tuple[torch.Tensor, torch.Tensor]:
    # pose为(E,W,7)：世界坐标xyz和wxyz四元数，顺序与接触数据一致。
    pose = self.robot.data.geom_pose_w[:, self.geom_ids]
    local_axis = torch.zeros_like(pose[..., :3])
    local_axis[..., 2] = 1
    # MuJoCo圆柱几何的轴是局部Z；几何姿态已包含轮轴旋转，不能直接套关节Y轴。
    axis = quat_apply(pose[..., 3:7], local_axis)
    return pose[..., :3], cylinder_extents(axis, self.sizes)


class PitTerrainLevels(_PitWheels):
  """Run before reset, without depending on the later-created metrics manager."""

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    del cfg
    super().__init__(env)
    terrain = env.scene.terrain
    assert terrain is not None and terrain.cfg.terrain_generator is not None
    generator = terrain.cfg.terrain_generator
    pit = generator.sub_terrains["pit"]
    assert isinstance(pit, BoxInvertedPyramidStairsTerrainCfg)
    low, high = pit.step_height_range
    # 与地形按row/(rows-1)分级一致，包含最浅/最深两个端点。
    # 此时metrics manager尚未创建，构造器只缓存几何和深度，不读取其状态。
    self.heights = torch.linspace(low, high, generator.num_rows, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice,
    command_name: str = "twist",
    promote_distance: float = 3.0,
  ) -> dict[str, torch.Tensor]:
    terrain = env.scene.terrain
    assert terrain is not None and terrain.env_origins is not None
    # 只更新本次真正重置的env_ids；其他环境即使正在行走，也不能被连带升降级。
    ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    if len(ids) == 0:
      return {"mean": terrain.terrain_levels.float().mean()}
    centers, extents = self.wheels()
    origins = env.scene.env_origins
    # origin的Z已经是坑底，加上该行坑深才是世界坐标中的坑沿高度。
    # 所有结算都在reset清空机器人/传感器状态及移动origin之前完成。
    rim_z = origins[:, 2] + self.heights[terrain.terrain_levels]
    contact = self.contact.data
    assert contact.found is not None and contact.force is not None
    assert contact.normal is not None and contact.pos is not None
    supported = rim_supported(
      centers,
      extents,
      origins[:, :2],
      rim_z,
      contact.found,
      contact.force,
      contact.normal,
      contact.pos,
    )
    root = self.robot.data.root_link_pos_w
    # 用root距坑中心的水平位移，不是累计轨迹长度；原地绕圈不会累加出晋级距离。
    distance = torch.linalg.vector_norm(root[:, :2] - origins[:, :2], dim=-1)
    command = env.command_manager.get_command(command_name)
    assert command is not None
    # 本轮vx固定，|vx|×实际步数×控制周期给出期望行程，不强行按完整20 s计算短episode。
    expected = command[:, 0].abs() * env.episode_length_buf * env.step_dt
    finite = (
      torch.isfinite(self.robot.data.root_link_pose_w).all(dim=-1)
      & torch.isfinite(self.robot.data.root_link_vel_w).all(dim=-1)
      & torch.isfinite(self.robot.data.joint_pos).all(dim=-1)
      & torch.isfinite(self.robot.data.joint_vel).all(dim=-1)
    )
    # terminated代表翻倒等真正失败；超时/边界truncate不自动算失败，也不自动算成功。
    failed = env.termination_manager.terminated | ~finite
    # 没结束的episode不结算：中途手动reset冻结等级，初次reset及过短episode也跳过。
    # 若auto_reset=False，已结束episode由调用者显式reset时仍可结算；play在配置层关课程。
    eligible = env.termination_manager.dones & (env.episode_length_buf > 2)
    if env.common_step_counter == 0:
      eligible = torch.zeros_like(eligible)
    up, down = level_changes(
      distance[ids],
      expected[ids],
      supported[ids],
      failed[ids],
      eligible[ids],
      promote_distance,
    )
    # 复用现有更新器：最低级不再降，最高级完成后随机重访已有等级。
    # 此处只更新等级/origin；随后环境reset事件才把机器人出生到新的坑底。
    terrain.update_env_origins(ids, up, down)
    # 返回值仅用于课程日志；不能拿这些批统计量反过来替单个环境决定晋级或当成功率。
    return {
      "mean": terrain.terrain_levels.float().mean(),
      "max": terrain.terrain_levels.float().max(),
      "promoted": up.float().mean(),
      "demoted": down.float().mean(),
    }


class PitTileBoundary(_PitWheels):
  """Truncate before root or a wheel reaches neighboring tiles; not success."""

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
    del cfg
    super().__init__(env)

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    centers, extents = self.wheels()
    origin_xy = env.scene.env_origins[:, :2]
    # 检查轮子的最外缘（中心距+半包络），而非只看轮心；任一轴/任一轮触线即截断。
    wheel_edge = (
      ((centers[..., :2] - origin_xy[:, None]).abs() + extents[..., :2] >= 4.9)
      .any(dim=-1)
      .any(dim=-1)
    )
    # 方形地块用X/Y最大绝对位移：root达到4 m或轮外缘达到4.9 m时留出安全余量。
    root_edge = (self.robot.data.root_link_pos_w[:, :2] - origin_xy).abs().amax(-1) >= 4
    # 前两步跳过，避免reset初期旧派生状态误触发；配置把此布尔量归入time_out截断。
    return (wheel_edge | root_edge) & (env.episode_length_buf > 2)
