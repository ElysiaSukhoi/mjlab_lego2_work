from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .velocity_command import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> dict[str, torch.Tensor]:
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None

  # Compute the distance the robot walked.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
    dim=1,
  )

  # Robots that walked far enough progress to harder terrains.
  move_up = distance > terrain_generator.size[0] / 2

  # Robots that walked less than half of their required distance go to
  # simpler terrains.
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  )
  move_down *= ~move_up

  # On the initial reset (before any env step) the robot is still at its spawn
  # pose rather than a walked-to position, so ``distance`` is meaningless and
  # would spuriously promote every env from level 0 to 1, ignoring
  # ``max_init_terrain_level``. Freeze levels on that first reset.
  if env.common_step_counter == 0:
    move_up = torch.zeros_like(move_up)
    move_down = torch.zeros_like(move_down)

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  # Compute per-terrain-type mean levels.
  levels = terrain.terrain_levels.float()
  result: dict[str, torch.Tensor] = {
    "mean": torch.mean(levels),
    "max": torch.max(levels),
  }

  # In curriculum mode num_cols == num_terrains (one column per type),
  # so the column index directly maps to the sub-terrain name.
  sub_terrain_names = list(terrain_generator.sub_terrains.keys())
  terrain_origins = terrain.terrain_origins
  assert terrain_origins is not None
  num_cols = terrain_origins.shape[1]
  if num_cols == len(sub_terrain_names):
    types = terrain.terrain_types
    for i, name in enumerate(sub_terrain_names):
      mask = types == i
      if mask.any():
        result[name] = torch.mean(levels[mask])

  return result


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  for stage in velocity_stages:
    if env.common_step_counter >= stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
  return {
    "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
  }


def ang_vel_z_rel_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
  cmd_threshold: float = 0.1,
) -> torch.Tensor:
  """Per-env relative yaw-rate tracking error ``|cmd - act| / max(|cmd|, eps)``."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  cmd = command[:, 2]
  err = torch.abs(cmd - asset.data.root_link_ang_vel_b[:, 2])
  return err / torch.clamp(torch.abs(cmd), min=cmd_threshold)


class foot_friction_slide_curriculum:
  """Raise slide friction only after sustained hard-env yaw tracking.

  Hard friction climbs through ``friction_levels``. At level ``i`` (hard =
  ``levels[i]``), the paired event samples:

  - easy (``1 - hard_ratio``): ``U(levels[0], levels[i-1])``
  - hard (``hard_ratio``): fixed at ``levels[i]`` (point mass)

  Each reset contributes gated hard-env ``track_ang`` samples into a running
  accumulator. Only when the accumulator reaches ``min_gated_envs`` is one EMA
  update performed (then the buffer is cleared). This matches how resets arrive
  a few envs at a time, while still requiring a statistically meaningful pool
  before trusting the metric.

  Command gating (at reset, last segment before command resample):

  - Always require ``|cmd_z| >= cmd_gate`` (nonzero yaw command).
  - If ``require_spin_inplace``: keep only spin-in-place samples. Prefer the
    command term's ``is_spin_env`` flag (from ``rel_spin_envs``); if absent,
    fall back to ``||cmd_xy|| <= lin_cmd_gate``.

  Promote when cooldown elapsed, EMA >= ``track_ang_threshold``, optional both
  +ω/−ω EMAs look good, and ``ok_streak`` of successful flushes is long enough.
  Demote when EMA stays below ``demote_threshold`` after cooldown.
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    event_name: str = cfg.params.get("event_name", "foot_friction_slide")
    self._event_cfg = env.event_manager.get_term_cfg(event_name)
    self._command_name: str = cfg.params.get("command_name", "twist")
    self._reward_name: str = cfg.params.get("reward_name", "track_ang_vel_z_exp")
    self._friction_levels: list[float] = list(
      cfg.params.get("friction_levels", [0.4, 0.7, 1.0, 1.3, 1.6, 2.0])
    )
    self._track_ang_threshold: float = float(
      cfg.params.get("track_ang_threshold", 1.7)
    )
    self._demote_threshold: float = float(
      cfg.params.get("demote_threshold", 1.4)
    )
    self._hard_ratio: float = float(cfg.params.get("hard_ratio", 0.6))
    self._min_promote_steps: int = int(cfg.params.get("min_promote_steps", 2000 * 24))
    self._min_demote_steps: int = int(
      cfg.params.get("min_demote_steps", self._min_promote_steps)
    )
    self._cmd_gate: float = float(cfg.params.get("cmd_gate", 0.3))
    self._require_spin_inplace: bool = bool(
      cfg.params.get("require_spin_inplace", False)
    )
    self._lin_cmd_gate: float = float(cfg.params.get("lin_cmd_gate", 0.1))
    self._ema_alpha: float = float(cfg.params.get("ema_alpha", 0.05))
    self._min_gated_envs: int = int(cfg.params.get("min_gated_envs", 64))
    self._require_both_yaw: bool = bool(cfg.params.get("require_both_yaw", True))
    self._min_side_envs: int = int(cfg.params.get("min_side_envs", 16))
    self._promote_ok_needed: int = int(cfg.params.get("promote_ok_needed", 50))
    start_level = int(cfg.params.get("start_level", 0))
    self._level = max(0, min(start_level, len(self._friction_levels) - 1))
    self._last_level_change_step = 0
    self._ema: float | None = None
    self._ema_pos: float | None = None
    self._ema_neg: float | None = None
    self._ok_streak = 0
    self._acc_sum = 0.0
    self._acc_n = 0
    self._acc_pos_sum = 0.0
    self._acc_pos_n = 0
    self._acc_neg_sum = 0.0
    self._acc_neg_n = 0
    self._apply_event_params()

  def _hard_friction(self) -> float:
    return self._friction_levels[self._level]

  def _pools(self) -> tuple[list[float], list[float]]:
    """Return endpoint lists for easy range and hard point-mass at this level."""
    levels = self._friction_levels
    i = self._level
    hard = [levels[i], levels[i]]
    if i <= 0:
      return [levels[0]], hard
    return [levels[0], levels[i - 1]], hard

  def _apply_event_params(self) -> None:
    easy_values, hard_values = self._pools()
    self._event_cfg.params["easy_values"] = easy_values
    self._event_cfg.params["hard_values"] = hard_values
    self._event_cfg.params["hard_ratio"] = self._hard_ratio

  def _spin_inplace_mask(
    self, env: ManagerBasedRlEnv, env_ids: torch.Tensor, command: torch.Tensor
  ) -> torch.Tensor:
    """True where the last command segment is spin-in-place."""
    term = env.command_manager.get_term(self._command_name)
    is_spin = getattr(term, "is_spin_env", None) if term is not None else None
    if isinstance(is_spin, torch.Tensor):
      return is_spin[env_ids]
    # Fallback: zero (or near-zero) planar velocity + nontrivial yaw.
    lin_norm = torch.linalg.norm(command[env_ids, :2], dim=1)
    return lin_norm <= self._lin_cmd_gate

  def _gated_masks(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    *,
    prefer_hard: bool = True,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (all_gated, pos_gated, neg_gated) boolean masks over env_ids."""
    n = env_ids.numel()
    device = env_ids.device
    command = env.command_manager.get_command(self._command_name)
    base = torch.ones(n, dtype=torch.bool, device=device)
    pos = torch.zeros(n, dtype=torch.bool, device=device)
    neg = torch.zeros(n, dtype=torch.bool, device=device)
    if command is not None and self._cmd_gate > 0.0:
      cmd_z = command[env_ids, 2]
      base = torch.abs(cmd_z) >= self._cmd_gate
      pos = cmd_z >= self._cmd_gate
      neg = cmd_z <= -self._cmd_gate
    elif command is not None:
      cmd_z = command[env_ids, 2]
      pos = cmd_z > 0.0
      neg = cmd_z < 0.0

    if self._require_spin_inplace and command is not None:
      spin = self._spin_inplace_mask(env, env_ids, command)
      base = base & spin
      pos = pos & spin
      neg = neg & spin

    if prefer_hard:
      event_func = self._event_cfg.func
      is_hard = getattr(event_func, "is_hard", None)
      if is_hard is not None and self._level > 0:
        hard = is_hard[env_ids]
        base = base & hard
        pos = pos & hard
        neg = neg & hard
    return base, pos, neg

  def _per_env_track_ang(
    self, env: ManagerBasedRlEnv, env_ids: torch.Tensor
  ) -> torch.Tensor | None:
    rm = env.reward_manager
    if self._reward_name not in getattr(rm, "active_terms", []):
      return None
    return rm._episode_sums[self._reward_name][env_ids] / env.max_episode_length_s

  def _update_ema(self, value: float, which: str = "all") -> float:
    attr = {"all": "_ema", "pos": "_ema_pos", "neg": "_ema_neg"}[which]
    prev = getattr(self, attr)
    if prev is None:
      setattr(self, attr, value)
      return value
    updated = (1.0 - self._ema_alpha) * prev + self._ema_alpha * value
    setattr(self, attr, updated)
    return updated

  def _clear_accumulators(self) -> None:
    self._acc_sum = 0.0
    self._acc_n = 0
    self._acc_pos_sum = 0.0
    self._acc_pos_n = 0
    self._acc_neg_sum = 0.0
    self._acc_neg_n = 0

  def _reset_tracking_stats(self) -> None:
    self._ema = None
    self._ema_pos = None
    self._ema_neg = None
    self._ok_streak = 0
    self._clear_accumulators()

  def _set_level(self, level: int, step: int) -> None:
    self._level = level
    self._last_level_change_step = step
    self._reset_tracking_stats()
    self._apply_event_params()

  def _flush_accumulators(self, env: ManagerBasedRlEnv) -> tuple[float, float, float]:
    """Flush pooled samples into EMA / streak / promote-demote. Returns means."""
    mean_track_ang = self._acc_sum / self._acc_n
    self._update_ema(mean_track_ang, "all")

    mean_pos = 0.0
    mean_neg = 0.0
    if self._acc_pos_n >= self._min_side_envs:
      mean_pos = self._acc_pos_sum / self._acc_pos_n
      self._update_ema(mean_pos, "pos")
    if self._acc_neg_n >= self._min_side_envs:
      mean_neg = self._acc_neg_sum / self._acc_neg_n
      self._update_ema(mean_neg, "neg")

    self._clear_accumulators()

    ema = self._ema if self._ema is not None else mean_track_ang
    sides_ok = True
    if self._require_both_yaw:
      sides_ok = (
        self._ema_pos is not None
        and self._ema_neg is not None
        and self._ema_pos >= self._track_ang_threshold
        and self._ema_neg >= self._track_ang_threshold
      )

    batch_ok = sides_ok and ema >= self._track_ang_threshold
    if batch_ok:
      self._ok_streak += 1
    else:
      self._ok_streak = 0

    steps_since = env.common_step_counter - self._last_level_change_step
    can_promote = (
      self._level < len(self._friction_levels) - 1
      and steps_since >= self._min_promote_steps
      and self._ok_streak >= self._promote_ok_needed
      and ema >= self._track_ang_threshold
      and sides_ok
    )
    can_demote = (
      self._level > 0
      and steps_since >= self._min_demote_steps
      and ema < self._demote_threshold
    )
    if can_promote:
      self._set_level(self._level + 1, env.common_step_counter)
    elif can_demote:
      self._set_level(self._level - 1, env.common_step_counter)

    return mean_track_ang, mean_pos, mean_neg

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str = "foot_friction_slide",
    command_name: str = "twist",
    reward_name: str = "track_ang_vel_z_exp",
    friction_levels: list[float] | None = None,
    easy_friction: float = 0.4,
    hard_ratio: float = 0.6,
    track_ang_threshold: float = 1.7,
    demote_threshold: float = 1.4,
    min_promote_steps: int = 2000 * 24,
    min_demote_steps: int = 2000 * 24,
    start_level: int = 0,
    cmd_gate: float = 0.3,
    require_spin_inplace: bool = False,
    lin_cmd_gate: float = 0.1,
    ema_alpha: float = 0.05,
    min_gated_envs: int = 64,
    require_both_yaw: bool = True,
    min_side_envs: int = 16,
    promote_ok_needed: int = 50,
    # Legacy kwargs kept so old configs/tests do not explode on **params.
    metric_name: str | None = None,
    error_threshold: float | None = None,
  ) -> dict[str, torch.Tensor]:
    del (
      event_name,
      command_name,
      reward_name,
      friction_levels,
      easy_friction,
      hard_ratio,
      track_ang_threshold,
      demote_threshold,
      min_promote_steps,
      min_demote_steps,
      start_level,
      cmd_gate,
      require_spin_inplace,
      lin_cmd_gate,
      ema_alpha,
      min_gated_envs,
      require_both_yaw,
      min_side_envs,
      promote_ok_needed,
      metric_name,
      error_threshold,
    )

    mean_track_ang = 0.0
    mean_pos = 0.0
    mean_neg = 0.0
    batch_gated_n = 0

    if env.common_step_counter > 0 and isinstance(env_ids, torch.Tensor):
      if env_ids.numel() > 0:
        per_env = self._per_env_track_ang(env, env_ids)
        if per_env is not None:
          # Level >0: only hard envs count. Never fall back to easy.
          # Level 0: easy==hard (all μ=levels[0]), so prefer_hard is a no-op.
          base, pos, neg = self._gated_masks(env, env_ids, prefer_hard=True)
          batch_gated_n = int(base.sum().item())
          pos_n = int(pos.sum().item())
          neg_n = int(neg.sum().item())

          if batch_gated_n > 0:
            self._acc_sum += float(per_env[base].sum().item())
            self._acc_n += batch_gated_n
          if pos_n > 0:
            self._acc_pos_sum += float(per_env[pos].sum().item())
            self._acc_pos_n += pos_n
          if neg_n > 0:
            self._acc_neg_sum += float(per_env[neg].sum().item())
            self._acc_neg_n += neg_n

          if self._acc_n >= self._min_gated_envs:
            mean_track_ang, mean_pos, mean_neg = self._flush_accumulators(env)

    easy_values, hard_values = self._pools()
    ema_val = self._ema if self._ema is not None else mean_track_ang
    pos_log = mean_pos if mean_pos != 0.0 else (self._ema_pos or 0.0)
    neg_log = mean_neg if mean_neg != 0.0 else (self._ema_neg or 0.0)
    return {
      "level": torch.tensor(float(self._level)),
      "hard_friction": torch.tensor(self._hard_friction()),
      "easy_min": torch.tensor(min(easy_values)),
      "easy_max": torch.tensor(max(easy_values)),
      "hard_min": torch.tensor(min(hard_values)),
      "hard_max": torch.tensor(max(hard_values)),
      "track_ang": torch.tensor(float(mean_track_ang)),
      "track_ang_ema": torch.tensor(float(ema_val)),
      "track_ang_pos": torch.tensor(float(pos_log)),
      "track_ang_neg": torch.tensor(float(neg_log)),
      "ok_streak": torch.tensor(float(self._ok_streak)),
      "gated_n": torch.tensor(float(batch_gated_n)),
      "acc_n": torch.tensor(float(self._acc_n)),
    }
