"""Tests for foot_friction_slide curriculum and easy/hard friction mixture."""

from unittest.mock import Mock

import pytest
import torch

from mjlab.envs.mdp.dr.geom import geom_friction_easy_hard
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.tasks.velocity.mdp.curriculums import foot_friction_slide_curriculum


def _make_event_cfg() -> EventTermCfg:
  return EventTermCfg(
    mode="reset",
    func=Mock(),
    params={
      "easy_values": [0.4],
      "hard_values": [0.4],
      "hard_ratio": 0.6,
      "axes": [0],
      "shared_random": True,
    },
  )


def _make_curriculum(
  *,
  step: int,
  track_ang: float,
  is_hard: torch.Tensor | None = None,
  min_promote_steps: int = 0,
  friction_levels: list[float] | None = None,
  track_ang_threshold: float = 1.76,
  demote_threshold: float = 1.4,
  start_level: int = 0,
  cmd_z: float | torch.Tensor = 0.5,
  cmd_xy: tuple[float, float] | torch.Tensor = (0.0, 0.0),
  is_spin: torch.Tensor | None = None,
  require_spin_inplace: bool = False,
  lin_cmd_gate: float = 0.1,
  max_episode_length_s: float = 20.0,
  num_envs: int = 4,
  min_gated_envs: int = 1,
  require_both_yaw: bool = False,
  min_side_envs: int = 1,
  promote_ok_needed: int = 1,
  ema_alpha: float = 1.0,
  min_demote_steps: int = 0,
):
  levels = friction_levels or [0.4, 0.7, 1.0, 1.3, 1.6, 2.0]
  event_cfg = _make_event_cfg()
  event_func = Mock()
  event_func.is_hard = (
    is_hard if is_hard is not None else torch.zeros(num_envs, dtype=torch.bool)
  )
  event_cfg.func = event_func

  env = Mock()
  env.num_envs = num_envs
  env.device = "cpu"
  env.common_step_counter = step
  env.max_episode_length_s = max_episode_length_s
  env.event_manager.get_term_cfg.return_value = event_cfg

  # episode_sum = Episode_Reward_scale * max_episode_length_s
  rm = Mock()
  rm.active_terms = ["track_ang_vel_z_exp"]
  if isinstance(track_ang, torch.Tensor):
    sums = track_ang * max_episode_length_s
  else:
    sums = torch.full((num_envs,), track_ang * max_episode_length_s)
  rm._episode_sums = {"track_ang_vel_z_exp": sums}
  env.reward_manager = rm

  cmd = torch.zeros(num_envs, 3)
  if isinstance(cmd_z, torch.Tensor):
    cmd[:, 2] = cmd_z
  else:
    cmd[:, 2] = cmd_z
  if isinstance(cmd_xy, torch.Tensor):
    cmd[:, :2] = cmd_xy
  else:
    cmd[:, 0] = cmd_xy[0]
    cmd[:, 1] = cmd_xy[1]
  env.command_manager.get_command.return_value = cmd

  cmd_term = Mock()
  if is_spin is not None:
    cmd_term.is_spin_env = is_spin
  else:
    cmd_term.is_spin_env = None
  env.command_manager.get_term.return_value = cmd_term

  params = {
    "event_name": "foot_friction_slide",
    "command_name": "twist",
    "reward_name": "track_ang_vel_z_exp",
    "hard_ratio": 0.6,
    "track_ang_threshold": track_ang_threshold,
    "demote_threshold": demote_threshold,
    "friction_levels": levels,
    "min_promote_steps": min_promote_steps,
    "min_demote_steps": min_demote_steps,
    "start_level": start_level,
    "cmd_gate": 0.25,
    "require_spin_inplace": require_spin_inplace,
    "lin_cmd_gate": lin_cmd_gate,
    "ema_alpha": ema_alpha,
    "min_gated_envs": min_gated_envs,
    "require_both_yaw": require_both_yaw,
    "min_side_envs": min_side_envs,
    "promote_ok_needed": promote_ok_needed,
  }
  cfg = CurriculumTermCfg(func=foot_friction_slide_curriculum, params=params)
  instance = foot_friction_slide_curriculum(cfg, env)
  return instance, env, event_cfg, params


def test_starts_at_level_zero():
  instance, _env, event_cfg, params = _make_curriculum(step=0, track_ang=0.0)
  state = instance(_env, torch.arange(4), **params)
  assert state["level"].item() == pytest.approx(0.0)
  assert state["hard_friction"].item() == pytest.approx(0.4)
  assert event_cfg.params["easy_values"] == [0.4]
  assert event_cfg.params["hard_values"] == [0.4, 0.4]


def test_promotes_when_track_ang_above_threshold():
  instance, env, event_cfg, params = _make_curriculum(
    step=100, track_ang=1.80, min_promote_steps=0
  )
  state = instance(env, torch.arange(4), **params)
  assert state["level"].item() == pytest.approx(1.0)
  assert state["hard_friction"].item() == pytest.approx(0.7)
  assert event_cfg.params["easy_values"] == [0.4, 0.4]
  # Hard is pinned to the current level (point mass).
  assert event_cfg.params["hard_values"] == [0.7, 0.7]
  assert state["track_ang"].item() == pytest.approx(1.80, abs=1e-5)


def test_pools_at_hard_1_6():
  instance, env, event_cfg, params = _make_curriculum(
    step=100,
    track_ang=1.80,
    min_promote_steps=0,
    is_hard=torch.ones(4, dtype=torch.bool),
  )
  # level 0 → promote to 4 to reach hard=1.6 with denser levels.
  for _ in range(4):
    instance(env, torch.arange(4), **params)
    env.common_step_counter += 1
  assert instance._level == 4
  assert event_cfg.params["easy_values"] == [0.4, 1.3]
  assert event_cfg.params["hard_values"] == [1.6, 1.6]


def test_no_promote_when_track_ang_low():
  instance, env, event_cfg, params = _make_curriculum(
    step=100, track_ang=1.50, min_promote_steps=0
  )
  state = instance(env, torch.arange(4), **params)
  assert state["level"].item() == pytest.approx(0.0)
  assert event_cfg.params["hard_values"] == [0.4, 0.4]


def test_promote_cooldown_blocks_rapid_climb():
  instance, env, event_cfg, params = _make_curriculum(
    step=100, track_ang=1.90, min_promote_steps=1000
  )
  state = instance(env, torch.arange(4), **params)
  assert state["level"].item() == pytest.approx(0.0)
  assert event_cfg.params["hard_values"] == [0.4, 0.4]


def test_single_batch_does_not_promote_without_streak():
  instance, env, _event_cfg, params = _make_curriculum(
    step=100,
    track_ang=1.90,
    min_promote_steps=0,
    promote_ok_needed=5,
  )
  state = instance(env, torch.arange(4), **params)
  assert state["level"].item() == pytest.approx(0.0)
  assert state["ok_streak"].item() == pytest.approx(1.0)


def test_demotes_when_ema_collapses():
  instance, env, event_cfg, params = _make_curriculum(
    step=100,
    track_ang=1.0,
    is_hard=torch.ones(4, dtype=torch.bool),
    min_promote_steps=0,
    start_level=2,
    demote_threshold=1.4,
    min_demote_steps=0,
  )
  state = instance(env, torch.arange(4), **params)
  assert state["level"].item() == pytest.approx(1.0)
  assert event_cfg.params["hard_values"] == [0.7, 0.7]


def test_level_gt0_ignores_easy_only_batch():
  """Good easy turning must not update EMA / promote when hard envs are absent."""
  is_hard = torch.tensor([False, False, False, False])
  instance, env, _event_cfg, params = _make_curriculum(
    step=100,
    track_ang=1.90,
    is_hard=is_hard,
    min_promote_steps=0,
    start_level=1,
    promote_ok_needed=1,
  )
  state = instance(env, torch.arange(4), **params)
  assert state["level"].item() == pytest.approx(1.0)
  assert state["gated_n"].item() == pytest.approx(0.0)
  assert instance._ema is None
  assert instance._ok_streak == 0


def test_accumulates_across_resets_before_ema_flush():
  """Small reset batches pool until min_gated_envs, then flush once."""
  instance, env, _event_cfg, params = _make_curriculum(
    step=100,
    track_ang=1.80,
    min_promote_steps=0,
    min_gated_envs=8,
    promote_ok_needed=2,  # need two flushes to promote
    num_envs=4,
  )
  # First call: 4 gated < 8 → accumulate only.
  state = instance(env, torch.arange(4), **params)
  assert state["level"].item() == pytest.approx(0.0)
  assert state["acc_n"].item() == pytest.approx(4.0)
  assert instance._ema is None
  # Second call: 4+4 >= 8 → flush EMA, no promote yet (streak=1).
  env.common_step_counter = 101
  state = instance(env, torch.arange(4), **params)
  assert state["level"].item() == pytest.approx(0.0)
  assert instance._ema == pytest.approx(1.80, abs=1e-5)
  assert instance._ok_streak == 1
  assert state["acc_n"].item() == pytest.approx(0.0)
  # Two more calls → second flush → promote.
  env.common_step_counter = 102
  instance(env, torch.arange(4), **params)
  env.common_step_counter = 103
  state = instance(env, torch.arange(4), **params)
  assert state["level"].item() == pytest.approx(1.0)


def test_require_both_yaw_blocks_asymmetric_skill():
  cmd_z = torch.tensor([0.5, 0.5, -0.5, -0.5])
  # Good on +ω only; −ω is poor.
  track = torch.tensor([1.85, 1.85, 1.0, 1.0])
  instance, env, _event_cfg, params = _make_curriculum(
    step=100,
    track_ang=track,
    cmd_z=cmd_z,
    min_promote_steps=0,
    require_both_yaw=True,
    min_side_envs=1,
    promote_ok_needed=1,
    ema_alpha=1.0,
  )
  state = instance(env, torch.arange(4), **params)
  assert state["level"].item() == pytest.approx(0.0)


def test_require_both_yaw_promotes_when_symmetric():
  cmd_z = torch.tensor([0.5, 0.5, -0.5, -0.5])
  track = torch.tensor([1.85, 1.85, 1.85, 1.85])
  instance, env, event_cfg, params = _make_curriculum(
    step=100,
    track_ang=track,
    cmd_z=cmd_z,
    min_promote_steps=0,
    require_both_yaw=True,
    min_side_envs=1,
    promote_ok_needed=1,
    ema_alpha=1.0,
  )
  state = instance(env, torch.arange(4), **params)
  assert state["level"].item() == pytest.approx(1.0)
  assert event_cfg.params["hard_values"] == [0.7, 0.7]


def test_spin_inplace_gate_excludes_driving_turn_via_is_spin():
  """Driving+yaw must not gate; only is_spin_env samples count."""
  is_spin = torch.tensor([True, False, True, False])
  cmd_xy = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.8, 0.2]])
  instance, env, _event_cfg, params = _make_curriculum(
    step=100,
    track_ang=1.90,
    cmd_z=0.5,
    cmd_xy=cmd_xy,
    is_spin=is_spin,
    require_spin_inplace=True,
    min_promote_steps=0,
    promote_ok_needed=1,
  )
  state = instance(env, torch.arange(4), **params)
  assert state["gated_n"].item() == pytest.approx(2.0)
  assert state["level"].item() == pytest.approx(1.0)


def test_spin_inplace_fallback_uses_lin_cmd_gate():
  """Without is_spin_env, ||cmd_xy|| <= lin_cmd_gate selects spin-like cmds."""
  cmd_xy = torch.tensor([[0.0, 0.0], [0.5, 0.0], [0.05, 0.0], [1.0, 0.0]])
  cmd_z = torch.tensor([0.5, 0.5, 0.5, 0.5])
  instance, env, _event_cfg, params = _make_curriculum(
    step=100,
    track_ang=1.90,
    cmd_z=cmd_z,
    cmd_xy=cmd_xy,
    is_spin=None,
    require_spin_inplace=True,
    lin_cmd_gate=0.1,
    min_promote_steps=0,
    promote_ok_needed=1,
  )
  state = instance(env, torch.arange(4), **params)
  # env0 and env2 only (lin ≈ 0 / 0.05)
  assert state["gated_n"].item() == pytest.approx(2.0)


def test_geom_friction_easy_hard_continuous_ranges(monkeypatch):
  written = []

  def _fake_geom_friction(env, env_ids, ranges, **kwargs):
    del env, kwargs
    written.append((set(env_ids.tolist()), ranges))

  monkeypatch.setattr("mjlab.envs.mdp.dr.geom.geom_friction", _fake_geom_friction)

  env = Mock()
  env.num_envs = 200
  env.device = "cpu"
  cfg = EventTermCfg(mode="reset", func=geom_friction_easy_hard, params={})
  term = geom_friction_easy_hard(cfg, env)
  torch.manual_seed(0)
  env_ids = torch.arange(200)
  term(
    env,
    env_ids,
    easy_values=[0.4, 1.2],
    hard_values=[1.2, 1.6],
    hard_ratio=0.6,
  )

  hard_count = int(term.is_hard.sum().item())
  assert 90 <= hard_count <= 150
  assert len(written) == 2
  ranges_by_kind = {r: ids for ids, r in written}
  assert ranges_by_kind[(0.4, 1.2)] == set(
    i for i in range(200) if not term.is_hard[i]
  )
  assert ranges_by_kind[(1.2, 1.6)] == set(i for i in range(200) if term.is_hard[i])


def test_geom_friction_easy_hard_all_easy_when_ranges_equal(monkeypatch):
  monkeypatch.setattr(
    "mjlab.envs.mdp.dr.geom.geom_friction",
    lambda *args, **kwargs: None,
  )
  env = Mock()
  env.num_envs = 8
  env.device = "cpu"
  cfg = EventTermCfg(mode="reset", func=geom_friction_easy_hard, params={})
  term = geom_friction_easy_hard(cfg, env)
  term(
    env,
    torch.arange(8),
    easy_values=[0.4],
    hard_values=[0.4],
    hard_ratio=0.6,
  )
  assert not term.is_hard.any()
