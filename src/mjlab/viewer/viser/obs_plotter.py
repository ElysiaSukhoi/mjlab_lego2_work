"""Hierarchical observation plotter for the Viser play viewer."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import viser
import viser.uplot

from mjlab.viewer.viser.term_plotter import _color_for

# Display order for top-level and sub-groups.
_GROUP_ORDER = ("actor", "critic")
_CATEGORY_ORDER = ("Joint", "Wheel", "Other")


@dataclass
class ObsSeries:
  """One scalar observation channel for plotting."""

  key: str
  """Stable plot key, e.g. ``actor/joint_pos/LF_HFE``."""
  group: str
  """Observation group: ``actor`` / ``critic`` / other."""
  category: str
  """``Joint`` / ``Wheel`` / ``Other``."""
  label: str
  """Short checkbox label inside the category folder."""
  values: list[float]
  """Current scalar value as a length-1 list (plotter API)."""


@dataclass
class _TermState:
  name: str
  color: str
  enabled: bool = False
  history: deque[float] = field(default_factory=lambda: deque(maxlen=300))
  plot: viser.GuiUplotHandle | None = None


def _is_wheel_name(name: str) -> bool:
  return "wheel" in name.lower()


def _is_joint_name(name: str) -> bool:
  upper = name.upper()
  return any(tok in upper for tok in ("HFE", "KFE", "HIP", "THIGH", "CALF", "KNEE"))


def _category_for_dim(term_name: str, dim_label: str) -> str:
  """Classify one observation dimension as Joint / Wheel / Other."""
  if _is_wheel_name(dim_label) or _is_wheel_name(term_name):
    return "Wheel"
  if term_name in {"joint_pos", "joint_vel", "actions"} or _is_joint_name(dim_label):
    return "Joint"
  return "Other"


def _axis_labels(n: int, prefixes: tuple[str, ...] = ("x", "y", "z")) -> list[str]:
  if n <= len(prefixes):
    return list(prefixes[:n])
  return [f"{i}" for i in range(n)]


def _dim_labels_for_term(env: Any, group_name: str, term_name: str, n: int) -> list[str]:
  """Best-effort per-dimension labels from obs/action configs."""
  obs_mgr = env.observation_manager
  try:
    term_cfg = obs_mgr.get_term_cfg(group_name, term_name)
  except (ValueError, KeyError):
    term_cfg = None

  if term_name in {"joint_pos", "joint_vel"} and term_cfg is not None:
    asset_cfg = (term_cfg.params or {}).get("asset_cfg")
    if asset_cfg is not None and getattr(asset_cfg, "joint_names", None):
      names = list(asset_cfg.joint_names)
      if len(names) == n:
        return names

  if term_name == "actions" and hasattr(env, "action_manager"):
    labels: list[str] = []
    for act_name in env.action_manager.active_terms:
      term = env.action_manager.get_term(act_name)
      target_names = getattr(term, "_target_names", None)
      if target_names:
        labels.extend(list(target_names))
      else:
        labels.extend(f"{act_name}[{i}]" for i in range(term.action_dim))
    if len(labels) == n:
      return labels

  if term_name == "command":
    if n == 2:
      return ["lin_vel_x", "ang_vel_z"]
    if n == 3:
      return ["lin_vel_x", "lin_vel_y", "ang_vel_z"]

  if term_name in {"base_ang_vel", "base_lin_vel"}:
    return _axis_labels(n)
  if term_name == "projected_gravity":
    return _axis_labels(n)

  return [str(i) for i in range(n)]


def expand_observation_terms(env: Any, env_idx: int) -> list[ObsSeries]:
  """Expand observation manager terms into labeled scalar series."""
  raw = list(env.observation_manager.get_active_iterable_terms(env_idx))
  series: list[ObsSeries] = []
  for full_name, values in raw:
    vals = list(values)
    if "-" in full_name:
      group_name, term_name = full_name.split("-", 1)
    else:
      group_name, term_name = "obs", full_name

    if not vals:
      vals = [0.0]

    dim_labels = _dim_labels_for_term(env, group_name, term_name, len(vals))
    if len(dim_labels) < len(vals):
      dim_labels = dim_labels + [str(i) for i in range(len(dim_labels), len(vals))]

    for i, (v, dim_label) in enumerate(zip(vals, dim_labels, strict=True)):
      category = _category_for_dim(term_name, dim_label)
      if len(vals) == 1:
        key = f"{group_name}/{term_name}"
        label = term_name
      else:
        key = f"{group_name}/{term_name}/{dim_label}"
        label = f"{term_name}/{dim_label}"
      series.append(
        ObsSeries(
          key=key,
          group=group_name,
          category=category,
          label=label,
          values=[float(v)],
        )
      )
  return series


class ViserObsPlotter:
  """Observation plotter with Actor/Critic → Joint/Wheel/Other folders."""

  def __init__(
    self,
    server: viser.ViserServer,
    series: list[ObsSeries],
    history_length: int = 150,
    env_idx: int = 0,
  ) -> None:
    self._server = server
    self._history_length = history_length
    self._env_idx = env_idx
    self._x_array = np.arange(-history_length + 1, 1, dtype=np.float64)
    self._empty = np.array([], dtype=np.float64)

    self._series_meta = {s.key: s for s in series}
    self._terms: dict[str, _TermState] = {}
    for i, s in enumerate(series):
      self._terms[s.key] = _TermState(
        name=s.key,
        color=_color_for(i),
        history=deque(maxlen=history_length),
      )

    self._checkboxes: dict[str, viser.GuiInputHandle] = {}
    self._folder_handles: list[Any] = []
    self._build_selector_gui(series)
    self._plots_folder = self._server.gui.add_folder("Plots", expand_by_default=True)

  def _build_selector_gui(self, series: list[ObsSeries]) -> None:
    """Nested folders: group (actor/critic) → category (Joint/Wheel/Other)."""
    grouped: dict[str, dict[str, list[ObsSeries]]] = defaultdict(
      lambda: defaultdict(list)
    )
    for s in series:
      grouped[s.group][s.category].append(s)

    group_order = [g for g in _GROUP_ORDER if g in grouped] + [
      g for g in grouped if g not in _GROUP_ORDER
    ]

    with self._server.gui.add_folder("Select observations", expand_by_default=True):
      self._env_label = self._server.gui.add_markdown(self._env_label_text())
      self._filter_input = self._server.gui.add_text(
        "Filter",
        initial_value="",
        hint="Name must contain this string",
      )

      @self._filter_input.on_update
      def _(_) -> None:
        self._apply_filter()

      bulk = self._server.gui.add_button_group("Select", options=["All", "None"])

      @bulk.on_click
      def _(event) -> None:
        enable = event.target.value == "All"
        for key, state in self._terms.items():
          cb = self._checkboxes[key]
          if cb.visible:
            state.enabled = enable
            cb.value = enable
        self._sync_plots()

      for group_name in group_order:
        title = group_name.capitalize()
        with self._server.gui.add_folder(title, expand_by_default=True) as group_folder:
          self._folder_handles.append(group_folder)
          cats = grouped[group_name]
          cat_order = [c for c in _CATEGORY_ORDER if c in cats] + [
            c for c in cats if c not in _CATEGORY_ORDER
          ]
          for category in cat_order:
            with self._server.gui.add_folder(
              category, expand_by_default=(category != "Other")
            ) as cat_folder:
              self._folder_handles.append(cat_folder)
              for s in cats[category]:
                state = self._terms[s.key]
                cb = self._server.gui.add_checkbox(
                  s.label,
                  initial_value=state.enabled,
                  hint=f"{s.key}  color={state.color}",
                )
                self._checkboxes[s.key] = cb

                @cb.on_update
                def _(event, _key=s.key) -> None:
                  self._terms[_key].enabled = event.target.value
                  self._sync_plots()

  def _apply_filter(self) -> None:
    filter_str = self._filter_input.value.lower()
    for key, state in self._terms.items():
      meta = self._series_meta[key]
      cb = self._checkboxes[key]
      visible = (
        not filter_str
        or filter_str in key.lower()
        or filter_str in meta.label.lower()
        or filter_str in meta.group.lower()
        or filter_str in meta.category.lower()
      )
      cb.visible = visible
      if not visible:
        if state.plot is not None:
          state.plot.remove()
          state.plot = None
      elif state.enabled and state.plot is None:
        self._create_plot(state)

  def _env_label_text(self) -> str:
    return (
      f"<small><em>Observations for environment #{self._env_idx} "
      f"(Actor / Critic → Joint / Wheel / Other)</em></small>"
    )

  def update_env_idx(self, env_idx: int) -> None:
    self._env_idx = env_idx
    self._env_label.content = self._env_label_text()

  def _sync_plots(self) -> None:
    for key, state in self._terms.items():
      cb = self._checkboxes[key]
      should_show = state.enabled and cb.visible
      if should_show and state.plot is None:
        self._create_plot(state)
      elif not should_show and state.plot is not None:
        state.plot.remove()
        state.plot = None

  def _create_plot(self, state: _TermState) -> None:
    h = state.history
    hist_len = len(h)
    if hist_len > 0:
      x = self._x_array[-hist_len:]
      y = np.fromiter(h, dtype=np.float64, count=hist_len)
    else:
      x = self._empty
      y = self._empty

    with self._plots_folder:
      state.plot = self._server.gui.add_uplot(
        data=(x, y),
        series=(
          viser.uplot.Series(label="Steps"),
          viser.uplot.Series(label=state.name, stroke=state.color, width=2),
        ),
        scales={
          "x": viser.uplot.Scale(
            time=False, auto=False, range=(-self._history_length, 0)
          ),
          "y": viser.uplot.Scale(auto=True),
        },
        legend=viser.uplot.Legend(show=True),
        title=state.name,
        aspect=2.0,
        visible=True,
      )

  def update(self, series: list[ObsSeries]) -> None:
    any_enabled = False
    for s in series:
      state = self._terms.get(s.key)
      if state is None:
        continue
      val = float(s.values[0])
      if np.isfinite(val):
        state.history.append(val)
      if state.enabled:
        any_enabled = True

    if not any_enabled:
      return

    for state in self._terms.values():
      if not state.enabled or state.plot is None:
        continue
      h = state.history
      hist_len = len(h)
      if hist_len > 0:
        x = self._x_array[-hist_len:]
        y = np.fromiter(h, dtype=np.float64, count=hist_len)
        state.plot.data = (x, y)

  def clear_histories(self) -> None:
    for state in self._terms.values():
      state.history.clear()
      if state.plot is not None:
        state.plot.data = (self._empty, self._empty)

  def cleanup(self) -> None:
    for state in self._terms.values():
      if state.plot is not None:
        state.plot.remove()
    for cb in self._checkboxes.values():
      cb.remove()
    self._terms.clear()
    self._checkboxes.clear()
