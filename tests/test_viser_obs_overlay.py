"""Tests for hierarchical observation expansion in the Viser Observations tab."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.viewer.viser.obs_plotter import (
  _category_for_dim,
  expand_observation_terms,
)


def test_category_joint_vs_wheel():
  assert _category_for_dim("joint_pos", "LF_HFE") == "Joint"
  assert _category_for_dim("joint_vel", "LF_Wheel") == "Wheel"
  assert _category_for_dim("actions", "RF_KFE") == "Joint"
  assert _category_for_dim("actions", "RH_Wheel") == "Wheel"
  assert _category_for_dim("base_ang_vel", "x") == "Other"
  assert _category_for_dim("command", "lin_vel_x") == "Other"


def test_expand_observation_terms_groups_and_labels():
  obs_mgr = MagicMock()
  obs_mgr.get_active_iterable_terms.return_value = [
    ("actor-joint_pos", [0.1, 0.2]),
    ("actor-joint_vel", [1.0, 2.0, 3.0]),
    ("actor-base_ang_vel", [0.0, 0.1, 0.2]),
    ("critic-command", [0.5, -0.3]),
  ]

  leg_cfg = SceneEntityCfg("robot", joint_names=("LF_HFE", "LF_KFE"))
  vel_cfg = SceneEntityCfg(
    "robot", joint_names=("LF_HFE", "LF_KFE", "LF_Wheel")
  )

  def get_term_cfg(group, term):
    params = {}
    if term == "joint_pos":
      params["asset_cfg"] = leg_cfg
    elif term == "joint_vel":
      params["asset_cfg"] = vel_cfg
    return SimpleNamespace(params=params)

  obs_mgr.get_term_cfg.side_effect = get_term_cfg
  env = SimpleNamespace(observation_manager=obs_mgr)

  series = expand_observation_terms(env, env_idx=0)
  by_key = {s.key: s for s in series}

  assert by_key["actor/joint_pos/LF_HFE"].group == "actor"
  assert by_key["actor/joint_pos/LF_HFE"].category == "Joint"
  assert by_key["actor/joint_vel/LF_Wheel"].category == "Wheel"
  assert by_key["actor/base_ang_vel/x"].category == "Other"
  assert by_key["critic/command/lin_vel_x"].group == "critic"
  assert by_key["critic/command/ang_vel_z"].values == [-0.3]
