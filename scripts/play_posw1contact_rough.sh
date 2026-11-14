#!/usr/bin/env bash
# Play posw1contact_rough with that run's training terrain params.
# Does NOT modify env_cfgs.py — overrides are applied at runtime.
#
# Usage:
#   bash scripts/play_posw1contact_rough.sh
#   bash scripts/play_posw1contact_rough.sh --num-envs 4
#   bash scripts/play_posw1contact_rough.sh --viewer native   # needs working DISPLAY
#   CKPT=logs/.../model_5000.pt bash scripts/play_posw1contact_rough.sh
#
# Default viewer=viser (browser) — avoids GLFW/X11 failures on broken DISPLAY.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="${HOME}/.local/bin:${PATH}"

RUN_DIR="${RUN_DIR:-logs/rsl_rl/lego2_velocity/rough/2026-08-31_21-40-25_posw1contact_rough}"
CKPT="${CKPT:-${RUN_DIR}/model_9999.pt}"

exec uv run python - "$@" <<'PY'
"""Runtime override: match 2026-08-31 posw1contact_rough terrain without editing env_cfgs."""
from __future__ import annotations

import sys
from dataclasses import replace

import mjlab.terrains as terrain_gen
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.scripts.play import PlayConfig, run_play
from mjlab.tasks.registry import load_env_cfg as _load_env_cfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

# --- terrain from that run's params/env.yaml ---
_POSW1CONTACT_ROUGH_TERRAIN = TerrainGeneratorCfg(
  curriculum=False,
  size=(10.0, 10.0),
  border_width=0.0,
  num_rows=10,
  num_cols=20,
  difficulty_range=(0.0, 1.0),
  merge_curriculum_rows=False,
  sub_terrains={
    "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.4),
    "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
      proportion=0.6,
      noise_range=(-0.025, 0.025),
      noise_step=0.05,
      downsampled_scale=0.2,
      horizontal_scale=0.2,
      vertical_scale=0.005,
      border_width=0.0,
      scale_with_difficulty=False,
      flat_patch_sampling=None,
    ),
  },
)


def _load_env_cfg_patched(task_id: str, play: bool = False):
  cfg = _load_env_cfg(task_id, play=play)
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = replace(_POSW1CONTACT_ROUGH_TERRAIN)
  cfg.scene.terrain.max_init_terrain_level = 5
  # Match training spawn (not flat-patch).
  cfg.events["reset_base"] = EventTermCfg(
    func=envs_mdp.reset_root_state_uniform,
    mode="reset",
    params={
      "pose_range": {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (0.01, 0.05),
        "yaw": (-3.14, 3.14),
      },
      "velocity_range": {},
    },
  )
  print(
    "[INFO] Applied posw1contact_rough terrain: "
    "curriculum=False, noise=(-0.025,0.025), hs=0.2, ds=0.2, border=0"
  )
  return cfg


# run_play binds load_env_cfg from its module namespace.
import mjlab.scripts.play as play_mod

play_mod.load_env_cfg = _load_env_cfg_patched  # type: ignore[attr-defined]

import os

ckpt = os.environ.get(
  "CKPT",
  "logs/rsl_rl/lego2_velocity/rough/2026-08-31_21-40-25_posw1contact_rough/model_9999.pt",
)

# Minimal CLI: --num-envs N, --viewer {auto,native,viser}
num_envs = 1
viewer = "viser"  # default: avoid GLFW when DISPLAY is broken (:1)
extra = list(sys.argv[1:])
if "--num-envs" in extra:
  i = extra.index("--num-envs")
  num_envs = int(extra[i + 1])
  del extra[i : i + 2]
if "--viewer" in extra:
  i = extra.index("--viewer")
  viewer = extra[i + 1]
  del extra[i : i + 2]

cfg = PlayConfig(
  agent="trained",
  checkpoint_file=ckpt,
  num_envs=num_envs,
  viewer=viewer,  # type: ignore[arg-type]
)
if extra:
  print(f"[WARN] Ignored extra args: {extra}")

print(f"[INFO] viewer={viewer}  checkpoint={ckpt}")
run_play("Mjlab-Velocity-Rough-Lego2", cfg)
PY
