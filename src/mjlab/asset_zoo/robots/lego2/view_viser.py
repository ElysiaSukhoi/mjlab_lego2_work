"""View Lego2 in the browser via Viser (no X11 / GLFW needed).

Run from the mjlab repo root:

  uv run python src/mjlab/asset_zoo/robots/lego2/view_viser.py
"""

from __future__ import annotations

import time
from pathlib import Path

import mujoco
import viser
from mjviser import ViserMujocoScene

XML_PATH = Path(__file__).resolve().parent / "xmls" / "lego2.xml"

# Standing pose used by lego2_constants.INIT_STATE.
INIT_JOINT_POS = {
  "RF_HFE": 0.72,
  "LF_HFE": 0.72,
  "RH_HFE": -0.72,
  "LH_HFE": -0.72,
  "RF_KFE": -1.44,
  "LF_KFE": -1.44,
  "RH_KFE": 1.44,
  "LH_KFE": 1.44,
}


def main() -> None:
  model = mujoco.MjModel.from_xml_path(str(XML_PATH))
  data = mujoco.MjData(model)

  data.qpos[0:3] = [0.0, 0.0, 0.32]
  data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
  for name, value in INIT_JOINT_POS.items():
    data.qpos[model.joint(name).qposadr[0]] = value
  mujoco.mj_forward(model, data)

  server = viser.ViserServer(label="Lego2")
  scene = ViserMujocoScene(server, model, num_envs=1)
  scene.create_scene_gui(
    camera_distance=2.0,
    camera_azimuth=135.0,
    camera_elevation=20.0,
  )
  scene.update_from_mjdata(data)

  print("=" * 60)
  print("Lego2 Viser viewer")
  print("  Open the URL printed by Viser (usually http://localhost:8080)")
  print("  Ctrl+C to quit")
  print("=" * 60)

  try:
    while True:
      time.sleep(0.1)
  except KeyboardInterrupt:
    print("\nShutting down...")
    server.stop()


if __name__ == "__main__":
  main()
