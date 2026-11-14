"""Check Lego2 cylinder inertia geometry on CPU, without running training.

This validates the existing uniform-solid-cylinder approximation, not the
physical robot's CAD mass properties. Thigh/calf and lidar calibration is
outside the scope of this check.
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ASSET_DIR = Path(__file__).resolve().parents[2] / "src/mjlab/asset_zoo/robots/lego2"


def rotation(quat: np.ndarray) -> np.ndarray:
  matrix = np.empty(9)
  mujoco.mju_quat2Mat(matrix, quat)
  return matrix.reshape(3, 3)


def check(xml_path: Path, urdf_path: Path) -> None:
  model = mujoco.MjModel.from_xml_path(str(xml_path))
  urdf = ET.parse(urdf_path).getroot()
  for leg in ("RF", "LF", "RH", "LH"):
    for suffix in ("hip", "FOOT"):
      name = f"{leg}_{suffix}"
      body_id = model.body(name).id
      geom_name = (
        f"{leg}_hip_collision" if suffix == "hip" else (f"{leg}_foot_collision")
      )
      geom_id = model.geom(geom_name).id
      radius, half_length = model.geom_size[geom_id, :2]
      mass = model.body_mass[body_id]
      axial = mass * radius**2 / 2
      transverse = mass * (3 * radius**2 + (2 * half_length) ** 2) / 12
      axis = rotation(model.geom_quat[geom_id])[:, 2]
      expected = transverse * np.eye(3) + (axial - transverse) * np.outer(axis, axis)
      inertial_rotation = rotation(model.body_iquat[body_id])
      actual = (
        inertial_rotation @ np.diag(model.body_inertia[body_id]) @ inertial_rotation.T
      )
      np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-10)
      eigenvalues = np.linalg.eigvalsh(actual)
      assert eigenvalues[0] > 0, f"{name}: nonpositive inertia"
      assert eigenvalues[0] + eigenvalues[1] >= eigenvalues[2], name

      link = urdf.find(f"link[@name='{name}']")
      assert link is not None, name
      inertial = link.find("inertial")
      assert inertial is not None, name
      values = inertial.find("inertia")
      mass_element = inertial.find("mass")
      assert values is not None and mass_element is not None, name
      assert float(mass_element.attrib["value"]) == mass, name
      a = values.attrib
      urdf_tensor = np.array(
        [
          [float(a["ixx"]), float(a["ixy"]), float(a["ixz"])],
          [float(a["ixy"]), float(a["iyy"]), float(a["iyz"])],
          [float(a["ixz"]), float(a["iyz"]), float(a["izz"])],
        ]
      )
      np.testing.assert_allclose(urdf_tensor, expected, atol=1e-11, rtol=1e-10)
      origin = inertial.find("origin")
      assert origin is not None, name
      np.testing.assert_allclose(
        np.fromstring(origin.get("rpy", "0 0 0"), sep=" "), 0, atol=1e-12
      )
      np.testing.assert_allclose(
        np.fromstring(origin.get("xyz", "0 0 0"), sep=" "),
        model.body_ipos[body_id],
        atol=1e-12,
      )
      if suffix == "FOOT":
        np.testing.assert_allclose(
          model.body_ipos[body_id], model.geom_pos[geom_id], atol=1e-12
        )
        joint_id = model.joint(f"{leg}_Wheel").id
        np.testing.assert_allclose(
          abs(np.dot(model.jnt_axis[joint_id], axis)), 1, atol=1e-10
        )
      print(
        f"PASS {name}: COM={model.body_ipos[body_id].tolist()} "
        f"Ixx/Iyy/Izz={np.diag(actual).round(12).tolist()}"
      )
  print(
    f"PASS total mass={model.body_mass.sum():.6f} kg, "
    f"nq={model.nq}, nv={model.nv}; CPU-only cylinder checks."
  )
  print("NOT VALIDATED: measured CAD inertia/COM, thigh/calf symmetry, climbing.")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--xml", type=Path, default=ASSET_DIR / "xmls/lego2.xml")
  parser.add_argument(
    "--urdf", type=Path, default=ASSET_DIR / "urdf/urdf_legowheel2.urdf"
  )
  args = parser.parse_args()
  check(args.xml, args.urdf)


if __name__ == "__main__":
  main()
