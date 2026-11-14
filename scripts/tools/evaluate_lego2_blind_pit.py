"""Fixed-height blind-pit evaluation, with terminal-state traces and exact n."""

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions.actions import JointPositionAction
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.config.lego2.blind_pit_env_cfg import lego2_blind_pit_eval_cfg
from mjlab.tasks.velocity.config.lego2.blind_pit_metrics import PitProgress
from mjlab.utils.os import dump_yaml


@dataclass
class EvalConfig:
  output: Path
  checkpoint: Path | None = None
  heights: tuple[float, ...] = (0.03, 0.06, 0.09)
  episodes: int = 10
  seed: int = 7
  device: str = "cpu"
  # Only for tool smoke checks. A capped episode is explicitly excluded from n.
  max_steps: int | None = None


def sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def run(cfg: EvalConfig) -> dict:
  if cfg.episodes < 1 or (cfg.max_steps is not None and cfg.max_steps < 1):
    raise ValueError("episodes and max_steps must be positive")
  if len(set(cfg.heights)) != len(cfg.heights):
    raise ValueError("Duplicate heights are not supported")
  for height in cfg.heights:
    lego2_blind_pit_eval_cfg(height)  # Validate all heights before creating files.
  if not cfg.heights:
    raise ValueError("At least one height is required")
  if cfg.checkpoint is not None and not cfg.checkpoint.is_file():
    raise FileNotFoundError(cfg.checkpoint)
  cfg.output.mkdir(parents=True, exist_ok=False)
  root = Path(__file__).resolve().parents[2]
  sources = [
    *sorted((root / "src/mjlab/tasks/velocity/config/lego2").glob("*.py")),
    *sorted((root / "src/mjlab/asset_zoo/robots/lego2").rglob("*.xml")),
    *sorted((root / "src/mjlab/asset_zoo/robots/lego2").rglob("*.urdf")),
    root / "src/mjlab/asset_zoo/robots/lego2/lego2_constants.py",
    Path(__file__).resolve(),
  ]
  report = {
    "mode": "policy" if cfg.checkpoint else "zero_action_diagnostic",
    "seed": cfg.seed,
    "device": cfg.device,
    "checkpoint": str(cfg.checkpoint.resolve()) if cfg.checkpoint else None,
    "checkpoint_sha256": sha256(cfg.checkpoint) if cfg.checkpoint else None,
    "source_sha256": {str(p.relative_to(root)): sha256(p) for p in sources},
    "definition": "four wheels rim-supported, tilt<45deg continuously >=0.5s",
    "heights": [],
  }
  for height in cfg.heights:
    print(f"Fixed pit depth: {height:.6f} m; requested episodes: {cfg.episodes}")
    env_cfg = lego2_blind_pit_eval_cfg(height)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = cfg.seed
    dump_yaml(cfg.output / f"env_{height:.6f}.yaml", asdict(env_cfg))
    env = ManagerBasedRlEnv(env_cfg, device=cfg.device)
    try:
      agent_cfg = load_rl_cfg("Mjlab-Blind-Pit-Lego2")
      dump_yaml(cfg.output / "agent.yaml", asdict(agent_cfg))
      wrapper = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
      policy = None
      if cfg.checkpoint is not None:
        runner_cls = load_runner_cls("Mjlab-Blind-Pit-Lego2") or MjlabOnPolicyRunner
        runner = runner_cls(wrapper, asdict(agent_cfg), device=cfg.device)
        print(f"[eval] Loading checkpoint: {cfg.checkpoint}", flush=True)
        runner.load(
          str(cfg.checkpoint),
          load_cfg={"actor": True},
          strict=True,
          map_location=cfg.device,
        )
        policy = runner.get_inference_policy(device=cfg.device)
        print("[eval] Checkpoint loaded", flush=True)
      progress = PitProgress(env_cfg.metrics["pit_stable_pass"], env)
      robot = env.scene["robot"]
      pos_term = env.action_manager.get_term("joint_pos")
      assert isinstance(pos_term, JointPositionAction)
      ids = pos_term.target_ids
      actuator_ids = robot.indexing.ctrl_ids.cpu().numpy()
      limits = torch.tensor(
        env.sim.mj_model.actuator_forcerange[actuator_ids], device=env.device
      )
      samples = []
      with (cfg.output / f"trace_{height:.6f}.jsonl").open("x") as trace:
        for episode in range(cfg.episodes):
          started = time.monotonic()
          label = f"[eval] height={height:.2f}m episode={episode + 1}/{cfg.episodes}"
          print(f"{label} resetting", flush=True)
          obs, _ = wrapper.reset()
          progress.reset()
          reached = dict.fromkeys(
            ("root_out", "front_up", "rear_up", "complete"), False
          )
          speed_errors = []
          ended, passed = False, False
          step = -1
          origins = env.scene.env_origins
          command = env.command_manager.get_command("twist")
          assert command is not None
          step_budget = cfg.max_steps or env.max_episode_length
          print(f"{label} running, step budget={step_budget}", flush=True)
          last_progress = time.monotonic()
          for step in range(step_budget):
            with torch.inference_mode():
              action = policy(obs) if policy else torch.zeros(1, 12, device=env.device)
              obs, reward, done, _ = wrapper.step(action)
            passed = bool(progress(env)[0])
            for field in reached:
              reached[field] |= bool(progress.values[field][0])
            command = env.command_manager.get_command("twist")
            assert command is not None
            error = command[0, :2] - robot.data.root_link_lin_vel_b[0, :2]
            speed_errors.append(float(error.square().sum()))
            centers, extents = progress.wheels()
            origins = env.scene.env_origins
            rim_z = origins[0, 2] + height
            target = robot.data.joint_pos_target[:, ids]
            hard = robot.data.joint_pos_limits[:, ids]
            assert hard is not None
            over = (target < hard[..., 0]) | (target > hard[..., 1])
            force = robot.data.actuator_force[0]
            saturated = (force <= limits[:, 0] + 1e-3) | (force >= limits[:, 1] - 1e-3)
            rates = {
              k: v[0] for k, v in env.reward_manager.get_active_iterable_terms(0)
            }
            weights = {k: env.reward_manager.get_term_cfg(k).weight for k in rates}
            contact = progress.contact.data
            assert contact.found is not None
            assert contact.force is not None and contact.normal is not None
            row = {
              "episode": episode,
              "step": step + 1,
              "stable_pass": passed,
              "stages": {k: bool(v[0]) for k, v in progress.values.items()},
              "wheel_names": progress.contact.primary_names,
              "joint_names": robot.joint_names,
              "actuator_names": robot.actuator_names,
              "position_target_names": pos_term.target_names,
              "raw_action": action[0].tolist(),
              "command": command[0].tolist(),
              "root": robot.data.root_link_pose_w[0].tolist(),
              "projected_gravity": robot.data.projected_gravity_b[0].tolist(),
              "speed_error_xy": error.tolist(),
              "wheel_bottom_above_pit_floor": (
                centers[0, :, 2] - extents[0, :, 2] - origins[0, 2]
              ).tolist(),
              "joint_pos": robot.data.joint_pos[0].tolist(),
              "joint_vel": robot.data.joint_vel[0].tolist(),
              "position_target": target[0].tolist(),
              "velocity_target": robot.data.joint_vel_target[0].tolist(),
              "target_over_hard_limit": over[0].tolist(),
              "actuator_force": force.tolist(),
              "torque_saturated": saturated.tolist(),
              "wheel_bottom_above_rim": (
                centers[0, :, 2] - extents[0, :, 2] - rim_z
              ).tolist(),
              "contact_force": contact.force[0].tolist(),
              "contact_normal": contact.normal[0].tolist(),
              "contact_found": contact.found[0].tolist(),
              "reward_raw": {k: rates[k] / weights[k] for k in rates},
              "reward_weighted_rate": rates,
              "reward_dt_contribution": {k: v * env.step_dt for k, v in rates.items()},
              "reward": float(reward[0]),
            }
            trace.write(json.dumps(row) + "\n")
            now = time.monotonic()
            # 只有当前步及轨迹写入完成才报进度；耗时包含仿真、GPU同步和诊断输出。
            if step == 0 or (step + 1) % 100 == 0 or now - last_progress >= 15:
              trace.flush()
              print(
                f"{label} step={step + 1}/{step_budget} "
                f"elapsed={now - started:.1f}s stable_pass={passed}",
                flush=True,
              )
              last_progress = now
            if done[0]:
              ended = True
              break
          displacement = float(
            torch.linalg.vector_norm(robot.data.root_link_pos_w[0, :2] - origins[0, :2])
          )
          samples.append(
            {
              "episode": episode,
              "completed": ended,
              "steps": step + 1,
              "passed": passed if ended else False,
              **reached,
              "fell_over": bool(env.termination_manager.get_term("fell_over")[0]),
              "failed": bool(env.termination_manager.terminated[0]),
              "truncated": bool(env.termination_manager.time_outs[0]),
              "stalled": ended
              and not passed
              and displacement
              < 0.5 * abs(float(command[0, 0])) * (step + 1) * env.step_dt,
              "speed_rmse": (sum(speed_errors) / len(speed_errors)) ** 0.5,
            }
          )
          trace.flush()
          print(
            f"{label} finished steps={step + 1} completed={ended} "
            f"passed={samples[-1]['passed']} elapsed={time.monotonic() - started:.1f}s",
            flush=True,
          )
      complete = [s for s in samples if s["completed"]]
      report["heights"].append(
        {
          "height_m": height,
          "requested_n": cfg.episodes,
          "n": len(complete),
          "passed": sum(s["passed"] for s in complete),
          "pass_rate": sum(s["passed"] for s in complete) / len(complete)
          if complete
          else None,
          "samples": samples,
        }
      )
    finally:
      env.close()
  (cfg.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
  print(f"[eval] Report saved: {cfg.output / 'report.json'}", flush=True)
  return report


if __name__ == "__main__":
  run(tyro.cli(EvalConfig))
