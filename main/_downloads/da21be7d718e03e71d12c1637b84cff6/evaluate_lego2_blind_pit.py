"""用固定坑深回放策略、记录轨迹和成功率，不更新PPO。

按reward_version选择v1/v2的eval配置，按checkpoint加载策略，并用PitProgress判断阶段。
逐回合保存真实初态，供compare_lego2_reward_v2.py核对两个模型是否公平配对。
报告同时记录速度、非轮接触和机身净空；人工截断片段不算完整评估样本。
默认仍用v1；关闭课程本身不固定坑深，实际高度由heights参数生成。
"""

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions.actions import JointPositionAction
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.sensor import ContactSensor
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.config.lego2.blind_pit_env_cfg import lego2_blind_pit_eval_cfg
from mjlab.tasks.velocity.config.lego2.blind_pit_metrics import PitProgress
from mjlab.tasks.velocity.config.lego2.blind_pit_reward_v2_env_cfg import (
  lego2_blind_pit_reward_v2_eval_cfg,
)
from mjlab.tasks.velocity.config.lego2.blind_pit_rewards import _level_ground_clearance
from mjlab.utils.os import dump_yaml


@dataclass
class EvalConfig:
  """一次固定高度评估的设置，决定用哪个模型、坑深、速度及多少组初态。"""

  output: Path
  checkpoint: Path | None = None
  heights: tuple[float, ...] = (0.03, 0.06, 0.09)  # 固定真实坑深，单位米。
  episodes: int = 10  # 每个高度请求的回合数，完整结束的才进入成功率分母。
  seed: int = 7  # 建立环境时的随机种子。
  device: str = "cpu"
  forward_speed: float | None = None  # 可固定前进速度；None沿用0.3～0.6米/秒抽样。
  # 每回合用base+回合编号重新设种子；None保留旧的连续随机序列。
  paired_seed_base: int | None = None
  # 软件快速检查用的步数上限；没等到自然结束的片段不计入n。
  max_steps: int | None = None
  reward_version: Literal["v1", "v2"] = "v1"  # 选择评分配置，不会把旧权重训练成v2。


def sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def run(cfg: EvalConfig) -> dict:
  """逐高度、逐回合运行模型，保存动作/接触/奖励轨迹和完整回合结果。

  先保存实际reset状态，再统计通过；max_steps提前截断的片段不进入成功率分母。
  """
  factory = (
    lego2_blind_pit_reward_v2_eval_cfg
    if cfg.reward_version == "v2"
    else lego2_blind_pit_eval_cfg
  )
  task = "Mjlab-Blind-Pit-Lego2" + ("-RewardV2" if cfg.reward_version == "v2" else "")
  if cfg.episodes < 1 or (cfg.max_steps is not None and cfg.max_steps < 1):
    raise ValueError("episodes and max_steps must be positive")
  if len(set(cfg.heights)) != len(cfg.heights):
    raise ValueError("Duplicate heights are not supported")
  if cfg.paired_seed_base is not None and cfg.paired_seed_base < 0:
    raise ValueError("paired_seed_base must be nonnegative")
  for height in cfg.heights:
    factory(height, cfg.forward_speed)
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
  report: dict[str, Any] = {
    "mode": "policy" if cfg.checkpoint else "zero_action_diagnostic",
    "reward_version": cfg.reward_version,
    "seed": cfg.seed,
    "fixed_forward_speed_m_s": cfg.forward_speed,
    "paired_seed_base": cfg.paired_seed_base,
    "device": cfg.device,
    "checkpoint": str(cfg.checkpoint.resolve()) if cfg.checkpoint else None,
    "checkpoint_sha256": sha256(cfg.checkpoint) if cfg.checkpoint else None,
    "source_sha256": {str(p.relative_to(root)): sha256(p) for p in sources},
    "definition": "four wheels rim-supported, tilt<45deg continuously >=0.5s",
    "heights": [],
  }
  for height in cfg.heights:
    print(f"Fixed pit depth: {height:.6f} m; requested episodes: {cfg.episodes}")
    env_cfg = factory(height, cfg.forward_speed)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = cfg.seed
    dump_yaml(cfg.output / f"env_{height:.6f}.yaml", asdict(env_cfg))
    env = ManagerBasedRlEnv(env_cfg, device=cfg.device)
    try:
      agent_cfg = load_rl_cfg(task)
      dump_yaml(cfg.output / "agent.yaml", asdict(agent_cfg))
      wrapper = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
      policy = None
      if cfg.checkpoint is not None:
        runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
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
      nonfoot = env.scene["undesired_ground_contact"]
      assert isinstance(nonfoot, ContactSensor)
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
          episode_seed = None
          if cfg.paired_seed_base is not None:
            episode_seed = cfg.paired_seed_base + episode
            wrapper.seed(episode_seed)
          obs, _ = wrapper.reset()
          progress.reset()
          # 保存真实reset帧；跨速度比较出生/历史，而非仅凭seed宣称已配对。
          initial_state = {
            "root_pose_w": robot.data.root_link_pose_w[0].tolist(),
            "root_lin_vel_w": robot.data.root_link_lin_vel_w[0].tolist(),
            "root_ang_vel_w": robot.data.root_link_ang_vel_w[0].tolist(),
            "joint_pos": robot.data.joint_pos[0].tolist(),
            "joint_vel": robot.data.joint_vel[0].tolist(),
            "actor_observation": obs["actor"][0].tolist(),
          }
          reached = dict.fromkeys(
            ("root_out", "front_up", "rear_up", "complete"), False
          )
          speed_errors = []
          quality = dict.fromkeys(
            (
              "eligible_frames",
              "low_clearance_frames",
              "nonfoot_frames",
              "post_complete_frames",
              "post_complete_nonfoot_frames",
              "post_complete_low_clearance_frames",
            ),
            0,
          )
          post_heights = []
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
            nonfoot_data = nonfoot.data
            assert contact.found is not None
            assert contact.force is not None and contact.normal is not None
            assert nonfoot_data.found is not None and nonfoot_data.force is not None
            clearance, eligible = _level_ground_clearance(env, "terrain_scan", 0.03)
            low = bool(eligible[0] & (clearance[0] < 0.34))
            nf = bool(
              (
                (nonfoot_data.found[0] > 0) & (nonfoot_data.force[0].norm(dim=-1) > 1)
              ).any()
            )
            quality["eligible_frames"] += int(eligible[0])
            quality["low_clearance_frames"] += int(low)
            quality["nonfoot_frames"] += int(nf)
            if reached["complete"]:
              quality["post_complete_frames"] += 1
              quality["post_complete_nonfoot_frames"] += int(nf)
              quality["post_complete_low_clearance_frames"] += int(low)
              post_heights.append(float(robot.data.root_link_pos_w[0, 2] - rim_z))
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
              "contact_pos": contact.pos[0].tolist()
              if contact.pos is not None
              else None,
              "continuous_support_s": float(progress.hold[0]),
              "base_clearance_m": float(clearance[0]) if eligible[0] else None,
              "base_clearance_eligible": bool(eligible[0]),
              "nonfoot_active": nf,
              "nonfoot_names": nonfoot.primary_names,
              "nonfoot_contact_found": nonfoot_data.found[0].tolist(),
              "nonfoot_contact_force": nonfoot_data.force[0].tolist(),
              "root_relative_xy": (
                robot.data.root_link_pos_w[0, :2] - origins[0, :2]
              ).tolist(),
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
              "reset_seed": episode_seed,
              "initial_state": initial_state,
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
              "quality": {
                **quality,
                "post_complete_min_root_rim_m": min(post_heights)
                if post_heights
                else None,
              },
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
