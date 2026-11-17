"""速度类任务的训练器，在通用MjlabOnPolicyRunner上补充模型导出。

任务注册文件把该类交给train/play；PPO采样和更新仍由rsl_rl训练器执行。
save先保存训练checkpoint，再尝试导出ONNX并附机器人相关信息。
当前v2独立训练入口也使用本类；网络和优化参数仍读rl_cfg.py。
"""

import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  """保存速度任务模型，并在训练checkpoint之外尝试生成可部署的ONNX文件。"""

  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    """先保存可续训checkpoint，再导出ONNX；导出失败会提示，训练仍继续。"""
    super().save(path, infos)
    policy_dir, filename, onnx_path = self._get_export_paths(path)
    try:
      self.export_policy_to_onnx(str(policy_dir), filename)
      run_name: str = (
        wandb.run.name
        if self.logger.logger_type in ("wandb", "WandbLogWriter") and wandb.run
        else "local"
      )  # type: ignore[assignment]
      metadata = get_base_metadata(self.env.unwrapped, run_name)
      attach_metadata_to_onnx(str(onnx_path), metadata)
      if (
        self.logger.logger_type in ("wandb", "WandbLogWriter")
        and self.cfg["upload_model"]
      ):
        wandb.save(str(onnx_path), base_path=str(policy_dir))
    except Exception as e:
      print(f"[WARN] ONNX export failed (training continues): {e}")
