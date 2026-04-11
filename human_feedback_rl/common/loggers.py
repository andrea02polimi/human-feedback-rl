import sys
from typing import Any, Dict, Optional

import wandb
from stable_baselines3.common.logger import HumanOutputFormat
from stable_baselines3.common.logger import Logger as SB3Logger


def setup_wandb_axes() -> None:
    """
    Define per-group custom x-axes in wandb.

    - agent/*        → x-axis: agent/total_timesteps
    - reward_model/* → x-axis: reward_model/global_epochs
    - rollout/*      → x-axis: rollout/iteration
    """
    if wandb.run is None:
        return
    wandb.define_metric("agent/total_timesteps")
    wandb.define_metric("agent/*", step_metric="agent/total_timesteps")
    wandb.define_metric("reward_model/global_epochs")
    wandb.define_metric("reward_model/*", step_metric="reward_model/global_epochs")
    wandb.define_metric("rollout/iteration")
    wandb.define_metric("rollout/*", step_metric="rollout/iteration")


class SB3MetricsLogger(SB3Logger):
    """
    SB3 Logger subclass that captures SAC training metrics for wandb.

    Overrides ``dump()`` to snapshot ``name_to_value`` before the parent
    clears it.  Call ``flush_to_wandb()`` after ``agent.train()`` to send
    the captured metrics to wandb under the ``agent/`` prefix with
    ``agent/total_timesteps`` as x-axis.
    """

    def __init__(self) -> None:
        super().__init__(folder=None, output_formats=[HumanOutputFormat(sys.stdout)])
        self._captured: Dict[str, Any] = {}
        self._captured_step: int = 0

    def record(self, key: str, value: Any, exclude: Optional[str] = None) -> None:
        # Capture each metric as it is recorded by SAC.train().
        # This works even in SB3 versions where train() does not call dump().
        self._captured[key] = value
        super().record(key, value, exclude=exclude)

    def dump(self, step: int = 0) -> None:
        # If SAC does call dump(), store the step for the x-axis.
        self._captured_step = step
        super().dump(step=step)

    def flush_to_wandb(self, fallback_step: int = 0) -> None:
        """Forward captured metrics to wandb and clear the buffer."""
        if not self._captured or wandb.run is None:
            self._captured.clear()
            return
        metrics: Dict[str, float] = {}
        for k, v in self._captured.items():
            try:
                metrics[f"agent/{k}"] = float(v)
            except (TypeError, ValueError):
                pass
        if metrics:
            step = self._captured_step if self._captured_step > 0 else fallback_step
            metrics["agent/total_timesteps"] = step
            wandb.log(metrics)
        self._captured.clear()
        self._captured_step = 0


class UnifiedLogger:
    """Logs scalar metrics to wandb (when active) and stdout."""

    def __init__(self, use_wandb: bool = True):
        self._use_wandb = use_wandb

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if self._use_wandb and wandb.run is not None:
            wandb.log(metrics, step=step)

    def prefix(self, prefix: str) -> "PrefixLogger":
        return PrefixLogger(prefix=prefix, inner=self)


class PrefixLogger:
    """Prepends a fixed prefix to every metric key before delegating to UnifiedLogger."""

    def __init__(self, prefix: str, inner: UnifiedLogger):
        self.prefix = prefix
        self.inner = inner

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        prefixed = {f"{self.prefix}/{k}": v for k, v in metrics.items()}
        self.inner.log(prefixed, step=step)