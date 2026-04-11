from typing import Any, Dict, Optional

import wandb


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