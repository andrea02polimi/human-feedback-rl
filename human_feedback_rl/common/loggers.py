from stable_baselines3.common.logger import HumanOutputFormat
from stable_baselines3.common.logger import KVWriter
import sys
import wandb


class PrefixedLogger:
    """Wraps an SB3 logger and prepends a prefix to every recorded key."""

    def __init__(self, logger, prefix: str):
        self._logger = logger
        self.prefix = prefix

    def record(self, key, value, exclude=None):
        self._logger.record(f"{self.prefix}/{key}", value, exclude)

    def record_mean(self, key, value, exclude=None):
        self._logger.record_mean(f"{self.prefix}/{key}", value, exclude)

    def dump(self, step=0):
        self._logger.dump(step)

    def __getattr__(self, name):
        return getattr(self._logger, name)


class ExcludedLogger:
    """Wraps an SB3 logger and injects a fixed exclude tag into every record call."""

    def __init__(self, logger, exclude):
        self._logger = logger
        self._exclude = exclude

    def _merge(self, exclude):
        if exclude is None:
            return self._exclude
        a = (self._exclude,) if isinstance(self._exclude, str) else tuple(self._exclude)
        b = (exclude,) if isinstance(exclude, str) else tuple(exclude)
        return tuple(set(a) | set(b))

    def record(self, key, value, exclude=None):
        self._logger.record(key, value, self._merge(exclude))

    def record_mean(self, key, value, exclude=None):
        self._logger.record_mean(key, value, self._merge(exclude))

    def dump(self, step=0):
        self._logger.dump(step)

    def __getattr__(self, name):
        return getattr(self._logger, name)


class FilteredHumanOutput(KVWriter):
    """HumanOutputFormat that respects SB3 exclude flags (keys excluded from 'stdout' are hidden)."""

    def __init__(self):
        self._inner = HumanOutputFormat(sys.stdout)

    def write(self, key_values: dict, key_excluded: dict, step: int = 0) -> None:
        visible = {
            k for k in key_values
            if key_excluded.get(k) is None or "stdout" not in key_excluded[k]
        }
        if visible:
            self._inner.write(key_values, key_excluded, step)

    def close(self) -> None:
        self._inner.close()


class WandbWriter(KVWriter):
    def write(self, key_values: dict, key_excluded: dict, step: int = 0):
        metrics = {}
        for k, v in key_values.items():
            excluded = key_excluded.get(k)
            if excluded is not None:
                formats = (excluded,) if isinstance(excluded, str) else excluded
                if "wandb" in formats:
                    continue
            metrics[k] = v
        wandb.log(metrics)

    def close(self):
        wandb.finish()
