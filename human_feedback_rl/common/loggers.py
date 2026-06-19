import sys

import wandb
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter
from stable_baselines3.common.logger import Logger as SB3Logger


HUMAN_OUTPUT_MAX_LENGTH = 96


def make_human_output_format(stream=sys.stdout) -> HumanOutputFormat:
    """Create a text logger wide enough for nested diagnostic metric names."""
    return HumanOutputFormat(stream, max_length=HUMAN_OUTPUT_MAX_LENGTH)


ITERATION_METRIC_PREFIXES = (
    "reward/",
    "reward_val/",
    "replay_relabel_debug/",
    "rollout/",
    "time/",
)

# These values remain in W&B history and can still be added to custom panels.
# Hiding them only keeps the automatically generated workspace focused.
HIDDEN_METRIC_PREFIXES = (
    "agent/action_rate/",
    "agent/time/",
)

HIDDEN_METRICS = (
    "reward/grad_norm_max",
    "reward/return_abs_mean",
    "reward/return_min",
    "reward/return_max",
    "replay_relabel_debug/sample_size",
    "replay_relabel_debug/stored_reward_mean",
    "replay_relabel_debug/current_reward_mean",
    "replay_relabel_debug/stored_reward_std",
    "replay_relabel_debug/current_reward_std",
    "replay_relabel_debug/delta_mean",
    "replay_relabel_debug/delta_std",
    "replay_relabel_debug/relabel_enabled",
    "replay_relabel_debug/critic_uses_current_reward",
    "time/loggings",
)

VALIDATION_DATASETS = ("current_rollout", "debug_dataset")
VALIDATION_STAGES = ("pre_update", "post_update")
HIDDEN_VALIDATION_SUFFIXES = (
    "reward_min",
    "reward_max",
    "reward_running",
    "reward_arrived",
    "reward_collided",
    "reward_offroad",
    "reward_timeout",
    "ensemble_std_running",
    "spearman_returns_defined",
)


def configure_wandb_metrics(run) -> None:
    """Assign semantic X axes and trim W&B's automatically generated panels."""
    run.define_metric("iterations", hidden=True, summary="max")
    run.define_metric(
        "agent/time/total_timesteps", hidden=True, summary="max"
    )
    run.define_metric(
        "agent/*", step_metric="agent/time/total_timesteps", step_sync=True
    )
    for prefix in ITERATION_METRIC_PREFIXES:
        run.define_metric(
            f"{prefix}*", step_metric="iterations", step_sync=True
        )

    for prefix in HIDDEN_METRIC_PREFIXES:
        run.define_metric(f"{prefix}*", hidden=True)
    for metric in HIDDEN_METRICS:
        run.define_metric(metric, hidden=True)
    for dataset in VALIDATION_DATASETS:
        for stage in VALIDATION_STAGES:
            prefix = f"reward_val/{dataset}/{stage}"
            for suffix in HIDDEN_VALIDATION_SUFFIXES:
                run.define_metric(f"{prefix}/{suffix}", hidden=True)


class Logger(SB3Logger):
    """SB3 Logger extended with record_sum (accumulates values before dump)."""

    def warn(self, msg: str) -> None:
        super().warn(f"\033[33m{msg}\033[0m")

    def record_sum(self, key, value, exclude=None):
        if key in self.name_to_value:
            self.name_to_value[key] += value
            self.name_to_count[key] += 1
        else:
            self.name_to_value[key] = value
            self.name_to_count[key] = 1
        self.name_to_excluded[key] = exclude

    def dump_keys(self, keys, step=0):
        """Write and clear only the given keys, leaving all other buffered data intact."""
        key_values   = {k: self.name_to_value[k]   for k in keys if k in self.name_to_value}
        key_excluded = {k: self.name_to_excluded[k] for k in keys if k in self.name_to_excluded}
        for fmt in self.output_formats:
            fmt.write(key_values, key_excluded, step)
        for k in keys:
            self.name_to_value.pop(k, None)
            self.name_to_excluded.pop(k, None)
            self.name_to_count.pop(k, None)


# ── WandB writer ──────────────────────────────────────────────────────────────

class WandbWriter(KVWriter):
    def write(self, key_values: dict, key_excluded: dict, step: int = 0) -> None:
        metrics = {k: v for k, v in key_values.items()
                   if key_excluded.get(k) is None or "wandb" not in key_excluded[k]}
        if metrics:
            wandb.log(metrics)

    def close(self) -> None:
        wandb.finish()


# ── prefix wrapper ────────────────────────────────────────────────────────────

class PrefixedLogger:
    """Wraps any SB3-compatible logger and prepends a prefix to every key.

    Records are buffered locally. dump() pushes the buffer into the parent
    logger and then flushes only those keys to all output formats, leaving
    any data accumulated by other components in the parent untouched.
    """

    def __init__(self, logger, prefix: str):
        self._logger = logger
        self.prefix  = prefix.rstrip("/")
        self._data: list = []  # (method_name, key, value, exclude)

    def _pk(self, key: str) -> str:
        return f"{self.prefix}/{key}"

    def record(self, key, value, exclude=None):
        self._data.append(("record", self._pk(key), value, exclude))

    def record_mean(self, key, value, exclude=None):
        self._data.append(("record_mean", self._pk(key), value, exclude))

    def record_sum(self, key, value, exclude=None):
        self._data.append(("record_sum", self._pk(key), value, exclude))

    def dump(self, step=0):
        keys = [key for _, key, _, _ in self._data]
        for method, key, value, exclude in self._data:
            getattr(self._logger, method)(key, value, exclude)
        self._data.clear()
        self._logger.dump_keys(keys, step)

    def __getattr__(self, name):
        return getattr(self._logger, name)


# ── exclude-format wrapper ────────────────────────────────────────────────────

class ExcludeFormatLogger:
    """Wraps a logger and always excludes a given format from every record call."""

    def __init__(self, logger, exclude: str):
        self._logger  = logger
        self._exclude = exclude

    def _merge(self, exclude):
        if exclude is None:
            return self._exclude
        if isinstance(exclude, str):
            return (exclude, self._exclude)
        return tuple(set(exclude) | {self._exclude})

    def record(self, key, value, exclude=None):
        self._logger.record(key, value, self._merge(exclude))

    def record_mean(self, key, value, exclude=None):
        self._logger.record_mean(key, value, self._merge(exclude))

    def record_sum(self, key, value, exclude=None):
        self._logger.record_sum(key, value, self._merge(exclude))

    def dump(self, step=0):
        self._logger.dump(step)

    def __getattr__(self, name):
        return getattr(self._logger, name)


# ── null logger ───────────────────────────────────────────────────────────────

class NullLogger:
    """Discards all calls. Used as a fallback when no logger is injected."""
    def record(self, key, value, exclude=None): pass
    def record_mean(self, key, value, exclude=None): pass
    def record_sum(self, key, value, exclude=None): pass
    def dump(self, step=0): pass
    def log(self, *args, **kwargs): pass
    def warn(self, *args, **kwargs): pass
