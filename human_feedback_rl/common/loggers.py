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

# Keep the automatically generated workspace deliberately small. All other
# values are still written to W&B history and can be added to custom panels.
# A maxent_2 run logs 15 common metrics plus its two loss-specific diagnostics:
# 17 automatic plots, which can be grouped into eight custom workspace panels.
VISIBLE_METRICS = (
    # Policy outcome and environment performance.
    "agent/event_rate/successes",
    "agent/event_rate/collisions",
    "agent/event_rate/off_road",
    "agent/rewards/ep_fast_return",
    # SAC stability.
    "agent/train/critic_loss",
    "agent/train/ent_coef",
    # Reward-model stability.
    "reward/loss",
    "reward/weight_norm",
    # Replay reward non-stationarity.
    "replay_relabel_debug/staleness_ratio",
    "replay_relabel_debug/sign_flip_frac",
    "replay_relabel_debug/stored_current_corr",
    # Generalization: recent policy distribution versus fixed debug data.
    "reward_val/current_rollout/post_update/spearman_returns",
    "reward_val/debug_dataset/post_update/spearman_returns",
    "reward_val/current_rollout/post_update/gap_arrived_collided",
    "reward_val/debug_dataset/post_update/gap_arrived_collided",
    # Historical MaxEnt loss diagnostics.
    "reward/maxent_effective_sample_fraction",
    "reward/maxent_top1_softmax_weight",
    "reward/maxent2_effective_sample_fraction",
    "reward/maxent2_expert_softmax_mass",
    # Corrected MaxEnt loss diagnostics.
    "reward/maxent_corrected_effective_sample_fraction",
    "reward/maxent_corrected_top1_softmax_weight",
    # Demo loss diagnostics. Only the pair matching the configured loss is
    # emitted, so unused definitions do not create empty panels.
    "reward/demo_margin",
    "reward/demo_scale_std",
    "reward/demo_corrected_margin",
    "reward/demo_corrected_scale_std",
)
VISIBLE_METRIC_SET = frozenset(VISIBLE_METRICS)

SEMANTIC_METRIC_PREFIXES = ("agent/", *ITERATION_METRIC_PREFIXES)
PREDEFINED_METRICS = frozenset({
    "iterations",
    "agent/time/total_timesteps",
    *VISIBLE_METRICS,
})


def configure_wandb_metrics(run) -> None:
    """Assign semantic X axes and expose only the core automatic panels."""
    run.define_metric("iterations", hidden=True, summary="max")
    run.define_metric(
        "agent/time/total_timesteps", hidden=True, summary="max"
    )
    run.define_metric(
        "agent/*",
        step_metric="agent/time/total_timesteps",
        step_sync=True,
        hidden=True,
    )
    for prefix in ITERATION_METRIC_PREFIXES:
        run.define_metric(
            f"{prefix}*",
            step_metric="iterations",
            step_sync=True,
            hidden=True,
        )

    # Exact definitions take precedence over the hidden prefix globs above.
    # Do not pass hidden=False: W&B represents visibility by the absence of the
    # hidden option in the exact metric definition.
    for metric in VISIBLE_METRICS:
        is_agent_metric = metric.startswith("agent/")
        run.define_metric(
            metric,
            step_metric=(
                "agent/time/total_timesteps" if is_agent_metric else "iterations"
            ),
            step_sync=True,
        )


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
    def __init__(self) -> None:
        # W&B 0.27 creates wildcard-derived metric records with defined=False.
        # The automatic workspace may ignore their hidden flag, so secondary
        # metrics are defined explicitly before their first log call.
        self._defined_metrics = set(PREDEFINED_METRICS)

    @staticmethod
    def _step_metric(metric: str) -> str:
        return (
            "agent/time/total_timesteps"
            if metric.startswith("agent/")
            else "iterations"
        )

    def _define_secondary_metrics(self, metrics) -> None:
        run = wandb.run
        if run is None:
            return
        for metric in metrics:
            if metric in self._defined_metrics or not metric.startswith(
                SEMANTIC_METRIC_PREFIXES
            ):
                continue
            run.define_metric(
                metric,
                step_metric=self._step_metric(metric),
                step_sync=True,
                hidden=metric not in VISIBLE_METRIC_SET,
            )
            self._defined_metrics.add(metric)

    def write(self, key_values: dict, key_excluded: dict, step: int = 0) -> None:
        metrics = {k: v for k, v in key_values.items()
                   if key_excluded.get(k) is None or "wandb" not in key_excluded[k]}
        if metrics:
            self._define_secondary_metrics(metrics)
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
