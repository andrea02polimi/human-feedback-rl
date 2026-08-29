"""SB3 callbacks: episode metrics, and dumping at a fixed interval.

SB3 flushes its logger on its own schedule, which does not line up with the
iterations of reward learning; these make the two agree.
"""

from stable_baselines3.common.callbacks import BaseCallback


class FixedIntervalDumpCallback(BaseCallback):
    """Dump the SB3 logger every ``dump_interval`` environment timesteps.

    Off-policy algorithms (SAC) dump on episode ends, so runs with different
    seeds log at different ``total_timesteps`` values and W&B grouped panels
    with a custom x-axis cannot aggregate them into min/max bands. Timesteps
    advance in identical ``n_envs`` increments for every seed, so dumping on a
    fixed timestep grid puts every seed's points on the same x values.

    Use together with ``learn(log_interval=None)`` so the episode-based dump
    is disabled; otherwise the off-grid episode dumps reintroduce misaligned
    points.
    """

    def __init__(self, dump_interval: int):
        super().__init__()
        if dump_interval <= 0:
            raise ValueError(f"dump_interval must be positive, got {dump_interval}")
        self.dump_interval = dump_interval
        self._last_bucket = None

    def _on_training_start(self) -> None:
        if self._last_bucket is None:
            self._last_bucket = self.num_timesteps // self.dump_interval

    def _on_step(self) -> bool:
        bucket = self.num_timesteps // self.dump_interval
        if bucket > self._last_bucket:
            self._last_bucket = bucket
            # dump_logs() is the public name; _dump_logs() the pre-2.7 one.
            getattr(self.model, "dump_logs", getattr(self.model, "_dump_logs", None))()
        return True


class CustomLoggingCallback(BaseCallback):
    """Logs per-episode metrics emitted by the sumo-rl-ego environment.

    Reads ``info["ego_status"]`` (a plain string, see
    :mod:`human_feedback_rl.common.status`) plus the episode metrics that the
    environment attaches to ``info`` on episode end.
    """

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]

        for i, done in enumerate(dones):
            if not done:
                continue

            info = infos[i]

            ep = info.get("metrics", {}).get("episode", {})
            for key, value in ep.items():
                self.logger.record_mean(key, value)

            ep_return = info.get("episode", {}).get("r", None)
            if ep_return is not None:
                self.logger.record_mean("rewards/ep_env_return", float(ep_return))

            self.logger.record_mean("performance/ep_length", float(info.get("step", 0)))
            self.logger.record_mean("performance/ep_duration", float(info.get("sim_time", 0.0)))

            ego_status = info.get("ego_status", "running")
            self.logger.record_mean("event_rate/collisions", int(ego_status == "collided"))
            self.logger.record_mean("event_rate/off_road", int(ego_status == "offroad"))
            self.logger.record_mean("event_rate/timeouts", int(ego_status == "timeout"))
            self.logger.record_mean("event_rate/successes", int(ego_status == "arrived"))

        return True
