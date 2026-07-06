from stable_baselines3.common.callbacks import BaseCallback


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
