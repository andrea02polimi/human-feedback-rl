from stable_baselines3.common.callbacks import BaseCallback

from sumo_gym_ego import EgoStatus


class CustomLoggingCallback(BaseCallback):
    """
    SB3 callback that records per-episode driving metrics.

    If main_logger is provided (a MainLogger instance), metrics are written there
    with the canonical env/* prefix and reach WandB at the outer iteration dump.
    Otherwise falls back to the SB3 internal logger (stdout/CSV only).
    """

    def __init__(self, main_logger=None):
        super().__init__()
        self.main_logger = main_logger

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]

        for i, done in enumerate(dones):
            if not done:
                continue

            info       = infos[i]
            ep         = info.get("metrics", {}).get("episode", {})
            ep_length  = info.get("step", 0)
            ep_duration = info.get("sim_time", 0.0)
            ego_status = info.get("ego_status", EgoStatus.RUNNING)

            if self.main_logger is not None:
                for key, value in ep.items():
                    self.main_logger.record(f"env/{key}", value)
                self.main_logger.record("env/ep_length",     float(ep_length))
                self.main_logger.record("env/ep_duration",   float(ep_duration))
                self.main_logger.record("env/collision_rate", float(ego_status == EgoStatus.COLLIDED.value))
                self.main_logger.record("env/off_road_rate",  float(ego_status == EgoStatus.OFF_ROAD.value))
                self.main_logger.record("env/timeout_rate",   float(ego_status == EgoStatus.TIMEOUT.value))
                self.main_logger.record("env/success_rate",   float(ego_status == EgoStatus.ARRIVED.value))
            else:
                # Fallback: original SB3-logger path (goes to stdout/CSV only)
                for key, value in ep.items():
                    self.logger.record(key, value)
                self.logger.record("performance/ep_length",   float(ep_length))
                self.logger.record("performance/ep_duration", float(ep_duration))
                self.logger.record("event_rate/collisions",   int(ego_status == EgoStatus.COLLIDED.value))
                self.logger.record("event_rate/off_road",     int(ego_status == EgoStatus.OFF_ROAD.value))
                self.logger.record("event_rate/timeouts",     int(ego_status == EgoStatus.TIMEOUT.value))
                self.logger.record("event_rate/successes",    int(ego_status == EgoStatus.ARRIVED.value))

        return True