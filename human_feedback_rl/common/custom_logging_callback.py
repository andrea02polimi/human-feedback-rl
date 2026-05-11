import numpy as np
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback

from sumo_gym_ego import EgoStatus



class CustomLoggingCallback(BaseCallback):

    def _on_step(self) -> bool:

        infos = self.locals["infos"]
        dones = self.locals["dones"]

        for i, done in enumerate(dones):

            if not done:
                continue

            info = infos[i]

            # --- extract episode metrics ---
            ep = info.get("metrics", {}).get("episode", {})

            for key, value in ep.items():
                self.logger.record_mean(key, value)

            # --- episode return ---
            ep_return = info.get("episode", {}).get("r", None)
            if ep_return is not None:
                self.logger.record_mean("rewards/ep_env_return", float(ep_return))

            # --- external metrics ---
            ep_length = info.get("step", 0)
            ep_duration = info.get("sim_time", 0.0)
            self.logger.record_mean("performance/ep_length", float(ep_length))
            self.logger.record_mean("performance/ep_duration", float(ep_duration))

            # --- events ---
            ego_status = info.get("ego_status", EgoStatus.RUNNING)
            self.logger.record_mean("event_rate/collisions", int(ego_status == EgoStatus.COLLIDED.value))
            self.logger.record_mean("event_rate/off_road", int(ego_status == EgoStatus.OFF_ROAD.value))
            self.logger.record_mean("event_rate/timeouts", int(ego_status == EgoStatus.TIMEOUT.value))
            self.logger.record_mean("event_rate/successes", int(ego_status == EgoStatus.ARRIVED.value))

        return True
    
