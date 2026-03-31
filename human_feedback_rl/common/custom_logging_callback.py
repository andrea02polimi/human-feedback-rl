import numpy as np
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback



class CustomLoggingCallback(BaseCallback):

    def __init__(self, window_size=100):
        super().__init__()
        self.W = window_size

        self.buffer = {
            "performance/speed_mean": deque(maxlen=self.W),
            "performance/return": deque(maxlen=self.W),
            "performance/length": deque(maxlen=self.W),
            "performance/duration": deque(maxlen=self.W),

            "action_rate/ss": deque(maxlen=self.W),
            "action_rate/lcl": deque(maxlen=self.W),
            "action_rate/lcr": deque(maxlen=self.W),
            "action_rate/acc": deque(maxlen=self.W),
            "action_rate/dec": deque(maxlen=self.W),

            "event_rate/collisions": deque(maxlen=self.W),
            "event_rate/off_road": deque(maxlen=self.W),
            "event_rate/timeouts": deque(maxlen=self.W),
            "event_rate/successes": deque(maxlen=self.W),
        }

    def _on_step(self) -> bool:

        infos = self.locals["infos"]
        dones = self.locals["dones"]

        for i, done in enumerate(dones):

            if not done:
                continue

            info = infos[i]

            # --- extract episode metrics ---
            ep = info.get("metrics", {}).get("episode", {})

            self.buffer["performance/speed_mean"].append(ep.get("performance/speed_mean", 0.0))
            self.buffer["performance/return"].append(ep.get("performance/return", 0.0))
            self.buffer["performance/length"].append(info.get("step", 0))
            self.buffer["performance/duration"].append(info.get("time", 0.0))

            self.buffer["action_rate/ss"].append(ep.get("action_rate/ss", 0.0))
            self.buffer["action_rate/lcl"].append(ep.get("action_rate/lcl", 0.0))
            self.buffer["action_rate/lcr"].append(ep.get("action_rate/lcr", 0.0))
            self.buffer["action_rate/acc"].append(ep.get("action_rate/acc", 0.0))
            self.buffer["action_rate/dec"].append(ep.get("action_rate/dec", 0.0))

            # --- events ---
            event = info.get("event", "running")

            self.buffer["event_rate/collisions"].append(int(event == "collided"))
            self.buffer["event_rate/off_road"].append(int(event == "off_road"))
            self.buffer["event_rate/timeouts"].append(int(event == "timeout"))
            self.buffer["event_rate/successes"].append(int(event == "arrived"))

        return True
    
    def _on_rollout_end(self) -> None:
        
        # --- log ---
        for k, v in self.buffer.items():
            if len(v) > 0:
                self.logger.record(k, np.mean(v))

        # self.logger.dump(self.num_timesteps)
