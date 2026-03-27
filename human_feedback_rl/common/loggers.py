import wandb
import numpy as np


class UnifiedLogger:
    def __init__(self):
        self.data = {}

    def record(self, key, value):
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(value)


    def dump(self, step):
        log_dict = {}

        for key, values in self.data.items():
            mean_value = float(np.mean(values))
            log_dict[key] = mean_value

        if wandb.run is not None:
            wandb.log(log_dict)

        self.data.clear()
        

class PrefixLogger:
    def __init__(self, unified_logger, prefix=None):
        self.unified_logger = unified_logger
        self.prefix = prefix

    def record(self, key, value, *args, **kwargs):
        if self.prefix:
            self.unified_logger.record(f"{self.prefix}/{key}", value)
        else:
            self.unified_logger.record(key, value)

    def dump(self, step=0):
        self.unified_logger.dump(step)
