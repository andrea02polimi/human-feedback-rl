import wandb
import numpy as np
from stable_baselines3.common.logger import Logger as SB3Logger


class UnifiedLogger:
    """
    Central metric store. Record values throughout an iteration, then call
    dump() once to average them and emit a single wandb.log() call.

    Usage::

        logger = UnifiedLogger()
        logger.record("loss", 0.4)
        logger.record("loss", 0.3)  # multiple records are averaged
        logger.dump()               # logs {"loss": 0.35} to wandb
    """
    def __init__(self):
        self.data = {}

    def record(self, key, value):
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(value)


    def dump(self, step=None):
        log_dict = {}

        for key, values in self.data.items():
            mean_value = float(np.mean(values))
            log_dict[key] = mean_value

        if wandb.run is not None:
            wandb.log(log_dict)
            # print(f"log_dict: {log_dict}")

        self.data.clear()
        

class PrefixLogger:
    """
    Thin wrapper around UnifiedLogger that prepends a prefix to every key
    and optionally remaps key names before forwarding to the unified store.

    Usage::

        log = PrefixLogger(logger, prefix="reward_model", key_map={"loss": "train_loss"})
        log.record("loss", 0.5)   # stored as "reward_model/train_loss"
    """
    def __init__(self, unified_logger, prefix=None, key_map=None):
        self.unified_logger = unified_logger
        self.prefix = prefix
        self.key_map = key_map or {}

    def record(self, key, value, *args, **kwargs):
        key = self.key_map.get(key, key)
        if self.prefix:
            self.unified_logger.record(f"{self.prefix}/{key}", value)
        else:
            self.unified_logger.record(key, value)

    def dump(self, *args, **kwargs):
        self.unified_logger.dump()

