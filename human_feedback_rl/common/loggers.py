import wandb
import numpy as np
from stable_baselines3.common.logger import Logger as SB3Logger, KVWriter


class MainLogger:
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
        
    def log(self, text):
        print(f"{text}")

    def warn(self, text):
        print(f"WARNING: {text}")


class PrefixWrapper:
    """
    Thin wrapper around UnifiedLogger that prepends a prefix to every key
    and optionally remaps key names before forwarding to the unified store.

    Usage::

        log = PrefixLogger(logger, prefix="reward_model", key_map={"loss": "train_loss"})
        log.record("loss", 0.5)   # stored as "reward_model/train_loss"
    """
    def __init__(self, main_logger, prefix=None, key_map=None):
        self.main_logger = main_logger
        self.prefix = prefix
        self.key_map = key_map or {}

    def record(self, key, value, *args, **kwargs):
        key = self.key_map.get(key, key)
        if self.prefix:
            self.main_logger.record(f"{self.prefix}/{key}", value)
        else:
            self.main_logger.record(key, value)

    def dump(self, *args, **kwargs):
        self.main_logger.dump()

    def log(self, text):
        self.main_logger.log(text)

    def warn(self, text):
        self.main_logger.warn(text)


class _SB3WandBBridge(KVWriter):
    """
    Intercepts SB3's logger.dump() to forward selected PPO training metrics
    into our MainLogger.  Does NOT flush MainLogger — the outer iteration loop
    is responsible for calling dump() at the right cadence.
    """

    _KEY_MAP = {
        "train/explained_variance": "ppo/explained_variance",
    }

    def __init__(self, main_logger: "MainLogger"):
        self.main_logger = main_logger

    def write(self, key_values, key_excluded, step: int = 0) -> None:
        for sb3_key, our_key in self._KEY_MAP.items():
            val = key_values.get(sb3_key)
            if val is not None:
                self.main_logger.record(our_key, float(val))

    def close(self) -> None:
        pass


def make_sb3_logger(main_logger: "MainLogger") -> SB3Logger:
    """Return an SB3 Logger that forwards PPO metrics to our MainLogger."""
    return SB3Logger(folder=None, output_formats=[_SB3WandBBridge(main_logger)])
