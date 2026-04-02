import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper, VecEnv

from .reward_model import EnsembleRewardModel


class _RunningMeanStd:
    """Welford's online algorithm for running mean and variance."""

    def __init__(self):
        self.mean = 0.0
        self.var = 0.0
        self.count = 0

    def update(self, values: np.ndarray) -> None:
        for x in values.flat:
            self.count += 1
            delta = x - self.mean
            self.mean += delta / self.count
            self.var += (x - self.mean) * delta  # M2

    @property
    def std(self) -> float:
        if self.count < 2:
            return 1.0
        return float(np.sqrt(self.var / (self.count - 1))) or 1.0


# ---------------------------------------------------------------------------
# Environment reward wrapper
# ---------------------------------------------------------------------------

class EnvRewardWrapper(VecEnvWrapper):
    """
    VecEnvWrapper that replaces environment rewards with EnsembleRewardModel
    predictions. Uses the pre-step observation for reward prediction to align
    with how the reward predictor was trained on (obs_t, a_t) pairs.

    Rewards are normalized to mean 0 / std 1 via a running estimate before
    being passed to the agent (Christiano et al. 2017, Section 2.2).
    """

    def __init__(self, venv: VecEnv, reward_model: EnsembleRewardModel):
        super().__init__(venv)
        self.reward_model = reward_model
        self._obs: np.ndarray | None = None
        self._actions: np.ndarray | None = None
        self._reward_stats = _RunningMeanStd()

    def reset_stats(self):
        if self._reward_stats.count > 1:
            new_count = self._reward_stats.count // 2
            self._reward_stats.var = self._reward_stats.var * (new_count - 1) / (self._reward_stats.count - 1)
            self._reward_stats.count = new_count

    def reset(self):
        obs = self.venv.reset()
        self._obs = np.asarray(obs, dtype=np.float32)
        return obs

    def step_async(self, actions):
        self._actions = np.asarray(actions)
        self.venv.step_async(actions)

    def step_wait(self):
        obs, _env_rewards, dones, infos = self.venv.step_wait()

        if self._obs is not None and self._actions is not None:
            rewards = self.reward_model.predict(self._obs, self._actions)
            self._reward_stats.update(rewards)
            rewards = ((rewards - self._reward_stats.mean) / self._reward_stats.std).astype(np.float32)
        else:
            rewards = np.zeros(len(obs), dtype=np.float32)

        self._obs = np.asarray(obs, dtype=np.float32)
        return obs, rewards, dones, infos