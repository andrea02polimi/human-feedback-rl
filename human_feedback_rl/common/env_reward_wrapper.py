from . import *
import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper, VecEnv


# ---------------------------------------------------------------------------
# Environment reward wrapper
# ---------------------------------------------------------------------------

class EnvRewardWrapper(VecEnvWrapper):
    """
    VecEnvWrapper that replaces environment rewards with EnsembleRewardModel
    predictions. Uses the pre-step observation for reward prediction to align
    with how the reward predictor was trained on (obs_t, a_t) pairs.
    """

    def __init__(self, venv: VecEnv, reward_model: EnsembleRewardModel):
        super().__init__(venv)
        self.reward_model = reward_model
        self._current_obs: np.ndarray | None = None
        self._last_actions: np.ndarray | None = None

    def reset(self):
        obs = self.venv.reset()
        self._current_obs = np.asarray(obs, dtype=np.float32)
        return obs

    def step_async(self, actions):
        self._last_actions = np.asarray(actions)
        self.venv.step_async(actions)

    def step_wait(self):
        obs, _env_rewards, dones, infos = self.venv.step_wait()

        if self._current_obs is not None and self._last_actions is not None:
            rewards = self.reward_model.predict(
                self._current_obs, self._last_actions
            ).astype(np.float32)
        else:
            rewards = np.zeros(len(obs), dtype=np.float32)

        self._current_obs = np.asarray(obs, dtype=np.float32)
        return obs, rewards, dones, infos