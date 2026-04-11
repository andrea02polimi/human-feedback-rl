import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper

from .reward_model import EnsembleRewardModel


class EnvRewardWrapper(VecEnvWrapper):
    """
    VecEnv wrapper that replaces the true environment reward with the reward
    predicted by the learned ensemble model.

    The original reward is preserved in info['true_reward'] so that it can
    still be accessed for evaluation or for generating synthetic preferences.
    """

    def __init__(self, venv, reward_model: EnsembleRewardModel):
        super().__init__(venv)
        self.reward_model = reward_model
        self._last_action: np.ndarray | None = None

    def step_async(self, actions: np.ndarray) -> None:
        self._last_action = actions
        self.venv.step_async(actions)

    def step_wait(self):
        obs, true_rewards, dones, infos = self.venv.step_wait()

        assert self._last_action is not None, (
            "step_async must be called before step_wait"
        )
        predicted_rewards = self.reward_model.predict_reward(obs, self._last_action)

        for i, info in enumerate(infos):
            info["true_reward"] = float(true_rewards[i])

        return obs, predicted_rewards, dones, infos

    def reset(self) -> np.ndarray:
        return self.venv.reset()