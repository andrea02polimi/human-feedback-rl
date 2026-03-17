"""
SB3-compatible VecEnv wrapper that replaces environment rewards with
reward-predictor predictions.

"""

import numpy as np

from stable_baselines3.common.vec_env import VecEnvWrapper

from human_feedback_rl.reward_models.ensemble import RewardPredictorEnsemble


class PredictedRewardVecWrapper(VecEnvWrapper):
    """
    VecEnv wrapper that replaces environment rewards with reward-predictor
    predictions.

    Before reward_predictor_ready_event is set, returns zero rewards so the
    policy never observes the environment reward (Christiano et al. §2.2).

    True environment rewards are stored in infos['true_reward'] so the
    SegmentCollectorCallback can label each Segment correctly.
    """

    def __init__(
        self,
        venv,
        reward_predictor: RewardPredictorEnsemble,
        reward_predictor_ready_event,
    ):
        super().__init__(venv)
        self.reward_predictor = reward_predictor
        self.reward_predictor_ready_event = reward_predictor_ready_event

    def reset(self):
        return self.venv.reset()

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, true_rewards, dones, infos = self.venv.step_wait()

        # Pass true reward through info so segments can be labelled correctly.
        for i, info in enumerate(infos):
            info["true_reward"] = float(true_rewards[i])

        if self.reward_predictor_ready_event.is_set():
            predicted = self.reward_predictor.reward(obs)
            rewards = predicted if np.all(np.isfinite(predicted)) else np.zeros(len(obs))
        else:
            rewards = np.zeros(len(obs), dtype=np.float32)

        return obs, rewards.astype(np.float32), dones, infos

    def reload(self, checkpoint_dir: str) -> None:
        """Load the latest reward predictor checkpoint from disk."""
        latest = RewardPredictorEnsemble.latest_checkpoint(checkpoint_dir)
        if latest:
            try:
                self.reward_predictor.load(latest)
            except Exception:
                pass
