from typing import Any, Optional

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize

from .env_wrappers import ego_status_to_onehot


class RewardDiagnosticsReplayBuffer(ReplayBuffer):
    """Standard SB3 replay sampling plus reward-staleness diagnostics."""

    STATUS_DIM = 7

    def __init__(self, *args, relabel_rewards: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.next_statuses = np.zeros(
            (self.buffer_size, self.n_envs, self.STATUS_DIM), dtype=np.float32
        )
        self.reward_model = None
        self.relabel_rewards = relabel_rewards

    def set_reward_model(self, reward_model) -> None:
        self.reward_model = reward_model

    def set_relabel_rewards(self, enabled: bool) -> None:
        self.relabel_rewards = bool(enabled)

    def __getstate__(self):
        state = self.__dict__.copy()
        # The model is checkpointed separately and reattached by DemoAlgorithm.
        state["reward_model"] = None
        return state

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        self.next_statuses[self.pos] = np.asarray(
            [ego_status_to_onehot(info.get("ego_status", "running")) for info in infos],
            dtype=np.float32,
        )
        super().add(obs, next_obs, action, reward, done, infos)

    def _unscale_actions(self, actions: np.ndarray) -> np.ndarray:
        if not isinstance(self.action_space, spaces.Box):
            return actions
        low, high = self.action_space.low, self.action_space.high
        return low + 0.5 * (actions + 1.0) * (high - low)

    def _predict_rewards(
        self,
        batch_inds: np.ndarray,
        env_indices: np.ndarray,
    ) -> np.ndarray:
        if self.reward_model is None:
            raise RuntimeError(
                "Reward relabelling requires an attached reward model. "
                "Call set_reward_model() after loading the replay buffer."
            )

        observations = self.observations[batch_inds, env_indices]
        actions = self._unscale_actions(self.actions[batch_inds, env_indices])
        statuses = self.next_statuses[batch_inds, env_indices]
        dones = self.dones[batch_inds, env_indices]
        return self.reward_model.predict(observations, actions, statuses, dones)

    def sample_reward_staleness(self, batch_size: int, rng: np.random.Generator):
        """Return stored and current rewards for actual replay-buffer entries."""
        upper_bound = self.buffer_size if self.full else self.pos
        if upper_bound == 0 or self.reward_model is None:
            return None

        n_samples = min(batch_size, upper_bound * self.n_envs)
        flat_indices = rng.choice(upper_bound * self.n_envs, size=n_samples, replace=False)
        batch_inds = flat_indices // self.n_envs
        env_indices = flat_indices % self.n_envs
        stored = self.rewards[batch_inds, env_indices].copy()
        current = self._predict_rewards(batch_inds, env_indices)
        return stored, current


class RewardRelabelReplayBuffer(RewardDiagnosticsReplayBuffer):
    """Replay buffer with configurable lazy relabelling and staleness diagnostics."""

    def __init__(self, *args, relabel_rewards: bool = True, **kwargs):
        super().__init__(*args, relabel_rewards=relabel_rewards, **kwargs)

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> ReplayBufferSamples:
        env_indices = np.random.randint(0, high=self.n_envs, size=len(batch_inds))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                self.observations[(batch_inds + 1) % self.buffer_size, env_indices], env
            )
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices], env)

        if self.relabel_rewards:
            rewards = self._predict_rewards(batch_inds, env_indices)
        else:
            rewards = self.rewards[batch_inds, env_indices]
        rewards = rewards.reshape(-1, 1)
        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices], env),
            self.actions[batch_inds, env_indices],
            next_obs,
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(rewards, env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))
