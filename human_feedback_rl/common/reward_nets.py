import abc
from typing import Iterable, Tuple

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn

from stable_baselines3.common.vec_env import VecEnv



class RewardNet(nn.Module, abc.ABC):
    """Minimal abstract reward network."""

    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    @abc.abstractmethod
    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor
    ) -> th.Tensor:
        """
        Compute rewards for a batch of transitions.
        Output shape: (batch_size,)
        """

    def preprocess(
        self,
        state: np.ndarray,
        action: np.ndarray
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """Convert NumPy arrays to torch tensors."""
        state_th = th.as_tensor(state, dtype=th.float32)
        action_th = th.as_tensor(action, dtype=th.float32)

        return state_th, action_th

    @th.no_grad()
    def predict(
        self,
        state: np.ndarray,
        action: np.ndarray
    ) -> np.ndarray:
        """Compute rewards without gradients."""
        state_th, action_th = self.preprocess(state, action)
        rewards = self.forward(state_th, action_th)
        return rewards.cpu().numpy()
    



class SimpleRewardNet(RewardNet):
    def __init__(self, observation_space, action_space, hidden_size: int = 256):
        super().__init__(observation_space, action_space)

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]

        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, state, action):
        x = th.cat([state, action], dim=-1)
        return self.net(x).squeeze(-1)



class RewardEnsemble(RewardNet):
    """Ensemble of reward networks that returns the mean reward."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        members: Iterable[RewardNet],
    ):
        members = list(members)
        if len(members) < 1:
            raise ValueError("RewardEnsemble needs at least 1 members.")

        super().__init__(observation_space, action_space)
        self.members = nn.ModuleList(members)

    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor,
    ) -> th.Tensor:
        """
        Mean reward across ensemble members.
        Output shape: (batch_size,)
        """
        member_rewards = [
            member(state, action)
            for member in self.members
        ]
        rewards_stack = th.stack(member_rewards, dim=0)  # (num_members, batch_size)
        return rewards_stack.mean(dim=0)

    @th.no_grad()
    def predict_all(
        self,
        state: np.ndarray,
        action: np.ndarray
    ) -> np.ndarray:
        """
        Reward of each ensemble member.
        Output shape: (batch_size, num_members)
        """
        state_th, action_th = self.preprocess(state, action)

        member_rewards = [
            member(state_th, action_th).cpu().numpy()
            for member in self.members
        ]
        return np.stack(member_rewards, axis=1)

    @th.no_grad()
    def predict_mean_std(
        self,
        state: np.ndarray,
        action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return mean and std across ensemble members."""
        all_rewards = self.predict_all(state, action)
        mean = all_rewards.mean(axis=1)
        std = all_rewards.std(axis=1)
        return mean, std


def make_reward_ensemble(venv: VecEnv, n_ensembles: int = 3) -> RewardEnsemble:
    """Factory: build a SimpleRewardNet ensemble matching the given environment spaces."""
    obs_space = venv.observation_space
    act_space = venv.action_space
    members = [SimpleRewardNet(obs_space, act_space) for _ in range(n_ensembles)]
    return RewardEnsemble(obs_space, act_space, members)