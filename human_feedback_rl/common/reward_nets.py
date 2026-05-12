import abc
from typing import Iterable, Optional, Tuple

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
        action: th.Tensor,
        next_status: Optional[th.Tensor] = None,
        done: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        """
        Compute rewards for a batch of transitions.
        Output shape: (batch_size,)
        """

    def preprocess(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_status: Optional[np.ndarray] = None,
        done: Optional[np.ndarray] = None,
    ) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor], Optional[th.Tensor]]:
        state_th = th.as_tensor(state, dtype=th.float32)
        action_th = th.as_tensor(action, dtype=th.float32)
        next_status_th = th.as_tensor(next_status, dtype=th.float32) if next_status is not None else None
        done_th = th.as_tensor(done, dtype=th.float32) if done is not None else None
        return state_th, action_th, next_status_th, done_th

    @th.no_grad()
    def predict(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_status: Optional[np.ndarray] = None,
        done: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        state_th, action_th, next_status_th, done_th = self.preprocess(state, action, next_status, done)
        rewards = self.forward(state_th, action_th, next_status_th, done_th)
        return rewards.cpu().numpy()




class SimpleRewardNet(RewardNet):
    def __init__(
        self,
        observation_space,
        action_space,
        net_arch: list[int] = [64, 64],
        activation_fn: str = "tanh",
    ):
        super().__init__(observation_space, action_space)

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]

        act = {"relu": nn.ReLU, "tanh": nn.Tanh}[activation_fn]

        layers = []
        in_dim = obs_dim + act_dim
        for out_dim in net_arch:
            layers += [nn.Linear(in_dim, out_dim), act()]
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, state, action, _next_status=None, _done=None):
        x = th.cat([state, action], dim=1)
        return self.net(x).squeeze(-1)




class SumoSimpleRewardNet(RewardNet):
    """
    Reward network for SUMO ego-vehicle tasks.

    Input: observation, action, and the vehicle status of the next state
    encoded as a 7-dim one-hot vector
    [arrived, collided, off_road, timeout, running, teleported, removed_unknown],
    plus a scalar done flag.
    """

    STATUS_DIM = 7  # arrived, collided, off_road, timeout, running, teleported, removed_unknown

    def __init__(
        self,
        observation_space,
        action_space,
        net_arch: list[int] = [64, 64],
        activation_fn: str = "tanh",
    ):
        super().__init__(observation_space, action_space)

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        in_dim = obs_dim + act_dim + self.STATUS_DIM + 1  # +1 for done flag

        act = {"relu": nn.ReLU, "tanh": nn.Tanh}[activation_fn]

        layers = []
        for out_dim in net_arch:
            layers += [nn.Linear(in_dim, out_dim), act()]
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, state, action, next_status=None, done=None):
        x = th.cat([state, action, next_status, done.unsqueeze(-1)], dim=1)
        return self.net(x).squeeze(-1)




class RewardEnsemble(RewardNet):
    """Ensemble of reward networks that returns the mean reward.

    Supports online reward normalization via set_mean / set_std.
    Normalization is applied only in predict() so the agent (PPO/SAC) sees
    normalized rewards, while forward() always returns raw values so the
    reward-model loss is not affected.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        members: Iterable[RewardNet],
    ):
        members = list(members)
        if len(members) < 2:
            raise ValueError("RewardEnsemble needs at least 2 members.")

        super().__init__(observation_space, action_space)
        self.members = nn.ModuleList(members)
        self._norm_mean: float = 0.0
        self._norm_std:  float = 1.0

    def set_mean(self, mean: float) -> None:
        self._norm_mean = float(mean)

    def set_std(self, std: float) -> None:
        self._norm_std = float(std)

    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_status: Optional[th.Tensor] = None,
        done: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        """Raw mean reward across ensemble members. Output shape: (batch_size,)"""
        member_rewards = [
            member(state, action, next_status, done)
            for member in self.members
        ]
        rewards_stack = th.stack(member_rewards, dim=0)  # (num_members, batch_size)
        return rewards_stack.mean(dim=0)

    @th.no_grad()
    def predict(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_status: Optional[np.ndarray] = None,
        done: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Normalized ensemble mean reward used by the agent.

        Applies (raw - mean) / std where mean/std are set via set_mean/set_std.
        Default mean=0, std=1 → no-op until normalization stats are injected.
        """
        raw = super().predict(state, action, next_status, done)
        return (raw - self._norm_mean) / (self._norm_std + 1e-8)

    @th.no_grad()
    def predict_unnormalized(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_status: Optional[np.ndarray] = None,
        done: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Raw ensemble mean without normalization.

        Used to compute normalization statistics from the rollout before
        injecting them via set_mean / set_std.
        """
        return super().predict(state, action, next_status, done)

    @th.no_grad()
    def predict_all(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_status: Optional[np.ndarray] = None,
        done: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Reward of each ensemble member. Output shape: (batch_size, num_members)"""
        state_th, action_th, next_status_th, done_th = self.preprocess(state, action, next_status, done)
        member_rewards = [
            member(state_th, action_th, next_status_th, done_th).cpu().numpy()
            for member in self.members
        ]
        return np.stack(member_rewards, axis=1)

    @th.no_grad()
    def predict_mean_std(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_status: Optional[np.ndarray] = None,
        done: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return mean and std across ensemble members."""
        all_rewards = self.predict_all(state, action, next_status, done)
        mean = all_rewards.mean(axis=1)
        std = all_rewards.std(axis=1)
        return mean, std
