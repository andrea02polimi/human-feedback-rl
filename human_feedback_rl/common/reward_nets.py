import abc
from typing import Iterable, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3.common.vec_env import VecEnv


def make_net(in_dim: int, net_arch: list[int], activation_fn: str) -> nn.Sequential:
    """Build an MLP with the given architecture and activation function."""
    act = {"relu": nn.ReLU, "tanh": nn.Tanh}[activation_fn]
    layers = []
    for out_dim in net_arch:
        layers += [nn.Linear(in_dim, out_dim), act()]
        in_dim = out_dim
    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)



class RewardNet(nn.Module, abc.ABC):
    """Abstract reward network: maps (state, action, next_status, done) → scalar reward."""

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
        """Compute rewards for a batch of transitions. Output shape: (batch_size,)"""

    def preprocess(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_status: Optional[np.ndarray] = None,
        done: Optional[np.ndarray] = None,
    ) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor], Optional[th.Tensor]]:
        """Convert numpy inputs to float32 tensors."""
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
        """Inference-time reward prediction. Returns numpy array, no gradients."""
        state_th, action_th, next_status_th, done_th = self.preprocess(state, action, next_status, done)
        return self.forward(state_th, action_th, next_status_th, done_th).cpu().numpy()

    def fragment_avg_reward(self, fragment) -> th.Tensor:
        """Average reward over a fragment. Returns a scalar tensor (with grad)."""
        obs         = th.tensor(np.array([t.observation  for t in fragment]), dtype=th.float32)
        actions     = th.tensor(np.array([t.action       for t in fragment]), dtype=th.float32)
        next_status = th.tensor(np.array([t.next_status  for t in fragment]), dtype=th.float32)
        done        = th.tensor(np.array([float(t.done)  for t in fragment]), dtype=th.float32)
        return self(obs, actions, next_status, done).sum() / len(fragment)


class SumoRewardNet(RewardNet):
    """MLP reward network for SUMO tasks.

    Input: (state, action, next_status, done), where next_status is a 7-dim one-hot
    encoding [arrived, collided, off_road, timeout, running, teleported, removed_unknown].
    """

    STATUS_DIM = 7

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        net_arch: list[int] = [128, 128],
        activation_fn: str = "tanh",
    ):
        super().__init__(observation_space, action_space)

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        in_dim = obs_dim + act_dim + self.STATUS_DIM + 1  # +1 for done
        self.net = make_net(in_dim, net_arch, activation_fn)

    def forward(self, state, action, next_status=None, done=None):
        x = th.cat([state, action, next_status, done.unsqueeze(-1)], dim=1)
        return self.net(x).squeeze(-1)


class RewardEnsemble(RewardNet):
    """Ensemble of reward networks returning the mean reward across members."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        members: Iterable[RewardNet],
    ):
        members = list(members)
        if not members:
            raise ValueError("RewardEnsemble needs at least 1 member.")
        super().__init__(observation_space, action_space)
        self.members = nn.ModuleList(members)

    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_status: Optional[th.Tensor] = None,
        done: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        """Raw mean reward across ensemble members. Output shape: (batch_size,)"""
        return th.stack([m(state, action, next_status, done) for m in self.members]).mean(dim=0)

    @th.no_grad()
    def predict_all(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_status: Optional[np.ndarray] = None,
        done: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Per-member rewards. Output shape: (batch_size, num_members)"""
        state_th, action_th, ns_th, done_th = self.preprocess(state, action, next_status, done)
        return np.stack(
            [m(state_th, action_th, ns_th, done_th).cpu().numpy() for m in self.members],
            axis=1,
        )

    @th.no_grad()
    def predict_mean_std(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_status: Optional[np.ndarray] = None,
        done: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Mean and std of rewards across ensemble members."""
        all_rewards = self.predict_all(state, action, next_status, done)
        return all_rewards.mean(axis=1), all_rewards.std(axis=1)


class NormalizedRewardNet(RewardNet):
    """Wraps any RewardNet and applies (raw - mean) / std normalization in predict().

    forward() stays raw so loss computation is unaffected.
    Stats are injected via set_mean() / set_std().
    """

    def __init__(self, net: RewardNet):
        super().__init__(net.observation_space, net.action_space)
        self.net = net
        self._mean: float = 0.0
        self._std: float = 1.0

    def set_mean(self, mean: float) -> None:
        self._mean = float(mean)

    def set_std(self, std: float) -> None:
        self._std = float(std)

    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_status: Optional[th.Tensor] = None,
        done: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        return self.net(state, action, next_status, done)

    @th.no_grad()
    def predict(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_status: Optional[np.ndarray] = None,
        done: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Normalized prediction: (raw - mean) / std."""
        raw = super().predict(state, action, next_status, done)
        return (raw - self._mean) / (self._std + 1e-8)

    @th.no_grad()
    def predict_unnormalized(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_status: Optional[np.ndarray] = None,
        done: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Raw prediction, bypassing normalization."""
        return super().predict(state, action, next_status, done)



def make_reward_ensemble(
    venv: VecEnv,
    n_ensembles: int = 1,
    net_arch: Optional[List[int]] = None,
    activation_fn: str = None,
) -> RewardEnsemble:
    obs_space = venv.observation_space
    act_space = venv.action_space
    members = [
        SumoRewardNet(obs_space, act_space, net_arch=net_arch, activation_fn=activation_fn)
        for _ in range(n_ensembles)
    ]
    return RewardEnsemble(obs_space, act_space, members)