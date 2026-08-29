"""The learned reward: one network, an ensemble of them, and normalization.

SumoRewardNet reads an observation, an action and the ego status.
RewardEnsemble holds several and exposes their spread, which is what active
fragment selection needs. NormalizedRewardNet applies an agent-facing affine
in `predict` while leaving `forward` raw, so reward learning is untouched.
"""

import abc
from typing import Iterable, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3.common.vec_env import VecEnv

from .status import STATUS_DIM


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

    Input: (state, action, next_status, done), where next_status is the one-hot
    status encoding defined in :mod:`human_feedback_rl.common.status`.
    """

    STATUS_DIM = STATUS_DIM

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

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ) -> None:
        # Checkpoints written before v0.2 wrapped each member in its own
        # NormalizedRewardNet: weights lived under members.{i}.net.net.* with
        # per-member _mean/_std buffers. Remap them to the flat member layout.
        for i in range(len(self.members)):
            member_prefix = f"{prefix}members.{i}."
            for key in [k for k in state_dict if k.startswith(member_prefix)]:
                rest = key[len(member_prefix):]
                if rest in ("_mean", "_std"):
                    del state_dict[key]
                elif rest.startswith("net.net."):
                    state_dict[member_prefix + "net." + rest[len("net.net."):]] = state_dict.pop(key)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

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

class NormalizedRewardNet(RewardNet):
    """Apply an agent-only affine transformation in ``predict``.

    ``forward`` is deliberately raw so inference statistics cannot alter the
    reward-learning objective. Statistics are buffers and survive checkpoints.
    """

    def __init__(self, net: RewardNet, alpha: float = 1):
        super().__init__(net.observation_space, net.action_space)
        self.net = net
        self.alpha = alpha
        self.register_buffer("_mean", th.tensor(0.0, dtype=th.float32))
        self.register_buffer("_std", th.tensor(1.0, dtype=th.float32))

    def set_mean(self, mean: float) -> None:
        updated = (1 - self.alpha) * float(self._mean) + self.alpha * float(mean)
        self._mean.fill_(updated)

    def set_std(self, std: float) -> None:
        updated = (1 - self.alpha) * float(self._std) + self.alpha * float(std)
        self._std.fill_(max(updated, 1e-8))

    @property
    def normalization_mean(self) -> float:
        """Mean subtracted from agent-facing predictions."""
        return float(self._mean)

    @property
    def normalization_std(self) -> float:
        """Standard deviation used for agent-facing predictions."""
        return float(self._std)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ) -> None:
        # Older checkpoints predate persistent normalization statistics.
        state_dict.setdefault(prefix + "_mean", self._mean)
        state_dict.setdefault(prefix + "_std", self._std)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

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
        raw = self.net.predict(state, action, next_status, done)
        return (raw - float(self._mean)) / (float(self._std) + 1e-8)

    @th.no_grad()
    def predict_unnormalized(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_status: Optional[np.ndarray] = None,
        done: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Raw prediction, bypassing normalization."""
        return self.net.predict(state, action, next_status, done)

    @th.no_grad()
    def predict_all(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_status: Optional[np.ndarray] = None,
        done: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Delegate to the inner ensemble's predict_all."""
        return self.net.predict_all(state, action, next_status, done)

    @property
    def members(self):
        return self.net.members


def make_reward_ensemble(
    venv: VecEnv,
    n_ensembles: int = 1,
    net_arch: Optional[List[int]] = None,
    activation_fn: str = None,
    alpha: float = 1,
) -> RewardEnsemble:

    obs_space = venv.observation_space
    act_space = venv.action_space

    net_arch = net_arch or [128, 128]
    activation_fn = activation_fn or "tanh"
    members = [
        SumoRewardNet(obs_space, act_space, net_arch=net_arch, activation_fn=activation_fn)
        for _ in range(n_ensembles)
    ]
    return NormalizedRewardNet(RewardEnsemble(obs_space, act_space, members), alpha)
