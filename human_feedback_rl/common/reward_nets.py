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

class ResidualBlock(nn.Module):
    """Simple residual block with optional skip connection."""
    def __init__(self, dim: int, activation_fn: str):
        super().__init__()
        act = {"relu": nn.ReLU, "tanh": nn.Tanh}[activation_fn]
        self.fc = nn.Linear(dim, dim)
        self.activation = act()
    
    def forward(self, x):
        return x + self.activation(self.fc(x))


def make_residual_net(in_dim: int, net_arch: list[int], activation_fn: str) -> nn.Sequential:
    """Build an MLP with residual connections."""
    act = {"relu": nn.ReLU, "tanh": nn.Tanh}[activation_fn]
    layers = []
    
    # Project input to first hidden dimension
    layers.append(nn.Linear(in_dim, net_arch[0]))
    layers.append(act())
    
    # Residual blocks for hidden layers
    for dim in net_arch:
        layers.append(ResidualBlock(dim, activation_fn))
    
    # Output layer
    layers.append(nn.Linear(net_arch[-1], 1))
    return nn.Sequential(*layers)

class RewardNet(nn.Module, abc.ABC):
    """Abstract reward network: maps (state, action, next_status, done) → scalar reward."""
    uses_next_state = False

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
        # self.net = make_residual_net(in_dim, net_arch, activation_fn)

    def forward(self, state, action, next_status=None, done=None):
        x = th.cat([state, action, next_status, done.unsqueeze(-1)], dim=1)
        return self.net(x).squeeze(-1)


class StateActionDiscriminatorNet(RewardNet):
    """Discriminator over (state, action), matching the GAIL paper."""

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
        self.net = make_net(obs_dim + act_dim, net_arch, activation_fn)

    def forward(self, state, action, next_status=None, done=None):
        x = th.cat([state, action], dim=1)
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

    @property
    def uses_next_state(self) -> bool:
        return any(getattr(m, "uses_next_state", False) for m in self.members)

    def set_policy(self, policy) -> None:
        for member in self.members:
            if hasattr(member, "set_policy"):
                member.set_policy(policy)

    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_status: Optional[th.Tensor] = None,
        done: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        """Raw mean reward across ensemble members. Output shape: (batch_size,)"""
        return th.stack([m(state, action, next_status, done) for m in self.members]).mean(dim=0)

    def shaped_reward(self, state, action, next_state, done):
        return th.stack([m.shaped_reward(state, action, next_state, done) for m in self.members]).mean(dim=0)

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

    def __init__(self, net: RewardNet, alfa: float = 1):
        super().__init__(net.observation_space, net.action_space)
        self.net = net
        self.alfa = alfa
        self._mean: float = 0.0
        self._std: float = 1.0

    def set_mean(self, mean: float) -> None:
        self._mean = (1 - self.alfa)*self._mean + self.alfa*float(mean)

    def set_std(self, std: float) -> None:
        self._std = (1 - self.alfa)*self._std + self.alfa*float(std)

    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_status: Optional[th.Tensor] = None,
        done: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        return (self.net(state, action, next_status, done) - self._mean) / (self._std + 1e-8)

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


class GailRewardNet(RewardNet):
    """GAIL discriminator wrapper.

    forward() returns discriminator logits for BCE training. predict() returns
    the policy reward used by the original GAIL objective, -log D(s, a), where
    D is the discriminator probability that a transition came from the policy.
    """

    def __init__(self, discriminator: RewardNet):
        super().__init__(discriminator.observation_space, discriminator.action_space)
        self.discriminator = discriminator

    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_status: Optional[th.Tensor] = None,
        done: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        return self.discriminator(state, action, next_status, done)

    @th.no_grad()
    def predict(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_status: Optional[np.ndarray] = None,
        done: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        state_th, action_th, next_status_th, done_th = self.preprocess(state, action, next_status, done)
        logits = self.forward(state_th, action_th, next_status_th, done_th)
        return th.nn.functional.softplus(-logits).cpu().numpy()

    @th.no_grad()
    def predict_all(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_status: Optional[np.ndarray] = None,
        done: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        state_th, action_th, ns_th, done_th = self.preprocess(state, action, next_status, done)
        return np.stack(
            [th.nn.functional.softplus(-m(state_th, action_th, ns_th, done_th)).cpu().numpy() for m in self.members],
            axis=1,
        )

    @property
    def members(self):
        return self.discriminator.members


def make_reward_ensemble(
    venv: VecEnv,
    n_ensembles: int = 1,
    net_arch: Optional[List[int]] = None,
    activation_fn: str = None,
    alfa: float = 1,
) -> RewardEnsemble:
    
    obs_space = venv.observation_space
    act_space = venv.action_space

    members = [
        NormalizedRewardNet(
            SumoRewardNet(
                obs_space, 
                act_space, 
                net_arch=net_arch, 
                activation_fn=activation_fn
            ), alfa)
        for _ in range(n_ensembles)
    ]
    return NormalizedRewardNet(RewardEnsemble(obs_space, act_space, members), alfa)


def make_gail_discriminator_ensemble(
    venv: VecEnv,
    n_ensembles: int = 1,
    net_arch: Optional[List[int]] = None,
    activation_fn: str = None,
    alfa: float = 1,
) -> GailRewardNet:
    del alfa

    obs_space = venv.observation_space
    act_space = venv.action_space

    members = [
        StateActionDiscriminatorNet(
            obs_space,
            act_space,
            net_arch=net_arch,
            activation_fn=activation_fn,
        )
        for _ in range(n_ensembles)
    ]
    return GailRewardNet(RewardEnsemble(obs_space, act_space, members))


class AirlRewardNet(RewardNet):
    uses_next_state = True

    def __init__(self, observation_space, action_space, net_arch=[128, 128], activation_fn="tanh", gamma=0.997):
        super().__init__(observation_space, action_space)
        self.gamma = gamma
        self.policy = None
        self._mean: float = 0.0

        self.obs_dim = observation_space.shape[0]
        self.act_dim = action_space.shape[0]

        # +1 input feature: an absorbing-state indicator (Kostrikov et al. 2019,
        # "Discriminator-Actor-Critic"). Real states carry indicator 0; the single
        # absorbing state carries indicator 1 with zeroed features. Terminating
        # episodes (collision/arrival) transition into it, so episode termination
        # carries no class information for the discriminator (removes survival bias).
        self.reward_net = make_net(self.obs_dim + 1 + self.act_dim, net_arch, activation_fn)
        self.value_net = make_net(self.obs_dim + 1, net_arch, activation_fn)

    def set_mean(self, mean: float) -> None:
        self._mean = float(mean)

    def _augment(self, state, absorbing: bool = False):
        """Append the absorbing-state indicator column to a batch of states."""
        ind = th.full((state.shape[0], 1), 1.0 if absorbing else 0.0,
                      dtype=state.dtype, device=state.device)
        return th.cat([state, ind], dim=1)

    def reward(self, state, action, absorbing: bool = False):
        x = th.cat([self._augment(state, absorbing), action], dim=1)
        return self.reward_net(x).squeeze(-1)

    def _value(self, state, absorbing: bool = False):
        return self.value_net(self._augment(state, absorbing)).squeeze(-1)

    def shaped_reward(self, state, action, next_state, terminated):
        """f = r(s,a) + γ V(s') − V(s).

        At a true terminal (collision/arrival) the episode transitions into the
        absorbing state, so we bootstrap V(absorbing) rather than zeroing. Timeouts
        are NOT terminals: ``terminated`` excludes them (computed upstream), so they
        correctly bootstrap V(real next state).
        """
        r = self.reward(state, action)
        v_s = self._value(state)
        v_next = self._value(next_state)
        v_abs = self._value(th.zeros_like(state), absorbing=True)
        v_target = th.where(terminated.bool(), v_abs, v_next)
        return r + self.gamma * v_target - v_s

    def absorbing_reward(self, n: int, device, dtype):
        """Discriminator logit for ``n`` absorbing→absorbing self-loops.

        The shaped reward of the self-loop reduces to its learned reward r_abs
        (value terms cancel: r_abs + γV_abs − V_abs = r_abs), and the policy
        log-prob is taken as 0 (deterministic self-loop), so the logit is r_abs.
        """
        s = th.zeros((n, self.obs_dim), device=device, dtype=dtype)
        a = th.zeros((n, self.act_dim), device=device, dtype=dtype)
        return self.reward(s, a, absorbing=True)

    def set_policy(self, policy) -> None:
        self.policy = policy

    def _policy_log_prob(self, state, action):
        if self.policy is None:
            return th.zeros(state.shape[0], dtype=state.dtype, device=state.device)
        _, log_prob, _ = self.policy.evaluate_actions(state, action)
        return log_prob.detach()

    def discriminator_logit(self, state, action, next_state, terminated):
        return self.shaped_reward(state, action, next_state, terminated) - self._policy_log_prob(state, action)

    def forward(self, state, action, next_state=None, done=None):
        # `done` here carries the true-terminal mask (timeouts excluded upstream).
        return self.discriminator_logit(state, action, next_state, done) - self._mean


def make_airl_reward_ensemble(
    venv: VecEnv,
    n_ensembles: int = 1,
    net_arch: Optional[List[int]] = None,
    activation_fn: str = None,
    gamma: float = 0.997,
    alfa: float = 1,
) -> RewardEnsemble:
    del alfa

    obs_space = venv.observation_space
    act_space = venv.action_space

    members = [
        AirlRewardNet(
            obs_space,
            act_space,
            net_arch=net_arch,
            activation_fn=activation_fn,
            gamma=gamma,
        )
        for _ in range(n_ensembles)
    ]
    return RewardEnsemble(obs_space, act_space, members)
