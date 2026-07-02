"""Standalone stochastic policies for imitation, decoupled from any RL library.

``SquashedGaussianPolicy`` is a self-contained tanh-squashed diagonal Gaussian
over a bounded ``Box`` action space. It exists so the weighted-behavior-cloning
algorithm (``DemoAlgorithm2``, section 13 of the report) can own a policy without
dragging in an SB3 RL algorithm (SAC/PPO) purely as a network container.

Why a *squashed* (bounded) density rather than an unbounded Gaussian + clipping:
the section-13 reward is a log-ratio of densities ``R* = log(p_E / q_t)``, so the
policy must be a proper density on the *same* bounded support as the expert
actions for that ratio — and the weighted-ML projection onto it — to be
measure-consistent. A tanh-squashed Gaussian is exactly such a density on the
action box (with the tanh + scale Jacobians); an unbounded Gaussian whose samples
are clipped is not (it puts boundary atoms that its own log-density ignores).

Duck-typing: ``predict`` and ``action_log_prob`` mirror the small slice of the
SB3 policy interface used by
:func:`~human_feedback_rl.common.trajectory_generators.rollout_agent`,
``policy_action_log_probs`` and the imitation diagnostics, so the existing tested
rollout/metric code works unchanged.
"""

from typing import Optional, Tuple

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn


_ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}


def _make_trunk(in_dim: int, net_arch, activation_fn: str) -> Tuple[nn.Sequential, int]:
    activation = _ACTIVATIONS[activation_fn]
    layers = []
    last = in_dim
    for width in net_arch:
        layers += [nn.Linear(last, width), activation()]
        last = width
    return nn.Sequential(*layers), last


class SquashedGaussianPolicy(nn.Module):
    """Tanh-squashed diagonal Gaussian policy over a ``Box`` action space.

    Sampling: ``u ~ N(mean(s), std(s))``, ``a = center + scale * tanh(u)`` maps the
    pre-squash Gaussian into ``[low, high]``. ``log pi(a|s)`` is the corresponding
    density in the *environment* action measure (tanh and affine-scale Jacobians
    included), so it is directly comparable to the trajectory returns of the
    reward model.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        net_arch=(64, 64),
        activation_fn: str = "tanh",
        log_std_init: float = -0.5,
        log_std_bounds: Tuple[float, float] = (-5.0, 2.0),
        device: str = "cpu",
        eps: float = 1e-6,
    ):
        super().__init__()
        if not isinstance(action_space, spaces.Box):
            raise TypeError("SquashedGaussianPolicy requires a Box action space.")

        self.device = th.device(device)
        self.eps = float(eps)
        self.log_std_low, self.log_std_high = log_std_bounds

        obs_dim = int(np.prod(observation_space.shape))
        act_dim = int(np.prod(action_space.shape))
        self._act_shape = action_space.shape

        self.trunk, last = _make_trunk(obs_dim, net_arch, activation_fn)
        self.mean_head = nn.Linear(last, act_dim)
        self.log_std_head = nn.Linear(last, act_dim)
        nn.init.constant_(self.log_std_head.bias, float(log_std_init))

        low = np.asarray(action_space.low, dtype=np.float32)
        high = np.asarray(action_space.high, dtype=np.float32)
        # Affine map from tanh-space [-1, 1] to the environment action box.
        self.register_buffer("_center", th.as_tensor((high + low) / 2.0))
        self.register_buffer("_scale", th.as_tensor((high - low) / 2.0))
        self._action_low = low
        self._action_high = high

        self.to(self.device)

    # ------------------------------------------------------------------ #
    # Core distribution
    # ------------------------------------------------------------------ #

    def _obs_tensor(self, observation) -> th.Tensor:
        obs = np.asarray(observation, dtype=np.float32)
        obs = obs.reshape(obs.shape[0], -1) if obs.ndim > 1 else obs.reshape(1, -1)
        return th.as_tensor(obs, dtype=th.float32, device=self.device)

    def _dist_params(self, obs_tensor: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        features = self.trunk(obs_tensor)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features).clamp(self.log_std_low, self.log_std_high)
        return mean, log_std

    def log_prob(self, obs_tensor: th.Tensor, action_env: th.Tensor) -> th.Tensor:
        """Differentiable ``log pi(a|s)`` for env-space actions. Shape ``(N,)``.

        Inverts the squash to recover the pre-tanh gaussian sample, evaluates the
        gaussian density there, and subtracts the tanh and affine-scale log-Jacobians.
        """
        mean, log_std = self._dist_params(obs_tensor)
        std = log_std.exp()

        a_tanh = ((action_env - self._center) / self._scale).clamp(
            -1.0 + self.eps, 1.0 - self.eps
        )
        pre_tanh = th.atanh(a_tanh)

        normal = th.distributions.Normal(mean, std)
        log_prob = normal.log_prob(pre_tanh).sum(dim=-1)
        # Tanh Jacobian: d a_tanh / d u = 1 - tanh(u)^2 = 1 - a_tanh^2.
        log_prob = log_prob - th.log(1.0 - a_tanh.pow(2) + self.eps).sum(dim=-1)
        # Affine-scale Jacobian: d a / d a_tanh = scale (constant).
        log_prob = log_prob - th.log(self._scale).sum()
        return log_prob

    def _sample(self, obs_tensor: th.Tensor, deterministic: bool) -> th.Tensor:
        mean, log_std = self._dist_params(obs_tensor)
        if deterministic:
            pre_tanh = mean
        else:
            std = log_std.exp()
            pre_tanh = mean + std * th.randn_like(std)
        return self._center + self._scale * th.tanh(pre_tanh)

    # ------------------------------------------------------------------ #
    # SB3-compatible duck interface (used by rollout / metrics code)
    # ------------------------------------------------------------------ #

    def predict(
        self,
        observation,
        state: Optional[np.ndarray] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, None]:
        was_training = self.training
        self.eval()
        with th.no_grad():
            actions = self._sample(self._obs_tensor(observation), deterministic)
        self.train(was_training)
        actions = actions.cpu().numpy().reshape((-1,) + self._act_shape)
        actions = np.clip(actions, self._action_low, self._action_high)
        return actions, state

    def action_log_prob(self, observation, actions) -> np.ndarray:
        """Non-differentiable ``log pi(a|s)`` as numpy, for the buffering/metrics code."""
        was_training = self.training
        self.eval()
        with th.no_grad():
            obs_tensor = self._obs_tensor(observation)
            act = np.asarray(actions, dtype=np.float32).reshape(obs_tensor.shape[0], -1)
            log_prob = self.log_prob(obs_tensor, th.as_tensor(act, device=self.device))
        self.train(was_training)
        return log_prob.cpu().numpy().reshape(-1)
