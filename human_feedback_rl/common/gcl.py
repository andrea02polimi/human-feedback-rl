"""Guided Cost Learning building blocks.

The code in this module keeps the paper-specific pieces separate from the
training loop: a neural cost parametrization, trajectory tensor helpers, simple
trajectory-distribution estimates for importance sampling, and the two
regularizers proposed by Finn, Levine and Abbeel (2016).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn

from human_feedback_rl.common.reward_nets import RewardNet
from human_feedback_rl.common.types import Trajectory


STATUS_DIM = 7
RUNNING_STATUS = np.array([0, 0, 0, 0, 1, 0, 0], dtype=np.float32)
STATUS_NAMES = (
    "arrived",
    "collided",
    "offroad",
    "timeout",
    "running",
    "teleported",
    "removed_unknown",
)
STATUS_ALIASES = {
    "off_road": "offroad",
    "removed": "removed_unknown",
    "unknown": "removed_unknown",
}


def _activation(name: str):
    activations = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}
    if name not in activations:
        raise ValueError(f"Unknown activation function: {name!r}")
    return activations[name]


def _flat_dim(space: gym.Space) -> int:
    if isinstance(space, gym.spaces.Discrete):
        return int(space.n)
    if not hasattr(space, "shape") or space.shape is None:
        raise TypeError(f"Unsupported space type: {type(space).__name__}")
    return int(np.prod(space.shape))


def _terminal_cost_vector(terminal_costs: Optional[dict[str, float]]) -> np.ndarray:
    values = {name: 0.0 for name in STATUS_NAMES}
    for key, value in (terminal_costs or {}).items():
        canonical = STATUS_ALIASES.get(str(key), str(key))
        if canonical not in values:
            valid = ", ".join(STATUS_NAMES)
            raise ValueError(f"Unknown terminal status {key!r}; valid statuses: {valid}")
        values[canonical] = float(value)
    return np.asarray([values[name] for name in STATUS_NAMES], dtype=np.float32)


class GuidedCostNet(RewardNet):
    """Neural cost model for Guided Cost Learning.

    The paper uses ``c_theta(x_t, u_t) = ||A f_theta(x_t) + b||^2 + w_u ||u_t||^2``.
    SUMO terminal events are only visible after a transition, so this model can
    optionally include ``next_status`` and ``done`` in the learned feature input.
    ``forward`` returns ``-cost`` so the same module can be plugged into the
    existing reward-replacement wrapper and optimized by SB3 as a reward model.
    Use :meth:`cost` for the IOC objective.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        feature_hidden_sizes: Optional[Sequence[int]] = None,
        feature_dim: int = 64,
        quadratic_dim: int = 32,
        activation_fn: str = "relu",
        action_weight: float = 1e-3,
        include_action_in_features: bool = True,
        include_status: bool = True,
        include_done: bool = True,
        reward_scale: float = 1.0,
        terminal_costs: Optional[dict[str, float]] = None,
        terminal_cost_on_done_only: bool = True,
        identity_init: bool = True,
        device: str = "cpu",
    ):
        super().__init__(observation_space, action_space)
        self.device = th.device(device)
        self.action_weight = float(action_weight)
        self.include_action_in_features = bool(include_action_in_features)
        self.include_status = bool(include_status)
        self.include_done = bool(include_done)
        self.reward_scale = float(reward_scale)
        self.terminal_cost_on_done_only = bool(terminal_cost_on_done_only)
        self.register_buffer(
            "_terminal_costs",
            th.as_tensor(_terminal_cost_vector(terminal_costs), dtype=th.float32),
        )

        self.obs_dim = _flat_dim(observation_space)
        self.action_dim = _flat_dim(action_space)
        self.discrete_actions = isinstance(action_space, gym.spaces.Discrete)

        input_dim = self.obs_dim
        if self.include_action_in_features:
            input_dim += self.action_dim
        if self.include_status:
            input_dim += STATUS_DIM
        if self.include_done:
            input_dim += 1

        hidden_sizes = list(feature_hidden_sizes or [64, 64])
        act = _activation(activation_fn)

        layers: List[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.extend([nn.Linear(last_dim, int(hidden_dim)), act()])
            last_dim = int(hidden_dim)
        layers.extend([nn.Linear(last_dim, int(feature_dim)), act()])
        self.feature_net = nn.Sequential(*layers)
        self.quadratic = nn.Linear(int(feature_dim), int(quadratic_dim), bias=True)

        if identity_init:
            self._identity_friendly_init(input_dim)

        self.to(self.device)

    def _identity_friendly_init(self, input_dim: int) -> None:
        """Initialize wide ReLU layers close to an identity feature map.

        This mirrors Appendix C of the paper when dimensions allow it, while
        remaining harmless for narrower layers.
        """
        linear_layers = [m for m in self.feature_net if isinstance(m, nn.Linear)]
        if not linear_layers:
            return

        first = linear_layers[0]
        nn.init.zeros_(first.weight)
        nn.init.zeros_(first.bias)
        n_pos = min(input_dim, first.out_features)
        first.weight.data[:n_pos, :n_pos] = th.eye(n_pos)
        n_neg = min(input_dim, max(0, first.out_features - n_pos))
        if n_neg:
            first.weight.data[n_pos : n_pos + n_neg, :n_neg] = -th.eye(n_neg)

        for layer in linear_layers[1:]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            n = min(layer.in_features, layer.out_features)
            if n > 0:
                layer.weight.data[:n, :n] += th.eye(n)

        nn.init.xavier_uniform_(self.quadratic.weight)
        nn.init.zeros_(self.quadratic.bias)

    def preprocess(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_status: Optional[np.ndarray] = None,
        done: Optional[np.ndarray] = None,
    ):
        state_th = th.as_tensor(state, dtype=th.float32, device=self.device)
        action_th = th.as_tensor(action, dtype=th.float32, device=self.device)
        next_status_th = (
            th.as_tensor(next_status, dtype=th.float32, device=self.device)
            if next_status is not None
            else None
        )
        done_th = (
            th.as_tensor(done, dtype=th.float32, device=self.device)
            if done is not None
            else None
        )
        return state_th, action_th, next_status_th, done_th

    def _encode_action(self, action: th.Tensor) -> th.Tensor:
        if self.discrete_actions:
            action_idx = action.long().reshape(-1)
            return th.nn.functional.one_hot(action_idx, num_classes=self.action_dim).float()
        return action.reshape(action.shape[0], -1).float()

    def _feature_input(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_status: Optional[th.Tensor],
        done: Optional[th.Tensor],
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        batch_size = state.shape[0]
        state = state.reshape(batch_size, -1).float()
        action_encoded = self._encode_action(action)
        parts = [state]

        if self.include_action_in_features:
            parts.append(action_encoded)

        if self.include_status:
            if next_status is None:
                status = th.as_tensor(RUNNING_STATUS, dtype=th.float32, device=self.device)
                next_status = status.repeat(batch_size, 1)
            parts.append(next_status.reshape(batch_size, STATUS_DIM).float())
        elif next_status is None:
            status = th.as_tensor(RUNNING_STATUS, dtype=th.float32, device=self.device)
            next_status = status.repeat(batch_size, 1)
        else:
            next_status = next_status.reshape(batch_size, STATUS_DIM).float()

        if self.include_done:
            if done is None:
                done = th.zeros(batch_size, dtype=th.float32, device=self.device)
            parts.append(done.reshape(batch_size, 1).float())
        elif done is None:
            done = th.zeros(batch_size, dtype=th.float32, device=self.device)
        else:
            done = done.reshape(batch_size).float()

        return (
            th.cat(parts, dim=1),
            action_encoded,
            next_status.reshape(batch_size, STATUS_DIM).float(),
            done.reshape(batch_size).float(),
        )

    def _fixed_terminal_cost(self, next_status: th.Tensor, done: th.Tensor) -> th.Tensor:
        terminal_cost = (next_status * self._terminal_costs).sum(dim=1)
        if self.terminal_cost_on_done_only:
            terminal_cost = terminal_cost * done.reshape(-1).float()
        return terminal_cost

    def cost(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_status: Optional[th.Tensor] = None,
        done: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        x, action_encoded, status_tensor, done_tensor = self._feature_input(
            state, action, next_status, done
        )
        features = self.feature_net(x)
        learned_cost = self.quadratic(features).pow(2).sum(dim=1)
        action_cost = self.action_weight * action_encoded.pow(2).sum(dim=1)
        terminal_cost = self._fixed_terminal_cost(status_tensor, done_tensor)
        return learned_cost + action_cost + terminal_cost

    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_status: Optional[th.Tensor] = None,
        done: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        return -self.reward_scale * self.cost(state, action, next_status, done)

    @th.no_grad()
    def predict_cost(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_status: Optional[np.ndarray] = None,
        done: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        state_th, action_th, status_th, done_th = self.preprocess(
            state, action, next_status, done
        )
        return self.cost(state_th, action_th, status_th, done_th).cpu().numpy()


@dataclass
class FlattenedTrajectories:
    observations: np.ndarray
    actions: np.ndarray
    statuses: np.ndarray
    dones: np.ndarray
    lengths: List[int]
    log_policy_probs: List[Optional[float]]


def flatten_trajectories(trajectories: Sequence[Trajectory]) -> FlattenedTrajectories:
    observations = []
    actions = []
    statuses = []
    dones = []
    lengths = []
    log_policy_probs = []

    for traj in trajectories:
        lengths.append(len(traj))
        logps = []
        for transition in traj:
            observations.append(np.asarray(transition.observation, dtype=np.float32))
            actions.append(np.asarray(transition.action, dtype=np.float32))
            status = transition.next_status
            statuses.append(
                np.asarray(status, dtype=np.float32) if status is not None else RUNNING_STATUS
            )
            dones.append(float(transition.done))
            if transition.log_policy_prob is not None:
                logps.append(float(transition.log_policy_prob))
        log_policy_probs.append(sum(logps) if len(logps) == len(traj) else None)

    if observations:
        obs_arr = np.stack(observations).astype(np.float32)
        act_arr = np.stack(actions).astype(np.float32)
        status_arr = np.stack(statuses).astype(np.float32)
        done_arr = np.asarray(dones, dtype=np.float32)
    else:
        obs_arr = np.empty((0,), dtype=np.float32)
        act_arr = np.empty((0,), dtype=np.float32)
        status_arr = np.empty((0, STATUS_DIM), dtype=np.float32)
        done_arr = np.empty((0,), dtype=np.float32)

    return FlattenedTrajectories(
        observations=obs_arr,
        actions=act_arr,
        statuses=status_arr,
        dones=done_arr,
        lengths=lengths,
        log_policy_probs=log_policy_probs,
    )


def trajectory_step_costs(
    cost_model: GuidedCostNet,
    trajectories: Sequence[Trajectory],
) -> List[th.Tensor]:
    flat = flatten_trajectories(trajectories)
    if flat.observations.size == 0:
        return []

    state_th, action_th, status_th, done_th = cost_model.preprocess(
        flat.observations,
        flat.actions,
        flat.statuses,
        flat.dones,
    )
    all_costs = cost_model.cost(state_th, action_th, status_th, done_th)

    split_costs = []
    start = 0
    for length in flat.lengths:
        split_costs.append(all_costs[start : start + length])
        start += length
    return split_costs


def reduce_trajectory_costs(
    step_costs: Sequence[th.Tensor],
    reduction: str = "sum",
) -> th.Tensor:
    if not step_costs:
        raise ValueError("Cannot reduce an empty trajectory batch.")
    if reduction == "sum":
        return th.stack([costs.sum() for costs in step_costs])
    if reduction == "mean":
        return th.stack([costs.mean() for costs in step_costs])
    raise ValueError(f"Unknown trajectory cost reduction: {reduction!r}")


def local_constant_rate_regularizer(step_costs: Sequence[th.Tensor]) -> th.Tensor:
    terms = []
    for costs in step_costs:
        if costs.numel() >= 3:
            second_diff = costs[2:] - 2.0 * costs[1:-1] + costs[:-2]
            terms.append(second_diff.pow(2).mean())
    if not terms:
        device = step_costs[0].device if step_costs else th.device("cpu")
        return th.zeros((), device=device)
    return th.stack(terms).mean()


def monotonic_regularizer(
    step_costs: Sequence[th.Tensor],
    margin: float = 1.0,
) -> th.Tensor:
    """Squared hinge from Section 5 of the paper.

    The default implements the printed formula ``relu(c_t - c_{t-1} - 1)^2``.
    """
    terms = []
    for costs in step_costs:
        if costs.numel() >= 2:
            delta = costs[1:] - costs[:-1]
            terms.append(th.relu(delta - float(margin)).pow(2).mean())
    if not terms:
        device = step_costs[0].device if step_costs else th.device("cpu")
        return th.zeros((), device=device)
    return th.stack(terms).mean()


class StepGaussianTrajectoryDistribution:
    """Diagonal Gaussian approximation for variable-length SUMO trajectories.

    The original paper estimates a Gaussian trajectory distribution for
    demonstrations when the true sampler is unknown. SUMO episodes have variable
    length, so this approximation models individual transition features plus a
    separate Gaussian over trajectory length.
    """

    def __init__(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        length_mean: float,
        length_std: float,
        name: str = "distribution",
    ):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        self.var = np.square(self.std)
        self.length_mean = float(length_mean)
        self.length_std = max(float(length_std), 1e-6)
        self.name = name
        self._log_norm = -0.5 * np.log(2.0 * math.pi * self.var).sum()
        self._length_log_norm = -0.5 * math.log(2.0 * math.pi * self.length_std**2)

    @classmethod
    def fit(
        cls,
        trajectories: Sequence[Trajectory],
        regularization: float = 1e-4,
        max_transitions: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        name: str = "distribution",
    ) -> "StepGaussianTrajectoryDistribution":
        features = trajectory_features(trajectories)
        if features.shape[0] == 0:
            raise ValueError("Cannot fit a trajectory distribution on no transitions.")
        if max_transitions is not None and features.shape[0] > max_transitions:
            rng = rng if rng is not None else np.random.default_rng()
            idx = rng.choice(features.shape[0], size=max_transitions, replace=False)
            features = features[idx]
        mean = features.mean(axis=0)
        std = features.std(axis=0) + float(regularization)
        lengths = np.asarray([len(traj) for traj in trajectories], dtype=np.float64)
        return cls(
            mean=mean,
            std=np.maximum(std, float(regularization)),
            length_mean=float(lengths.mean()),
            length_std=float(lengths.std() + regularization),
            name=name,
        )

    def log_prob(self, trajectory: Trajectory) -> float:
        features = trajectory_features([trajectory])
        if features.shape[0] == 0:
            return -np.inf
        diff = features.astype(np.float64) - self.mean
        transition_lp = self._log_norm - 0.5 * np.square(diff / self.std).sum(axis=1)
        length_diff = (len(trajectory) - self.length_mean) / self.length_std
        length_lp = self._length_log_norm - 0.5 * length_diff**2
        return float(transition_lp.sum() + length_lp)


def trajectory_features(trajectories: Sequence[Trajectory]) -> np.ndarray:
    rows = []
    for traj in trajectories:
        for transition in traj:
            obs = np.asarray(transition.observation, dtype=np.float32).reshape(-1)
            act = np.asarray(transition.action, dtype=np.float32).reshape(-1)
            status = transition.next_status
            status_arr = (
                np.asarray(status, dtype=np.float32).reshape(-1)
                if status is not None
                else RUNNING_STATUS
            )
            done = np.asarray([float(transition.done)], dtype=np.float32)
            rows.append(np.concatenate([obs, act, status_arr, done], axis=0))
    if not rows:
        return np.empty((0, 0), dtype=np.float32)
    return np.stack(rows).astype(np.float32)


def fusion_log_prob(
    trajectory: Trajectory,
    distributions: Sequence[StepGaussianTrajectoryDistribution],
) -> float:
    if not distributions:
        return 0.0
    logps = np.asarray([dist.log_prob(trajectory) for dist in distributions], dtype=np.float64)
    max_logp = np.max(logps)
    if not np.isfinite(max_logp):
        return -np.inf
    return float(max_logp + np.log(np.exp(logps - max_logp).mean()))


def minibatch(items: Sequence, batch_size: int, rng: np.random.Generator, replace: bool = False):
    if len(items) == 0:
        return []
    size = min(int(batch_size), len(items)) if not replace else int(batch_size)
    idx = rng.choice(len(items), size=size, replace=replace)
    return [items[int(i)] for i in idx]


def finite_mean(values: Iterable[float], default: float = float("nan")) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else default
