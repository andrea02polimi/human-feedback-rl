"""
Christiano et al. (2017) "Deep Reinforcement Learning from Human Preferences"
adapted for the SUMO EgoVehicle environment.

All algorithm-specific components are self-contained in this module.
Only WandB logging utilities are imported from the common package.
"""

from __future__ import annotations

import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper

from human_feedback_rl.common.custom_logging_callback import CustomLoggingCallback


# ===========================================================================
# Data structures
# ===========================================================================

@dataclass
class Transition:
    obs: np.ndarray
    action: Any          # int for discrete, np.ndarray for continuous
    true_reward: float


@dataclass
class Segment:
    transitions: List[Transition]

    def true_return(self, discount: float = 1.0) -> float:
        return sum(discount ** i * t.true_reward for i, t in enumerate(self.transitions))

    def __len__(self) -> int:
        return len(self.transitions)


@dataclass
class SegmentPair:
    left: Segment
    right: Segment


@dataclass
class Preference:
    """(1.0, 0.0) → left preferred; (0.0, 1.0) → right preferred."""
    label: Tuple[float, float]


class PreferenceDataset:
    """Stores (SegmentPair, Preference) tuples with optional FIFO capacity."""

    def __init__(self, capacity: Optional[int] = None):
        self.capacity = capacity
        self.pairs: List[SegmentPair] = []
        self.preferences: List[Preference] = []

    def push(self, pairs: List[SegmentPair], preferences: List[Preference]) -> None:
        self.pairs.extend(pairs)
        self.preferences.extend(preferences)
        if self.capacity is not None and len(self.pairs) > self.capacity:
            excess = len(self.pairs) - self.capacity
            self.pairs = self.pairs[excess:]
            self.preferences = self.preferences[excess:]

    def __len__(self) -> int:
        return len(self.pairs)


# ===========================================================================
# Query schedules
# ===========================================================================

def _distribute(weights: List[float], total: int) -> List[int]:
    """Distribute `total` among buckets proportional to weights, rounding to ints."""
    w_sum = sum(weights)
    raw = [w / w_sum * total for w in weights]
    floored = [int(r) for r in raw]
    diff = total - sum(floored)
    order = sorted(range(len(floored)), key=lambda k: raw[k] - floored[k], reverse=True)
    for k in order[:diff]:
        floored[k] += 1
    return floored


def _schedule_constant(n: int, total: int) -> List[int]:
    return _distribute([1.0] * n, total)


def _schedule_hyperbolic(n: int, total: int) -> List[int]:
    return _distribute([1.0 / (k + 1) for k in range(n)], total)


def _schedule_inverse_quadratic(n: int, total: int) -> List[int]:
    return _distribute([1.0 / (k + 1) ** 2 for k in range(n)], total)


QUERY_SCHEDULES: Dict[str, Any] = {
    "constant": _schedule_constant,
    "hyperbolic": _schedule_hyperbolic,
    "inverse_quadratic": _schedule_inverse_quadratic,
}


# ===========================================================================
# Reward network
# ===========================================================================

class RewardNet(nn.Module):
    """MLP mapping (obs, action) → scalar reward."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """(B, obs_dim), (B, action_dim) → (B,)"""
        return self.net(torch.cat([obs, action], dim=-1)).squeeze(-1)


# ===========================================================================
# Ensemble reward model
# ===========================================================================

class EnsembleRewardModel:
    """K independent RewardNets. Supports discrete (one-hot) and continuous actions."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_ensembles: int,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        device: str = "cpu",
        discrete_actions: bool = True,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_ensembles = n_ensembles
        self.discrete_actions = discrete_actions
        self.device = torch.device(device)

        self.nets = [
            RewardNet(obs_dim, action_dim, hidden_dim).to(self.device)
            for _ in range(n_ensembles)
        ]
        self.optimizers = [
            torch.optim.Adam(net.parameters(), lr=lr)
            for net in self.nets
        ]

    def _encode_actions(self, actions) -> torch.Tensor:
        if self.discrete_actions:
            a = np.asarray(actions, dtype=np.int64).reshape(-1)
            enc = np.zeros((len(a), self.action_dim), dtype=np.float32)
            enc[np.arange(len(a)), a] = 1.0
            return torch.as_tensor(enc, device=self.device)
        return torch.as_tensor(
            np.asarray(actions, dtype=np.float32).reshape(-1, self.action_dim),
            device=self.device,
        )

    def _obs_tensor(self, obs: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)

    @torch.no_grad()
    def predict(self, obs: np.ndarray, actions, pessimism: float = 0.0) -> np.ndarray:
        """Mean reward across ensemble, optionally penalised by pessimism × std. Returns (B,)."""
        obs_t = self._obs_tensor(obs)
        act_t = self._encode_actions(actions)
        preds = torch.stack([net(obs_t, act_t) for net in self.nets])  # (K, B)
        mean = preds.mean(dim=0)
        if pessimism > 0.0:
            return (mean - pessimism * preds.std(dim=0)).cpu().numpy()
        return mean.cpu().numpy()

    def segment_returns(self, segment: Segment, ensemble_idx: int) -> torch.Tensor:
        """Sum of rewards over a segment for one ensemble member. Differentiable."""
        obs = np.stack([t.obs for t in segment.transitions])
        actions = [t.action for t in segment.transitions]
        obs_t = self._obs_tensor(obs)
        act_t = self._encode_actions(actions)
        return self.nets[ensemble_idx](obs_t, act_t).sum()


# ===========================================================================
# Reward wrapper
# ===========================================================================

class _RunningMeanStd:
    """Welford's online algorithm for running mean and variance."""

    def __init__(self):
        self.mean = 0.0
        self._M2 = 0.0
        self.count = 0

    def update(self, values: np.ndarray) -> None:
        for x in np.asarray(values).flat:
            self.count += 1
            delta = x - self.mean
            self.mean += delta / self.count
            self._M2 += delta * (x - self.mean)

    @property
    def std(self) -> float:
        if self.count < 2:
            return 1.0
        return float(np.sqrt(self._M2 / (self.count - 1))) or 1.0


class RewardWrapper(VecEnvWrapper):
    """
    Replaces environment rewards with EnsembleRewardModel predictions.

    Stores true env rewards under infos[i]['true_reward'] so the rollout
    collector can use them to label segment pairs synthetically.

    Predicted rewards are normalized to zero-mean / unit-variance via
    Welford's online algorithm (Christiano et al. 2017, §2.2).
    """

    def __init__(
        self, venv: VecEnv, reward_model: EnsembleRewardModel, pessimism: float = 0.0
    ):
        super().__init__(venv)
        self.reward_model = reward_model
        self.pessimism = pessimism
        self._obs: Optional[np.ndarray] = None
        self._actions: Optional[np.ndarray] = None
        self._stats = _RunningMeanStd()

    def reset(self) -> np.ndarray:
        obs = self.venv.reset()
        self._obs = np.asarray(obs, dtype=np.float32)
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions)
        self.venv.step_async(actions)

    def step_wait(self):
        obs, true_rewards, dones, infos = self.venv.step_wait()

        for i, info in enumerate(infos):
            info['true_reward'] = float(true_rewards[i])

        if self._obs is not None and self._actions is not None:
            pred = self.reward_model.predict(self._obs, self._actions, self.pessimism)
            self._stats.update(pred)
            rewards = ((pred - self._stats.mean) / self._stats.std).astype(np.float32)
        else:
            rewards = np.zeros(len(obs), dtype=np.float32)

        self._obs = np.asarray(obs, dtype=np.float32)
        return obs, rewards, dones, infos


# ===========================================================================
# Rollout collector callback
# ===========================================================================

class RolloutCollectorCallback(BaseCallback):
    """
    Captures (obs_t, action_t, true_reward_t) tuples during agent.learn().

    Builds complete episode trajectories per env, split at done=True.
    Call flush() after agent.learn() to retrieve all completed episodes
    and reset the internal buffer.

    Relies on SB3 calling the callback BEFORE updating self.model._last_obs,
    so self.model._last_obs holds the observation before the current step.
    """

    def __init__(self):
        super().__init__()
        self._active: Dict[int, List[Transition]] = defaultdict(list)
        self._completed: List[Segment] = []

    def _on_step(self) -> bool:
        obs_before = self.model._last_obs   # shape (n_envs, obs_dim); updated after callback
        actions = self.locals['actions']
        dones = self.locals['dones']
        infos = self.locals['infos']        # contains 'true_reward' from RewardWrapper

        for env_idx in range(len(dones)):
            true_reward = infos[env_idx].get('true_reward', 0.0)
            self._active[env_idx].append(
                Transition(
                    obs=obs_before[env_idx].astype(np.float32, copy=True),
                    action=actions[env_idx],
                    true_reward=float(true_reward),
                )
            )
            if dones[env_idx] and self._active[env_idx]:
                self._completed.append(Segment(self._active[env_idx]))
                self._active[env_idx] = []

        return True

    def flush(self) -> List[Segment]:
        """Return all completed trajectories collected since the last flush()."""
        completed = self._completed
        self._completed = []
        self._active.clear()
        return completed


# ===========================================================================
# Fragmenter
# ===========================================================================

def _fragment_trajectories(
    trajectories: List[Segment],
    fragment_length: int,
    n_pairs: int,
    rng: np.random.Generator,
) -> List[SegmentPair]:
    """
    Split trajectories into fixed-length non-overlapping segments and randomly
    pair them. Partial tail segments (shorter than fragment_length) are discarded.
    Returns at most n_pairs SegmentPairs.
    """
    segments: List[Segment] = []
    for traj in trajectories:
        ts = traj.transitions
        for start in range(0, len(ts), fragment_length):
            segments.append(Segment(ts[start: start + fragment_length]))

    if len(segments) < 2:
        return []

    indices = rng.permutation(len(segments))
    segments = [segments[i] for i in indices]

    pairs: List[SegmentPair] = []
    for i in range(0, len(segments) - 1, 2):
        if len(pairs) >= n_pairs:
            break
        pairs.append(SegmentPair(left=segments[i], right=segments[i + 1]))

    return pairs


# ===========================================================================
# Synthetic gatherer
# ===========================================================================

class SyntheticGatherer:
    """
    Labels segment pairs using the true environment reward (Bradley-Terry model).

        P(left > right) = sigmoid((R_left − R_right) / temperature)

    where R is the total discounted true reward of each segment. When
    sample=True, labels are drawn from Bernoulli(p) rather than using soft
    probabilities.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        discount: float = 1.0,
        threshold: float = 50.0,
        sample: bool = True,
        rng: Optional[np.random.Generator] = None,
    ):
        self.temperature = temperature
        self.discount = discount
        self.threshold = threshold
        self.sample = sample
        self.rng = rng if rng is not None else np.random.default_rng()

    def label(self, pairs: List[SegmentPair]) -> List[Preference]:
        return [self._label_one(pair) for pair in pairs]

    def _label_one(self, pair: SegmentPair) -> Preference:
        r_left = pair.left.true_return(self.discount) / len(pair.left)
        r_right = pair.right.true_return(self.discount) / len(pair.right)

        if self.temperature == 0.0:
            p_left = 1.0 if r_left > r_right else (0.5 if r_left == r_right else 0.0)
        else:
            diff = np.clip(
                (r_left - r_right) / self.temperature,
                -self.threshold,
                self.threshold,
            )
            p_left = float(1.0 / (1.0 + math.exp(-diff)))

        if self.sample:
            l = float(self.rng.random() < p_left)
        else:
            l = p_left

        return Preference(label=(l, 1.0 - l))


# ===========================================================================
# Reward trainer
# ===========================================================================

_ACCURACY_SAMPLE = 200  # max pairs used for accuracy estimation


class RewardTrainer:
    """
    Trains an EnsembleRewardModel on a PreferenceDataset.

    Each ensemble member is trained on an independent bootstrap sample of the
    dataset (bagging), improving ensemble diversity and uncertainty calibration.

    Loss: cross-entropy on Bradley-Terry logits (sum of per-step predicted rewards).
    """

    def __init__(
        self,
        reward_model: EnsembleRewardModel,
        epochs: int = 4,
        batch_size: int = 64,
    ):
        self.reward_model = reward_model
        self.epochs = epochs
        self.batch_size = batch_size

    def train(
        self, dataset: PreferenceDataset, epoch_multiplier: float = 1.0
    ) -> Dict[str, float]:
        if len(dataset) == 0:
            return {"reward_model/loss": 0.0, "reward_model/accuracy": 0.0}

        n_epochs = max(1, round(self.epochs * epoch_multiplier))
        last_loss = 0.0
        for _ in range(n_epochs):
            last_loss = self._train_one_epoch(dataset)

        accuracy = self._compute_accuracy(dataset)
        return {"reward_model/loss": last_loss, "reward_model/accuracy": accuracy}

    def _train_one_epoch(self, dataset: PreferenceDataset) -> float:
        n = len(dataset)
        bootstrap = [
            list(np.random.choice(n, size=n, replace=True))
            for _ in range(self.reward_model.n_ensembles)
        ]
        total_loss = 0.0
        n_batches = 0
        for batch_start in range(0, n, self.batch_size):
            batch_per_member = [b[batch_start: batch_start + self.batch_size] for b in bootstrap]
            total_loss += self._train_batch(dataset, batch_per_member)
            n_batches += 1
        return total_loss / max(n_batches, 1)

    def _train_batch(
        self, dataset: PreferenceDataset, batch_per_member: List[List[int]]
    ) -> float:
        rm = self.reward_model
        batch_loss = 0.0
        for k, (net, opt, indices) in enumerate(
            zip(rm.nets, rm.optimizers, batch_per_member)
        ):
            opt.zero_grad()
            losses = []
            for idx in indices:
                pair = dataset.pairs[idx]
                pref = dataset.preferences[idx]
                r_left = rm.segment_returns(pair.left, k) / len(pair.left)
                r_right = rm.segment_returns(pair.right, k) / len(pair.right)
                logits = torch.stack([r_left, r_right]).unsqueeze(0)  # (1, 2)
                target = torch.tensor(
                    [0 if pref.label[0] > pref.label[1] else 1],
                    device=rm.device,
                )
                losses.append(F.cross_entropy(logits, target))
            loss = torch.stack(losses).mean()
            loss.backward()
            opt.step()
            batch_loss += loss.item()
        return batch_loss / rm.n_ensembles

    @torch.no_grad()
    def _compute_accuracy(self, dataset: PreferenceDataset) -> float:
        rm = self.reward_model
        indices = list(range(len(dataset)))
        if len(indices) > _ACCURACY_SAMPLE:
            indices = random.sample(indices, _ACCURACY_SAMPLE)
        correct = 0
        for idx in indices:
            pair = dataset.pairs[idx]
            pref = dataset.preferences[idx]
            r_l = np.mean([rm.segment_returns(pair.left, k).item() / len(pair.left) for k in range(rm.n_ensembles)])
            r_r = np.mean([rm.segment_returns(pair.right, k).item() / len(pair.right) for k in range(rm.n_ensembles)])
            correct += int((r_l > r_r) == (pref.label[0] > pref.label[1]))
        return correct / len(indices) if indices else 0.0


# ===========================================================================
# Main algorithm
# ===========================================================================

class ChristianoAlgorithm:
    """
    Christiano et al. (2017) — Deep Reinforcement Learning from Human Preferences.

    Each iteration:
    1. Train the RL agent on the reward-model-wrapped environment via agent.learn().
       A callback captures (obs, action, true_reward) for every step.
    2. Fragment completed trajectories into fixed-length segments and pair them.
    3. Label pairs synthetically by comparing true environment returns.
    4. Train the reward model on the accumulated preference dataset.

    The RewardWrapper automatically uses the updated model in the next iteration.
    """

    def __init__(
        self,
        env: VecEnv,
        agent,
        rng: np.random.Generator,
        device: str = "cpu",
        n_ensembles: int = 3,
        hidden_dim: int = 256,
        lr_reward_model: float = 3e-4,
        fragment_length: int = 10,
        comparison_queue_size: Optional[int] = None,
        transition_oversampling: float = 3.0,   # unused with learn()-based collection
        query_schedule: str = "constant",
        num_iterations: int = 20,
        reward_trainer_epochs: int = 4,
        reward_model_batch_size: int = 64,
        preference_temperature: float = 1.0,
        preference_sample: bool = True,
        preference_discount_factor: float = 1.0,
        preference_threshold: float = 50.0,
        pessimism: float = 0.0,
    ):
        self.rng = rng
        self.num_iterations = num_iterations
        self.fragment_length = fragment_length
        self._query_schedule_name = query_schedule

        obs_dim = int(np.prod(env.observation_space.shape))
        act_space = env.action_space
        if hasattr(act_space, 'n'):
            action_dim = int(act_space.n)
            discrete_actions = True
        else:
            action_dim = int(np.prod(act_space.shape))
            discrete_actions = False

        self.reward_model = EnsembleRewardModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_ensembles=n_ensembles,
            hidden_dim=hidden_dim,
            lr=lr_reward_model,
            device=device,
            discrete_actions=discrete_actions,
        )
        self.reward_wrapper = RewardWrapper(env, self.reward_model, pessimism=pessimism)
        self.dataset = PreferenceDataset(capacity=comparison_queue_size)
        self.gatherer = SyntheticGatherer(
            temperature=preference_temperature,
            discount=preference_discount_factor,
            threshold=preference_threshold,
            sample=preference_sample,
            rng=rng,
        )
        self.reward_trainer = RewardTrainer(
            reward_model=self.reward_model,
            epochs=reward_trainer_epochs,
            batch_size=reward_model_batch_size,
        )

        # Redirect the agent to the reward-wrapped environment
        agent.set_env(self.reward_wrapper)
        self.agent = agent

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def train(
        self,
        total_timesteps: int,
        total_comparisons: int,
        checkpoint_dir: str,
    ) -> None:
        os.makedirs(checkpoint_dir, exist_ok=True)

        query_allocation = self._make_query_allocation(total_comparisons)
        timesteps_per_iter = total_timesteps // self.num_iterations

        collector = RolloutCollectorCallback()
        callbacks = [collector, CustomLoggingCallback()]

        for i in range(self.num_iterations):
            print(
                f"\n[Christiano] Iteration {i + 1}/{self.num_iterations} "
                f"— {timesteps_per_iter} timesteps, {query_allocation[i]} queries"
            )

            # 1. Train agent; collector captures rollout data via callback
            self.agent.learn(
                total_timesteps=timesteps_per_iter,
                callback=callbacks,
                reset_num_timesteps=(i == 0),
            )

            # 2. Retrieve completed episode trajectories
            trajectories = collector.flush()

            # 3. Fragment → pair → label
            n_queries = query_allocation[i]
            pairs = _fragment_trajectories(
                trajectories, self.fragment_length, n_queries, self.rng
            )

            if not pairs:
                print("  [Warning] No segment pairs generated — skipping reward model update.")
            else:
                preferences = self.gatherer.label(pairs)
                self.dataset.push(pairs, preferences)

                # 4. Train reward model
                metrics = self.reward_trainer.train(self.dataset)

                print(
                    f"  Dataset: {len(self.dataset)} pairs | "
                    f"RM loss: {metrics['reward_model/loss']:.4f} | "
                    f"RM accuracy: {metrics['reward_model/accuracy']:.2%}"
                )

                if wandb.run is not None:
                    wandb.log({
                        "iteration": i + 1,
                        "dataset/size": len(self.dataset),
                        "dataset/n_queries": n_queries,
                        **metrics,
                    })

            # 5. Save checkpoint
            ckpt_path = os.path.join(checkpoint_dir, f"agent_iter_{i + 1:03d}")
            self.agent.save(ckpt_path)

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _make_query_allocation(self, total_comparisons: int) -> List[int]:
        """Distribute total_comparisons uniformly across all iterations."""
        schedule_fn = QUERY_SCHEDULES[self._query_schedule_name]
        return schedule_fn(self.num_iterations, total_comparisons)
