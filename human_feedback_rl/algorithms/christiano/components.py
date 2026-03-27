"""
Supporting components for ChristianoAlgorithm.

    - EnsembleRewardModel    : ensemble of reward MLPs
    - PreferenceModelFromReward : Bradley-Terry preference model
    - RewardTrainerChristiano : trains reward model on preference dataset
    - ActiveFragmenter       : fragments trajectories into labelled segment pairs
    - EnvRewardWrapper       : VecEnvWrapper that substitutes predicted rewards
"""

import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper

from human_feedback_rl.common.core import (
    Segment,
    SegmentPair,
    Trajectory,
    PreferenceDataset,
)


# ---------------------------------------------------------------------------
# Reward network
# ---------------------------------------------------------------------------

class RewardNet(nn.Module):
    """Single MLP reward predictor: (obs, one_hot_action) -> scalar."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, action_enc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs        : (B, obs_dim)
            action_enc : (B, action_dim)
        Returns:
            rewards    : (B,)
        """
        x = torch.cat([obs, action_enc], dim=-1)
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Ensemble reward model
# ---------------------------------------------------------------------------

class EnsembleRewardModel:
    """
    Ensemble of K independent RewardNets.

    Actions are expected to be discrete integers; they are one-hot encoded
    before being fed to the network.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_ensembles: int,
        lr: float = 2e-4,
        hidden_dim: int = 64,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_ensembles = n_ensembles
        self.device = torch.device(device)

        self.nets = [
            RewardNet(obs_dim, action_dim, hidden_dim).to(self.device)
            for _ in range(n_ensembles)
        ]
        self.optimizers = [
            torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
            for net in self.nets
        ]

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def _encode_actions(self, actions: np.ndarray) -> torch.Tensor:
        """One-hot encode a batch of integer actions. shape: (B, action_dim)."""
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        enc = np.zeros((len(actions), self.action_dim), dtype=np.float32)
        enc[np.arange(len(actions)), actions] = 1.0
        return torch.as_tensor(enc, device=self.device)

    def _obs_tensor(self, obs: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        Mean reward across ensemble for a batch of (obs, action) pairs.

        Args:
            obs     : (B, obs_dim)
            actions : (B,) integer actions
        Returns:
            rewards : (B,) float32
        """
        obs_t = self._obs_tensor(obs)
        act_t = self._encode_actions(actions)
        with torch.no_grad():
            preds = torch.stack([net(obs_t, act_t) for net in self.nets], dim=0)
        return preds.mean(dim=0).cpu().numpy()

    def ensemble_variance(self, obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Variance of predictions across ensemble members, shape (B,)."""
        obs_t = self._obs_tensor(obs)
        act_t = self._encode_actions(actions)
        with torch.no_grad():
            preds = torch.stack([net(obs_t, act_t) for net in self.nets], dim=0)
        return preds.var(dim=0).cpu().numpy()

    # ------------------------------------------------------------------
    # Differentiable segment returns (used by trainer)
    # ------------------------------------------------------------------

    def segment_returns(self, segment: Segment, net_idx: int) -> torch.Tensor:
        """
        Sum of predicted rewards over a segment for ensemble member `net_idx`.
        Differentiable — used inside the training loss.

        Returns:
            scalar tensor
        """
        obs = np.stack([t.obs for t in segment.transitions])
        actions = np.array([t.action for t in segment.transitions], dtype=np.int64)
        obs_t = self._obs_tensor(obs)
        act_t = self._encode_actions(actions)
        return self.nets[net_idx](obs_t, act_t).sum()


# ---------------------------------------------------------------------------
# Preference model
# ---------------------------------------------------------------------------

class PreferenceModelFromReward:
    """
    Bradley-Terry preference model built on top of EnsembleRewardModel.

    P(seg1 > seg2) = exp(R1) / (exp(R1) + exp(R2))

    where R_k = sum_t r_net_k(obs_t, a_t) for each ensemble member k,
    and the final preference probability uses the mean across members.
    """

    def __init__(self, reward_model: EnsembleRewardModel):
        self.reward_model = reward_model

    def preference_probs(self, seg1: Segment, seg2: Segment) -> Tuple[float, float]:
        """
        Returns (p1, p2) where p1 = P(seg1 preferred), p2 = P(seg2 preferred).
        Uses mean reward across ensemble members.
        """
        rm = self.reward_model
        r1_list, r2_list = [], []
        for k in range(rm.n_ensembles):
            with torch.no_grad():
                r1_list.append(rm.segment_returns(seg1, k).item())
                r2_list.append(rm.segment_returns(seg2, k).item())

        r1 = float(np.mean(r1_list))
        r2 = float(np.mean(r2_list))

        logits = torch.tensor([r1, r2])
        probs = torch.softmax(logits, dim=0)
        return float(probs[0].item()), float(probs[1].item())

    def preference_logits_for_net(
        self, seg1: Segment, seg2: Segment, net_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Differentiable (R1, R2) for one ensemble member. Used by the trainer."""
        rm = self.reward_model
        r1 = rm.segment_returns(seg1, net_idx)
        r2 = rm.segment_returns(seg2, net_idx)
        return r1, r2


# ---------------------------------------------------------------------------
# Reward trainer
# ---------------------------------------------------------------------------

class RewardTrainerChristiano:
    """
    Trains EnsembleRewardModel on a PreferenceDataset.

    Loss: mean cross-entropy preference loss (Christiano et al. eq. 1)
    applied independently to each ensemble member.
    """

    def __init__(
        self,
        preference_model: PreferenceModelFromReward,
        batch_size: int = 32,
        n_epochs: int = 10,
    ):
        self.preference_model = preference_model
        self.batch_size = batch_size
        self.n_epochs = n_epochs

    def train(self, dataset: PreferenceDataset) -> float:
        """
        Train on the preference dataset.

        Returns:
            mean loss over the training run.
        """
        if len(dataset) == 0:
            return 0.0

        rm = self.preference_model.reward_model
        total_loss = 0.0
        n_steps = 0

        for _ in range(self.n_epochs):
            indices = list(range(len(dataset)))
            random.shuffle(indices)

            for start in range(0, len(indices), self.batch_size):
                batch_idx = indices[start : start + self.batch_size]

                for opt in rm.optimizers:
                    opt.zero_grad()

                batch_loss = 0.0
                for i in batch_idx:
                    pair = dataset.pairs[i]
                    pref = dataset.targets[i]
                    # label (1,0) -> target=0 (seg1 preferred)
                    # label (0,1) -> target=1 (seg2 preferred)
                    target = torch.tensor(
                        [pref.label.index(max(pref.label))],
                        dtype=torch.long,
                        device=rm.device,
                    )

                    for k in range(rm.n_ensembles):
                        r1, r2 = self.preference_model.preference_logits_for_net(
                            pair.seg1, pair.seg2, k
                        )
                        logits = torch.stack([r1, r2]).unsqueeze(0)
                        loss = F.cross_entropy(logits, target)
                        loss.backward()
                        batch_loss += loss.item()

                for opt in rm.optimizers:
                    opt.step()

                total_loss += batch_loss / len(batch_idx)
                n_steps += 1

        return total_loss / max(n_steps, 1)


# ---------------------------------------------------------------------------
# Fragmenter
# ---------------------------------------------------------------------------

class ActiveFragmenter:
    """
    Extracts fixed-length segments from trajectories and pairs them.

    If the reward model has already been trained, pairs are selected by
    ensemble disagreement (active learning). Otherwise random selection is used.
    """

    def __init__(
        self,
        reward_model: EnsembleRewardModel,
        length_segment: int,
        num_max_segment_pairs: int,
    ):
        self.reward_model = reward_model
        self.length_segment = length_segment
        self.num_max_segment_pairs = num_max_segment_pairs

    def fragment(self, trajectories: List[Trajectory]) -> List[SegmentPair]:
        segments = self._extract_segments(trajectories)
        if len(segments) < 2:
            return []

        n_pairs = min(self.num_max_segment_pairs, len(segments) // 2)

        # Active selection: prefer segment pairs with high ensemble disagreement
        if len(segments) > n_pairs * 2:
            pairs = self._active_pairs(segments, n_pairs)
        else:
            pairs = self._random_pairs(segments, n_pairs)

        return pairs

    # ------------------------------------------------------------------

    def _extract_segments(self, trajectories: List[Trajectory]) -> List[Segment]:
        segments = []
        for traj in trajectories:
            transitions = traj.transitions
            T = len(transitions)
            if T < self.length_segment:
                continue
            start = 0
            while start + self.length_segment <= T:
                segments.append(Segment(transitions[start : start + self.length_segment]))
                start += self.length_segment
        return segments

    def _random_pairs(self, segments: List[Segment], n_pairs: int) -> List[SegmentPair]:
        idxs = list(range(len(segments)))
        random.shuffle(idxs)
        pairs = []
        for i in range(0, min(n_pairs * 2, len(idxs) - 1), 2):
            pairs.append(SegmentPair(seg1=segments[idxs[i]], seg2=segments[idxs[i + 1]]))
        return pairs

    def _active_pairs(self, segments: List[Segment], n_pairs: int) -> List[SegmentPair]:
        """Score candidate pairs by ensemble disagreement and take the top ones."""
        rm = self.reward_model

        # Score each segment by mean per-step ensemble variance
        def _seg_score(seg: Segment) -> float:
            obs = np.stack([t.obs for t in seg.transitions])
            actions = np.array([t.action for t in seg.transitions], dtype=np.int64)
            return float(rm.ensemble_variance(obs, actions).mean())

        scored = [(s, _seg_score(s)) for s in segments]
        # Sort descending by uncertainty
        scored.sort(key=lambda x: x[1], reverse=True)

        # Take top-K uncertain segments
        top_k = min(n_pairs * 2, len(scored))
        top_segs = [s for s, _ in scored[:top_k]]

        pairs = []
        for i in range(0, len(top_segs) - 1, 2):
            pairs.append(SegmentPair(seg1=top_segs[i], seg2=top_segs[i + 1]))
            if len(pairs) >= n_pairs:
                break
        return pairs


# ---------------------------------------------------------------------------
# Environment reward wrapper
# ---------------------------------------------------------------------------

class EnvRewardWrapper(VecEnvWrapper):
    """
    VecEnvWrapper that replaces environment rewards with EnsembleRewardModel
    predictions. Uses the pre-step observation for reward prediction to align
    with how the reward predictor was trained on (obs_t, a_t) pairs.
    """

    def __init__(self, venv: VecEnv, reward_model: EnsembleRewardModel):
        super().__init__(venv)
        self.reward_model = reward_model
        self._current_obs: np.ndarray | None = None
        self._last_actions: np.ndarray | None = None

    def reset(self):
        obs = self.venv.reset()
        self._current_obs = np.asarray(obs, dtype=np.float32)
        return obs

    def step_async(self, actions):
        self._last_actions = np.asarray(actions)
        self.venv.step_async(actions)

    def step_wait(self):
        obs, _env_rewards, dones, infos = self.venv.step_wait()

        if self._current_obs is not None and self._last_actions is not None:
            rewards = self.reward_model.predict(
                self._current_obs, self._last_actions
            ).astype(np.float32)
        else:
            rewards = np.zeros(len(obs), dtype=np.float32)

        self._current_obs = np.asarray(obs, dtype=np.float32)
        return obs, rewards, dones, infos