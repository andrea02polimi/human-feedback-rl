from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from .core import Preference, PreferenceDataset


# ---------------------------------------------------------------------------
# Running mean/std  (Welford's online algorithm)
# ---------------------------------------------------------------------------


class RunningMeanStd:
    """
    Online estimate of mean and variance using Welford's algorithm.
    Used to compute running z-score normalization for predicted rewards.
    """

    def __init__(self, epsilon: float = 1e-8) -> None:
        self.mean: float = 0.0
        self.var: float = 1.0
        self.count: int = 0
        self.epsilon = epsilon

    def update(self, values: np.ndarray) -> None:
        """Update running statistics with a batch of scalar values."""
        values = np.asarray(values, dtype=np.float64).ravel()
        if values.size == 0:
            return
        batch_count = int(values.size)
        batch_mean = float(np.mean(values))
        batch_var = float(np.var(values))

        total_count = self.count + batch_count
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, values: np.ndarray) -> np.ndarray:
        """Normalize by std only: x / std. Preserves sign and absolute scale."""
        std = float(np.sqrt(max(self.var, 0.0))) + self.epsilon
        return np.asarray(values, dtype=np.float32) / std


class RewardNet(nn.Module):
    """
    MLP reward model mapping (obs, action) -> scalar reward.

    Architecture follows Christiano et al. 2017: two hidden layers with
    tanh activations. Tanh is used instead of ReLU for smoother gradients
    and bounded pre-activations.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs:    (batch, obs_dim)
            action: (batch, action_dim)
        Returns:
            reward: (batch,)
        """
        x = torch.cat([obs, action], dim=-1)
        return self.net(x).squeeze(-1)


class EnsembleRewardModel:
    """
    Ensemble of RewardNets trained with the Bradley-Terry preference model.

    Loss for a preference (σ1, σ2, μ):
        R1 = Σ_t r̂(o_t^1, a_t^1)
        R2 = Σ_t r̂(o_t^2, a_t^2)
        loss = BCE(R1 - R2, μ)          [binary cross-entropy with logits]

    where μ=1 means σ1 is preferred, μ=0 means σ2 is preferred, μ=0.5 is a tie.

    Each network in the ensemble is trained on independent random mini-batches,
    following the procedure from the paper.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_networks: int = 3,
        hidden_size: int = 256,
        lr: float = 3e-4,
        l2_reg: float = 1e-4,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_networks = n_networks
        self.device = torch.device(device)

        self.networks: List[RewardNet] = [
            RewardNet(obs_dim, action_dim, hidden_size).to(self.device)
            for _ in range(n_networks)
        ]
        self.optimizers = [
            torch.optim.Adam(net.parameters(), lr=lr, weight_decay=l2_reg)
            for net in self.networks
        ]

        for net in self.networks:
            net.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_reward(self, obs: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Predict reward as mean across ensemble.

        Args:
            obs:    (batch, obs_dim) or (obs_dim,)
            action: (batch, action_dim) or (action_dim,)
        Returns:
            reward: (batch,) or scalar float
        """
        scalar_input = obs.ndim == 1
        if scalar_input:
            obs = obs[np.newaxis]
            action = action[np.newaxis]

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        act_t = torch.tensor(action, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            preds = np.stack(
                [net(obs_t, act_t).cpu().numpy() for net in self.networks]
            )  # (n_networks, batch)

        mean_reward = preds.mean(axis=0)  # (batch,)
        return float(mean_reward[0]) if scalar_input else mean_reward

    def train(
        self,
        dataset: PreferenceDataset,
        n_steps: int,
        batch_size: int,
        rng: np.random.Generator,
    ) -> Dict[str, float]:
        """
        Train each network independently for n_steps gradient steps.

        Each network samples its own random mini-batch from the dataset at every
        step, following the independent training procedure in the paper.

        Returns:
            dict with training metrics (average loss across networks).
        """
        if len(dataset) < batch_size:
            return {}

        total_loss = 0.0

        for net, optimizer in zip(self.networks, self.optimizers):
            net.train()
            net_loss = 0.0
            for _ in range(n_steps):
                batch = dataset.sample(batch_size, rng)
                optimizer.zero_grad()
                loss = self._preference_loss(net, batch)
                loss.backward()
                optimizer.step()
                net_loss += loss.item()
            net.eval()
            total_loss += net_loss / n_steps

        return {"reward_model/loss": total_loss / self.n_networks}

    def accuracy(
        self,
        dataset: PreferenceDataset,
        batch_size: int,
        rng: np.random.Generator,
    ) -> float:
        """
        Fraction of non-tie preferences correctly predicted by the ensemble mean.
        Returns nan if there are no non-tie samples in the batch.
        """
        if len(dataset) < batch_size:
            return float("nan")

        batch = dataset.sample(batch_size, rng)
        non_tie = [p for p in batch if p.label != 0.5]
        if not non_tie:
            return float("nan")

        correct = 0
        for pref in non_tie:
            r1 = self._segment_return(pref.seg1.obs, pref.seg1.actions)
            r2 = self._segment_return(pref.seg2.obs, pref.seg2.actions)
            predicted = 1.0 if r1 > r2 else 0.0
            if predicted == pref.label:
                correct += 1

        return correct / len(non_tie)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _segment_return(self, obs: np.ndarray, actions: np.ndarray) -> float:
        """Sum of predicted rewards over a segment, averaged across ensemble."""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        act_t = torch.tensor(actions, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            total = sum(
                net(obs_t, act_t).sum().item() for net in self.networks
            )
        return total / self.n_networks

    def _preference_loss(
        self,
        net: RewardNet,
        batch: List[Preference],
    ) -> torch.Tensor:
        """
        Bradley-Terry cross-entropy loss for a batch of preferences.

        logit = R1 - R2  where R_i = Σ_t r̂(o_t^i, a_t^i)
        loss  = BCE_with_logits(logit, μ)
        """
        logits, labels = [], []

        for pref in batch:
            obs1 = torch.tensor(pref.seg1.obs, dtype=torch.float32, device=self.device)
            act1 = torch.tensor(pref.seg1.actions, dtype=torch.float32, device=self.device)
            obs2 = torch.tensor(pref.seg2.obs, dtype=torch.float32, device=self.device)
            act2 = torch.tensor(pref.seg2.actions, dtype=torch.float32, device=self.device)

            r1 = net(obs1, act1).sum()
            r2 = net(obs2, act2).sum()

            logits.append(r1 - r2)
            labels.append(pref.label)

        logits_t = torch.stack(logits)
        labels_t = torch.tensor(labels, dtype=torch.float32, device=self.device)

        return nn.functional.binary_cross_entropy_with_logits(logits_t, labels_t)