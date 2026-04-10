import numpy as np
import torch
import torch.nn as nn

from .core import Segment


# ---------------------------------------------------------------------------
# Ego status encoding
# ---------------------------------------------------------------------------

STATUS_ORDER = ["running", "arrived", "collided", "off_road"]


def encode_ego_status(status_str: str) -> np.ndarray:
    """One-hot encode ego_status string → float32 array of shape (4,)."""
    vec = np.zeros(4, dtype=np.float32)
    if status_str in STATUS_ORDER:
        vec[STATUS_ORDER.index(status_str)] = 1.0
    return vec


# ---------------------------------------------------------------------------
# Reward network
# ---------------------------------------------------------------------------

class RewardNet(nn.Module):
    def __init__(self, obs_dim, action_dim, info_dim, hidden_dim=128):
        super().__init__()
        input_dim = obs_dim + action_dim + info_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, action_enc: torch.Tensor, info: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs        : (B, obs_dim)
            action_enc : (B, action_dim)  one-hot if discrete, raw if continuous
            info       : (B, info_dim)    one-hot ego_status
        Returns:
            rewards    : (B,)
        """
        x = torch.cat([obs, action_enc, info], dim=-1)
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Ensemble reward model
# ---------------------------------------------------------------------------

class EnsembleRewardModel:
    """
    Ensemble of K independent RewardNets.

    Supports both discrete actions (one-hot encoded) and continuous actions
    (passed through as-is). Set discrete_actions=False for continuous spaces.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_ensembles: int,
        lr: float = 2e-4,
        hidden_dim: int = 64,
        info_dim: int = 4,
        device: str = "cpu",
        discrete_actions: bool = True,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_ensembles = n_ensembles
        self.info_dim = info_dim
        self.device = torch.device(device)
        self.discrete_actions = discrete_actions

        self.nets = [
            RewardNet(obs_dim, action_dim, info_dim, hidden_dim).to(self.device)
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
        """
        Encode actions for network input.
        Discrete: one-hot vector of shape (B, action_dim).
        Continuous: raw float array of shape (B, action_dim).
        """
        if self.discrete_actions:
            actions = np.asarray(actions, dtype=np.int64).reshape(-1)
            enc = np.zeros((len(actions), self.action_dim), dtype=np.float32)
            enc[np.arange(len(actions)), actions] = 1.0
            return torch.as_tensor(enc, device=self.device)
        else:
            return torch.as_tensor(
                np.asarray(actions, dtype=np.float32).reshape(-1, self.action_dim),
                device=self.device,
            )

    def _obs_tensor(self, obs: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)

    def _info_tensor(self, info: np.ndarray, batch_size: int) -> torch.Tensor:
        """Convert info array to tensor. Uses zeros if info is None."""
        if info is None:
            return torch.zeros(batch_size, self.info_dim, device=self.device)
        return torch.as_tensor(np.asarray(info, dtype=np.float32), device=self.device)

    def _forward_all(self, obs: np.ndarray, actions: np.ndarray, info: np.ndarray = None) -> torch.Tensor:
        """Run all ensemble members. Returns (n_ensembles, B) tensor. No grad."""
        obs_t = self._obs_tensor(obs)
        act_t = self._encode_actions(actions)
        info_t = self._info_tensor(info, obs_t.shape[0])
        with torch.no_grad():
            return torch.stack([net(obs_t, act_t, info_t) for net in self.nets])

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, obs: np.ndarray, actions: np.ndarray, pessimism: float = 0.0, info: np.ndarray = None) -> np.ndarray:
        """Mean reward across ensemble, optionally penalized by ensemble std.

        Args:
            pessimism: If > 0, returns mean - pessimism * std. This penalizes
                       states where ensemble members disagree (OOD regions),
                       reducing reward hacking by the policy.
            info: (B, info_dim) one-hot ego_status array. Zeros if None.
        Returns:
            (B,) float32 array of predicted rewards.
        """
        all_preds = self._forward_all(obs, actions, info)  # (n_ensembles, B)
        mean = all_preds.mean(dim=0)
        if pessimism > 0.0:
            std = all_preds.std(dim=0)
            return (mean - pessimism * std).cpu().numpy()
        return mean.cpu().numpy()

    def ensemble_variance(self, obs: np.ndarray, actions: np.ndarray, info: np.ndarray = None) -> np.ndarray:
        """Variance of predictions across ensemble members, shape (B,)."""
        return self._forward_all(obs, actions, info).var(dim=0).cpu().numpy()

    # ------------------------------------------------------------------
    # Differentiable segment returns (used by trainer)
    # ------------------------------------------------------------------

    def segment_returns(self, segment: Segment, ensemble_idx: int) -> torch.Tensor:
        """
        Sum of predicted rewards over a segment for ensemble member `ensemble_idx`.
        Differentiable — used inside the training loss.
        """
        obs = np.stack([t.obs for t in segment.transitions])
        actions = np.array([t.action for t in segment.transitions])
        info = np.stack([
            encode_ego_status(t.info.get("ego_status", "running"))
            if t.info is not None else np.zeros(self.info_dim, dtype=np.float32)
            for t in segment.transitions
        ])
        obs_t = self._obs_tensor(obs)
        act_t = self._encode_actions(actions)
        info_t = torch.as_tensor(info, device=self.device)
        return self.nets[ensemble_idx](obs_t, act_t, info_t).sum()
