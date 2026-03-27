from . import *
import torch
import torch.nn as nn

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

