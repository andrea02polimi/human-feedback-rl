import torch
import torch.nn as nn


class SumoRewardNetwork(nn.Module):
    """
    Action-conditioned reward network for flat vector observations.

    Input = concat(obs, action_features) where action_features is
    pre-encoded by the ensemble:
      - discrete actions:   one_hot(action, n_actions)  → dim = n_actions
      - continuous actions: raw action vector            → dim = action_dim

    Follows the contract required by RewardPredictorEnsemble:
        forward(obs, action_features) -> (N,)   scalar reward per step

    Args:
        obs_dim:            dimensionality of the observation vector
        action_feature_dim: dimensionality of the (already encoded) action features
        hidden_dim:         width of the two hidden layers (default 64)
        dropout:            dropout rate (default 0.2)
    """

    def __init__(self, obs_dim: int, action_feature_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.action_feature_dim = action_feature_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, action_features: torch.Tensor) -> torch.Tensor:
        # obs: (N, obs_dim), action_features: (N, action_feature_dim) — already encoded
        x = torch.cat([obs.float(), action_features.float()], dim=-1)
        return self.net(x).squeeze(-1)  # (N,)
