import torch
import torch.nn as nn


class SumoRewardNetwork(nn.Module):
    """
    Reward network for flat vector observations (SUMO / highway environments).

    Follows the contract required by RewardPredictorEnsemble:
        forward(obs) -> (N,)   scalar reward per step

    Args:
        obs_dim:    dimensionality of the observation vector
        hidden_dim: width of the two hidden layers (default 64)
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (N, obs_dim) — float tensor (already normalised upstream)
        return self.net(obs.float()).squeeze(-1)  # (N,)
