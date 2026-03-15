import torch
import torch.nn as nn


class AgentPolicyNetwork(nn.Module):
    """
    Shared-trunk actor-critic network for A2C (Christiano et al. 2017).

    forward(obs) → (logits, values)
        logits : (N, n_actions)  — unnormalised action scores for the actor
        values : (N,)            — state-value estimates V(s) for the critic
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor  = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        h = self.shared(x.float())
        return self.actor(h), self.critic(h).squeeze(-1)   # logits, values
