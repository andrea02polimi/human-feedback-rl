from typing import List

import torch as th
import torch.nn as nn
from .types import FragmentPair

from .reward_nets import RewardNet
from .bradley_terry import BradleyTerry


class PreferenceModelFromReward(nn.Module):
    """Wraps a RewardNet to produce Bradley-Terry preference probabilities over fragment pairs."""

    def __init__(self, reward_model: RewardNet):
        super().__init__()
        self.reward_model = reward_model

    def forward(self, fragment_pairs: List[FragmentPair]) -> th.Tensor:
        """Returns BT preference probs of shape (batch, 2)."""
        r1 = th.stack([self.reward_model.fragment_avg_reward(p.frag1) for p in fragment_pairs])
        r2 = th.stack([self.reward_model.fragment_avg_reward(p.frag2) for p in fragment_pairs])
        return BradleyTerry.torch(r1, r2)
