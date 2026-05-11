from typing import List

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from .types import FragmentPair, Fragment

from .reward_nets import RewardNet


class PreferenceModelFromReward(nn.Module):
    def __init__(self, reward_model: RewardNet):
        super().__init__()
        self.reward_model = reward_model

    def forward(self, fragment_pairs: List[FragmentPair]) -> th.Tensor:
        r1_list, r2_list = [], []
        for pair in fragment_pairs:
            r1_list.append(self._sum_rewards(pair.frag1))
            r2_list.append(self._sum_rewards(pair.frag2))

        r1 = th.stack(r1_list)               # (batch_size,)
        r2 = th.stack(r2_list)               # (batch_size,)

        logits = th.stack([r1, r2], dim=1)   # (batch_size, 2)
        return F.softmax(logits, dim=1)       # Bradley-Terry probs

    def _sum_rewards(self, fragment: Fragment) -> th.Tensor:
        obs         = th.tensor(np.array([t.observation  for t in fragment]), dtype=th.float32)
        actions     = th.tensor(np.array([t.action       for t in fragment]), dtype=th.float32)
        next_status = th.tensor(np.array([t.next_status  for t in fragment]), dtype=th.float32)
        done        = th.tensor(np.array([float(t.done)  for t in fragment]), dtype=th.float32)
        return self.reward_model(obs, actions, next_status, done).sum() / len(fragment)