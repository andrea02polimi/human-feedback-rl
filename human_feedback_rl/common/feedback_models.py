from . import *
from typing import Tuple
import numpy as np
import torch


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



