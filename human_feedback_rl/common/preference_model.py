import numpy as np
import torch
from typing import Tuple

from .core import Segment, Preference


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

    def __init__(self, reward_model):
        self.reward_model = reward_model

    def preference_probs(self, seg1: Segment, seg2: Segment) -> Preference:
        """
        Returns Preference with (p1, p2) where p1 = P(seg1 preferred).
        Uses mean reward across ensemble members.
        """
        rm = self.reward_model
        with torch.no_grad():
            r1_vals = [rm.segment_returns(seg1, k).item() for k in range(rm.n_ensembles)]
            r2_vals = [rm.segment_returns(seg2, k).item() for k in range(rm.n_ensembles)]

        r1 = float(np.mean(r1_vals)) / len(seg1.transitions)
        r2 = float(np.mean(r2_vals)) / len(seg2.transitions)

        probs = torch.softmax(torch.tensor([r1, r2]), dim=0)
        return Preference((float(probs[0].item()), float(probs[1].item())))

    def preference_logits_for_net(
        self, seg1: Segment, seg2: Segment, ensemble_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Differentiable (R1, R2) for one ensemble member. Used by the trainer."""
        rm = self.reward_model
        r1 = rm.segment_returns(seg1, ensemble_idx) / len(seg1.transitions)
        r2 = rm.segment_returns(seg2, ensemble_idx) / len(seg2.transitions)
        return r1, r2