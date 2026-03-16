"""
Expert oracle for synthetic preference labeling.

Supports two modes:
  "env_reward"  — prefer segment with higher sum of true environment rewards
  "qnet"        — prefer segment with higher sum of V(s) = max_a Q(s, a)

Moved and adapted from human_feedback_rl/christiano/expert_pref_interface.py.
Does NOT inherit from PrefInterface.
"""

from typing import Optional, Tuple

import torch

from .base import BaseOracle


class ExpertOracle(BaseOracle):
    """
    Synthetic preference oracle supporting two scoring modes.

    Args:
        mode:         "env_reward" | "qnet"
        expert_model: SB3 DQN with a .q_net attribute — required when mode="qnet"
    """

    def __init__(self, mode: str = "env_reward", expert_model=None):
        self.mode = mode
        self.expert_model = expert_model

        if mode == "qnet" and expert_model is None:
            raise ValueError("mode='qnet' requires expert_model to be provided")

    # ------------------------------------------------------------------

    def label(self, seg1, seg2) -> Optional[Tuple[float, float]]:
        """
        Score both segments and return a soft preference (p1, p2) where p1+p2=1.0.
        """
        if self.mode == "qnet":
            score1 = self._score_qnet(seg1)
            score2 = self._score_qnet(seg2)
        else:  # "env_reward"
            score1 = self._score_env_reward(seg1)
            score2 = self._score_env_reward(seg2)

        probs = torch.softmax(torch.tensor([score1, score2]), dim=0)
        p1, p2 = probs.tolist()
        return (p1, p2)

    # ------------------------------------------------------------------

    def _score_env_reward(self, seg) -> float:
        """Sum of true environment rewards over the segment (seg.env_rewards)."""
        return float(sum(getattr(seg, "env_rewards", [])))

    def _score_qnet(self, seg) -> float:
        """Sum of V(s) = max_a Q(s, a) over all frames using the expert DQN."""
        total = 0.0
        for frame in seg.frames:
            obs = torch.as_tensor(frame, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_vals = self.expert_model.q_net(obs)[0]
            total += q_vals.max().item()
        return total
