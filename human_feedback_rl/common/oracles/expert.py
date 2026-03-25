"""
Expert oracle for synthetic preference labeling.

Supports three scoring modes:
  "env_reward"  — prefer segment with higher sum of true environment rewards
  "qnet"        — prefer segment with higher sum of V(s) = max_a Q(s, a)
  "q_action"    — prefer segment with higher mean Q_exp(s, a_agent)
                  (captures both state quality and action quality)

Supports two labeling modes:
  "hard"  — (1,0) / (0,1) / (0.5,0.5)  [original Christiano et al.]
  "soft"  — softmax over scores
"""

from typing import List, Optional, Tuple

import torch

from .base import BaseOracle


class ExpertOracle(BaseOracle):
    """
    Synthetic preference oracle supporting three scoring modes and two label modes.

    Args:
        mode:         "env_reward" | "qnet" | "q_action"
        label_mode:   "hard" | "soft"
        expert_model: SB3 DQN with a .q_net attribute — required when mode != "env_reward"
    """

    def __init__(
        self,
        mode: str = "env_reward",
        label_mode: str = "hard",
        expert_model=None,
    ):
        if mode in ("qnet", "q_action") and expert_model is None:
            raise ValueError(f"mode={mode!r} requires expert_model to be provided")
        if label_mode not in ("soft", "hard"):
            raise ValueError(f"label_mode must be 'soft' or 'hard', got {label_mode!r}")
        self.mode         = mode
        self.label_mode   = label_mode
        self.expert_model = expert_model

    # ------------------------------------------------------------------

    def label(self, seg1, seg2) -> Optional[Tuple[float, float]]:
        """
        Score both segments and return a preference (p1, p2) where p1+p2=1.0.

        hard mode: (1,0) / (0,1) / (0.5,0.5) — original Christiano et al.
        soft mode: softmax over raw scores.
        """
        if self.mode == "qnet":
            score1 = self._score_qnet(seg1)
            score2 = self._score_qnet(seg2)
        elif self.mode == "q_action":
            score1 = self._score_q_action(seg1)
            score2 = self._score_q_action(seg2)
        else:  # "env_reward"
            score1 = self._score_env_reward(seg1)
            score2 = self._score_env_reward(seg2)

        if self.label_mode == "soft":
            probs = torch.softmax(torch.tensor([score1, score2]), dim=0)
            return tuple(probs.tolist())

        # hard labels
        if abs(score1 - score2) < 1e-6:
            return (0.5, 0.5)
        return (1.0, 0.0) if score1 > score2 else (0.0, 1.0)

    # ------------------------------------------------------------------

    def _score_env_reward(self, seg) -> float:
        """Sum of true environment rewards over the segment (seg.env_rewards)."""
        return float(sum(getattr(seg, "env_rewards", [])))

    def _frame_qnet_values(self, seg) -> List[float]:
        """Return per-frame V(s) = max_a Q(s,a) for all frames in the segment."""
        values = []
        for frame in seg.frames:
            obs = torch.as_tensor(frame, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_vals = self.expert_model.q_net(obs)[0]
            values.append(q_vals.max().item())
        return values

    def _score_qnet(self, seg) -> float:
        """Sum of V(s) = max_a Q(s, a) over all frames using the expert DQN."""
        return sum(self._frame_qnet_values(seg))

    def _score_q_action(self, seg) -> float:
        """Mean of Q_exp(s_t, a_agent_t) over the segment.

        Captures both state quality (via Q-values) and action quality
        (which action the agent actually took). Unlike qnet, sub-optimal
        actions in a given state are penalised because Q(s,a) < max_a Q(s,a).
        Requires seg.actions to be set.
        """
        if not hasattr(seg, "actions") or not seg.actions:
            return 0.0
        total = 0.0
        for frame, action in zip(seg.frames, seg.actions):
            obs = torch.as_tensor(frame, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_vals = self.expert_model.q_net(obs)[0]
            total += q_vals[int(action)].item()
        return total / len(seg.frames)
