"""
Log-probability oracle for synthetic preference labeling.

Scores a segment as the average log-probability of the agent's actions
under the expert's Boltzmann policy:

    score(σ) = (1/T) Σ_t log π_exp(a_t | s_t)

where π_exp(a|s) = softmax(Q_exp(s, ·) / τ)[a].

Higher score → agent's actions were more "expert-like" in this segment.
Requires segments to carry a .actions attribute (list of int).
"""

from typing import Optional, Tuple

import torch

from .base import BaseOracle


class LogProbOracle(BaseOracle):
    """
    Preference oracle based on average log-probability of agent actions
    under the expert's Boltzmann policy.

    Args:
        label_mode:   "hard" | "soft"
        expert_model: SB3 DQN with a .q_net attribute
        temperature:  Boltzmann temperature τ for softmax over Q-values.
                      Lower τ → more peaked distribution → higher contrast
                      between expert and non-expert actions.
    """

    def __init__(
        self,
        label_mode: str = "soft",
        expert_model=None,
        temperature: float = 1.0,
    ):
        if expert_model is None:
            raise ValueError("LogProbOracle requires expert_model")
        if label_mode not in ("soft", "hard"):
            raise ValueError(f"label_mode must be 'soft' or 'hard', got {label_mode!r}")
        self.label_mode   = label_mode
        self.expert_model = expert_model
        self.temperature  = temperature

    def label(self, seg1, seg2) -> Optional[Tuple[float, float]]:
        if not hasattr(seg1, "actions") or not hasattr(seg2, "actions"):
            return None

        score1 = self._score(seg1)
        score2 = self._score(seg2)

        if self.label_mode == "soft":
            probs = torch.softmax(torch.tensor([score1, score2]), dim=0)
            return tuple(probs.tolist())

        if abs(score1 - score2) < 1e-6:
            return (0.5, 0.5)
        return (1.0, 0.0) if score1 > score2 else (0.0, 1.0)

    def _score(self, seg) -> float:
        """(1/T) Σ log π_exp(a_agent | s) under Boltzmann policy."""
        total = 0.0
        T = len(seg.frames)
        for frame, action in zip(seg.frames, seg.actions):
            obs = torch.as_tensor(frame, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_vals = self.expert_model.q_net(obs)[0]           # (n_actions,)
            log_probs = torch.log_softmax(q_vals / self.temperature, dim=0)
            total += log_probs[int(action)].item()
        return total / T
