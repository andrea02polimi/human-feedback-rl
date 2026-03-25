"""
Log-probability oracle for PPO experts.

Scores a segment as the average log-probability of the agent's actions
under the expert PPO policy's distribution:

    score(σ) = (1/T) Σ_t log π_exp(a_t | s_t)

where π_exp is the PPO policy's stochastic distribution (e.g. Gaussian
for continuous actions). Log-probs are obtained via SB3's
policy.evaluate_actions(), which works for any ActorCriticPolicy
(PPO, A2C, SAC, etc.).

Higher score → agent's actions were more "expert-like" in this segment.
Requires segments to carry a .actions attribute (list of arrays or ints).
"""

from typing import Optional, Tuple

import numpy as np
import torch

from .base import BaseOracle


class PPOLogProbOracle(BaseOracle):
    """
    Preference oracle based on log-probability of agent actions under
    the expert PPO policy.

    Unlike LogProbOracle (which approximates the distribution via
    Boltzmann over Q-values), this oracle queries the policy distribution
    directly via policy.evaluate_actions().

    Args:
        label_mode:   "hard" | "soft"
        expert_model: SB3 PPO (or any ActorCriticPolicy) with a .policy attribute
        temperature:  optional scaling: score = log_prob / temperature.
                      Default 1.0 (no scaling).
    """

    def __init__(
        self,
        label_mode: str = "soft",
        expert_model=None,
        temperature: float = 1.0,
    ):
        if expert_model is None:
            raise ValueError("PPOLogProbOracle requires expert_model")
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
        """(1/T) Σ log π_exp(a_agent | s), scaled by 1/temperature."""
        T = len(seg.frames)
        frames_t  = torch.as_tensor(np.array(seg.frames),   dtype=torch.float32)   # (T, obs_dim)
        actions_t = torch.as_tensor(np.array(seg.actions),  dtype=torch.float32)   # (T,) or (T, act_dim)
        if actions_t.ndim == 1:
            # Discrete: evaluate_actions expects (T, 1) for some envs, but
            # SB3 Categorical distribution accepts (T,) directly.
            pass
        with torch.no_grad():
            _, log_probs, _ = self.expert_model.policy.evaluate_actions(frames_t, actions_t)
        # log_probs: (T,)
        return (log_probs.sum().item() / T) / self.temperature
