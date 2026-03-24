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

OOD filtering (qnet mode only):
  Tracks a running distribution of per-frame V(s) = max_a Q(s,a) values seen
  so far (Welford's online algorithm). After a warmup period, any preference
  pair where *either* segment contains a frame whose V(s) exceeds
  mean + ood_k * std is discarded (label() returns None). This prevents
  OOD Q-value extrapolation from poisoning the reward predictor.

  Parameters:
    ood_k        — z-score threshold (default 3.0; None = disabled)
    ood_warmup   — number of frames to accumulate before filtering starts
                   (default 5000; lets the oracle see the early distribution)
"""

from typing import List, Optional, Tuple

import torch

from .base import BaseOracle


class ExpertOracle(BaseOracle):
    """
    Synthetic preference oracle supporting two scoring modes and two label modes.

    Args:
        mode:         "env_reward" | "qnet"
        label_mode:   "hard" | "soft"
        expert_model: SB3 DQN with a .q_net attribute — required when mode="qnet"
        ood_k:        OOD z-score threshold for qnet filtering (None = disabled)
        ood_warmup:   frames to observe before OOD filtering activates
    """

    def __init__(
        self,
        mode: str = "env_reward",
        label_mode: str = "hard",
        expert_model=None,
        ood_k: Optional[float] = None,
        ood_warmup: int = 5000,
    ):
        if mode in ("qnet", "q_action") and expert_model is None:
            raise ValueError(f"mode={mode!r} requires expert_model to be provided")
        if label_mode not in ("soft", "hard"):
            raise ValueError(f"label_mode must be 'soft' or 'hard', got {label_mode!r}")
        self.mode         = mode
        self.label_mode   = label_mode
        self.expert_model = expert_model
        self.ood_k        = ood_k
        self.ood_warmup   = ood_warmup

        # Welford's online stats for per-frame V(s) values
        self._n_frames: int   = 0
        self._mean: float     = 0.0
        self._M2: float       = 0.0   # sum of squared deviations

        # Counters for WandB logging (read externally via properties)
        self._n_labeled:  int = 0
        self._n_filtered: int = 0

    # ------------------------------------------------------------------
    # Public stats (read by preference_worker for WandB logging)

    @property
    def ood_mean(self) -> float:
        return self._mean

    @property
    def ood_std(self) -> float:
        return (self._M2 / self._n_frames) ** 0.5 if self._n_frames > 1 else 0.0

    @property
    def ood_filter_rate(self) -> float:
        total = self._n_labeled + self._n_filtered
        return self._n_filtered / total if total > 0 else 0.0

    # ------------------------------------------------------------------

    def label(self, seg1, seg2) -> Optional[Tuple[float, float]]:
        """
        Score both segments and return a preference (p1, p2) where p1+p2=1.0.

        hard mode: (1,0) / (0,1) / (0.5,0.5) — original Christiano et al.
        soft mode: softmax over raw scores.

        Returns None if OOD filtering is active and either segment contains
        a frame with V(s) above mean + ood_k * std.
        """
        if self.mode == "qnet":
            frame_vals1 = self._frame_qnet_values(seg1)
            frame_vals2 = self._frame_qnet_values(seg2)

            # Update running stats with all frames from both segments
            for v in frame_vals1 + frame_vals2:
                self._update_stats(v)

            # OOD filter: discard pair if any frame is too far from distribution
            if self._is_ood_pair(frame_vals1, frame_vals2):
                self._n_filtered += 1
                return None

            score1 = sum(frame_vals1)
            score2 = sum(frame_vals2)

        elif self.mode == "q_action":
            score1 = self._score_q_action(seg1)
            score2 = self._score_q_action(seg2)
        else:  # "env_reward"
            score1 = self._score_env_reward(seg1)
            score2 = self._score_env_reward(seg2)

        self._n_labeled += 1

        if self.label_mode == "soft":
            probs = torch.softmax(torch.tensor([score1, score2]), dim=0)
            return tuple(probs.tolist())

        # hard labels
        if abs(score1 - score2) < 1e-6:
            return (0.5, 0.5)
        return (1.0, 0.0) if score1 > score2 else (0.0, 1.0)

    # ------------------------------------------------------------------

    def _update_stats(self, value: float) -> None:
        """Welford's online algorithm for running mean and variance."""
        self._n_frames += 1
        delta = value - self._mean
        self._mean += delta / self._n_frames
        delta2 = value - self._mean
        self._M2 += delta * delta2

    def _is_ood_pair(self, vals1: List[float], vals2: List[float]) -> bool:
        """Return True if OOD filtering is enabled and either segment is OOD."""
        if self.ood_k is None:
            return False
        if self._n_frames < self.ood_warmup:
            return False
        std = self.ood_std
        if std < 1e-6:
            return False
        threshold = self._mean + self.ood_k * std
        return max(vals1) > threshold or max(vals2) > threshold

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
        T = len(seg.frames)
        for frame, action in zip(seg.frames, seg.actions):
            obs = torch.as_tensor(frame, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_vals = self.expert_model.q_net(obs)[0]
            total += q_vals[int(action)].item()
        return total / T
