import torch
from learning_from_human_preferences.preferences.pref_interface import PrefInterface


class ExpertPrefInterface(PrefInterface):
    """
    Preference interface that generates synthetic labels from true environment
    rewards, as described in Christiano et al. (2017) Section 3.1.

    For each segment, sums the true environment rewards stored in
    seg.env_rewards to obtain a quality score.  The preference is the softmax
    of [score1, score2], so the better segment receives a higher probability.

    Args:
        max_segs:     maximum number of segments queued (passed to parent)
        log_dir:      directory for parent-class logging
        expert_model: SB3 DQN model — not used by the current oracle, kept
                      for future re-activation of q-net scoring (see below)
    """

    def __init__(self, max_segs: int, log_dir: str, expert_model=None):
        # synthetic_prefs=True disables VideoRenderer in the parent class
        super().__init__(synthetic_prefs=True, max_segs=max_segs, log_dir=log_dir)
        # Stored for future use when q-net scoring is re-enabled.
        self.expert_model = expert_model

    # ------------------------------------------------------------------

    def ask_user(self, seg1, seg2):
        """
        Compare two segments using the sum of true environment rewards.

        Args:
            seg1, seg2: Segment objects with a .frames list and a .env_rewards
                        list (one float per frame, attached by _policy_worker)

        Returns:
            (p1, p2) preference tuple, where p1 + p2 = 1.0
        """
        # score1 = self._score_by_env_reward(seg1)
        # score2 = self._score_by_env_reward(seg2)

        # ── Q-net scoring (DQN expert) — disabled, kept for future reference ──
        score1 = self._score_segment(seg1)
        score2 = self._score_segment(seg2)

        probs = torch.softmax(torch.tensor([score1, score2]), dim=0)
        p1, p2 = probs.tolist()
        return (p1, p2)

    # ------------------------------------------------------------------

    def _score_by_env_reward(self, seg) -> float:
        """
        Sum of true environment rewards over the segment.

        This is the synthetic oracle from Christiano et al. (2017): the
        segment that accumulated more environment reward is preferred.
        seg.env_rewards is attached by _policy_worker at segment creation time.
        """
        return float(sum(getattr(seg, "env_rewards", [])))

    # ── Q-net scoring — disabled, kept for future reference ─────────────────
    def _score_segment(self, seg) -> float:
        """Sum V(s) = max_a Q(s, a) over all frames in the segment."""
        total = 0.0
        for frame in seg.frames:
            obs = torch.as_tensor(frame, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_vals = self.expert_model.q_net(obs)[0]
            total += q_vals.max().item()
        return total
