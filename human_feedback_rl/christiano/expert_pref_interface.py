import torch
from learning_from_human_preferences.preferences.pref_interface import PrefInterface


class ExpertPrefInterface(PrefInterface):
    """
    Synthetic preference interface supporting two oracle modes.

    oracle="env_reward"  (Christiano et al. 2017, Section 3.1)
        Sums the true environment rewards stored in seg.env_rewards.
        Requires no expert model — env rewards are attached by _policy_worker.

    oracle="qnet"
        Sums V(s) = max_a Q(s, a) over all frames using the expert DQN.
        Requires expert_model (SB3 DQN with a .q_net attribute).

    In both modes the preference is softmax([score1, score2]), so the better
    segment receives a higher probability rather than a hard 0/1 label.

    Args:
        max_segs:     maximum number of segments queued (passed to parent)
        log_dir:      directory for parent-class logging
        oracle:       "env_reward" | "qnet"
        expert_model: SB3 DQN — required only when oracle="qnet"
    """

    def __init__(
        self,
        max_segs: int,
        log_dir: str,
        oracle: str = "env_reward",
        expert_model=None,
    ):
        super().__init__(synthetic_prefs=True, max_segs=max_segs, log_dir=log_dir)
        self.oracle       = oracle
        self.expert_model = expert_model

        if oracle == "qnet" and expert_model is None:
            raise ValueError("oracle='qnet' requires expert_model to be provided")

    # ------------------------------------------------------------------

    def ask_user(self, seg1, seg2):
        """
        Score both segments with the configured oracle and return a soft
        preference (p1, p2) where p1 + p2 = 1.0.
        """
        if self.oracle == "qnet":
            score1 = self._score_by_qnet(seg1)
            score2 = self._score_by_qnet(seg2)
        else:   # "env_reward"
            score1 = self._score_by_env_reward(seg1)
            score2 = self._score_by_env_reward(seg2)

        probs = torch.softmax(torch.tensor([score1, score2]), dim=0)
        p1, p2 = probs.tolist()
        return (p1, p2)

    # ------------------------------------------------------------------

    def _score_by_env_reward(self, seg) -> float:
        """Sum of true environment rewards over the segment (seg.env_rewards)."""
        return float(sum(getattr(seg, "env_rewards", [])))

    def _score_by_qnet(self, seg) -> float:
        """Sum of V(s) = max_a Q(s, a) over all frames using the expert DQN."""
        total = 0.0
        for frame in seg.frames:
            obs = torch.as_tensor(frame, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_vals = self.expert_model.q_net(obs)[0]
            total += q_vals.max().item()
        return total
