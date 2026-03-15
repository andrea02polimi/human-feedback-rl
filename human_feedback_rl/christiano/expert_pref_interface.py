import torch
from learning_from_human_preferences.preferences.pref_interface import PrefInterface


class ExpertPrefInterface(PrefInterface):
    """
    Preference interface that uses a trained DQN expert to generate
    synthetic preferences instead of showing video to a human annotator.

    For each segment, the expert computes V(s) = max_a Q(s, a) for every
    observation frame and sums them to obtain a segment quality score.
    The preference is the softmax of [score1, score2].

    Args:
        expert_model: SB3 DQN model with a `.q_net` attribute
        max_segs:     maximum number of segments queued (passed to parent)
        log_dir:      directory for parent-class logging
    """

    def __init__(self, expert_model, max_segs: int, log_dir: str):
        # synthetic_prefs=True disables VideoRenderer in the parent class
        super().__init__(synthetic_prefs=True, max_segs=max_segs, log_dir=log_dir)
        self.expert_model = expert_model

    # ------------------------------------------------------------------

    def ask_user(self, seg1, seg2):
        """
        Compare two segments using the expert's Q-network.

        Args:
            seg1, seg2: Segment objects with a `.frames` list of np.ndarray

        Returns:
            (p1, p2) preference tuple, where p1 + p2 = 1.0
        """
        score1 = self._score_segment(seg1)
        score2 = self._score_segment(seg2)

        probs = torch.softmax(torch.tensor([score1, score2]), dim=0)
        p1, p2 = probs.tolist()
        return (p1, p2)

    # ------------------------------------------------------------------

    def _score_segment(self, seg) -> float:
        """Sum V(s) = max_a Q(s, a) over all frames in the segment."""
        total = 0.0
        for frame in seg.frames:
            obs = torch.as_tensor(frame, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_vals = self.expert_model.q_net(obs)[0]
            total += q_vals.max().item()
        return total
