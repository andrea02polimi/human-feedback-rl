from . import *
from typing import List
import random
import numpy as np


# ---------------------------------------------------------------------------
# Fragmenter
# ---------------------------------------------------------------------------

class ActiveFragmenter:
    """
    Extracts fixed-length segments from trajectories and pairs them.

    If the reward model has already been trained, pairs are selected by
    ensemble disagreement (active learning). Otherwise random selection is used.
    """

    def __init__(
        self,
        reward_model: EnsembleRewardModel,
        length_segment: int,
        num_max_segment_pairs: int,
    ):
        self.reward_model = reward_model
        self.length_segment = length_segment
        self.num_max_segment_pairs = num_max_segment_pairs

    def fragment(self, trajectories: List[Trajectory]) -> List[SegmentPair]:
        segments = self._extract_segments(trajectories)
        if len(segments) < 2:
            return []

        n_pairs = min(self.num_max_segment_pairs, len(segments) // 2)

        # Active selection: prefer segment pairs with high ensemble disagreement
        if len(segments) > n_pairs * 2:
            pairs = self._active_pairs(segments, n_pairs)
        else:
            pairs = self._random_pairs(segments, n_pairs)

        return pairs

    # ------------------------------------------------------------------

    def _extract_segments(self, trajectories: List[Trajectory]) -> List[Segment]:
        segments = []
        for traj in trajectories:
            transitions = traj.transitions
            T = len(transitions)
            if T < self.length_segment:
                continue
            start = 0
            while start + self.length_segment <= T:
                segments.append(Segment(transitions[start : start + self.length_segment]))
                start += self.length_segment
        return segments

    def _random_pairs(self, segments: List[Segment], n_pairs: int) -> List[SegmentPair]:
        idxs = list(range(len(segments)))
        random.shuffle(idxs)
        pairs = []
        for i in range(0, min(n_pairs * 2, len(idxs) - 1), 2):
            pairs.append(SegmentPair(seg1=segments[idxs[i]], seg2=segments[idxs[i + 1]]))
        return pairs

    def _active_pairs(self, segments: List[Segment], n_pairs: int) -> List[SegmentPair]:
        """Score candidate pairs by ensemble disagreement and take the top ones."""
        rm = self.reward_model

        # Score each segment by mean per-step ensemble variance
        def _seg_score(seg: Segment) -> float:
            obs = np.stack([t.obs for t in seg.transitions])
            actions = np.array([t.action for t in seg.transitions], dtype=np.int64)
            return float(rm.ensemble_variance(obs, actions).mean())

        scored = [(s, _seg_score(s)) for s in segments]
        # Sort descending by uncertainty
        scored.sort(key=lambda x: x[1], reverse=True)

        # Take top-K uncertain segments
        top_k = min(n_pairs * 2, len(scored))
        top_segs = [s for s, _ in scored[:top_k]]

        pairs = []
        for i in range(0, len(top_segs) - 1, 2):
            pairs.append(SegmentPair(seg1=top_segs[i], seg2=top_segs[i + 1]))
            if len(pairs) >= n_pairs:
                break
        return pairs
