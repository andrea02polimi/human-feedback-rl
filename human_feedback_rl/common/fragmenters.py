import numpy as np
import random

from typing import List, Tuple
from .core import Trajectory, Segment, SegmentPair
from .reward_model import encode_ego_status


# ---------------------------------------------------------------------------
# Fragmenter
# ---------------------------------------------------------------------------

class ActiveFragmenter:
    """
    Pipeline:
    1. Split trajectories into segments (last segment can be shorter)
    2. Compute reward and variance per segment (normalized by length)
    3. Sort segments by variance (descending)
    4. Take top segments and form N pairs
    """

    def __init__(self, reward_model, segment_length: int):
        self.reward_model = reward_model
        self.segment_length = segment_length

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fragment(self, trajectories: List[Trajectory], num_pairs: int) -> List[SegmentPair]:
        segments = self._extract_segments(trajectories)

        if len(segments) < 2:
            return []

        scored = self._score_segments(segments)
        scored.sort(key=lambda x: x[2], reverse=True)

        top_k = min(len(scored), 2 * num_pairs)
        top_segments = [s for s, _, _ in scored[:top_k]]

        return self._pair_sequentially(top_segments, num_pairs)

    # ------------------------------------------------------------------
    # Segment extraction
    # ------------------------------------------------------------------

    def _extract_segments(self, trajectories: List[Trajectory]) -> List[Segment]:
        segments: List[Segment] = []
        for traj in trajectories:
            segments.extend(self._split_trajectory(traj))
        return segments

    def _split_trajectory(self, traj: Trajectory) -> List[Segment]:
        transitions = traj.transitions
        T = len(transitions)
        return [
            Segment(transitions[start : min(start + self.segment_length, T)])
            for start in range(0, T, self.segment_length)
        ]

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_segments(self, segments: List[Segment]) -> List[Tuple[Segment, float, float]]:
        """Returns list of (segment, reward_score, variance_score).
        All segments are scored in two batched forward passes for efficiency.
        """
        dtype = np.int64 if self.reward_model.discrete_actions else np.float32
        lengths = [len(seg.transitions) for seg in segments]

        all_obs = np.concatenate([
            np.stack([t.obs for t in seg.transitions]) for seg in segments
        ])
        all_actions = np.concatenate([
            np.array([t.action for t in seg.transitions], dtype=dtype) for seg in segments
        ])
        all_infos = np.concatenate([
            np.stack([
                encode_ego_status(t.info.get("ego_status", "running"))
                if t.info is not None else np.zeros(4, dtype=np.float32)
                for t in seg.transitions
            ])
            for seg in segments
        ])

        all_rewards = self.reward_model.predict(all_obs, all_actions, info=all_infos)
        all_variances = self.reward_model.ensemble_variance(all_obs, all_actions, info=all_infos)

        scored = []
        idx = 0
        for seg, length in zip(segments, lengths):
            r = float(all_rewards[idx:idx + length].sum() / length)
            v = float(all_variances[idx:idx + length].sum() / length)
            scored.append((seg, r, v))
            idx += length
        return scored

    # ------------------------------------------------------------------
    # Pairing
    # ------------------------------------------------------------------

    def _pair_sequentially(self, segments: List[Segment], num_pairs) -> List[SegmentPair]:
        pairs: List[SegmentPair] = []

        for i in range(0, len(segments) - 1, 2):
            if len(pairs) >= num_pairs:
                break

            pairs.append(
                SegmentPair(seg1=segments[i], seg2=segments[i + 1])
            )

        return pairs
    
    def _pair_randomly(self, segments: List[Segment], num_pairs) -> List[SegmentPair]:
        segments = segments.copy()          # avoid modifying original list
        random.shuffle(segments)            # randomize order

        return self._pair_sequentially(segments, num_pairs)