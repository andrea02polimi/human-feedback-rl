from typing import List, Optional

import numpy as np

from .core import Segment, SegmentPair, Trajectory


class ActiveFragmenter:
    """
    Samples pairs of trajectory segments of a fixed length.

    Segments are drawn uniformly at random from a pool of trajectories,
    without crossing episode boundaries. Any trajectory shorter than
    segment_length is silently skipped.
    """

    def __init__(self, segment_length: Optional[int], rng: np.random.Generator):
        self.segment_length = segment_length
        self.rng = rng

    def sample_pairs(
        self, trajectories: List[Trajectory], n_pairs: int
    ) -> List[SegmentPair]:
        """
        Sample n_pairs pairs of segments from the given trajectories.
        May return fewer pairs if there are not enough valid trajectories.
        """
        segments = self._sample_segments(trajectories, n_pairs * 2)
        n = len(segments) // 2
        return [SegmentPair(segments[i * 2], segments[i * 2 + 1]) for i in range(n)]

    # ------------------------------------------------------------------

    def _sample_segments(
        self, trajectories: List[Trajectory], n: int
    ) -> List[Segment]:
        # Full-episode mode: use only completed trajectories
        if self.segment_length is None:
            valid = [t for t in trajectories if t.transitions and t.transitions[-1].done]
            if not valid:
                return []
            # sample n segments
            idxs = self.rng.integers(len(valid), size=n)
            return [Segment(valid[i].transitions) for i in idxs]

        # Fixed-length mode
        valid = [t for t in trajectories if len(t) >= self.segment_length]
        if not valid:
            return []

        segments = []
        max_attempts = n * 10
        attempts = 0

        while len(segments) < n and attempts < max_attempts:
            attempts += 1
            traj = valid[self.rng.integers(len(valid))]
            max_start = len(traj) - self.segment_length
            start = int(self.rng.integers(0, max_start + 1))
            transitions = traj.transitions[start : start + self.segment_length]
            segments.append(Segment(transitions))

        return segments