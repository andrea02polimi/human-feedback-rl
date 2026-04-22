import numpy as np
import random

from typing import List, Tuple
from .types import Trajectory, Fragment, FragmentPair



class RandomFragmenter:

    def __init__(
        self,
        rng: np.random.Generator,
        logger,
    ) -> None:
        
        self.rng = rng
        self.logger = logger

    def __call__(
        self,
        trajectories: List[Trajectory],
        fragment_length: int,
        num_pairs: int,
    ) -> List[FragmentPair]:
        
        fragments: List[Fragment] = []

        weights = [len(traj) for traj in trajectories]

        # number of transitions that will be contained in the fragments
        num_transitions = 2 * num_pairs * fragment_length
        if sum(weights) < num_transitions:
            self.logger.warn(
                "Fewer transitions available than needed for desired number "
                "of fragment pairs. Some transitions will appear multiple times.",
            )

        # we need two fragments for each comparison
        for _ in range(2 * num_pairs):
            # NumPy's annotation here is overly-conservative, but this works at runtime
            traj = self.rng.choice(
                trajectories,  # type: ignore[arg-type]
                p=np.array(weights) / sum(weights),
            )

            # if the traj is shorter than the fragment length, than takes the entire traj as the fragment
            n = len(traj)
            if n >= fragment_length:
                start = self.rng.integers(0, n - fragment_length, endpoint=True)
                end = start + fragment_length
            else:
                start = 0
                end = n

            fragment = Fragment(transitions=traj.transitions[start:end])
            
            fragments.append(fragment)

        # fragments is currently a list of single fragments. We want to pair up
        # fragments to get a list of (fragment1, fragment2) tuples. To do so,
        # we create a single iterator of the list and zip it with itself:
        iterator = iter(fragments)
        return [FragmentPair(frag1=f1, frag2=f2) for f1, f2 in zip(iterator, iterator)]


































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

    def fragment(self, trajectories: List[Trajectory], num_pairs: int) -> List[FragmentPair]:
        segments = self._extract_segments(trajectories)

        if len(segments) < 2:
            return []

        scored = self._score_segments(segments)
        scored.sort(key=lambda x: x[2], reverse=True)

        top_k = min(len(scored), 2 * num_pairs)
        top_segments = [s for s, _, _ in scored[:top_k]]

        return self._pair_sequentially(top_segments, num_pairs)

    # ------------------------------------------------------------------
    # Fragment extraction
    # ------------------------------------------------------------------

    def _extract_segments(self, trajectories: List[Trajectory]) -> List[Fragment]:
        segments: List[Fragment] = []
        for traj in trajectories:
            segments.extend(self._split_trajectory(traj))
        return segments

    def _split_trajectory(self, traj: Trajectory) -> List[Fragment]:
        transitions = traj.transitions
        T = len(transitions)
        return [
            Fragment(transitions[start : min(start + self.segment_length, T)])
            for start in range(0, T, self.segment_length)
        ]

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_segments(self, segments: List[Fragment]) -> List[Tuple[Fragment, float, float]]:
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

        all_rewards = self.reward_model.predict(all_obs, all_actions)
        all_variances = self.reward_model.ensemble_variance(all_obs, all_actions)

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

    def _pair_sequentially(self, segments: List[Fragment], num_pairs) -> List[FragmentPair]:
        pairs: List[FragmentPair] = []

        for i in range(0, len(segments) - 1, 2):
            if len(pairs) >= num_pairs:
                break

            pairs.append(
                FragmentPair(seg1=segments[i], seg2=segments[i + 1])
            )

        return pairs
    
    def _pair_randomly(self, segments: List[Fragment], num_pairs) -> List[FragmentPair]:
        segments = segments.copy()          # avoid modifying original list
        random.shuffle(segments)            # randomize order

        return self._pair_sequentially(segments, num_pairs)