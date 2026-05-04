import numpy as np

from typing import List, Tuple
from .types import Trajectory, Fragment, FragmentPair


def _sample_fragments(
    trajectories: List[Trajectory],
    fragment_length: int,
    n_fragments: int,
    rng: np.random.Generator,
    logger,
) -> Tuple[List[Fragment], List[int]]:
    """
    Length-weighted trajectory sampling shared by all fragmenters.
    Returns (fragments, chosen_traj_ids) so callers can compute uniqueness stats.
    """
    weights = [len(traj) for traj in trajectories]
    if sum(weights) < n_fragments * fragment_length:
        logger.warn(
            "Fewer transitions available than needed for desired number "
            "of fragment pairs. Some transitions will appear multiple times.",
        )

    fragments: List[Fragment] = []
    chosen_traj_ids: List[int] = []

    for _ in range(n_fragments):
        traj = rng.choice(
            np.array(trajectories, dtype=object),
            p=np.array(weights) / sum(weights),
        )
        chosen_traj_ids.append(id(traj))

        n = len(traj)
        if n >= fragment_length:
            start = rng.integers(0, n - fragment_length, endpoint=True)
            end = start + fragment_length
        else:
            start, end = 0, n

        fragments.append(Fragment(traj[start:end]))

    return fragments, chosen_traj_ids


class RandomFragmenter:
    """Samples fragment pairs for preference comparisons."""

    def __init__(self, rng: np.random.Generator, logger) -> None:
        self.rng = rng
        self.logger = logger

    def __call__(
        self,
        trajectories: List[Trajectory],
        fragment_length: int,
        num_pairs: int,
    ) -> List[FragmentPair]:
        fragments, chosen_traj_ids = _sample_fragments(
            trajectories, fragment_length, 2 * num_pairs, self.rng, self.logger
        )

        n_unique = len(set(chosen_traj_ids))
        traj_lengths = [len(t) for t in trajectories]
        print(
            f"[DEBUG Fragmenter] num_pairs={num_pairs} fragment_length={fragment_length} | "
            f"pool: {len(trajectories)} trajs (len min={min(traj_lengths)} mean={sum(traj_lengths)/len(traj_lengths):.1f} max={max(traj_lengths)}) | "
            f"unique trajs used for {2*num_pairs} fragments: {n_unique} ({100*n_unique/max(len(trajectories),1):.0f}%)"
        )

        pairs = []
        for i in range(0, len(fragments) - 1, 2):
            pairs.append(FragmentPair(frag1=fragments[i], frag2=fragments[i + 1]))
        return pairs


class SingleFragmenter:
    """
    Samples individual fragments (not pairs) from trajectories.
    Used by demo-based algorithms that query one set of segments per iteration.
    """

    def __init__(self, rng: np.random.Generator, logger) -> None:
        self.rng = rng
        self.logger = logger

    def __call__(
        self,
        trajectories: List[Trajectory],
        fragment_length: int,
        num_demos: int,
    ) -> List[Fragment]:
        fragments, _ = _sample_fragments(
            trajectories, fragment_length, num_demos, self.rng, self.logger
        )
        return fragments