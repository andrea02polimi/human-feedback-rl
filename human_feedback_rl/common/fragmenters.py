import numpy as np
import random

from typing import List, Optional, Tuple
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
        fragment_length: Optional[int],
        num_pairs: int,
    ) -> List[FragmentPair]:
        if fragment_length is None:
            return self._full_episode_pairs(trajectories, num_pairs)
        return self._fixed_length_pairs(trajectories, fragment_length, num_pairs)

    def _full_episode_pairs(
        self,
        trajectories: List[Trajectory],
        num_pairs: int,
    ) -> List[FragmentPair]:
        if not trajectories:
            return []

        traj_lengths = [len(t) for t in trajectories]
        print(
            f"[DEBUG Fragmenter] full-episode mode | num_pairs={num_pairs} | "
            f"pool: {len(trajectories)} episodes "
            f"(len min={min(traj_lengths)} mean={sum(traj_lengths)/len(traj_lengths):.1f} max={max(traj_lengths)})"
        )

        pairs = []
        for _ in range(num_pairs):
            if len(trajectories) >= 2:
                i, j = self.rng.choice(len(trajectories), size=2, replace=False)
            else:
                i = j = 0
            pairs.append(FragmentPair(
                frag1=Fragment(list(trajectories[i])),
                frag2=Fragment(list(trajectories[j])),
            ))
        return pairs

    def _fixed_length_pairs(
        self,
        trajectories: List[Trajectory],
        fragment_length: int,
        num_pairs: int,
    ) -> List[FragmentPair]:
        fragments: List[Fragment] = []

        weights = [len(traj) for traj in trajectories]

        num_transitions = 2 * num_pairs * fragment_length
        if sum(weights) < num_transitions:
            self.logger.warn(
                "Fewer transitions available than needed for desired number "
                "of fragment pairs. Some transitions will appear multiple times.",
            )

        chosen_traj_ids = []
        for _ in range(2 * num_pairs):
            traj = self.rng.choice(
                np.array(trajectories, dtype=object),
                p=np.array(weights) / sum(weights),
            )
            chosen_traj_ids.append(id(traj))

            n = len(traj)
            if n >= fragment_length:
                start = self.rng.integers(0, n - fragment_length, endpoint=True)
                end = start + fragment_length
            else:
                start = 0
                end = n

            fragments.append(Fragment(traj[start:end]))

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