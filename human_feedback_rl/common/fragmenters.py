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
        chosen_traj_ids = []
        for _ in range(2 * num_pairs):
            # NumPy's annotation here is overly-conservative, but this works at runtime
            traj = self.rng.choice(
                np.array(trajectories, dtype=object),
                p=np.array(weights) / sum(weights),
            )
            chosen_traj_ids.append(id(traj))

            # if the traj is shorter than the fragment length, than takes the entire traj as the fragment
            n = len(traj)
            if n >= fragment_length:
                start = self.rng.integers(0, n - fragment_length, endpoint=True)
                end = start + fragment_length
            else:
                start = 0
                end = n

            fragment = Fragment(traj[start:end])

            fragments.append(fragment)

        n_unique = len(set(chosen_traj_ids))
        traj_lengths = [len(t) for t in trajectories]
        print(
            f"[DEBUG Fragmenter] num_pairs={num_pairs} fragment_length={fragment_length} | "
            f"pool: {len(trajectories)} trajs (len min={min(traj_lengths)} mean={sum(traj_lengths)/len(traj_lengths):.1f} max={max(traj_lengths)}) | "
            f"unique trajs used for {2*num_pairs} fragments: {n_unique} ({100*n_unique/max(len(trajectories),1):.0f}%)"
        )

        # fragments is currently a list of single fragments. We want to pair up
        # fragments to get a list of (fragment1, fragment2) tuples. To do so,
        # we create a single iterator of the list and zip it with itself:
        pairs = []
        for i in range(0, len(fragments) - 1, 2): # range(start, stop, step)
            pairs.append(
                FragmentPair(
                    frag1=fragments[i],
                    frag2=fragments[i + 1]
                )
            )

        return pairs