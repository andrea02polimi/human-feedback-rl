import numpy as np
import random
import time

from dataclasses import dataclass
from typing import List, Tuple
from .types import Trajectory, Fragment, FragmentPair


@dataclass
class FragmenterMetrics:
    time_fragmenter: float



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
    ) -> Tuple[List[FragmentPair], FragmenterMetrics]:
        t0 = time.perf_counter()
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
                np.array(trajectories, dtype=object),
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

            fragment = Fragment(traj[start:end])
            
            fragments.append(fragment)

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

        return pairs, FragmenterMetrics(time_fragmenter=time.perf_counter() - t0)