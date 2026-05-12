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

    def _sample_fragments(
        self,
        trajectories: List[Trajectory],
        fragment_length: int,
        num_fragments: int,
    ) -> List[Fragment]:
        weights = [len(traj) // fragment_length + 1 for traj in trajectories]

        num_transitions = num_fragments * fragment_length
        if sum(len(traj) for traj in trajectories) < num_transitions:
            self.logger.warn(
                "Fewer transitions available than needed for desired number "
                "of fragments. Some transitions will appear multiple times.",
            )

        fragments: List[Fragment] = []
        for _ in range(num_fragments):
            traj = self.rng.choice(
                np.array(trajectories, dtype=object),
                p=np.array(weights) / sum(weights),
            )

            n = len(traj)
            if n >= fragment_length:
                start = self.rng.integers(0, n - fragment_length, endpoint=True)
                end = start + fragment_length
            else:
                start = 0
                end = n

            fragments.append(Fragment(traj[start:end]))

        return fragments


class RandomPairFragmenter(RandomFragmenter):

    def __call__(
        self,
        trajectories: List[Trajectory],
        fragment_length: int,
        num_pairs: int,
    ) -> Tuple[List[FragmentPair], FragmenterMetrics]:
        t0 = time.perf_counter()

        fragments = self._sample_fragments(trajectories, fragment_length, 2 * num_pairs)

        pairs = [
            FragmentPair(frag1=fragments[i], frag2=fragments[i + 1])
            for i in range(0, len(fragments) - 1, 2)
        ]

        return pairs, FragmenterMetrics(time_fragmenter=time.perf_counter() - t0)


class RandomSingleFragmenter(RandomFragmenter):

    def __call__(
        self,
        trajectories: List[Trajectory],
        fragment_length: int,
        num_fragments: int,
    ) -> Tuple[List[Fragment], FragmenterMetrics]:
        t0 = time.perf_counter()

        fragments = self._sample_fragments(trajectories, fragment_length, num_fragments)

        return fragments, FragmenterMetrics(time_fragmenter=time.perf_counter() - t0)
