import numpy as np
import random
import time

from dataclasses import dataclass
from typing import List, Tuple
from .types import Trajectory, Fragment, FragmentPair
from .reward_nets import RewardEnsemble


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



class HighVarianceFragmenter(RandomFragmenter):
    """Samples oversample * num_fragments random fragments, then keeps the
    num_fragments with the highest variance in predicted return across the
    reward ensemble members."""

    def __init__(
        self,
        rng: np.random.Generator,
        logger,
        reward_ensemble: RewardEnsemble,
        oversample: int = 5,
    ) -> None:
        super().__init__(rng, logger)
        self.reward_ensemble = reward_ensemble
        self.oversample = oversample

    def _fragment_variance(self, fragment: Fragment) -> float:
        """Return variance of predicted returns across ensemble members for one fragment."""
        obs = np.stack([t.observation for t in fragment])
        acts = np.stack([t.action for t in fragment])
        next_statuses = np.stack([t.next_status for t in fragment]) if fragment[0].next_status is not None else None
        dones = np.array([t.done for t in fragment], dtype=np.float32)

        # all_rewards: (fragment_len, num_members)
        all_rewards = self.reward_ensemble.predict_all(obs, acts, next_statuses, dones)
        # sum over timesteps → (num_members,) returns, then compute variance across members
        returns = all_rewards.sum(axis=0)
        return float(returns.var())

    def _select_high_variance(
        self,
        fragments: List[Fragment],
        num_keep: int,
    ) -> List[Fragment]:
        variances = np.array([self._fragment_variance(f) for f in fragments])
        top_indices = np.argsort(variances)[-num_keep:][::-1]
        return [fragments[i] for i in top_indices]


class HighVariancePairFragmenter(HighVarianceFragmenter):

    def __call__(
        self,
        trajectories: List[Trajectory],
        fragment_length: int,
        num_pairs: int,
    ) -> Tuple[List[FragmentPair], FragmenterMetrics]:
        t0 = time.perf_counter()

        candidates = self._sample_fragments(
            trajectories, fragment_length, self.oversample * 2 * num_pairs
        )
        fragments = self._select_high_variance(candidates, 2 * num_pairs)

        pairs = [
            FragmentPair(frag1=fragments[i], frag2=fragments[i + 1])
            for i in range(0, len(fragments) - 1, 2)
        ]

        return pairs, FragmenterMetrics(time_fragmenter=time.perf_counter() - t0)


class HighVarianceSingleFragmenter(HighVarianceFragmenter):

    def __call__(
        self,
        trajectories: List[Trajectory],
        fragment_length: int,
        num_fragments: int,
    ) -> Tuple[List[Fragment], FragmenterMetrics]:
        t0 = time.perf_counter()

        candidates = self._sample_fragments(
            trajectories, fragment_length, self.oversample * num_fragments
        )
        fragments = self._select_high_variance(candidates, num_fragments)

        return fragments, FragmenterMetrics(time_fragmenter=time.perf_counter() - t0)
