import numpy as np
from typing import List, Optional

from .types import Trajectory, Fragment, FragmentPair
from .reward_nets import RewardEnsemble


class RandomFragmenter:

    def __init__(
        self,
        rng: np.random.Generator,
        logger,
    ) -> None:
        self.rng    = rng
        self.logger = logger

    def _sample_fragments(
        self,
        trajectories: List[Trajectory],
        fragment_length: Optional[int],
        num_fragments: int,
    ) -> List[Fragment]:

        # Sample trajectory indices, not the trajectories themselves:
        # np.array(list_of_equal_length_lists, dtype=object) silently builds a
        # 2-D array, which breaks rng.choice.
        if fragment_length is None:
            if num_fragments > len(trajectories):
                self.logger.warn(
                    f"Requested {num_fragments} fragments but only "
                    f"{len(trajectories)} trajectories available. "
                    "Some trajectories will be sampled more than once.",
                )
            weights = np.ones(len(trajectories))
            fragments: List[Fragment] = []
            for _ in range(num_fragments):
                traj = trajectories[self.rng.choice(len(trajectories), p=weights / weights.sum())]
                fragments.append(Fragment(traj[:]))
            return fragments

        num_unique = sum(
            max(len(traj) - fragment_length + 1, 1) for traj in trajectories
        )
        if num_fragments > num_unique:
            self.logger.warn(
                f"Requested {num_fragments} fragments but only "
                f"{num_unique} unique fragments are available. "
                "Some fragments will be sampled more than once.",
            )

        weights = [len(traj) // fragment_length + 1 for traj in trajectories]

        fragments = []
        for _ in range(num_fragments):
            traj = trajectories[self.rng.choice(len(trajectories), p=np.array(weights) / sum(weights))]

            n = len(traj)
            if n >= fragment_length:
                start = self.rng.integers(0, n - fragment_length, endpoint=True)
                end   = start + fragment_length
            else:
                start, end = 0, n

            fragments.append(Fragment(traj[start:end]))

        return fragments


class RandomPairFragmenter(RandomFragmenter):

    def __call__(
        self,
        trajectories: List[Trajectory],
        fragment_length: Optional[int],
        num_pairs: int,
    ) -> List[FragmentPair]:
        
        fragments = self._sample_fragments(trajectories, fragment_length, 2 * num_pairs)

        pairs = [
            FragmentPair(frag1=fragments[i], frag2=fragments[i + 1])
            for i in range(0, len(fragments) - 1, 2)
        ]

        return pairs


class RandomSingleFragmenter(RandomFragmenter):

    def __call__(
        self,
        trajectories: List[Trajectory],
        fragment_length: Optional[int],
        num_fragments: int,
    ) -> List[Fragment]:
        
        return self._sample_fragments(trajectories, fragment_length, num_fragments)


class HighVarianceFragmenter(RandomFragmenter):
    """Samples oversample × num_fragments random fragments, then keeps the
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
        self.oversample      = oversample

    def _fragment_variance(self, fragment: Fragment) -> float:
        """Variance of predicted returns across ensemble members for one fragment."""
        obs          = np.stack([t.observation for t in fragment])
        acts         = np.stack([t.action for t in fragment])
        next_statuses = np.stack([t.next_status for t in fragment]) if fragment[0].next_status is not None else None
        dones        = np.array([t.done for t in fragment], dtype=np.float32)
        all_rewards  = self.reward_ensemble.predict_all(obs, acts, next_statuses, dones)
        return float(all_rewards.sum(axis=0).var())

    def _select_high_variance(self, fragments: List[Fragment], num_keep: int) -> List[Fragment]:
        seen, unique = set(), []
        for f in fragments:
            key = id(f[0])
            if key not in seen:
                seen.add(key)
                unique.append(f)
        variances   = np.array([self._fragment_variance(f) for f in unique])
        top_indices = np.argsort(variances)[-num_keep:][::-1]
        return [unique[i] for i in top_indices]


class HighVariancePairFragmenter(HighVarianceFragmenter):

    def __call__(
        self,
        trajectories: List[Trajectory],
        fragment_length: Optional[int],
        num_pairs: int,
    ) -> List[FragmentPair]:
        
        candidates = self._sample_fragments(trajectories, fragment_length, self.oversample * 2 * num_pairs)
        fragments  = self._select_high_variance(candidates, 2 * num_pairs)

        pairs = [
            FragmentPair(frag1=fragments[i], frag2=fragments[i + 1])
            for i in range(0, len(fragments) - 1, 2)
        ]

        return pairs
