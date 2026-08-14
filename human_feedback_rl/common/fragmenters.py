import numpy as np
import torch as th
from typing import List, Optional

from .batching import stacked_transitions
from .types import Trajectory, Fragment, FragmentPair
from .reward_nets import RewardEnsemble


def make_pair_fragmenter(kind: str, rng, logger, reward_ensemble=None, oversample: int = 5):
    """Build a pair fragmenter by name: "random" or "active" (ensemble disagreement)."""
    if kind == "active":
        if reward_ensemble is None:
            raise ValueError('fragmenter "active" requires a reward_ensemble.')
        # Con un membro solo il punteggio di acquisizione (disaccordo fra
        # membri) e' identicamente zero: "active" degenererebbe in silenzio in
        # uno casuale, e i risultati sarebbero attribuiti alla strategia
        # sbagliata. Meglio fallire subito.
        if len(getattr(reward_ensemble, "members", []) or []) < 2:
            raise ValueError(
                'fragmenter "active" needs at least 2 ensemble members; '
                "with one member the acquisition score is identically zero."
            )
        return HighVariancePairFragmenter(
            rng=rng, logger=logger, reward_ensemble=reward_ensemble, oversample=oversample
        )
    if kind == "random":
        return RandomPairFragmenter(rng=rng, logger=logger)
    raise ValueError(f"Unknown fragmenter type: {kind!r}")


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

        # Un frammento contiguo di lunghezza L dentro una traiettoria lunga T
        # puo' iniziare in max(T - L + 1, 1) posizioni. Lo stesso conteggio
        # serve due volte -- per l'avviso e per i pesi di campionamento -- e va
        # calcolato UNA sola volta. Quando erano due espressioni diverse i pesi
        # usavano len(traj) // L + 1, che con L=1 vale T+1 invece di T: le
        # traiettorie corte (cioe' gli episodi finiti in collisione) venivano
        # sovracampionate, e la distorsione cresceva con L -- a L=10 arrivava
        # al 26% fra la traiettoria piu' corta e la piu' lunga.
        per_traiettoria = np.array(
            [max(len(traj) - fragment_length + 1, 1) for traj in trajectories],
            dtype=float,
        )
        num_unique = int(per_traiettoria.sum())
        if num_fragments > num_unique:
            self.logger.warn(
                f"Requested {num_fragments} fragments but only "
                f"{num_unique} unique fragments are available. "
                "Some fragments will be sampled more than once.",
            )

        # Pesare per il numero di frammenti distinti, e poi estrarre lo start
        # uniformemente dentro la traiettoria, rende equiprobabile ogni
        # frammento del pool.
        weights = per_traiettoria / per_traiettoria.sum()

        fragments = []
        for _ in range(num_fragments):
            traj = trajectories[self.rng.choice(len(trajectories), p=weights)]

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

    def _fragment_variances(self, fragments: List[Fragment]) -> np.ndarray:
        """Variance of predicted returns across ensemble members, per fragment.

        One batched ``predict_all`` over the concatenated fragments instead of
        one call per fragment.
        """
        lengths = [len(f) for f in fragments]
        parts = [stacked_transitions(f) for f in fragments]
        obs, acts, statuses, dones = (
            th.cat([p[i] for p in parts]).numpy() for i in range(4)
        )
        all_rewards = self.reward_ensemble.predict_all(obs, acts, statuses, dones)
        boundaries = np.cumsum(lengths)[:-1]
        return np.array([
            chunk.sum(axis=0).var() for chunk in np.split(all_rewards, boundaries)
        ])

    def _select_high_variance(self, fragments: List[Fragment], num_keep: int) -> List[Fragment]:
        seen, unique = set(), []
        for f in fragments:
            key = id(f[0])
            if key not in seen:
                seen.add(key)
                unique.append(f)
        variances   = self._fragment_variances(unique)
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
