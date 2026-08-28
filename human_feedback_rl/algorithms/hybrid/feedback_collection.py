"""Asking the oracle for comparisons, and counting what comes back."""

import numpy as np
from human_feedback_rl.common.types import Preference


class FeedbackCollectionMixin:
    """Turns a query budget into labelled fragment pairs."""

    def _collect_feedback(self, num_queries: int) -> None:
        self._collect_preference_feedback(num_queries)
        if self.demo_mode == "preferences":
            self._collect_demo_preference_pairs(self.demo_pref_pairs_per_iteration)

    def _collect_preference_feedback(self, num_queries: int) -> None:
        if num_queries <= 0:
            return
        fragments = self.fragmenter(
            self.trajectories,
            self.preference_fragment_length,
            num_queries,
        )
        preferences = self.preference_gatherer(fragments)
        self.dataset_train.push(fragments, preferences)
        self.logger.record("dataset/n_train", len(self.dataset_train), exclude="stdout")
        self._count_duplicate_comparisons(fragments)

    def _collect_demo_preference_pairs(self, num_pairs: int) -> None:
        """Expert fragment > agent fragment pairs (Ibarz et al. 2018).

        The expert is preferred by assumption — no reward signal is used.
        """
        if num_pairs <= 0 or not self.trajectories:
            return
        from human_feedback_rl.common.types import FragmentPair

        expert_frags = self._single_fragmenter(
            self.expert_trajectories, self.preference_fragment_length, num_pairs
        )
        agent_frags = self._single_fragmenter(
            self.trajectories, self.preference_fragment_length, num_pairs
        )
        pairs = [FragmentPair(e, a) for e, a in zip(expert_frags, agent_frags)]
        preferences = [Preference(1.0, 0.0) for _ in pairs]
        self.dataset_demo_prefs_train.push(pairs, preferences)
        self.logger.record(
            "dataset/n_demo_prefs_train", len(self.dataset_demo_prefs_train), exclude="stdout"
        )

    def _count_duplicate_comparisons(self, pairs) -> None:
        """Count the comparisons that repeat.

        The fragmenter draws with replacement and does not deduplicate, so n_pref
        counts stored items rather than distinct ones. Signatures are taken over
        content: a reused id() would give a false positive.
        """
        def signature(frag):
            return tuple(
                (t.observation.tobytes(), np.asarray(t.action).tobytes(),
                 float(t.true_reward))
                for t in frag
            )

        for pair in pairs:
            f1, f2 = signature(pair.frag1), signature(pair.frag2)
            if f1 == f2:
                self._dup_self_pairs += 1
            pair_key = (f1, f2) if f1 <= f2 else (f2, f1)
            if pair_key in self._seen_pairs:
                self._dup_pairs += 1
            else:
                self._seen_pairs.add(pair_key)
            for f in (f1, f2):
                if f in self._seen_fragments:
                    self._dup_fragments += 1
                else:
                    self._seen_fragments.add(f)

        self.logger.record("dataset/dup_pairs", self._dup_pairs, exclude="stdout")
        self.logger.record("dataset/dup_self_pairs", self._dup_self_pairs, exclude="stdout")
        self.logger.record("dataset/dup_fragments", self._dup_fragments, exclude="stdout")
