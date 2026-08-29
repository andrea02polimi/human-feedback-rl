"""The synthetic oracle that answers the comparisons.

It scores a pair by true reward and returns a label: soft probabilities, a
hard bit, or a Bernoulli draw. `temperature` sets how decisive it is, and
describes the annotator, not the learner.
"""

import math
from typing import List, Optional

import numpy as np

from human_feedback_rl.common.loggers import NullLogger
from human_feedback_rl.common.types import FragmentPair, Preference


class PreferenceGathererFromReward:
    """Uses ground-truth rewards to generate preferences (synthetic oracle).

    Fragments are compared on their mean per-step true reward. ``labels_type``
    selects the label format:

    * ``binary`` — hard labels (1,0)/(0,1), (0.5,0.5) on ties.
    * ``soft`` — sigmoid of the reward difference at ``temperature``.
    * ``binary_bernoulli`` — hard labels sampled from the soft probability.
    """

    VALID_LABELS_TYPES = ("binary", "soft", "binary_bernoulli")
    # Historical misspelling kept working for old configs.
    _DEPRECATED_ALIASES = {"binary_bernulli": "binary_bernoulli"}

    def __init__(
        self,
        logger=None,
        labels_type: str = "soft",
        temperature: float = 20.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        labels_type = self._DEPRECATED_ALIASES.get(labels_type, labels_type)
        if labels_type not in self.VALID_LABELS_TYPES:
            raise ValueError(
                f"labels_type must be one of {self.VALID_LABELS_TYPES}, got {labels_type!r}"
            )
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        self.logger = logger if logger is not None else NullLogger()
        self.labels_type = labels_type
        self.temperature = temperature
        self.rng = rng if rng is not None else np.random.default_rng()

    def __call__(self, fragment_pairs: List[FragmentPair]) -> List[Preference]:
        preferences = []
        for p in fragment_pairs:
            r1 = p.frag1.total_reward() / p.frag1.length()
            r2 = p.frag2.total_reward() / p.frag2.length()

            if self.labels_type == "binary":
                if r1 > r2:
                    pref = Preference(1.0, 0.0)
                elif r1 < r2:
                    pref = Preference(0.0, 1.0)
                else:
                    pref = Preference(0.5, 0.5)
            elif self.labels_type == "soft":
                prob1 = 1.0 / (1.0 + math.exp((r2 - r1) / self.temperature))
                pref = Preference(prob1, 1.0 - prob1)
            else:  # binary_bernoulli
                prob1 = 1.0 / (1.0 + math.exp((r2 - r1) / self.temperature))
                pref1 = 1.0 if self.rng.random() < prob1 else 0.0
                pref = Preference(pref1, 1.0 - pref1)

            preferences.append(pref)

        return preferences
