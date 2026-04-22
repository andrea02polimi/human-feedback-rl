import numpy as np
from abc import ABC, abstractmethod
from typing import List

from human_feedback_rl.common.types import FragmentPair, Preference


class PreferenceGatherer(ABC):
    """Abstract interface for obtaining preferences."""

    @abstractmethod
    def gather(self, fragment_pairs: List[FragmentPair]) -> List[Preference]:
        """Return preferences for fragment pairs."""
        pass


class PreferenceGathererFromReward(PreferenceGatherer):
    """Uses ground-truth rewards to generate preferences (for testing)."""

    def gather(self, fragment_pairs: List[FragmentPair]) -> List[Preference]:
        preferences = []
        for pair in fragment_pairs:
            reward1 = pair.frag1.total_reward()
            reward2 = pair.frag2.total_reward()

            if reward1 > reward2:
                preferences.append(Preference(pref1=1.0, pref2=0.0))
            elif reward2 > reward1:
                preferences.append(Preference(pref1=0.0, pref2=1.0))
            else:
                preferences.append(Preference(pref1=0.5, pref2=0.5))

        return preferences