import time
from dataclasses import dataclass
from typing import List, Tuple

from human_feedback_rl.common.types import FragmentPair, Preference


@dataclass
class GathererMetrics:
    time_gatherer: float


class PreferenceGathererFromReward:
    """Uses ground-truth rewards to generate preferences (for testing)."""

    def __call__(self, fragment_pairs: List[FragmentPair]) -> Tuple[List[Preference], GathererMetrics]:
        t0 = time.perf_counter()
        preferences = []
        for pair in fragment_pairs:
            reward1 = pair.frag1.total_reward() / pair.frag1.length()
            reward2 = pair.frag2.total_reward() / pair.frag2.length()

            if reward1 > reward2:
                preferences.append(Preference(pref1=1.0, pref2=0.0))
            elif reward2 > reward1:
                preferences.append(Preference(pref1=0.0, pref2=1.0))
            else:
                preferences.append(Preference(pref1=0.5, pref2=0.5))

        return preferences, GathererMetrics(time_gatherer=time.perf_counter() - t0)