from typing import List

from human_feedback_rl.common.types import FragmentPair, Preference


class PreferenceGathererFromReward:
    """Uses ground-truth rewards to generate preferences (for testing)."""

    def __init__(self, normalize_by_length: bool = False):
        self.normalize_by_length = normalize_by_length

    def __call__(self, fragment_pairs: List[FragmentPair]) -> List[Preference]:
        preferences = []
        for pair in fragment_pairs:
            if self.normalize_by_length:
                reward1 = pair.frag1.total_reward() / max(len(pair.frag1), 1)
                reward2 = pair.frag2.total_reward() / max(len(pair.frag2), 1)
            else:
                reward1 = pair.frag1.total_reward()
                reward2 = pair.frag2.total_reward()

            if reward1 > reward2:
                preferences.append(Preference(pref1=1.0, pref2=0.0))
            elif reward2 > reward1:
                preferences.append(Preference(pref1=0.0, pref2=1.0))
            else:
                preferences.append(Preference(pref1=0.5, pref2=0.5))

        return preferences