import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from human_feedback_rl.common.types import FragmentPair, Fragment, Transition, Preference


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


class DemoGathererFromExpert:
    """For each fragment, replaces every action with the expert's action on that observation."""

    def __init__(self, expert) -> None:
        self.expert = expert

    def __call__(self, fragments: List[Fragment]) -> Tuple[List[Fragment], GathererMetrics]:
        t0 = time.perf_counter()
        expert_fragments = []
        for fragment in fragments:
            obs_batch = np.array([tr.observation for tr in fragment])
            expert_actions = self.expert.predict(obs_batch)
            expert_fragments.append(Fragment([
                Transition(
                    observation=tr.observation,
                    action=expert_actions[j],
                    true_reward=tr.true_reward,
                    next_status=tr.next_status,
                    done=tr.done,
                )
                for j, tr in enumerate(fragment)
            ]))
        return expert_fragments, GathererMetrics(time_gatherer=time.perf_counter() - t0)