import math
import random
import time
from typing import List

import numpy as np

from human_feedback_rl.common.loggers import NullLogger
from human_feedback_rl.common.types import FragmentPair, Fragment, Transition, Preference


class PreferenceGathererFromReward:
    """Uses ground-truth rewards to generate preferences (for testing)."""

    def __init__(self, logger = None, labels_type: str = "soft", temperature: float = 20.0) -> None:

        self.logger = logger if logger is not None else NullLogger()
        self.labels_type = labels_type
        self.temperature = temperature

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
                prob1 = 1.0 / (1.0 + math.exp((r2 - r1)/self.temperature))
                pref = Preference(prob1, 1.0 - prob1)

            elif self.labels_type == "binary_bernulli":
                prob1 = 1.0 / (1.0 + math.exp((r2 - r1)/self.temperature))
                pref1 = 1 if random.random() < prob1 else 0
                pref = Preference(pref1, 1.0 - pref1)
            else:
                print("errore inserimento labels_type: soft or binary or binary_bernulli")

            preferences.append(pref)

        return preferences
    


class DemoGathererFromExpert:
    """For each fragment, replaces every action with the expert's action on that observation."""

    def __init__(self, expert, logger=None) -> None:
        self.expert = expert
        self.logger = logger if logger is not None else NullLogger()

    def __call__(self, fragments: List[Fragment]) -> List[Fragment]:
        
        expert_fragments = []
        for fragment in fragments:
            obs_batch      = np.array([tr.observation for tr in fragment])
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

        return expert_fragments
