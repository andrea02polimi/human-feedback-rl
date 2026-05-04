from typing import List

import numpy as np

from human_feedback_rl.common.types import Fragment, FragmentPair, Preference, Transition


class PreferenceGathererFromReward:
    """Uses ground-truth rewards to generate preferences (for testing)."""

    def __call__(self, fragment_pairs: List[FragmentPair]) -> List[Preference]:
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


class DemonstrationGathererFromExpert:
    """
    Replaces each agent action in a fragment with the expert policy's action
    for the corresponding observation.

    The expert is queried one observation at a time (safe for any predict() API).
    Each observation is promoted to shape (1, obs_dim), the result squeezed back
    to (act_dim,) so the returned Fragment has the same structure as the input.
    """

    def __init__(self, expert_policy):
        self.expert_policy = expert_policy

    def __call__(self, fragments: List[Fragment]) -> List[Fragment]:
        demos: List[Fragment] = []

        for frag in fragments:
            new_transitions: List[Transition] = []

            for t in frag:
                obs_2d = np.atleast_2d(t.observation).astype(np.float32)  # (1, obs_dim)
                result = self.expert_policy.predict(obs_2d)

                if isinstance(result, (tuple, list)):
                    expert_action = np.asarray(result[0]).squeeze(0)
                else:
                    expert_action = np.asarray(result).squeeze(0)

                new_transitions.append(
                    Transition(
                        observation=t.observation,
                        action=expert_action.astype(np.float32),
                        true_reward=t.true_reward,
                    )
                )

            demos.append(Fragment(new_transitions))

        return demos