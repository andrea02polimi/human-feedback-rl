import math

import numpy as np
import pytest

from human_feedback_rl.common.gatherers import PreferenceGathererFromReward
from human_feedback_rl.common.types import FragmentPair, Trajectory, Transition


def _fragment(mean_reward, length=4):
    return Trajectory([
        Transition(observation=np.zeros(2), action=np.zeros(1), true_reward=mean_reward)
        for _ in range(length)
    ])


def _pair(r1, r2):
    return FragmentPair(_fragment(r1), _fragment(r2))


def test_binary_labels():
    g = PreferenceGathererFromReward(labels_type="binary")
    prefs = g([_pair(2.0, 1.0), _pair(1.0, 2.0), _pair(1.0, 1.0)])
    assert [(p.pref1, p.pref2) for p in prefs] == [(1.0, 0.0), (0.0, 1.0), (0.5, 0.5)]


def test_soft_labels_match_hand_computed_sigmoid():
    temperature = 1.5
    g = PreferenceGathererFromReward(labels_type="soft", temperature=temperature)
    (pref,) = g([_pair(2.0, 1.0)])
    expected = 1.0 / (1.0 + math.exp((1.0 - 2.0) / temperature))
    assert pref.pref1 == pytest.approx(expected)
    assert pref.pref1 + pref.pref2 == pytest.approx(1.0)


def test_bernoulli_labels_are_hard_and_match_rate():
    temperature = 2.0
    g = PreferenceGathererFromReward(
        labels_type="binary_bernoulli", temperature=temperature, rng=np.random.default_rng(0)
    )
    prefs = g([_pair(1.0, 0.0)] * 2000)
    assert all((p.pref1, p.pref2) in ((1.0, 0.0), (0.0, 1.0)) for p in prefs)
    rate = np.mean([p.pref1 for p in prefs])
    expected = 1.0 / (1.0 + math.exp(-1.0 / temperature))
    assert rate == pytest.approx(expected, abs=0.03)


def test_invalid_labels_type_raises():
    # Regression: used to print and silently reuse the previous preference.
    with pytest.raises(ValueError, match="labels_type"):
        PreferenceGathererFromReward(labels_type="not-a-mode")


def test_deprecated_bernulli_alias_accepted():
    g = PreferenceGathererFromReward(labels_type="binary_bernulli")
    assert g.labels_type == "binary_bernoulli"


def test_nonpositive_temperature_raises():
    with pytest.raises(ValueError, match="temperature"):
        PreferenceGathererFromReward(labels_type="soft", temperature=0.0)
