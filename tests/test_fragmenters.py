import numpy as np
import torch as th

from human_feedback_rl.common.fragmenters import (
    HighVariancePairFragmenter,
    RandomPairFragmenter,
    RandomSingleFragmenter,
)
from human_feedback_rl.common.loggers import NullLogger
from human_feedback_rl.common.reward_nets import RewardEnsemble

from conftest import ConstantRewardNet, make_trajectories


def test_equal_length_trajectories_do_not_crash(rng):
    # Regression: np.array(equal-length lists, dtype=object) built a 2-D array
    # and rng.choice failed.
    trajs = make_trajectories(rng, [6, 6, 6, 6])
    frag = RandomPairFragmenter(rng=rng, logger=NullLogger())
    pairs = frag(trajs, 3, 4)
    assert len(pairs) == 4


def test_fragments_have_requested_length(rng):
    trajs = make_trajectories(rng, [8, 12, 5])
    frag = RandomSingleFragmenter(rng=rng, logger=NullLogger())
    fragments = frag(trajs, 4, 10)
    assert all(len(f) == 4 for f in fragments)


def test_short_trajectory_yields_whole_trajectory(rng):
    trajs = make_trajectories(rng, [2])
    frag = RandomSingleFragmenter(rng=rng, logger=NullLogger())
    fragments = frag(trajs, 5, 3)
    assert all(len(f) == 2 for f in fragments)


def test_fragment_length_none_returns_whole_trajectories(rng):
    trajs = make_trajectories(rng, [4, 7])
    frag = RandomSingleFragmenter(rng=rng, logger=NullLogger())
    fragments = frag(trajs, None, 5)
    assert all(len(f) in (4, 7) for f in fragments)


def test_fragments_are_contiguous_subsequences(rng):
    trajs = make_trajectories(rng, [9, 11])
    frag = RandomSingleFragmenter(rng=rng, logger=NullLogger())
    for fragment in frag(trajs, 3, 8):
        source = next(t for t in trajs if any(x is fragment[0] for x in t))
        start = next(i for i, x in enumerate(source) if x is fragment[0])
        assert all(a is b for a, b in zip(fragment, source[start:start + 3]))


def test_high_variance_pair_fragmenter_prefers_disagreement(rng):
    # Two members disagreeing only through the (constant) offset produce zero
    # variance everywhere except where inputs differ; build members whose
    # disagreement scales with the observation so high-obs fragments win.
    class ScaledNet(ConstantRewardNet):
        def __init__(self, scale):
            super().__init__()
            self.scale = scale

        def forward(self, state, action, next_status=None, done=None):
            return self.scale * state.sum(dim=1)

    trajs = make_trajectories(rng, [6] * 6)
    # Inflate observations of half the trajectories so their fragments have
    # maximal ensemble variance; sampling 20 candidates guarantees several
    # inflated ones are available for the top-2 selection.
    for traj in trajs[:3]:
        for t in traj:
            t.observation = t.observation + 100.0

    ensemble = RewardEnsemble(
        ScaledNet(1.0).observation_space,
        ScaledNet(1.0).action_space,
        [ScaledNet(1.0), ScaledNet(-1.0)],
    )
    frag = HighVariancePairFragmenter(
        rng=rng, logger=NullLogger(), reward_ensemble=ensemble, oversample=10
    )
    pairs = frag(trajs, 3, 1)
    picked = [pairs[0].frag1, pairs[0].frag2]
    # Both selected fragments should come from inflated trajectories.
    assert all(abs(float(f[0].observation.sum())) > 50 for f in picked)
