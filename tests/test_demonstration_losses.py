import math

import numpy as np
import pytest
import torch as th

from human_feedback_rl.algorithms.hybrid.demonstration_losses import (
    RewardLossMixin,
    VALID_LOSSES,
    demo_1_loss,
    demo_2_loss,
)

from conftest import ConstantRewardNet, make_trajectories


def _logsumexp(values):
    return math.log(sum(math.exp(v) for v in values))


def test_demo_1_loss_is_difference_of_means():
    expert = th.tensor([3.0, 5.0])
    model = th.tensor([1.0, 2.0, 3.0])
    assert demo_1_loss(expert, model).item() == pytest.approx(-4.0 + 2.0)


def test_demo_2_loss_hand_computed():
    expert = th.tensor([2.0])
    model = th.tensor([0.0, 1.0])
    expected = -2.0 + _logsumexp([0.0, 1.0, 2.0]) - math.log(3)
    assert demo_2_loss(expert, model).item() == pytest.approx(expected)


class _Shim(RewardLossMixin):
    """Minimal stand-in for HybridAlgorithm carrying what the mixin needs."""

    def __init__(self, loss_type, expert_trajs, model_trajs):
        self.loss_type = loss_type
        self.batch_size_expert = len(expert_trajs)
        self.batch_size_model = len(model_trajs)
        # Demonstration minibatches draw from the training stream, kept apart
        # from the one that picks fragments and the one for the oracle.
        self._rng_train = np.random.default_rng(7)
        self.expert_trajectories = expert_trajs
        self.trajectories = model_trajs


@pytest.mark.parametrize("loss_type", VALID_LOSSES)
def test_reward_loss_dispatch_is_finite(rng, loss_type):
    expert = make_trajectories(rng, [4, 6])
    model = make_trajectories(rng, [5, 3, 4])
    shim = _Shim(loss_type, expert, model)
    loss = shim._reward_loss(ConstantRewardNet())
    assert th.isfinite(loss)
