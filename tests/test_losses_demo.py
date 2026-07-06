import math

import numpy as np
import pytest
import torch as th

from human_feedback_rl.algorithms.demo.losses import (
    RewardLossMixin,
    VALID_LOSSES,
    demo_corrected_loss,
    demo_loss,
    maxent2_loss,
    maxent_corrected_partition,
    maxent_loss,
)

from conftest import ConstantRewardNet, make_trajectories


def _logsumexp(values):
    return math.log(sum(math.exp(v) for v in values))


def test_maxent_loss_hand_computed():
    expert = th.tensor([1.0, 1.0])
    model = th.tensor([0.0, 0.0])
    # -mean(expert) + logsumexp(model) - log(n) = -1 + log(2) - log(2) = -1
    assert maxent_loss(expert, model).item() == pytest.approx(-1.0)


def test_maxent2_loss_hand_computed():
    expert = th.tensor([2.0])
    model = th.tensor([0.0, 1.0])
    expected = -2.0 + _logsumexp([0.0, 1.0, 2.0]) - math.log(3)
    assert maxent2_loss(expert, model).item() == pytest.approx(expected)


def test_demo_loss_is_difference_of_means():
    expert = th.tensor([3.0, 5.0])
    model = th.tensor([1.0, 2.0, 3.0])
    assert demo_loss(expert, model).item() == pytest.approx(-4.0 + 2.0)


def test_demo_corrected_loss_hand_computed():
    margins = th.tensor([2.0, -2.0])
    temperature = 2.0
    expected = (math.log(1 + math.exp(-1.0)) + math.log(1 + math.exp(1.0))) / 2
    assert demo_corrected_loss(margins, temperature).item() == pytest.approx(expected)


def test_maxent_corrected_partition_hand_computed():
    logits = th.tensor([0.0, 1.0, 2.0])
    expected = _logsumexp([0.0, 1.0, 2.0]) - math.log(3)
    assert maxent_corrected_partition(logits).item() == pytest.approx(expected)


class _Shim(RewardLossMixin):
    """Minimal stand-in for DemoAlgorithm carrying the attributes the mixin needs."""

    def __init__(self, loss_type, expert_trajs, model_trajs, fragment_length=None,
                 temperature=2.0):
        self.loss_type = loss_type
        self.fragment_length = fragment_length
        self.temperature = temperature
        self.batch_size_expert = len(expert_trajs)
        self.batch_size_model = len(model_trajs)
        self.rng = np.random.default_rng(7)
        self.expert_trajectories = expert_trajs
        self.trajectories = model_trajs
        self.agent = None  # log_policy_prob is stored on every transition
        self._maxent_corrected_steps = []


@pytest.mark.parametrize("loss_type", VALID_LOSSES)
def test_reward_loss_dispatch_is_finite(rng, loss_type):
    expert = make_trajectories(rng, [4, 6], with_log_probs=True)
    model = make_trajectories(rng, [5, 3, 4], with_log_probs=True)
    shim = _Shim(loss_type, expert, model)
    loss = shim._reward_loss(ConstantRewardNet())
    assert th.isfinite(loss)


def test_maxent_corrected_whole_trajectory_equals_manual_formula(rng):
    expert = make_trajectories(rng, [4, 5], with_log_probs=True)
    model = make_trajectories(rng, [3, 6], with_log_probs=True)
    member = ConstantRewardNet()
    shim = _Shim("maxent_corrected", expert, model, fragment_length=None)
    loss = shim._reward_loss(member).item()

    def traj_return(traj):
        return sum(
            float(t.observation.sum()) + 0.5 * float(t.action.sum()) for t in traj
        )

    tau = shim.temperature
    expert_term = -np.mean([traj_return(t) for t in expert]) / tau
    logits = [
        traj_return(t) / tau - sum(tr.log_policy_prob for tr in t) for t in model
    ]
    partition = math.log(np.mean([math.exp(l) for l in logits]))
    assert loss == pytest.approx(expert_term + partition, rel=1e-5)


def test_fragment_returns_chunking_preserves_total(rng):
    trajs = make_trajectories(rng, [5], with_log_probs=True)
    member = ConstantRewardNet()
    shim = _Shim("maxent_corrected", trajs, trajs, fragment_length=2)
    fragments = shim._fragment_returns(member, trajs)
    assert fragments.shape == (3,)  # chunks of (2, 2, 1)
    whole = shim._traj_sum_reward(member, trajs[0])
    assert fragments.sum().item() == pytest.approx(whole.item(), rel=1e-5)


def test_maxent_corrected_records_ess_diagnostics(rng):
    expert = make_trajectories(rng, [4], with_log_probs=True)
    model = make_trajectories(rng, [4, 4], with_log_probs=True)
    shim = _Shim("maxent_corrected", expert, model)
    shim._reward_loss(ConstantRewardNet())
    (step,) = shim._maxent_corrected_steps
    assert 0 < step["ess_fraction"] <= 1.0
    assert np.isfinite(step["logit_var"])
