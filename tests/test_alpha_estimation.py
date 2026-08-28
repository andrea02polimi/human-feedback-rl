"""The reliability weight alpha, from per-sample dispersion to the minibatch.

These tests are the contract of the method: they pin down the two definitions
of variance, the distinct roles of N and B, the exactness of the preference
decomposition, the definition adopted for demonstrations, the threshold below
which alpha stays pinned, and the keys that must appear in the logs.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch as th

from human_feedback_rl.algorithms.hybrid.alpha_estimation import (
    _dispersion,
    demonstration_sample_gradients,
    estimate_alpha,
    preference_sample_gradients,
)
from human_feedback_rl.algorithms.hybrid.demonstration_losses import demo_2_loss
from human_feedback_rl.common.batching import fragment_avg_rewards, fragment_sum_rewards
from human_feedback_rl.common.datasets import PreferenceBatch
from human_feedback_rl.common.preference_losses import (
    bradley_terry_probs,
    preference_labels_tensor,
    preference_nll,
)
from human_feedback_rl.common.types import FragmentPair, Preference

from conftest import make_trajectories


# --- the dispersion recipe --------------------------------------------------

def test_dispersion_uses_n_minus_1_and_divides_by_the_minibatch():
    """The two divisors play different roles and must not be confused.

    V estimates how much the data-generating process scatters (divisor N-1);
    S = V/B describes the noise of the gradient ACTUALLY applied, and B is the
    minibatch size, not the number of samples available.
    """
    g = th.tensor([[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]])   # mean (3,0)
    d = _dispersion(g, batch=2, eps=1e-12)
    expected_V = ((1 - 3) ** 2 + (3 - 3) ** 2 + (5 - 3) ** 2) / (3 - 1)
    assert d.n == 3 and d.batch == 2
    assert d.process_var == pytest.approx(expected_V)
    assert d.mean_var == pytest.approx(expected_V / 2)
    assert d.mean_norm_sq == pytest.approx(9.0)
    assert d.cv2 == pytest.approx(expected_V / 2 / 9.0)


def test_the_variance_of_the_mean_falls_as_the_minibatch_grows():
    """The sanity check: for the same process, S falls with B."""
    g = th.randn(50, 4, generator=th.Generator().manual_seed(0))
    s = [_dispersion(g, batch=b, eps=1e-12).mean_var for b in (2, 10, 50)]
    assert s[0] > s[1] > s[2]
    # while V does not depend on B
    v = {_dispersion(g, batch=b, eps=1e-12).process_var for b in (2, 10, 50)}
    assert len(v) == 1


def test_dispersion_is_undefined_below_two_samples():
    assert _dispersion(th.zeros(1, 3), batch=1, eps=1e-12) is None
    assert _dispersion(th.zeros(5, 3), batch=0, eps=1e-12) is None


def test_identical_gradients_give_zero_variance():
    g = th.ones(6, 3)
    assert _dispersion(g, batch=3, eps=1e-12).process_var == pytest.approx(0.0)


# --- the preference decomposition: exact ------------------------------------

def _pref_batch(rng, n, net):
    trajs = make_trajectories(rng, [4] * (2 * n))
    pairs = [FragmentPair(trajs[2 * i], trajs[2 * i + 1]) for i in range(n)]
    prefs = [Preference(1.0, 0.0) if i % 2 else Preference(0.0, 1.0) for i in range(n)]
    return PreferenceBatch(pairs, prefs)


def test_per_comparison_gradients_recompose_the_full_batch_gradient(
    rng, tiny_reward_ensemble
):
    """The preference decomposition is exact, not approximate."""
    member = tiny_reward_ensemble.members[0]
    params = [p for p in member.parameters() if p.requires_grad]
    batch = _pref_batch(rng, 6, member)
    labels = preference_labels_tensor(batch.preferences)

    per_sample = preference_sample_gradients(member, batch, labels, params)
    mean = per_sample.mean(dim=0)

    r1 = fragment_avg_rewards(member, [p.frag1 for p in batch.fragment_pairs])
    r2 = fragment_avg_rewards(member, [p.frag2 for p in batch.fragment_pairs])
    loss = preference_nll(bradley_terry_probs(r1, r2), labels)
    grads = th.autograd.grad(loss, params)
    full = th.cat([g.reshape(-1) for g in grads])

    assert th.allclose(mean, full, atol=1e-5)


# --- the definition adopted for demonstrations ------------------------------

def test_the_per_demonstration_gradient_is_the_one_expert_loss_gradient(
    rng, tiny_reward_ensemble
):
    """One expert at a time, with the whole rollout frozen."""
    member = tiny_reward_ensemble.members[0]
    params = [p for p in member.parameters() if p.requires_grad]
    experts = make_trajectories(rng, [5, 5, 5])
    rollout = make_trajectories(np.random.default_rng(7), [5, 5])

    per_sample = demonstration_sample_gradients(member, experts, rollout, params)

    for i, traj in enumerate(experts):
        loss_i = demo_2_loss(
            fragment_sum_rewards(member, [traj]),
            fragment_sum_rewards(member, rollout),
        )
        expected = th.cat([g.reshape(-1) for g in th.autograd.grad(loss_i, params)])
        assert th.allclose(per_sample[i], expected, atol=1e-5), f"sample {i}"


def test_the_rollout_is_shared_by_every_sample(rng, tiny_reward_ensemble):
    """Changing the rollout changes EVERY row: it is not a per-sample nuisance."""
    member = tiny_reward_ensemble.members[0]
    params = [p for p in member.parameters() if p.requires_grad]
    experts = make_trajectories(rng, [5, 5])
    a = demonstration_sample_gradients(
        member, experts, make_trajectories(np.random.default_rng(1), [5, 5]), params)
    b = demonstration_sample_gradients(
        member, experts, make_trajectories(np.random.default_rng(2), [5, 5]), params)
    assert not th.allclose(a, b)


# --- from CV^2 to alpha -----------------------------------------------------

def test_alpha_stays_pinned_to_one_below_the_comparison_threshold(
    rng, tiny_reward_ensemble
):
    """With very few comparisons the preference dispersion is not estimable."""
    member = tiny_reward_ensemble.members[0]
    params = [p for p in member.parameters() if p.requires_grad]
    batch = _pref_batch(rng, 3, member)          # 3 < 5
    est = estimate_alpha(
        member, params, batch, preference_labels_tensor(batch.preferences),
        make_trajectories(rng, [5, 5]), make_trajectories(rng, [5, 5]),
        batch_size_pref=8, batch_size_expert=8, min_prefs=5, eps=1e-8,
    )
    assert est.alpha == 1.0 and est.pinned is True
    assert est.pref is None and est.demo is None


def test_alpha_is_the_ratio_between_the_two_cv2(rng, tiny_reward_ensemble):
    member = tiny_reward_ensemble.members[0]
    params = [p for p in member.parameters() if p.requires_grad]
    batch = _pref_batch(rng, 8, member)
    est = estimate_alpha(
        member, params, batch, preference_labels_tensor(batch.preferences),
        make_trajectories(rng, [5, 5, 5]), make_trajectories(rng, [5, 5]),
        batch_size_pref=4, batch_size_expert=2, min_prefs=5, eps=1e-8,
    )
    assert est.pinned is False
    assert 0.0 <= est.alpha <= 1.0
    expected = est.pref.cv2 / (est.pref.cv2 + est.demo.cv2)
    assert est.alpha == pytest.approx(expected)


def test_the_minibatch_is_the_min_of_batch_size_and_samples(rng, tiny_reward_ensemble):
    """B = min(batch_size, N): you cannot average over more samples than exist."""
    member = tiny_reward_ensemble.members[0]
    params = [p for p in member.parameters() if p.requires_grad]
    batch = _pref_batch(rng, 6, member)
    est = estimate_alpha(
        member, params, batch, preference_labels_tensor(batch.preferences),
        make_trajectories(rng, [5, 5, 5]), make_trajectories(rng, [5, 5]),
        batch_size_pref=999, batch_size_expert=999, min_prefs=5, eps=1e-8,
    )
    assert est.pref.batch == 6 and est.pref.n == 6
    assert est.demo.batch == 3 and est.demo.n == 3


def test_alpha_rises_when_the_preferences_scatter_more():
    """What the weight means: whoever scatters more gets less of it."""
    tight = _dispersion(th.tensor([[1.0, 0.0], [1.1, 0.0]]), batch=2, eps=1e-12)
    wide = _dispersion(th.tensor([[1.0, 0.0], [9.0, 0.0]]), batch=2, eps=1e-12)
    alpha_noisy_prefs = wide.cv2 / (wide.cv2 + tight.cv2)
    alpha_noisy_demos = tight.cv2 / (tight.cv2 + wide.cv2)
    assert alpha_noisy_prefs > 0.5 > alpha_noisy_demos
