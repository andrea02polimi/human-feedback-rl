import math

import pytest
import torch as th

from human_feedback_rl.common.preference_losses import (
    bradley_terry_probs,
    preference_accuracy,
    preference_nll,
)


def test_bradley_terry_hand_computed():
    r1 = th.tensor([1.0, 0.0])
    r2 = th.tensor([0.0, 2.0])
    probs = bradley_terry_probs(r1, r2)
    assert probs[0, 0].item() == pytest.approx(1 / (1 + math.exp(-1.0)))
    assert probs[1, 0].item() == pytest.approx(1 / (1 + math.exp(2.0)))
    assert th.allclose(probs.sum(dim=1), th.ones(2))


def test_bradley_terry_symmetry():
    r1 = th.tensor([0.3, -1.2])
    r2 = th.tensor([1.1, 0.4])
    forward = bradley_terry_probs(r1, r2)
    backward = bradley_terry_probs(r2, r1)
    assert th.allclose(forward, backward.flip(dims=[1]))


def test_preference_nll_equals_neg_log_prob_for_hard_labels():
    probs = th.tensor([[0.8, 0.2], [0.3, 0.7]])
    labels = th.tensor([[1.0, 0.0], [0.0, 1.0]])
    expected = -(math.log(0.8) + math.log(0.7)) / 2
    assert preference_nll(probs, labels).item() == pytest.approx(expected)


def test_preference_nll_soft_labels():
    probs = th.tensor([[0.6, 0.4]])
    labels = th.tensor([[0.5, 0.5]])
    expected = -(0.5 * math.log(0.6) + 0.5 * math.log(0.4))
    assert preference_nll(probs, labels).item() == pytest.approx(expected)


def test_preference_accuracy():
    probs = th.tensor([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    labels = th.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert preference_accuracy(probs, labels).item() == pytest.approx(1 / 3)
