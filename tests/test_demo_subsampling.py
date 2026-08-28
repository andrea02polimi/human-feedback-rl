"""The property the budget curves rest on: same budget -> same demonstrations."""
from __future__ import annotations

import numpy as np
import pytest

from human_feedback_rl.common.demo_subsampling import (
    DEMO_SUBSAMPLE_SEED,
    dataset_fingerprint,
    indices_fingerprint,
    select_demo_indices,
    subsample_manifest,
)

N_AVAILABLE = 500
LENGTHS = [10 + i % 7 for i in range(N_AVAILABLE)]


def indices(n_trajectories=None, n_transitions=None, seed=None):
    return select_demo_indices(
        N_AVAILABLE, lengths=LENGTHS, n_trajectories=n_trajectories,
        n_transitions=n_transitions, seed=seed,
    )


def test_same_budget_same_demonstrations():
    """A demo method and a hybrid one asking for n=10 must get THE SAME 10."""
    a, b = indices(n_trajectories=10), indices(n_trajectories=10)
    assert np.array_equal(a, b)


def test_the_selection_ignores_the_training_seed():
    """Nothing in the selection may depend on run.seed."""
    assert np.array_equal(indices(n_trajectories=25), indices(n_trajectories=25, seed=None))
    assert indices_fingerprint(indices(n_trajectories=25)) == indices_fingerprint(
        select_demo_indices(N_AVAILABLE, lengths=LENGTHS, n_trajectories=25,
                            seed=DEMO_SUBSAMPLE_SEED))


def test_seed_none_is_the_shared_constant():
    assert np.array_equal(
        indices(n_trajectories=10),
        select_demo_indices(N_AVAILABLE, n_trajectories=10, seed=DEMO_SUBSAMPLE_SEED),
    )


@pytest.mark.parametrize("budgets", [(10, 100), (1, 10, 100, 500), (20, 50, 200)])
def test_budgets_are_nested(budgets):
    """A larger budget ADDS demonstrations, it does not swap them."""
    for smaller, larger in zip(budgets, budgets[1:]):
        a = set(indices(n_trajectories=smaller).tolist())
        b = set(indices(n_trajectories=larger).tolist())
        assert a <= b, f"{smaller} is not contained in {larger}"


def test_different_seeds_select_different_demonstrations():
    """The guarantee is a property of the shared seed, not a lucky draw."""
    a = set(indices(n_trajectories=50, seed=1).tolist())
    b = set(indices(n_trajectories=50, seed=2).tolist())
    assert a != b


def test_a_transition_budget_respects_the_cap_and_is_nested():
    sel = indices(n_transitions=200)
    assert sum(LENGTHS[i] for i in sel) <= 200
    smaller = set(indices(n_transitions=100).tolist())
    larger = set(indices(n_transitions=400).tolist())
    assert smaller <= larger


def test_a_transition_budget_gives_at_least_one_trajectory():
    """A cap below the shortest trajectory still returns something usable."""
    sel = indices(n_transitions=1)
    assert len(sel) == 1


def test_the_two_budgets_share_the_permutation():
    """They read the same order, so the two axes stay comparable."""
    per_traj = indices(n_trajectories=1)
    per_trans = indices(n_transitions=1)
    assert per_traj[0] == per_trans[0]


def test_without_a_budget_the_whole_dataset_is_taken():
    assert np.array_equal(select_demo_indices(N_AVAILABLE), np.arange(N_AVAILABLE))


@pytest.mark.parametrize("kwargs", [
    dict(n_trajectories=0),
    dict(n_trajectories=N_AVAILABLE + 1),
    dict(n_transitions=0),
    dict(n_trajectories=10, n_transitions=100),
])
def test_invalid_budgets_raise(kwargs):
    with pytest.raises(ValueError):
        indices(**kwargs)


def test_a_transition_budget_needs_the_lengths():
    with pytest.raises(ValueError, match="lengths"):
        select_demo_indices(N_AVAILABLE, n_transitions=100)


def test_the_fingerprint_identifies_the_set_not_the_order():
    a = [3, 1, 2]
    b = [2, 3, 1]
    assert indices_fingerprint(a) == indices_fingerprint(b)
    assert indices_fingerprint(a) != indices_fingerprint([1, 2, 4])


def test_the_dataset_fingerprint_sees_a_changed_dataset():
    assert dataset_fingerprint(LENGTHS) != dataset_fingerprint(LENGTHS[:-1])


def test_the_manifest_describes_the_selection():
    sel = indices(n_trajectories=10)
    m = subsample_manifest(sel, LENGTHS, seed=None, n_trajectories=10,
                           dataset_name="expert")
    assert m["n_selected"] == 10
    # names the training entry point relies on
    assert m["subsample_seed"] == DEMO_SUBSAMPLE_SEED
    assert m["budget_n_trajectories"] == 10
    assert m["fingerprint"] == indices_fingerprint(sel)
    assert m["n_transitions_selected"] == sum(LENGTHS[i] for i in sel)
    assert m["dataset_fingerprint"] == dataset_fingerprint(LENGTHS)


def test_two_methods_get_the_same_manifest_at_the_same_budget():
    """End-to-end shape of the check the runs make possible."""
    def manifest_for(training_seed):
        # the training seed plays no part in the selection
        sel = select_demo_indices(N_AVAILABLE, lengths=LENGTHS, n_trajectories=100)
        return subsample_manifest(sel, LENGTHS, seed=None, n_trajectories=100)

    assert manifest_for(1)["fingerprint"] == manifest_for(7)["fingerprint"]
