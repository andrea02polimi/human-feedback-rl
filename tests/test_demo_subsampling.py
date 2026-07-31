"""The property the budget curves depend on: same budget -> same demonstrations."""

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
LENGTHS = [10 + (i % 7) for i in range(N_AVAILABLE)]


def indices(n_trajectories=None, n_transitions=None, seed=None):
    return select_demo_indices(
        n_available=N_AVAILABLE,
        lengths=LENGTHS,
        n_trajectories=n_trajectories,
        n_transitions=n_transitions,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# The cross-arm guarantee
# ---------------------------------------------------------------------------

def test_same_budget_gives_the_same_demonstrations():
    """A demo arm and a hybrid arm asking for n=10 must get the SAME 10."""
    demo_arm = indices(n_trajectories=10)
    hybrid_arm = indices(n_trajectories=10)
    assert np.array_equal(demo_arm, hybrid_arm)
    assert indices_fingerprint(demo_arm) == indices_fingerprint(hybrid_arm)


def test_selection_ignores_the_training_seed():
    """Nothing about the selection may depend on run.seed.

    Arms run on different training seeds; if the demo draw followed the run
    seed, two arms at the same budget would train on different data.
    """
    assert np.array_equal(indices(n_trajectories=50), indices(n_trajectories=50))


def test_none_seed_is_the_shared_constant():
    assert np.array_equal(
        indices(n_trajectories=25, seed=None),
        indices(n_trajectories=25, seed=DEMO_SUBSAMPLE_SEED),
    )


@pytest.mark.parametrize("budgets", [(10, 100), (1, 10, 100, 500), (20, 50, 200)])
def test_budgets_are_nested(budgets):
    """A larger budget only adds demonstrations, never swaps them."""
    previous = None
    for budget in budgets:
        current = set(indices(n_trajectories=budget).tolist())
        assert len(current) == budget
        if previous is not None:
            assert previous <= current
        previous = current


def test_different_seeds_select_different_demonstrations():
    """The guarantee is a property of the shared seed, not an accident."""
    a = indices(n_trajectories=100, seed=DEMO_SUBSAMPLE_SEED)
    b = indices(n_trajectories=100, seed=DEMO_SUBSAMPLE_SEED + 1)
    assert not np.array_equal(a, b)
    assert indices_fingerprint(a) != indices_fingerprint(b)


# ---------------------------------------------------------------------------
# Transition-budgeted selection
# ---------------------------------------------------------------------------

def test_transition_budget_respects_the_cap_and_nests():
    small = indices(n_transitions=200)
    large = indices(n_transitions=2000)
    assert sum(LENGTHS[i] for i in small) <= 200
    assert sum(LENGTHS[i] for i in large) <= 2000
    assert set(small.tolist()) <= set(large.tolist())


def test_transition_budget_returns_at_least_one_trajectory():
    """A cap below the shortest trajectory still yields a usable set."""
    selected = indices(n_transitions=1)
    assert len(selected) == 1


def test_transition_and_trajectory_budgets_share_the_permutation():
    """Both budgets read the same order, so the two axes stay comparable."""
    by_transitions = indices(n_transitions=10_000)
    by_trajectories = indices(n_trajectories=len(by_transitions))
    assert np.array_equal(by_transitions, by_trajectories)


# ---------------------------------------------------------------------------
# Degenerate budgets and invalid input
# ---------------------------------------------------------------------------

def test_no_budget_returns_the_whole_dataset():
    assert np.array_equal(indices(), np.arange(N_AVAILABLE))


@pytest.mark.parametrize("kwargs", [
    dict(n_trajectories=0),
    dict(n_trajectories=N_AVAILABLE + 1),
    dict(n_transitions=0),
    dict(n_trajectories=10, n_transitions=100),
])
def test_invalid_budgets_raise(kwargs):
    with pytest.raises(ValueError):
        indices(**kwargs)


def test_transition_budget_requires_lengths():
    with pytest.raises(ValueError):
        select_demo_indices(n_available=N_AVAILABLE, n_transitions=100)


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------

def test_fingerprint_identifies_the_set_not_the_order():
    assert indices_fingerprint([3, 1, 2]) == indices_fingerprint([1, 2, 3])
    assert indices_fingerprint([1, 2, 3]) != indices_fingerprint([1, 2, 4])


def test_dataset_fingerprint_detects_a_changed_dataset():
    changed = list(LENGTHS)
    changed[0] += 1
    assert dataset_fingerprint(LENGTHS) != dataset_fingerprint(changed)


def test_manifest_describes_the_selection():
    selected = indices(n_trajectories=30)
    manifest = subsample_manifest(
        indices=selected,
        lengths=LENGTHS,
        seed=None,
        n_trajectories=30,
        dataset_name="fake.pkl",
    )
    assert manifest["subsample_seed"] == DEMO_SUBSAMPLE_SEED
    assert manifest["n_selected"] == 30
    assert manifest["n_transitions_selected"] == sum(LENGTHS[i] for i in selected)
    assert manifest["fingerprint"] == indices_fingerprint(selected)
    assert manifest["dataset_n_trajectories"] == N_AVAILABLE
    # The ordered indices are what makes nesting checkable after the fact.
    assert manifest["indices"] == [int(i) for i in selected]


def test_manifests_of_two_arms_match_at_the_same_budget():
    """End-to-end shape of the verification the runs make possible."""
    def manifest_for(seed):
        selected = indices(n_trajectories=100, seed=seed)
        return subsample_manifest(
            indices=selected, lengths=LENGTHS, seed=seed, n_trajectories=100
        )

    demo_arm = manifest_for(None)
    hybrid_arm = manifest_for(DEMO_SUBSAMPLE_SEED)
    assert demo_arm["fingerprint"] == hybrid_arm["fingerprint"]
    assert demo_arm["dataset_fingerprint"] == hybrid_arm["dataset_fingerprint"]
