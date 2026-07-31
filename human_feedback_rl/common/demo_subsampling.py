"""Reproducible selection of the expert-demonstration subsample.

Every arm that consumes demonstrations — the demo-only baselines and the
hybrid algorithm — must see the SAME demonstrations at the same budget.
Otherwise a budget curve compares algorithms and demonstration sets at once,
and a difference between two arms cannot be attributed to either. Two
properties make the selection shareable:

* it is the prefix of a single permutation that depends only on ``seed`` and
  the dataset size, so smaller budgets are nested inside larger ones
  (10 demos are contained in 100, which are contained in 1000);
* the seed is the fixed constant ``DEMO_SUBSAMPLE_SEED``, NOT the run seed,
  so two arms at the same budget agree even when they train on different
  seeds.

``subsample_manifest`` records what was actually selected. Each run writes it
next to its other artifacts and reports the fingerprint to W&B, so the two
properties can be *verified* after the fact rather than assumed; see
``scripts/verify_demo_subsample.py``.
"""

import hashlib
from typing import Dict, List, Optional, Sequence

import numpy as np

# The one seed every arm subsamples with. Deliberately independent of
# ``run.seed``: multi-seed runs vary the training stochasticity, not which
# demonstrations they are given.
DEMO_SUBSAMPLE_SEED = 1000


def select_demo_indices(
    n_available: int,
    lengths: Optional[Sequence[int]] = None,
    n_trajectories: Optional[int] = None,
    n_transitions: Optional[int] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Return the indices of the demonstrations to keep, in selection order.

    ``n_trajectories`` budgets in whole demonstrations, ``n_transitions`` in
    transitions (whole trajectories are taken while the cumulative length
    stays within the cap, always at least one); the two are mutually
    exclusive. With neither set the full dataset is returned.

    ``seed=None`` means :data:`DEMO_SUBSAMPLE_SEED`, so callers that forget to
    thread a seed through still get the shared subsample rather than a private
    one.
    """
    if n_trajectories is not None and n_transitions is not None:
        raise ValueError(
            "n_trajectories and n_transitions are mutually exclusive; "
            f"got {n_trajectories} and {n_transitions}."
        )
    if n_available < 1:
        raise ValueError(f"n_available must be >= 1, got {n_available}.")
    if n_trajectories is None and n_transitions is None:
        return np.arange(n_available)

    if seed is None:
        seed = DEMO_SUBSAMPLE_SEED
    # Permuting the whole dataset (not just the budget) is what makes the
    # budgets nested: every budget reads a prefix of the same order.
    order = np.random.default_rng(seed).permutation(n_available)

    if n_transitions is not None:
        if lengths is None:
            raise ValueError("lengths is required when budgeting by n_transitions.")
        if len(lengths) != n_available:
            raise ValueError(
                f"lengths has {len(lengths)} entries but n_available is {n_available}."
            )
        if n_transitions < 1:
            raise ValueError(f"n_transitions must be >= 1, got {n_transitions}.")
        selected: List[int] = []
        total = 0
        for index in order:
            length = int(lengths[index])
            # Whole trajectories only: stop before exceeding the cap, but never
            # return an empty set (the first trajectory may be longer than it).
            if selected and total + length > n_transitions:
                break
            selected.append(int(index))
            total += length
        return np.asarray(selected, dtype=int)

    if not 1 <= n_trajectories <= n_available:
        raise ValueError(
            f"n_trajectories must be in [1, {n_available}], got {n_trajectories}."
        )
    return order[:n_trajectories]


def indices_fingerprint(indices: Sequence[int]) -> str:
    """Hash the SET of selected demonstrations.

    Sorted before hashing: two runs that picked the same demonstrations match
    even if they enumerated them in a different order, which is exactly the
    "same demonstrations" question the fingerprint is meant to answer.
    """
    payload = ",".join(str(int(i)) for i in sorted(indices))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def dataset_fingerprint(lengths: Sequence[int]) -> str:
    """Hash the dataset shape (size and per-trajectory lengths).

    The permutation depends on the dataset size, so a subsample fingerprint is
    only comparable across runs that read the same underlying pickle. This
    catches a swapped or regenerated dataset silently changing the selection.
    """
    payload = ",".join(str(int(length)) for length in lengths)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def subsample_manifest(
    indices: Sequence[int],
    lengths: Sequence[int],
    seed: Optional[int],
    n_trajectories: Optional[int] = None,
    n_transitions: Optional[int] = None,
    dataset_name: str = "",
) -> Dict:
    """Describe a selection well enough to reproduce and compare it."""
    indices = [int(i) for i in indices]
    return {
        "dataset_name": dataset_name,
        "dataset_n_trajectories": len(lengths),
        "dataset_fingerprint": dataset_fingerprint(lengths),
        "subsample_seed": DEMO_SUBSAMPLE_SEED if seed is None else int(seed),
        "budget_n_trajectories": n_trajectories,
        "budget_n_transitions": n_transitions,
        "n_selected": len(indices),
        "n_transitions_selected": sum(int(lengths[i]) for i in indices),
        "fingerprint": indices_fingerprint(indices),
        # Selection order, i.e. the permutation prefix: keeping it makes the
        # nesting between budgets checkable, which the fingerprint alone
        # (a set hash) cannot show.
        "indices": indices,
    }
