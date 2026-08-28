"""Reproducible selection of the demonstration subsample.

Every method must see the same demonstrations at the same budget, or a
difference between methods could come from which trajectories were drawn. The
seed is a shared constant, independent of the training seed, and the whole
dataset is permuted before a prefix is taken, so budgets are nested.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Sequence

import numpy as np

# Shared constant: this is NOT the training seed.
DEMO_SUBSAMPLE_SEED = 1000


def select_demo_indices(
    n_available: int,
    lengths: Optional[Sequence[int]] = None,
    n_trajectories: Optional[int] = None,
    n_transitions: Optional[int] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Indices of the demonstrations to keep, in selection order.

    ``n_trajectories`` sets the budget in whole trajectories; ``n_transitions``
    sets it in transitions (whole trajectories are taken while the cumulative
    length fits the cap, always at least one). The two are mutually exclusive.
    With neither, the whole dataset is returned.

    ``seed=None`` means :data:`DEMO_SUBSAMPLE_SEED`, so forgetting to pass it
    still yields the shared subsample rather than a private one.
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
    # Permuting the whole dataset, not just the budget, is what makes the
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
            if selected and total + length > n_transitions:
                continue
            selected.append(int(index))
            total += length
            if total >= n_transitions:
                break
        return np.asarray(selected, dtype=int)

    if not 1 <= n_trajectories <= n_available:
        raise ValueError(
            f"n_trajectories must be in [1, {n_available}], got {n_trajectories}."
        )
    return order[:n_trajectories]


def indices_fingerprint(indices: Sequence[int]) -> str:
    """Hash of the selected SET, not of its order.

    Two methods at the same budget may list the same demonstrations in a
    different order: what has to match is the set.
    """
    payload = ",".join(str(int(i)) for i in sorted(indices))
    return hashlib.sha1(payload.encode()).hexdigest()


def dataset_fingerprint(lengths: Sequence[int]) -> str:
    """Hash of the dataset's shape: how many trajectories, and how long.

    It separates "same budget, same dataset" from "same budget, but someone
    regenerated the demonstrations".
    """
    payload = ",".join(str(int(x)) for x in lengths)
    return hashlib.sha1(payload.encode()).hexdigest()


def subsample_manifest(
    indices: Sequence[int],
    lengths: Sequence[int],
    seed: Optional[int],
    n_trajectories: Optional[int] = None,
    n_transitions: Optional[int] = None,
    dataset_name: str = "",
) -> Dict:
    """Describes a selection well enough to reproduce it and to compare it."""
    idx = [int(i) for i in indices]
    # The key names are an interface: the training entry point logs
    # subsample_seed, fingerprint, dataset_fingerprint and
    # n_transitions_selected. Renaming them breaks it.
    return {
        "dataset_name": dataset_name,
        "dataset_size": len(lengths),
        "dataset_fingerprint": dataset_fingerprint(lengths),
        "subsample_seed": DEMO_SUBSAMPLE_SEED if seed is None else int(seed),
        "budget_n_trajectories": n_trajectories,
        "budget_n_transitions": n_transitions,
        "n_selected": len(idx),
        "n_transitions_selected": int(sum(int(lengths[i]) for i in idx)),
        "indices": idx,
        "fingerprint": indices_fingerprint(idx),
    }
