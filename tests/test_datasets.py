import numpy as np
import pytest

from human_feedback_rl.common.datasets import PreferenceDataset
from human_feedback_rl.common.types import FragmentPair, Preference

from conftest import make_trajectories


def _pairs_and_prefs(rng, n):
    trajs = make_trajectories(rng, [3] * (2 * n))
    pairs = [FragmentPair(trajs[2 * i], trajs[2 * i + 1]) for i in range(n)]
    prefs = [Preference(1.0, 0.0) for _ in range(n)]
    return pairs, prefs


def test_push_len_and_eviction(rng):
    ds = PreferenceDataset(queue_size=5, rng=rng)
    pairs, prefs = _pairs_and_prefs(rng, 8)
    ds.push(pairs, prefs)
    assert len(ds) == 5  # circular buffer keeps the newest 5
    kept = ds.get_all().fragment_pairs
    assert kept == pairs[3:]


def test_push_length_mismatch_raises(rng):
    ds = PreferenceDataset(rng=rng)
    pairs, prefs = _pairs_and_prefs(rng, 2)
    with pytest.raises(ValueError):
        ds.push(pairs, prefs[:1])


def test_sample_without_replacement(rng):
    ds = PreferenceDataset(rng=rng)
    pairs, prefs = _pairs_and_prefs(rng, 6)
    ds.push(pairs, prefs)
    batch = ds.sample(4)
    assert len(batch.fragment_pairs) == 4
    ids = [id(p) for p in batch.fragment_pairs]
    assert len(set(ids)) == 4


def test_sample_empty_raises(rng):
    with pytest.raises(ValueError):
        PreferenceDataset(rng=rng).sample(3)


def test_empty_get_all_returns_empty_batch(rng):
    # Regression: used to raise a cryptic zip-unpacking error.
    batch = PreferenceDataset(rng=rng).get_all()
    assert batch.fragment_pairs == [] and batch.preferences == []


def test_get_yields_every_item_once(rng):
    ds = PreferenceDataset(rng=rng)
    pairs, prefs = _pairs_and_prefs(rng, 7)
    ds.push(pairs, prefs)
    seen = [p for batch in ds.get(3) for p in batch.fragment_pairs]
    assert sorted(map(id, seen)) == sorted(map(id, pairs))


def test_bootstrap_same_size_with_replacement(rng):
    ds = PreferenceDataset(rng=np.random.default_rng(1))
    pairs, prefs = _pairs_and_prefs(rng, 10)
    ds.push(pairs, prefs)
    boot = ds.bootstrap()
    assert len(boot) == len(ds)
    boot_ids = {id(p) for p in boot.get_all().fragment_pairs}
    # With replacement: over 10 draws from 10 items, duplicates are essentially certain.
    assert len(boot_ids) < 10
