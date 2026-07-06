import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Any, List, Optional

from .types import FragmentPair, Preference


class BaseDataset:
    """Deque-backed dataset retaining the last ``queue_size`` individual items."""

    def __init__(self, queue_size: int = 1_000_000, rng: Optional[np.random.Generator] = None):
        self.queue_size = queue_size
        self._data: deque = deque(maxlen=queue_size)
        self.rng = rng if rng is not None else np.random.default_rng()

    def push(self, *items: Any) -> None:
        self._data.extend(items)

    def _to_batch(self, items: List[Any]):
        return items

    def get(self, batch_size: int):
        """Yield all items in random sequential batches."""
        data = list(self._data)
        indices = self.rng.permutation(len(data))
        for start in range(0, len(data), batch_size):
            yield self._to_batch([data[i] for i in indices[start:start + batch_size]])

    def sample(self, batch_size: int):
        """Return a single random batch of at most batch_size items, without replacement."""
        if not self._data:
            raise ValueError(f"Cannot sample from an empty {type(self).__name__}.")
        data = list(self._data)
        indices = self.rng.choice(len(data), size=min(batch_size, len(data)), replace=False)
        return self._to_batch([data[i] for i in indices])

    def get_all(self):
        return self._to_batch(list(self._data))

    def __len__(self) -> int:
        return len(self._data)


@dataclass
class PreferenceBatch:
    fragment_pairs: List[FragmentPair]
    preferences: List[Preference]


class PreferenceDataset(BaseDataset):
    def __init__(self, queue_size: int = 1000, rng: Optional[np.random.Generator] = None):
        super().__init__(queue_size=queue_size, rng=rng)

    def push(self, fragment_pairs: List[FragmentPair], preferences: List[Preference]) -> None:
        if len(fragment_pairs) != len(preferences):
            raise ValueError(
                f"Got {len(fragment_pairs)} fragment pairs but {len(preferences)} preferences."
            )
        super().push(*zip(fragment_pairs, preferences))

    def _to_batch(self, items: List) -> PreferenceBatch:
        if not items:
            return PreferenceBatch([], [])
        fragment_pairs, preferences = zip(*items)
        return PreferenceBatch(list(fragment_pairs), list(preferences))

    def bootstrap(self) -> "PreferenceDataset":
        """Return a new dataset of the same size sampled with replacement."""
        data = list(self._data)
        n = len(data)
        indices = self.rng.choice(n, size=n, replace=True)
        boot = PreferenceDataset(queue_size=n, rng=self.rng)
        for i in indices:
            frag, pref = data[i]
            boot.push([frag], [pref])
        return boot
