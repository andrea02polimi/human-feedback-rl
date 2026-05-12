import numpy as np
from collections import deque
from typing import Any, Deque, List, Optional
from .types import FragmentPair, Fragment, Preference
from dataclasses import dataclass


class BaseDataset:
    def __init__(self, queue_size: int = 1000):
        self.queue_size = queue_size
        self._data: Deque[Any] = deque(maxlen=queue_size)

    def push(self, *items: Any) -> None:
        for item in items:
            self._data.append(item)

    def get(self, batch_size: int):
        """Yield all data in sequential batches."""
        data = list(self._data)
        indices = np.random.permutation(len(data))
        start_idx = 0
        while start_idx < len(data):
            batch_indices = indices[start_idx : start_idx + batch_size]
            yield [data[i] for i in batch_indices]
            start_idx += batch_size

    def sample(self, batch_size: int):
        """Return a single random batch of batch_size items."""
        data = list(self._data)
        indices = np.random.choice(len(data), size=min(batch_size, len(data)), replace=False)
        return [data[i] for i in indices]

    def get_all(self) -> List[Any]:
        return list(self._data)

    def __len__(self) -> int:
        return len(self._data)


@dataclass
class PreferenceBatch:
    fragment_pairs: List[FragmentPair]
    preferences: List[Preference]
    timestamps: List[int]


class PreferenceDataset(BaseDataset):
    def __init__(self, queue_size: int = 1000):
        super().__init__(queue_size=queue_size)

    def push(self, fragment_pairs: List[FragmentPair], preferences: List[Preference], timestamp: int) -> None:
        assert len(fragment_pairs) == len(preferences), "pairs and preferences must have the same length"
        items = [(frag, pref, timestamp) for frag, pref in zip(fragment_pairs, preferences)]
        super().push(*items)

    def _filter(self, timestamp: Optional[int] = None) -> List[Any]:
        data = list(self._data)
        if timestamp is not None:
            data = [item for item in data if item[2] == timestamp]
        return data

    def _to_batch(self, items: List) -> PreferenceBatch:
        fragment_pairs, preferences, timestamps = zip(*items)
        return PreferenceBatch(list(fragment_pairs), list(preferences), list(timestamps))

    def get(self, batch_size: int, timestamp: Optional[int] = None):
        """Yield all data in sequential batches, optionally filtered by timestamp."""
        data = self._filter(timestamp)
        indices = np.random.permutation(len(data))
        start_idx = 0
        while start_idx < len(data):
            batch_indices = indices[start_idx : start_idx + batch_size]
            yield self._to_batch([data[i] for i in batch_indices])
            start_idx += batch_size

    def sample(self, batch_size: int, timestamp: Optional[int] = None) -> PreferenceBatch:
        """Return a single random batch of batch_size items, optionally filtered by timestamp."""
        data = self._filter(timestamp)
        indices = np.random.choice(len(data), size=min(batch_size, len(data)), replace=False)
        return self._to_batch([data[i] for i in indices])

    def get_all(self, timestamp: Optional[int] = None) -> PreferenceBatch:
        """Return all items as a PreferenceBatch, optionally filtered by timestamp."""
        data = self._filter(timestamp)
        return self._to_batch(data)

    def bootstrap(self) -> "PreferenceDataset":
        """Return a new dataset of the same size sampled with replacement."""
        all_data = self.get_all()
        n = len(all_data.fragment_pairs)
        indices = np.random.choice(n, size=n, replace=True)
        boot = PreferenceDataset(queue_size=n)
        for i in indices:
            boot.push([all_data.fragment_pairs[i]], [all_data.preferences[i]], all_data.timestamps[i])
        return boot

    def max_timestamp(self) -> Optional[int]:
        """Return the highest timestamp present in the dataset."""
        if not self._data:
            return None
        return max(item[2] for item in self._data)


@dataclass
class DemoBatch:
    fragments: List[Fragment]
    expert_fragments: List[Fragment]
    timestamps: List[int]


class DemonstrationDataset(BaseDataset):
    def __init__(self, queue_size: int = 1000):
        super().__init__(queue_size=queue_size)

    def push(self, fragments: List[Fragment], expert_fragments: List[Fragment], timestamp: int) -> None:
        assert len(fragments) == len(expert_fragments)
        items = [(frag, frag_exp, timestamp) for frag, frag_exp in zip(fragments, expert_fragments)]
        super().push(*items)

    def _to_batch(self, items: List) -> DemoBatch:
        fragments, expert_fragments, timestamps = zip(*items)
        return DemoBatch(list(fragments), list(expert_fragments), list(timestamps))

    def get(self, batch_size: int):
        data = list(self._data)
        indices = np.random.permutation(len(data))
        start_idx = 0
        while start_idx < len(data):
            batch_indices = indices[start_idx : start_idx + batch_size]
            yield self._to_batch([data[i] for i in batch_indices])
            start_idx += batch_size

    def get_all(self) -> DemoBatch:
        return self._to_batch(list(self._data))

    def bootstrap(self) -> "DemonstrationDataset":
        all_data = list(self._data)
        n = len(all_data)
        indices = np.random.choice(n, size=n, replace=True)
        boot = DemonstrationDataset(queue_size=n)
        for i in indices:
            frag, frag_exp, ts = all_data[i]
            boot.push([frag], [frag_exp], ts)
        return boot

    def max_timestamp(self) -> Optional[int]:
        if not self._data:
            return None
        return max(item[2] for item in self._data)