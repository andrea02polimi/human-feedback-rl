import numpy as np
import random
from collections import deque
from typing import Any, Deque, List, Tuple
from .types import FragmentPair, Preference

from dataclasses import dataclass


class BaseDataset:
    def __init__(self, train_frac: float = 0.8, queue_size: int = 1000):
        assert 0.0 < train_frac < 1.0, "train_frac must be in (0, 1)"
        self.train_frac = train_frac
        self.queue_size = queue_size

        self.train_data: Deque[Any] = deque(maxlen=queue_size)
        self.val_data: Deque[Any] = deque(maxlen=queue_size)

    def push(self, *items: Any) -> None:
        for item in items:
            if random.random() < self.train_frac:
                self.train_data.append(item)
            else:
                self.val_data.append(item)

    def get(self, batch_size):
        data = list(self.train_data)
        indices = np.random.permutation(len(data))
        start_idx = 0
        while start_idx < len(data):
            batch_indices = indices[start_idx:start_idx + batch_size]
            yield [data[i] for i in batch_indices]
            start_idx += batch_size

    def get_train(self) -> List[Any]:
        data = list(self.train_data)
        return data

    def get_val(self) -> List[Any]:
        data = list(self.val_data)
        return data

    def get_all(self) -> List[Any]:
        data = list(self.train_data) + list(self.val_data)
        return data

    def __len__(self) -> int:
        return len(self.train_data) + len(self.val_data)




@dataclass
class PreferenceBatch:
    fragment_pairs: List[FragmentPair]
    preferences: List[Preference]
    timestamps: List[int]


class PreferenceDataset(BaseDataset):
    def __init__(self, train_frac: float = 0.8, queue_size: int = 1000):
        super().__init__(train_frac=train_frac, queue_size=queue_size)

    def push(self, fragment_pairs: List[FragmentPair], preferences: List[Preference], timestamp: int) -> None:
        assert len(fragment_pairs) == len(preferences), "pairs and preferences must have the same length"
        items = [(frag, pref, timestamp) for frag, pref in zip(fragment_pairs, preferences)]
        super().push(*items)
        
    def get(self, batch_size):
        data = list(self.train_data)
        indices = np.random.permutation(len(data))
        start_idx = 0
        while start_idx < len(data):
            batch_indices = indices[start_idx:start_idx + batch_size]
            batch = [data[i] for i in batch_indices]
            fragment_pairs, preferences, timestamps = zip(*batch)
            yield PreferenceBatch(list(fragment_pairs), list(preferences), list(timestamps))
            start_idx += batch_size