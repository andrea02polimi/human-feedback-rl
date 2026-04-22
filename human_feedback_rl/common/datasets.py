import random
from collections import deque
from typing import Any, Deque, List, Tuple
from .types import FragmentPair, Preference


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



class PreferenceDataset(BaseDataset):
    def __init__(self, train_frac: float = 0.8, queue_size: int = 1000):
        super().__init__(train_frac=train_frac, queue_size=queue_size)

    def push(self, pairs: List[FragmentPair], preferences: List[Preference]) -> None:
        assert len(pairs) == len(preferences), "pairs and preferences must have the same length"
        items = list(zip(pairs, preferences))
        super().push(*items)

    def get_train(self) -> List[Tuple[FragmentPair, Preference]]:
        return super().get_train()

    def get_val(self) -> List[Tuple[FragmentPair, Preference]]:
        return super().get_val()

    def get_all(self) -> List[Tuple[FragmentPair, Preference]]:
        return super().get_all()