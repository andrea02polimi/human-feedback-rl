from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class Transition:
    obs: np.ndarray
    action: np.ndarray
    true_reward: float
    done: bool


@dataclass
class Segment:
    transitions: List[Transition]

    @property
    def obs(self) -> np.ndarray:
        return np.stack([t.obs for t in self.transitions])

    @property
    def actions(self) -> np.ndarray:
        return np.stack([t.action for t in self.transitions])

    @property
    def true_return(self) -> float:
        return float(sum(t.true_reward for t in self.transitions))

    def __len__(self) -> int:
        return len(self.transitions)


@dataclass
class Trajectory:
    transitions: List[Transition] = field(default_factory=list)

    def add(self, transition: Transition) -> None:
        self.transitions.append(transition)

    def __len__(self) -> int:
        return len(self.transitions)


@dataclass
class SegmentPair:
    seg1: Segment
    seg2: Segment


@dataclass
class Preference:
    seg1: Segment
    seg2: Segment
    label: float  # 1.0: seg1 preferred | 0.0: seg2 preferred | 0.5: equal


class PreferenceDataset:
    def __init__(self, max_size: int = 3000):
        self._data: List[Preference] = []
        self.max_size = max_size

    def add(self, preference: Preference) -> None:
        self._data.append(preference)
        if len(self._data) > self.max_size:
            self._data.pop(0)

    def sample(self, batch_size: int, rng: np.random.Generator) -> List[Preference]:
        indices = rng.integers(0, len(self._data), size=batch_size)
        return [self._data[i] for i in indices]

    def __len__(self) -> int:
        return len(self._data)
