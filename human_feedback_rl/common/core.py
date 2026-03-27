from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Transition:
    obs: any
    action: any
    reward: float


@dataclass
class Trajectory:
    transitions: List[Transition]

    def total_reward(self) -> float:
        return sum(t.reward for t in self.transitions)


@dataclass
class Segment:
    transitions: List[Transition]

    def total_reward(self) -> float:
        return sum(t.reward for t in self.transitions)


@dataclass
class SegmentPair:
    seg1: Segment
    seg2: Segment


@dataclass
class Preference:
    label: Tuple[int, int]  # (1,0) or (0,1)


class PreferenceDataset:
    def __init__(self, n_max: int):
        self.n_max = n_max
        self.pairs: List[SegmentPair] = []
        self.targets: List[Preference] = []

    def push(self, pairs: List[SegmentPair], preferences: List[Preference]) -> None:
        self.pairs.extend(pairs)
        self.targets.extend(preferences)
        if len(self.pairs) > self.n_max:
            self.pairs = self.pairs[-self.n_max:]
            self.targets = self.targets[-self.n_max:]

    def __len__(self) -> int:
        return len(self.pairs)

    def __iter__(self):
        return zip(self.pairs, self.targets)