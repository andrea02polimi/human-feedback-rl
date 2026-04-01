from dataclasses import dataclass
from typing import Any, List, Tuple


@dataclass
class Transition:
    obs: Any
    action: Any
    reward: float


@dataclass
class Trajectory:
    transitions: List[Transition]

    def total_reward(self) -> float:
        return sum(t.reward for t in self.transitions)
    
    def length(self) -> int:
        return len(self.transitions)

    def add_transition(self, transition: Transition) -> None:
        self.transitions.append(transition)

Segment = Trajectory


@dataclass
class SegmentPair:
    seg1: Segment
    seg2: Segment


@dataclass
class Preference:
    label: Tuple[float, float]  # (1,0) or (0,1)


class PreferenceDataset:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.pairs: List[SegmentPair] = []
        self.preferences: List[Preference] = []

    def push(self, pairs: List[SegmentPair], preferences: List[Preference]) -> None:
        self.pairs.extend(pairs)
        self.preferences.extend(preferences)
        if len(self.pairs) > self.capacity:
            self.pairs = self.pairs[-self.capacity:]
            self.preferences = self.preferences[-self.capacity:]

    def __len__(self) -> int:
        return len(self.pairs)

    def __iter__(self):
        return zip(self.pairs, self.preferences)