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


@dataclass
class PreferenceDataset:
    pairs: List[SegmentPair]
    targets: List[Preference]