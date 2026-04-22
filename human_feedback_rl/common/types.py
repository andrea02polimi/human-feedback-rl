from dataclasses import dataclass
from typing import Any, List, Tuple, Sequence


@dataclass
class Transition:
    obs: Any        # o_t
    action: Any     # a_t
    reward: float   # r_t (true reward)


@dataclass
class Trajectory:
    transitions: List[Transition]

    def total_reward(self) -> float:
        return sum(t.reward for t in self.transitions)
    
    def length(self) -> int:
        return len(self.transitions)

    def add_transition(self, transition: Transition) -> None:
        self.transitions.append(transition)


Fragment = Trajectory


@dataclass
class FragmentPair:
    frag1: Fragment
    frag2: Fragment


@dataclass
class Preference:
    pref1: float
    pref2: float

    def __post_init__(self):
        # check bounds
        if not (0.0 <= self.pref1 <= 1.0):
            raise ValueError(f"pref1 must be in [0, 1], got {self.pref1}")

        if not (0.0 <= self.pref2 <= 1.0):
            raise ValueError(f"pref2 must be in [0, 1], got {self.pref2}")

        # check sum to 1
        if not abs((self.pref1 + self.pref2) - 1.0) < 1e-6:
            raise ValueError(
                f"pref1 + pref2 must sum to 1, got {self.pref1 + self.pref2}"
            )
