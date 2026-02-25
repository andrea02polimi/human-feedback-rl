"""
Core Data Types

Defines the fundamental data structures for the feedback system:
- Step: A state-action pair
- Trajectory: A sequence of steps
- History: Collection of trajectories observed by an Expert
"""

from typing import Sequence, List, Any, Iterator
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Step:
    """
    A single step in an environment: a state-action pair.

    Immutable to ensure consistency in history tracking.
    """
    state: Any
    action: Any

    def __repr__(self) -> str:
        return f"Step(state={self.state!r}, action={self.action!r})"


@dataclass
class Trajectory:
    """
    A sequence of steps representing an episode or partial episode.
    """
    steps: List[Step] = field(default_factory=list)

    def __init__(self, steps: Sequence[Step] = ()):
        self.steps = list(steps)

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self) -> Iterator[Step]:
        return iter(self.steps)

    def __getitem__(self, idx: int) -> Step:
        return self.steps[idx]

    def append(self, step: Step) -> None:
        self.steps.append(step)

    def __repr__(self) -> str:
        return f"Trajectory({len(self.steps)} steps)"


class History:
    """
    Maintains the history of trajectories observed by an Expert.

    Each Expert has its own History to track what it has evaluated.
    """

    def __init__(self):
        self._trajectories: List[Trajectory] = []

    def add(self, trajectory: Trajectory) -> None:
        self._trajectories.append(trajectory)

    @property
    def trajectories(self) -> List[Trajectory]:
        return list(self._trajectories)

    def __len__(self) -> int:
        return len(self._trajectories)

    def __iter__(self) -> Iterator[Trajectory]:
        return iter(self._trajectories)

    def clear(self) -> None:
        self._trajectories.clear()

    def __repr__(self) -> str:
        return f"History({len(self._trajectories)} trajectories)"