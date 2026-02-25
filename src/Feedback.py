"""
Feedback Module

Defines the Feedback types that an Expert can provide.
Feedback is the output of an Expert's evaluation.
"""

from abc import ABC
from typing import Any, TypeVar, Generic

T = TypeVar('T')


class Feedback(ABC, Generic[T]):
    """Base class for all feedback types."""

    def __init__(self, value: T):
        self._value = value

    @property
    def value(self) -> T:
        return self._value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._value!r})"


class CorrectionFeedback(Feedback[Any]):
    """
    Feedback that provides a corrected action for a given state.
    The value is the corrected action.
    """
    pass


class DemonstrationFeedback(Feedback[Any]):
    """
    Feedback that provides a demonstration (step or trajectory).
    The value is the demonstrated step or trajectory.
    """
    pass


class RewardFeedback(Feedback[float]):
    """
    Feedback that provides a scalar reward signal.
    The value is the reward (float).
    """

    def __init__(self, value: float):
        if not isinstance(value, (int, float)):
            raise TypeError(f"Reward must be numeric, got {type(value)}")
        super().__init__(float(value))


class PreferenceFeedback(Feedback[int]):
    """
    Feedback that provides preference over multiple options.
    The value is the index of the preferred option.
    """

    def __init__(self, preferred_index: int):
        if not isinstance(preferred_index, int) or preferred_index < 0:
            raise ValueError(f"Preferred index must be a non-negative integer, got {preferred_index}")
        super().__init__(preferred_index)