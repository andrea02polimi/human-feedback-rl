"""
Feedback Module

Defines the Feedback types that an Expert can provide.
Feedback is the output of an Expert's evaluation.
"""

from typing import Generic, List, TypeVar

from human_feedback_rl.interfaces.feedback import Feedback

T = TypeVar("T")

class CorrectionFeedback(Feedback[T], Generic[T]):
    """
    Feedback that provides a corrected action for a given state.
    The value is the corrected action.
    """
    pass


class DemonstrationFeedback(Feedback[T], Generic[T]):
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


class PreferenceFeedback(Feedback[List[float]]):

    def __init__(self, probabilities: List[float]):
        probabilities = [float(p) for p in probabilities]
        if len(probabilities) < 2:
            raise ValueError(
                f"SoftPreferenceFeedback requires at least 2 probabilities, "
                f"got {len(probabilities)}."
            )
        if not all(0.0 <= p <= 1.0 for p in probabilities):
            raise ValueError("All probabilities must be in [0.0, 1.0].")
        total = sum(probabilities)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Probabilities must sum to 1.0, got {total:.8f}. "
                f"Normalize your scores before constructing SoftPreferenceFeedback."
            )
        super().__init__(probabilities)

    @property
    def preferred_index(self) -> int:
        """Returns the index of the option with the highest probability (argmax)."""
        return int(max(range(len(self._value)), key=lambda i: self._value[i]))

    @property
    def n_options(self) -> int:
        """Number of options in the comparison."""
        return len(self._value)