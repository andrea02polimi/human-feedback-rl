"""
Feedback Module

Defines the Feedback types that an Expert can provide.
Feedback is the output of an Expert's evaluation.
"""

from typing import Generic, List, TypeVar

from src.interfaces.Feedback import Feedback

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


class PreferenceFeedback(Feedback[T], Generic[T]):
    """
    Base class for all preference feedback types.

    Preference feedback is relative: it always involves comparing two or more
    options and expressing some form of preference over them.

    Two concrete subclasses are provided:
    - HardPreferenceFeedback: a single winning index (deterministic)
    - SoftPreferenceFeedback: a probability distribution over options

    Using a base class allows learning algorithms to accept any preference
    representation via isinstance(feedback, PreferenceFeedback), while
    dispatching on the concrete type to apply the correct loss function
    (e.g. cross-entropy for hard labels, Bradley-Terry for soft labels).
    """
    pass


class HardPreferenceFeedback(PreferenceFeedback[int]):
    """
    A deterministic preference: the index of the single most preferred option.

    This is the simplest preference representation. It is appropriate when
    the expert's preference is unambiguous and no probability model is needed.

    Example use cases:
    - Oracle with direct access to the reward function (argmax).
    - Any setting where ties do not occur or are resolved arbitrarily.
    """

    def __init__(self, preferred_index: int):
        preferred_index = int(preferred_index)
        if preferred_index < 0:
            raise ValueError(f"Preferred index must be non-negative, got {preferred_index}")
        super().__init__(preferred_index)


class SoftPreferenceFeedback(PreferenceFeedback[List[float]]):
    """
    A probabilistic preference: a probability distribution over n options.

    This representation is strictly more general than HardPreferenceFeedback
    and supports the following preference models:

    Bradley-Terry (pairwise, n=2):
        value = [p, 1-p]  where p = P(option_0 is preferred over option_1).
        Tie: value = [0.5, 0.5].
        Strict preference for option 0: value = [1.0, 0.0].

    Plackett-Luce (full ranking, n >= 2):
        value = [p_0, p_1, ..., p_{n-1}], a full distribution over all options.
        Typically derived from a softmax over scores: p_i ∝ exp(score_i).

    Any other differentiable preference loss can consume this representation
    directly, since all that is required is a probability vector.
    """

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