"""
Abstract base classes for all available Experts.

Each concrete Expert must implement the corresponding abstract Expert class defined here.
"""

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Union, Tuple, Optional, List, TypeVar, Generic
from dataclasses import dataclass

from src.Core import Step, Trajectory, History
from src.interfaces.Feedback import Feedback
from src.Feedback import (
    CorrectionFeedback,
    DemonstrationFeedback,
    RewardFeedback,
    PreferenceFeedback,
    HardPreferenceFeedback,
    SoftPreferenceFeedback,
)

T = TypeVar('T')

# ==============================================================
# ENUMERATIONS
# ==============================================================

class FeedbackScope(Enum):
    """What level of abstraction the Expert operates on."""
    STEP = auto()  # Evaluates individual state-action pairs
    TRAJECTORY = auto()  # Evaluates sequences of steps


class FeedbackMode(Enum):
    """How the Expert evaluates: single object or comparison."""
    ABSOLUTE = auto()  # Evaluates a single step/trajectory
    RELATIVE = auto()  # Compares two or more steps/trajectories


# ==============================================================
# TYPE ALIASES
# ==============================================================

ObjectCount = Union[int, Tuple[int, Optional[int]]]


# ==============================================================
# CONFIGURATION
# ==============================================================

@dataclass(frozen=True)
class ExpertConfig:
    """
    Configuration for an Expert, encapsulating scope and mode.
    """
    scope: FeedbackScope
    mode: FeedbackMode

    @property
    def required_object_count(self) -> ObjectCount:
        """
        Determines how many objects are required based on mode.
        - ABSOLUTE: exactly 1 object
        - RELATIVE: at least 2 objects (no upper bound)
        """
        if self.mode == FeedbackMode.ABSOLUTE:
            return 1
        else:
            return 2, None


# Pre-defined configurations for convenience
STEP_ABSOLUTE = ExpertConfig(FeedbackScope.STEP, FeedbackMode.ABSOLUTE)
STEP_RELATIVE = ExpertConfig(FeedbackScope.STEP, FeedbackMode.RELATIVE)
TRAJECTORY_ABSOLUTE = ExpertConfig(FeedbackScope.TRAJECTORY, FeedbackMode.ABSOLUTE)
TRAJECTORY_RELATIVE = ExpertConfig(FeedbackScope.TRAJECTORY, FeedbackMode.RELATIVE)


# ==============================================================
# VALIDATION
# ==============================================================

class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


def validate_objects(objects: Any, config: ExpertConfig) -> List[Any]:
    """
    Validates input objects based on the Expert's configuration.

    Args:
        objects: Single object or sequence of objects to validate
        config: Expert configuration specifying requirements

    Returns:
        list of objects

    Raises:
        ValidationError: If object count doesn't match requirements
    """
    if not isinstance(objects, (list, tuple)):
        objects = [objects] # list with one single element
    else:
        objects = list(objects) # here to convert a tuple to a list

    n = len(objects)
    req = config.required_object_count

    if isinstance(req, int):
        if n != req:
            raise ValidationError(f"Expected exactly {req} object(s), got {n}")
    else:
        min_n, max_n = req
        if n < min_n:
            raise ValidationError(
                f"Expected at least {min_n} object(s), got {n}. "
                f"Relative feedback requires comparison between multiple objects."
            )
        if max_n is not None and n > max_n:
            raise ValidationError(f"Expected at most {max_n} object(s), got {n}")

    # Validate object types match scope
    expected_type = Step if config.scope == FeedbackScope.STEP else Trajectory
    for i, obj in enumerate(objects):
        if not isinstance(obj, expected_type):
            raise ValidationError(
                f"Object at index {i} has type {type(obj).__name__}, "
                f"expected {expected_type.__name__}"
            )

    return objects


# ==============================================================
# BASE EXPERT CLASS
# ==============================================================

class Expert(ABC, Generic[T]):
    """
    Abstract base class for all Experts.

    An Expert is an entity that:
    - Evaluates steps or trajectories providing a Feedback
    - Maintains a history of what it has evaluated

    Template Method Pattern:
    - `query()` is the public interface
    - `_evaluate()` is the protected method subclasses implement
    """

    def __init__(self, config: ExpertConfig):
        """
        Initialize the Expert.

        Args:
            config: Configuration specifying scope and mode
        """
        self._config = config
        self._history = History()

    @property
    def scope(self) -> FeedbackScope:
        """What level the Expert operates on (Step or Trajectory)."""
        return self._config.scope

    @property
    def mode(self) -> FeedbackMode:
        """How the Expert evaluates (Absolute or Relative)."""
        return self._config.mode

    @property
    def history(self) -> History:
        """History of trajectories this Expert has observed."""
        return self._history

    @property
    def required_object_count(self) -> ObjectCount:
        """How many objects this Expert requires for evaluation."""
        return self._config.required_object_count

    def query(self, objects: Any) -> Feedback:
        """
        Query the Expert for feedback on the given object(s).

        This is the public interface. It validates input, delegates to
        the subclass implementation, and updates history.

        Args:
            objects: Step(s) or Trajectory(ies) to evaluate

        Returns:
            Feedback from the Expert

        Raises:
            ValidationError: If objects don't meet requirements
        """
        # Validate and normalize
        validated = validate_objects(objects, self._config)

        # Delegate to subclass
        result = self._evaluate(validated)

        # Record in history (convert steps to single-step trajectories if needed) only successful evaluations
        self._record_history(validated)

        return result

    def _record_history(self, objects: List[T]) -> None:
        """Record evaluated objects in history."""
        if self._config.scope == FeedbackScope.STEP:
            # Wrap steps in trajectories for history
            for step in objects:
                self._history.add(Trajectory([step]))
        else:
            for traj in objects:
                self._history.add(traj)

    @abstractmethod
    def _evaluate(self, objects: List[T]) -> Feedback:
        """
        Perform the actual evaluation. Implemented by subclasses.

        Args:
            objects: Validated list of objects to evaluate

        Returns:
            Appropriate Feedback type
        """
        pass

# ----------------------
# CORRECTION EXPERTS
# ----------------------

class StepCorrectionExpert(ABC, Expert[Step]):
    """
    Expert that provides corrected actions for steps.

    Given a step (state, action), suggests what the action should have been.
    """

    def __init__(self):
        super().__init__(STEP_ABSOLUTE)

    @abstractmethod
    def _correction_fn(self, step: Step) -> Any:
        """
            Compute the corrected action for a given step.

            Implemented by subclasses to define the expert policy.

            Args:
                step: The step to correct (state, action).

            Returns:
                The action that should have been taken in the given state.
        """
        pass

    def _evaluate(self, objects: List[Step]) -> CorrectionFeedback:
        """
            Evaluate a step and return the corrected action.

            This method extracts the step, applies the correction function,
            and wraps the result in a CorrectionFeedback object.

            Args:
                objects: List containing the validated step to evaluate.

            Returns:
                CorrectionFeedback containing the corrected action.
        """
        step = objects[0]
        corrected_action = self._correction_fn(step)
        return CorrectionFeedback(corrected_action)


class TrajectoryCorrectionExpert(ABC, Expert[Trajectory]):
    """
    Expert that provides corrected trajectories.

    Given a trajectory, suggests what the trajectory should have been.
    """

    def __init__(self):
        super().__init__(TRAJECTORY_ABSOLUTE)

    @abstractmethod
    def _correction_fn(self, traj: Trajectory) -> Any:
        """
            Compute the corrected trajectory.

            Implemented by subclasses to define how a trajectory should be
            corrected according to the Expert's knowledge of the environment.

            Args:
                traj: The trajectory to correct.

            Returns:
                The trajectory that should have been produced instead.
        """
        pass

    def _evaluate(self, objects: List[Trajectory]) -> CorrectionFeedback:
        """
           Evaluate a trajectory and return its corrected version.

           This method extracts the trajectory, applies the correction
           function, and wraps the result in a CorrectionFeedback object.

           Args:
               objects: List containing the validated trajectory to evaluate.

           Returns:
               CorrectionFeedback containing the corrected trajectory.
        """
        traj = objects[0]
        corrected_traj = self._correction_fn(traj)
        return CorrectionFeedback(corrected_traj)


# ----------------------
# DEMONSTRATION EXPERTS
# ----------------------

class StepDemonstrationExpert(ABC, Expert[Step]):
    """
    Expert that provides demonstration steps.

    Given a step, shows the ideal step for that situation.
    """

    def __init__(self):
        super().__init__(STEP_ABSOLUTE)

    @abstractmethod
    def _demo_fn(self, step: Step) -> Any:
        """
            Generate a demonstration trajectory.

            Implemented by subclasses to provide an example of correct behavior
            in the environment.

            Args:
                step: The step to generate a demonstration for.

            Returns:
                A trajectory demonstrating the desired behavior.
        """
        pass

    def _evaluate(self, objects: List[Step]) -> DemonstrationFeedback:
        """
            Produce a demonstration trajectory.

            This method calls the demonstration function and wraps the result
            in a DemonstrationFeedback object.

            Args:
                objects: Validated input objects (typically unused).

            Returns:
                DemonstrationFeedback containing the generated trajectory.
        """
        step = objects[0]
        demo_step = self._demo_fn(step)
        return DemonstrationFeedback(demo_step)


class TrajectoryDemonstrationExpert(ABC, Expert[Trajectory]):
    """
    Expert that provides demonstration trajectories.

    Given a trajectory, shows the ideal trajectory.
    """

    def __init__(self):
        super().__init__(TRAJECTORY_ABSOLUTE)

    @abstractmethod
    def _demo_fn(self, step: Trajectory) -> Any:
        pass

    def _evaluate(self, objects: List[Trajectory]) -> DemonstrationFeedback:
        traj = objects[0]
        demo_traj = self._demo_fn(traj)
        return DemonstrationFeedback(demo_traj)


# ----------------------
# REWARD EXPERTS
# ----------------------

class StepRewardExpert(ABC, Expert[Step]):
    """
    Expert that provides scalar rewards for steps.
    """

    def __init__(self):
        super().__init__(STEP_ABSOLUTE)

    @abstractmethod
    def _reward_fn(self, step: Step) -> Any:
        pass

    def _evaluate(self, objects: List[Step]) -> RewardFeedback:
        step = objects[0]
        reward = self._reward_fn(step)
        return RewardFeedback(reward)


class TrajectoryRewardExpert(ABC, Expert[Trajectory]):
    """
    Expert that provides scalar rewards for trajectories.
    """

    def __init__(self):
        super().__init__(TRAJECTORY_ABSOLUTE)

    @abstractmethod
    def _reward_fn(self, step: Trajectory) -> Any:
        """
            Compute the reward for a given object.

            Implemented by subclasses to assign a scalar reward based on the
            Expert's knowledge of the environment.

            Args:
                step: Step or trajectory to evaluate.

            Returns:
                A scalar reward value.
        """
        pass

    def _evaluate(self, objects: List[Trajectory]) -> RewardFeedback:
        """
            Evaluate an object and return its reward.

            This method extracts the object, applies the reward function,
            and wraps the result in a RewardFeedback object.

            Args:
                objects: List containing the validated object to evaluate.

            Returns:
                RewardFeedback containing the computed reward.
        """
        traj = objects[0]
        reward = self._reward_fn(traj)
        return RewardFeedback(reward)


# ----------------------
# PREFERENCE EXPERTS
# ----------------------

def _wrap_preference(result: Any, n_options: int) -> PreferenceFeedback:
    """
    Converts the raw output of a preference_fn into a PreferenceFeedback object.

    Accepted return types from preference_fn:
    - int               -> HardPreferenceFeedback (index of preferred option)
    - List[float]       -> SoftPreferenceFeedback (probability distribution)
    - PreferenceFeedback -> returned as-is (caller manages the type)

    This helper centralises the dispatch logic so both StepPreferenceExpert
    and TrajectoryPreferenceExpert share the same behaviour.
    """
    if isinstance(result, PreferenceFeedback):
        return result
    elif isinstance(result, list):
        return SoftPreferenceFeedback(result)
    else:
        # Handles int and numpy integer types
        idx = int(result)
        if not (0 <= idx < n_options):
            raise ValueError(
                f"Preference index {idx} out of range for {n_options} options."
            )
        return HardPreferenceFeedback(idx)

class StepPreferenceExpert(ABC, Expert[Step]):
    """
    Expert that provides preferences over multiple steps.

    Requires at least 2 steps to compare.

    The preference_fn can return:
    - int               -> wrapped in HardPreferenceFeedback (deterministic preference)
    - List[float]       -> wrapped in SoftPreferenceFeedback (probability distribution,
                           supports Bradley-Terry, Plackett-Luce, etc.)
    - PreferenceFeedback -> returned directly (full control by the caller)
    """

    def __init__(self):
        super().__init__(STEP_RELATIVE)

    @abstractmethod
    def _preference_fn(self, steps: List[Step]) -> Any:
        """
            Determine the preferred step among multiple candidates.

            Implemented by subclasses to compare steps and select the one
            considered better according to the Expert.

            Args:
                steps: Steps to compare.

            Returns:
                Index of the preferred step in the input list.
        """
        pass

    def _evaluate(self, objects: List[Step]) -> PreferenceFeedback:
        """
            Evaluate multiple steps and return the preferred one.

            This method delegates the comparison to the preference function
            and wraps the result in a PreferenceFeedback object.

            Args:
                objects: List of validated steps to compare.

            Returns:
                PreferenceFeedback containing the index of the preferred object.
        """
        result = self._preference_fn(objects)
        return _wrap_preference(result, n_options=len(objects))



class TrajectoryPreferenceExpert(ABC, Expert[Trajectory]):
    """
    Expert that provides preferences over multiple trajectories.

    Requires at least 2 trajectories to compare.

    The preference_fn can return:
    - int               -> wrapped in HardPreferenceFeedback
    - List[float]       -> wrapped in SoftPreferenceFeedback
    - PreferenceFeedback -> returned directly
    """

    def __init__(self):
        super().__init__(TRAJECTORY_RELATIVE)

    @abstractmethod
    def _preference_fn(self, trajectories: List[Trajectory]) -> Any:
        """
            Determine the preferred trajectory among multiple candidates.

            Implemented by subclasses to compare trajectories and select the one
            considered better according to the Expert.

            Args:
                trajectories: Trajectories to compare.

            Returns:
                Index of the preferred trajectory in the input list.
        """
        pass

    def _evaluate(self, objects: List[Trajectory]) -> PreferenceFeedback:
        """
            Evaluate multiple steps and return the preferred one.

            This method delegates the comparison to the preference function
            and wraps the result in a PreferenceFeedback object.

            Args:
                objects: List of validated steps to compare.

            Returns:
                PreferenceFeedback containing the index of the preferred object.
        """
        result = self._preference_fn(objects)
        return _wrap_preference(result, n_options=len(objects))