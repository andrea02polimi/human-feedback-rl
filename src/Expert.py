"""
Expert (FeedbackModel) Module

Design Principles:
-----------------
1. Expert is a black-box entity that provides Feedback
2. Expert knows the environment
3. Each Expert type corresponds to a Feedback type
4. Two orthogonal dimensions:
   - Scope: Step (state-action pair) vs Trajectory (sequence of steps)
   - Mode: Absolute (single object) vs Relative (comparison of 2+ objects)
5. Each Expert maintains its own history of evaluations
6. Preference-based Experts require at least 2 objects to compare
7. Passive agent model: Expert is queried, doesn't proactively provide feedback

Architecture:
------------
We use composition over multiple inheritance to avoid diamond problems.
The Expert base class uses template method pattern for validation.
"""

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Sequence, Any, Callable, Union, Tuple, Optional, List, TypeVar, Generic
from dataclasses import dataclass

from src.Core import Step, Trajectory, History
from src.Feedback import (
    Feedback,
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
# CONFIGURATION (Composition over Inheritance)
# ==============================================================

@dataclass(frozen=True)
class ExpertConfig:
    """
    Configuration for an Expert, encapsulating scope and mode.

    Using composition instead of multiple inheritance avoids
    diamond inheritance issues and makes the design more flexible.
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
    Validates and normalizes input objects based on the Expert's configuration.

    Args:
        objects: Single object or sequence of objects to validate
        config: Expert configuration specifying requirements

    Returns:
        Normalized list of objects

    Raises:
        ValidationError: If object count doesn't match requirements
    """
    # Normalize to list
    if not isinstance(objects, (list, tuple)):
        objects = [objects]
    else:
        objects = list(objects)

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
    Abstract base class for all Experts (FeedbackModels).

    An Expert is a black-box entity that:
    - Knows the environment
    - Evaluates steps or trajectories
    - Provides feedback based on its internal criteria
    - Maintains a history of what it has evaluated

    Template Method Pattern:
    - `query()` is the public interface that handles validation
    - `_evaluate()` is the protected method subclasses implement
    """

    def __init__(self, env: Any, config: ExpertConfig):
        """
        Initialize the Expert.

        Args:
            env: The environment the Expert has knowledge of
            config: Configuration specifying scope and mode
        """
        self._env = env
        self._config = config
        self._history = History()

    @property
    def env(self) -> Any:
        """The environment this Expert knows."""
        return self._env

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


# ==============================================================
# CONCRETE EXPERT IMPLEMENTATIONS
# ==============================================================

# ----------------------
# CORRECTION EXPERTS
# ----------------------

class StepCorrectionExpert(Expert[Step]):
    """
    Expert that provides corrected actions for steps.

    Given a step (state, action), suggests what the action should have been.
    """

    def __init__(self, env: Any, correction_fn: Callable[[Step, Any, History], Any]):
        """
        Args:
            env: The environment
            correction_fn: Function(step, env, history) -> corrected_action
        """
        super().__init__(env, STEP_ABSOLUTE)
        self._correction_fn = correction_fn

    def _evaluate(self, objects: List[Step]) -> CorrectionFeedback:
        step = objects[0]
        corrected_action = self._correction_fn(step, self._env, self._history)
        return CorrectionFeedback(corrected_action)


class TrajectoryCorrectionExpert(Expert[Trajectory]):
    """
    Expert that provides corrected trajectories.

    Given a trajectory, suggests what the trajectory should have been.
    """

    def __init__(self, env: Any, correction_fn: Callable[[Trajectory, Any, History], Trajectory]):
        """
        Args:
            env: The environment
            correction_fn: Function(trajectory, env, history) -> corrected_trajectory
        """
        super().__init__(env, TRAJECTORY_ABSOLUTE)
        self._correction_fn = correction_fn

    def _evaluate(self, objects: List[Trajectory]) -> CorrectionFeedback:
        traj = objects[0]
        corrected_traj = self._correction_fn(traj, self._env, self._history)
        return CorrectionFeedback(corrected_traj)


# ----------------------
# DEMONSTRATION EXPERTS
# ----------------------

class StepDemonstrationExpert(Expert[Step]):
    """
    Expert that provides demonstration steps.

    Given a step, shows the ideal step for that situation.
    """

    def __init__(self, env: Any, demo_fn: Callable[[Step, Any, History], Step]):
        """
        Args:
            env: The environment
            demo_fn: Function(step, env, history) -> demonstrated_step
        """
        super().__init__(env, STEP_ABSOLUTE)
        self._demo_fn = demo_fn

    def _evaluate(self, objects: List[Step]) -> DemonstrationFeedback:
        step = objects[0]
        demo_step = self._demo_fn(step, self._env, self._history)
        return DemonstrationFeedback(demo_step)


class TrajectoryDemonstrationExpert(Expert[Trajectory]):
    """
    Expert that provides demonstration trajectories.

    Given a trajectory, shows the ideal trajectory.
    """

    def __init__(self, env: Any, demo_fn: Callable[[Trajectory, Any, History], Trajectory]):
        """
        Args:
            env: The environment
            demo_fn: Function(trajectory, env, history) -> demonstrated_trajectory
        """
        super().__init__(env, TRAJECTORY_ABSOLUTE)
        self._demo_fn = demo_fn

    def _evaluate(self, objects: List[Trajectory]) -> DemonstrationFeedback:
        traj = objects[0]
        demo_traj = self._demo_fn(traj, self._env, self._history)
        return DemonstrationFeedback(demo_traj)


# ----------------------
# REWARD EXPERTS
# ----------------------

class StepRewardExpert(Expert[Step]):
    """
    Expert that provides scalar rewards for steps.
    """

    def __init__(self, env: Any, reward_fn: Callable[[Step, Any, History], float]):
        """
        Args:
            env: The environment
            reward_fn: Function(step, env, history) -> reward (float)
        """
        super().__init__(env, STEP_ABSOLUTE)
        self._reward_fn = reward_fn

    def _evaluate(self, objects: List[Step]) -> RewardFeedback:
        step = objects[0]
        reward = self._reward_fn(step, self._env, self._history)
        return RewardFeedback(reward)


class TrajectoryRewardExpert(Expert[Trajectory]):
    """
    Expert that provides scalar rewards for trajectories.
    """

    def __init__(self, env: Any, reward_fn: Callable[[Trajectory, Any, History], float]):
        """
        Args:
            env: The environment
            reward_fn: Function(trajectory, env, history) -> reward (float)
        """
        super().__init__(env, TRAJECTORY_ABSOLUTE)
        self._reward_fn = reward_fn

    def _evaluate(self, objects: List[Trajectory]) -> RewardFeedback:
        traj = objects[0]
        reward = self._reward_fn(traj, self._env, self._history)
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

class StepPreferenceExpert(Expert[Step]):
    """
    Expert that provides preferences over multiple steps.

    Requires at least 2 steps to compare.

    The preference_fn can return:
    - int               -> wrapped in HardPreferenceFeedback (deterministic preference)
    - List[float]       -> wrapped in SoftPreferenceFeedback (probability distribution,
                           supports Bradley-Terry, Plackett-Luce, etc.)
    - PreferenceFeedback -> returned directly (full control by the caller)
    """

    def __init__(self, env: Any, preference_fn: Callable[[Sequence[Step], Any, History], Any]):
        """
        Args:
            env: The environment
            preference_fn: Function(steps, env, history) -> int | List[float] | PreferenceFeedback
        """
        super().__init__(env, STEP_RELATIVE)
        self._preference_fn = preference_fn

    def _evaluate(self, objects: List[Step]) -> PreferenceFeedback:
        result = self._preference_fn(objects, self._env, self._history)
        return _wrap_preference(result, n_options=len(objects))



class TrajectoryPreferenceExpert(Expert[Trajectory]):
    """
    Expert that provides preferences over multiple trajectories.

    Requires at least 2 trajectories to compare.

    The preference_fn can return:
    - int               -> wrapped in HardPreferenceFeedback
    - List[float]       -> wrapped in SoftPreferenceFeedback
    - PreferenceFeedback -> returned directly
    """

    def __init__(self, env: Any, preference_fn: Callable[[Sequence[Trajectory], Any, History], Any]):
        """
        Args:
            env: The environment
            preference_fn: Function(trajectories, env, history) -> int | List[float] | PreferenceFeedback
        """
        super().__init__(env, TRAJECTORY_RELATIVE)
        self._preference_fn = preference_fn

    def _evaluate(self, objects: List[Trajectory]) -> PreferenceFeedback:
        result = self._preference_fn(objects, self._env, self._history)
        return _wrap_preference(result, n_options=len(objects))


# ==============================================================
# FACTORY (Optional convenience)
# ==============================================================

class ExpertFactory:
    """
    Factory for creating Experts.

    Provides a cleaner API for creating experts without knowing
    the exact class names.
    """

    @staticmethod
    def create_correction_expert(
            env: Any,
            scope: FeedbackScope,
            correction_fn: Callable
    ) -> Expert:
        if scope == FeedbackScope.STEP:
            return StepCorrectionExpert(env, correction_fn)
        else:
            return TrajectoryCorrectionExpert(env, correction_fn)

    @staticmethod
    def create_demonstration_expert(
            env: Any,
            scope: FeedbackScope,
            demo_fn: Callable
    ) -> Expert:
        if scope == FeedbackScope.STEP:
            return StepDemonstrationExpert(env, demo_fn)
        else:
            return TrajectoryDemonstrationExpert(env, demo_fn)

    @staticmethod
    def create_reward_expert(
            env: Any,
            scope: FeedbackScope,
            reward_fn: Callable
    ) -> Expert:
        if scope == FeedbackScope.STEP:
            return StepRewardExpert(env, reward_fn)
        else:
            return TrajectoryRewardExpert(env, reward_fn)

    @staticmethod
    def create_preference_expert(
            env: Any,
            scope: FeedbackScope,
            preference_fn: Callable
    ) -> Expert:
        if scope == FeedbackScope.STEP:
            return StepPreferenceExpert(env, preference_fn)
        else:
            return TrajectoryPreferenceExpert(env, preference_fn)