"""
Feedback System Package

A framework for modeling expert feedback in reinforcement learning.

Key Concepts:
- Expert: A black-box entity that evaluates agent behavior and provides feedback
- Feedback: The output of an Expert's evaluation (corrections, demonstrations, rewards, preferences)
- Step: A state-action pair
- Trajectory: A sequence of steps

Two Orthogonal Dimensions:
- Scope: Step vs Trajectory
- Mode: Absolute (single object) vs Relative (comparison)

Usage:
------
    from feedback_system import (
        Step, Trajectory, History,
        StepRewardExpert, TrajectoryPreferenceExpert,
        FeedbackScope, FeedbackMode,
    )

    # Create an environment (your RL environment)
    env = MyEnvironment()

    # Define a reward function
    def my_reward_fn(step, env, history):
        return 1.0 if step.action == "correct" else 0.0

    # Create an expert
    expert = StepRewardExpert(env, my_reward_fn)

    # Query the expert
    step = Step(state="s1", action="a1")
    feedback = expert.query(step)
    print(feedback.value)  # The reward
"""

# Core data types
from src.Core import Step, Trajectory, History

# Feedback types
from src.Feedback import (
    Feedback,
    CorrectionFeedback,
    DemonstrationFeedback,
    RewardFeedback,
    PreferenceFeedback,
)

# Enumerations
from src.interfaces.Expert import FeedbackScope, FeedbackMode

# Configuration
from src.interfaces.Expert import (
    ExpertConfig,
    STEP_ABSOLUTE,
    STEP_RELATIVE,
    TRAJECTORY_ABSOLUTE,
    TRAJECTORY_RELATIVE,
)

# Base class
from src.interfaces.Expert import Expert, ValidationError

# Concrete Expert implementations
from src.interfaces.Expert import (
    # Correction
    StepCorrectionExpert,
    TrajectoryCorrectionExpert,
    # Demonstration
    StepDemonstrationExpert,
    TrajectoryDemonstrationExpert,
    # Reward
    StepRewardExpert,
    TrajectoryRewardExpert,
    # Preference
    StepPreferenceExpert,
    TrajectoryPreferenceExpert,
)

# Factory
from src.interfaces.Expert import ExpertFactory

__all__ = [
    # Core
    "Step",
    "Trajectory",
    "History",
    # Feedback
    "Feedback",
    "CorrectionFeedback",
    "DemonstrationFeedback",
    "RewardFeedback",
    "PreferenceFeedback",
    # Enums
    "FeedbackScope",
    "FeedbackMode",
    # Config
    "ExpertConfig",
    "STEP_ABSOLUTE",
    "STEP_RELATIVE",
    "TRAJECTORY_ABSOLUTE",
    "TRAJECTORY_RELATIVE",
    # Base
    "Expert",
    "ValidationError",
    # Experts
    "StepCorrectionExpert",
    "TrajectoryCorrectionExpert",
    "StepDemonstrationExpert",
    "TrajectoryDemonstrationExpert",
    "StepRewardExpert",
    "TrajectoryRewardExpert",
    "StepPreferenceExpert",
    "TrajectoryPreferenceExpert",
    # Factory
    "ExpertFactory",
]