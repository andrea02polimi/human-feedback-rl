from typing import Any, List

from human_feedback_rl.interfaces.Expert import StepPreferenceExpert,  TrajectoryPreferenceExpert
from human_feedback_rl.Core import Step, Trajectory


class ConcreteStepCorrectionExpert(StepPreferenceExpert):
    def __init__(self, env: Any, policy: Any):
        self._policy = policy
        super().__init__(env)

    def _preference_fn(self, step: List[Step]) -> Any:
        pass  # implementation...


class ConcreteTrajectoryCorrectionExpert(TrajectoryPreferenceExpert):
    def __init__(self, env: Any, policy: Any):
        self._policy = policy
        super().__init__(env)

    def _preference_fn(self, trajectories: List[Trajectory]) -> Any:
        pass  # implementation...