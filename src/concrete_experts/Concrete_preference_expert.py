from typing import Any

from src import Step, History, Trajectory, StepPreferenceExpert, TrajectoryPreferenceExpert


class ConcreteStepCorrectionExpert(StepPreferenceExpert):
    def __init__(self, env: Any, policy: Any):
        self._policy = policy
        super().__init__(env)

    def _preference_fn(self, step: Step, env: Any, history: History) -> Any:
        pass  # implementation...


class ConcreteTrajectoryCorrectionExpert(TrajectoryPreferenceExpert):
    def __init__(self, env: Any, policy: Any):
        self._policy = policy
        super().__init__(env)

    def _preference_fn(self, step: Trajectory, env: Any, history: History) -> Any:
        pass  # implementation...