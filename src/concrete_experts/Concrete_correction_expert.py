from typing import Any

from src import StepCorrectionExpert, Step, History, TrajectoryCorrectionExpert, Trajectory


class ConcreteStepCorrectionExpert(StepCorrectionExpert):
    def __init__(self, env: Any, policy: Any):
        self._policy = policy
        super().__init__(env)

    def _correction_fn(self, step: Step, env: Any, history: History) -> Any:
        pass # implementation...


class ConcreteTrajectoryCorrectionExpert(TrajectoryCorrectionExpert):
    def __init__(self, env: Any, policy: Any):
        self._policy = policy
        super().__init__(env)

    def _correction_fn(self, step: Trajectory, env: Any, history: History) -> Any:
        pass  # implementation...