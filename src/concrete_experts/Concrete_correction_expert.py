from typing import Any

from src.interfaces.Expert import StepCorrectionExpert,  TrajectoryCorrectionExpert
from src.Core import Step, Trajectory


class ConcreteStepCorrectionExpert(StepCorrectionExpert):
    def __init__(self, env: Any, policy: Any):
        self._policy = policy
        super().__init__(env)

    def _correction_fn(self, step: Step) -> Any:
        pass # implementation...


class ConcreteTrajectoryCorrectionExpert(TrajectoryCorrectionExpert):
    def __init__(self, env: Any, policy: Any):
        self._policy = policy
        super().__init__(env)

    def _correction_fn(self, trajectory: Trajectory) -> Any:
        pass  # implementation...