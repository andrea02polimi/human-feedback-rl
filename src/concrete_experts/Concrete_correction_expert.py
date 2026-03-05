from typing import Any

from src.interfaces.Expert import StepCorrectionExpert,  TrajectoryCorrectionExpert
from src.Core import Step, Trajectory


class ConcreteStepCorrectionExpert(StepCorrectionExpert):
    def __init__(self, policy: Any):
        self._policy = policy
        super().__init__()

    def _correction_fn(self, step: Step) -> Any:
        pass # implementation...


class ConcreteTrajectoryCorrectionExpert(TrajectoryCorrectionExpert):
    def __init__(self, policy: Any):
        self._policy = policy
        super().__init__()

    def _correction_fn(self, trajectory: Trajectory) -> Any:
        pass  # implementation...