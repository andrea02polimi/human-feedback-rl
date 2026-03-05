from typing import Any

from human_feedback_rl.interfaces.Expert import StepCorrectionExpert,  TrajectoryCorrectionExpert
from human_feedback_rl.Core import Step, Trajectory


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