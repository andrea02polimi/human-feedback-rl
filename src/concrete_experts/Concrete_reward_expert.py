from typing import Any

from src.interfaces.Expert import StepRewardExpert,  TrajectoryRewardExpert
from src.Core import Step, Trajectory


class ConcreteStepCorrectionExpert(StepRewardExpert):
    def __init__(self, env: Any, policy: Any):
        self._policy = policy
        super().__init__(env)

    def _reward_fn(self, step: Step) -> Any:
        pass  # implementation...


class ConcreteTrajectoryCorrectionExpert(TrajectoryRewardExpert):
    def __init__(self, env: Any, policy: Any):
        self._policy = policy
        super().__init__(env)

    def _reward_fn(self, trajectory: Trajectory) -> Any:
        pass  # implementation...