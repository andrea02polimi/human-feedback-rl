from typing import Any

from human_feedback_rl.interfaces.Expert import StepRewardExpert,  TrajectoryRewardExpert
from human_feedback_rl.Core import Step, Trajectory


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