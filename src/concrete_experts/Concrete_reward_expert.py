from typing import Any

from src import Step, History, Trajectory, StepRewardExpert, TrajectoryRewardExpert


class ConcreteStepCorrectionExpert(StepRewardExpert):
    def __init__(self, env: Any, policy: Any):
        self._policy = policy
        super().__init__(env)

    def _reward_fn(self, step: Step, env: Any, history: History) -> Any:
        pass  # implementation...


class ConcreteTrajectoryCorrectionExpert(TrajectoryRewardExpert):
    def __init__(self, env: Any, policy: Any):
        self._policy = policy
        super().__init__(env)

    def _reward_fn(self, step: Trajectory, env: Any, history: History) -> Any:
        pass  # implementation...