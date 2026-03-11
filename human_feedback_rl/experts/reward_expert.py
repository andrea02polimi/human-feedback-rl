from typing import Any

from human_feedback_rl.interfaces.expert import StepRewardExpert,  TrajectoryRewardExpert
from human_feedback_rl.core import Step, Trajectory


class ConcreteStepRewardExpert(StepRewardExpert):
    def __init__(self, env: Any, policy: Any):
        self._policy = policy
        self.env = env
        super().__init__()

    def _reward_fn(self, step: Step) -> Any:
        pass  # implementation...


class ConcreteTrajectoryRewardExpert(TrajectoryRewardExpert):
    def __init__(self, env: Any, policy: Any):
        self._policy = policy
        self.env = env
        super().__init__()

    def _reward_fn(self, trajectory: Trajectory) -> Any:
        pass  # implementation...