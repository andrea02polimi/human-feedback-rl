from typing import Any

from src import Step, History, Trajectory, StepDemonstrationExpert, TrajectoryDemonstrationExpert


class ConcreteStepDemonstrationExpert(StepDemonstrationExpert):
    def __init__(self, env: Any, policy: Any):
        self._policy = policy
        super().__init__(env)

    def _demo_fn(self, step: Step, env: Any, history: History) -> Any:
        pass  # implementation...


class ConcreteTrajectoryDemonstrationExpert(TrajectoryDemonstrationExpert):
    def __init__(self, env: Any, policy: Any):
        self._policy = policy
        super().__init__(env)

    def _demo_fn(self, step: Trajectory, env: Any, history: History) -> Any:
        pass  # implementation...