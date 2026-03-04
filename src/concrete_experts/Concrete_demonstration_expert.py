from typing import Any

from src.interfaces.Expert import StepDemonstrationExpert,  TrajectoryDemonstrationExpert
from src.Core import Step, Trajectory


class ConcreteStepDemonstrationExpert(StepDemonstrationExpert):
    def __init__(self, env: Any, policy: Any):
        self._policy = policy
        super().__init__(env)

    def _demo_fn(self, step: Step) -> Any:
        action, _ = self._policy.predict(step.state, deterministic=True)
        return action


class ConcreteTrajectoryDemonstrationExpert(TrajectoryDemonstrationExpert):
    def __init__(self, env: Any, policy: Any):
        self._policy = policy
        super().__init__(env)

    def _demo_fn(self, trajectory: Trajectory) -> Any:
        state = self.env.reset()
        steps = []
        done = False

        while not done:
            action, _ = self._policy.predict(state, deterministic=True)
            next_state, reward, done, info = self.env.step(action)
            steps.append(Step(state, action))
            state = next_state
        return Trajectory(steps)