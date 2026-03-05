from typing import Any

from human_feedback_rl.interfaces.Expert import StepDemonstrationExpert,  TrajectoryDemonstrationExpert
from human_feedback_rl.Core import Step, Trajectory


class ConcreteStepDemonstrationExpert(StepDemonstrationExpert):
    def __init__(self, policy: Any):
        self._policy = policy
        super().__init__()

    def _demo_fn(self, step: Step) -> Any:
        action, _ = self._policy.predict(step.state, deterministic=True)
        return action


class ConcreteTrajectoryDemonstrationExpert(TrajectoryDemonstrationExpert):
    def __init__(self, policy: Any):
        self._policy = policy
        super().__init__()

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