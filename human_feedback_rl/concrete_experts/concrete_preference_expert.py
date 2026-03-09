from typing import Any, List, Sequence

import torch

from human_feedback_rl.interfaces.expert import StepPreferenceExpert,  TrajectoryPreferenceExpert
from human_feedback_rl.core import Step, Trajectory


class ConcreteStepPreferenceExpert(StepPreferenceExpert):
    def __init__(self, env: Any, policy: Any):
        self._policy = policy
        self._env = env
        super().__init__()

    def _preference_fn(self, steps: Sequence[Step]):
        s = torch.as_tensor(steps[0].state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            q_values = self._policy.q_net(s)[0]

        a0 = steps[0].action
        a1 = steps[1].action

        score0 = q_values[a0].item()
        score1 = q_values[a1].item()

        probs = torch.softmax(torch.tensor([score0, score1]), dim=0)

        return probs.tolist()


class ConcreteTrajectoryPreferenceExpert(TrajectoryPreferenceExpert):

    def __init__(self, env: Any, policy: Any):
        self._env = env
        self._policy = policy
        super().__init__()

    def _preference_fn(self, trajectories: List[Trajectory]):

        traj1, traj2 = trajectories

        score1 = 0.0
        score2 = 0.0

        for step in traj1.steps:

            s = torch.as_tensor(step.state, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                q_values = self._policy.q_net(s)[0]

            score1 += q_values[step.action].item()

        for step in traj2.steps:

            s = torch.as_tensor(step.state, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                q_values = self._policy.q_net(s)[0]

            score2 += q_values[step.action].item()

        probs = torch.softmax(
            torch.tensor([score1, score2]),
            dim=0
        )

        return probs.tolist()