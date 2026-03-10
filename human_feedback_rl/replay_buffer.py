import random
import torch
from human_feedback_rl.core import Trajectory, Step


class ReplayBuffer:

    def __init__(self, capacity=100000):

        self.states = []
        self.actions = []
        self.rewards = []

        self.capacity = capacity

    def add_trajectory(self, trajectory):

        for step in trajectory.steps:

            self.states.append(step.state)
            self.actions.append(step.action)
            self.rewards.append(0.0)

        if len(self.states) > self.capacity:

            self.states = self.states[-self.capacity:]
            self.actions = self.actions[-self.capacity:]
            self.rewards = self.rewards[-self.capacity:]

    def relabel_rewards(self, reward_model, window=20000):

        start = max(0, len(self.states) - window)

        for i in range(start, len(self.states)):

            s = self.states[i]
            a = self.actions[i]

            with torch.no_grad():
                r = reward_model(s, a)

            r = torch.clamp(r, -5, 5)
            r = torch.tanh(r)

            # exponential smoothing
            self.rewards[i] = 0.9 * self.rewards[i] + 0.1 * r.item()

    def sample_batch(self, batch_size=256):

        n = len(self.states)

        if n == 0:
            raise RuntimeError("Replay buffer empty")

        batch_size = min(batch_size, n)

        idx = random.sample(range(n), batch_size)

        states = [self.states[i] for i in idx]
        actions = [self.actions[i] for i in idx]
        rewards = [self.rewards[i] for i in idx]

        return states, actions, rewards

    def sample_segments(self, segment_length=25):

        max_start = len(self.states) - segment_length - 1

        idx1 = random.randint(0, max_start)
        idx2 = random.randint(0, max_start)

        traj1 = Trajectory([
            Step(self.states[i], self.actions[i])
            for i in range(idx1, idx1 + segment_length)
        ])

        traj2 = Trajectory([
            Step(self.states[i], self.actions[i])
            for i in range(idx2, idx2 + segment_length)
        ])

        return traj1, traj2