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
            self.rewards.append(0)

        if len(self.states) > self.capacity:

            self.states = self.states[-self.capacity:]
            self.actions = self.actions[-self.capacity:]
            self.rewards = self.rewards[-self.capacity:]

    def relabel_rewards(self, reward_model):

        new_rewards = []

        for s, a in zip(self.states, self.actions):
            with torch.no_grad():
                r = torch.tanh(
                    reward_model(s, a)
                ).item()

            new_rewards.append(r)

        r = torch.tensor(new_rewards)

        r = (r - r.mean()) / (r.std() + 1e-8)

        self.rewards = r.tolist()

    def sample_segments(self, segment_length=25):

        max_start = len(self.states) - segment_length - 1

        if max_start <= 0:
            raise ValueError("Not enough data in replay buffer")

        idx1 = random.randint(0, max_start)
        idx2 = random.randint(0, max_start)

        for _ in range(10):
            if abs(idx1 - idx2) >= segment_length:
                break
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

    def sample_batch(self, batch_size=256):

        batch_size = min(batch_size, len(self.states))
        idx = random.sample(range(len(self.states)), batch_size)

        states = [self.states[i] for i in idx]
        actions = [self.actions[i] for i in idx]
        rewards = [self.rewards[i] for i in idx]

        return states, actions, rewards