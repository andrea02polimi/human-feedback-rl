"""Rollout collection, rollout logging, and policy training."""

import time

import numpy as np
from gymnasium import spaces

from human_feedback_rl.common.types import Trajectory


class RolloutMixin:
    """Agent/environment interaction methods used by ``DemoAlgorithm``."""

    def _sample_rollout(self, agent_steps: int, exploration_steps: int = 0) -> list:
        t0 = time.perf_counter()
        trajectories = self.trajectory_generator.sample(agent_steps, exploration_steps)
        t_sample = time.perf_counter() - t0

        t0 = time.perf_counter()
        true_rewards = [traj.total_reward() for traj in trajectories]
        model_rewards = [self._score_trajectory(traj) for traj in trajectories]
        lengths = [len(traj) for traj in trajectories]

        self.logger.record("rollout/mean_true_reward", float(np.mean(true_rewards)))
        self.logger.record("rollout/mean_model_reward", float(np.mean(model_rewards)))
        self.logger.record("rollout/mean_length", float(np.mean(lengths)))
        self._log_action_boundaries(trajectories)
        self.logger.record("time/sample_rollout", t_sample)
        self.logger.record_sum("time/loggings", time.perf_counter() - t0)
        return trajectories

    def _log_action_boundaries(self, trajectories) -> None:
        action_space = self.env.action_space
        if not isinstance(action_space, spaces.Box):
            return

        actions = np.asarray(
            [t.action for trajectory in trajectories for t in trajectory], dtype=np.float32
        ).reshape(-1, int(np.prod(action_space.shape)))
        low = action_space.low.reshape(1, -1)
        high = action_space.high.reshape(1, -1)
        at_bound = np.isclose(actions, low, rtol=0, atol=1e-6) | np.isclose(
            actions, high, rtol=0, atol=1e-6
        )
        self.logger.record(
            "rollout/action_at_bound_fraction", float(at_bound.any(axis=1).mean())
        )
        self.logger.record(
            "rollout/action_component_at_bound_fraction", float(at_bound.mean())
        )

    def _score_trajectory(self, traj: Trajectory) -> float:
        obs = np.array([t.observation for t in traj])
        acts = np.array([t.action for t in traj])
        status = np.array([t.next_status for t in traj])
        done = np.array([float(t.done) for t in traj])
        return self.reward_model.predict(obs, acts, status, done).sum()

    def _train_agent(self, steps: int, log_interval: int) -> None:
        t0 = time.perf_counter()
        self.trajectory_generator.train(steps=steps, log_interval=log_interval)
        self.logger.record("time/train_agent", time.perf_counter() - t0)
