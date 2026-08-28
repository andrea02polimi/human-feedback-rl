"""Shared base for reward-learning algorithms.

Owns what any reward-learning loop needs: the learned reward model, the
trajectory generator that rolls out and trains the agent on predicted rewards,
rollout logging, the query schedule and checkpointing. HybridAlgorithm builds
its ``train()`` loop on top.
"""

import os
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch as th

from human_feedback_rl.common import status
from human_feedback_rl.common.base_algorithm import BaseAlgorithm
from human_feedback_rl.common.batching import stacked_transitions
from human_feedback_rl.common.fragmenters import RandomSingleFragmenter
from human_feedback_rl.common.loggers import ExcludeFormatLogger, PrefixedLogger
from human_feedback_rl.common.reward_nets import RewardNet
from human_feedback_rl.common.trajectory_generators import TrajectoryGeneratorFromAgent
from human_feedback_rl.common.types import Trajectory


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

QUERY_SCHEDULES: Dict[str, Callable[[float], float]] = {
    "constant":          lambda t: 1.0,
    "hyperbolic":        lambda t: 1.0 / (1.0 + t),
    "inverse_quadratic": lambda t: 1.0 / (1.0 + t**2),
}


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseRewardLearningAlgorithm(BaseAlgorithm):
    """Shared base for the reward-learning algorithms.

    Adds to ``BaseAlgorithm`` a learned reward model, a trajectory generator for
    rollouts and agent updates, the query schedule, per-member reward training
    and checkpointing.

    Subclasses implement ``train()`` (the outer loop) using these utilities,
    define ``self.optimizers`` (one per reward-model ensemble member), and may
    override ``_save_checkpoint_extras`` to persist extra state.
    """

    STATUS_ARRIVED  = status.STATUS_ARRIVED
    STATUS_COLLIDED = status.STATUS_COLLIDED
    STATUS_OFFROAD  = status.STATUS_OFFROAD
    STATUS_TIMEOUT  = status.STATUS_TIMEOUT
    STATUS_RUNNING  = status.STATUS_RUNNING

    def __init__(
        self,
        env,
        agent,
        reward_model: RewardNet,
        exploration_frac: float = 0.0,
        exploration_eps: float = 0.5,
        rng: Optional[np.random.Generator] = None,
        log_folder: Optional[str] = None,
        output_formats: Optional[List] = None,
        debug_dataset: Optional[Dict] = None,
        sampling_venv=None,
        agent_log_timestep_interval: Optional[int] = None,
    ):
        super().__init__(env, agent, rng, log_folder=log_folder, output_formats=output_formats)

        self.exploration_frac = exploration_frac
        self.iteration = 0
        self.trajectories: List[Trajectory] = []
        self.debug_dataset = debug_dataset or {}

        # Query-scheduling state; the subclass assigns its configured values
        # (``build_query_schedule`` reads them).
        self.initial_queries = 0
        self.query_schedule: Callable[[float], float] = QUERY_SCHEDULES["constant"]
        self.query_schedule_name: str = "constant"

        self.reward_model = reward_model

        agent.set_logger(ExcludeFormatLogger(PrefixedLogger(self.logger, "agent"), exclude="stdout"))
        self.trajectory_generator = TrajectoryGeneratorFromAgent(
            venv=env,
            agent=agent,
            reward_model=reward_model,
            exploration_eps=exploration_eps,
            rng=self.rng,
            logger=self.logger,
            sampling_venv=sampling_venv,
            dump_timestep_interval=agent_log_timestep_interval,
        )

        # Used exclusively for reward-correlation logging (always single fragments).
        self._single_fragmenter = RandomSingleFragmenter(
            rng=self.rng,
            logger=self.logger,
        )


    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    def train_reward_members(self, member_step: Callable) -> List[Any]:
        """Run ``member_step(member, optimizer)`` for each ensemble member in turn.

        Returns the list of per-member results (e.g. gradient-norm traces).
        """
        return [
            member_step(member, optimizer)
            for member, optimizer in zip(self.reward_model.members, self.optimizers)
        ]

    def build_query_schedule(self, n_iterations: int, total_queries: int) -> List[int]:
        """One query count per iteration, summing to total_queries.

        initial_queries are added to iteration 0. When the budget is smaller than the
        number of iterations every share floors to zero and the remainders tie, so a
        constant schedule spreads its leftovers over evenly spaced indices rather than
        letting a stable sort push them all to the end. Other schedules keep the
        largest-remainder rule.
        """
        if n_iterations <= 0:
            return []
        remaining = max(total_queries - self.initial_queries, 0)

        t_vec = np.linspace(0, 1, n_iterations)
        weights = np.array([self.query_schedule(t) for t in t_vec])
        exact = weights / weights.sum() * remaining

        shares = np.floor(exact).astype(int)
        leftover = remaining - int(shares.sum())
        if leftover > 0:
            if self.query_schedule_name == "constant":
                # k = 1..leftover; the last index lands exactly on n_iterations-1
                top_up = [(k * n_iterations - 1) // leftover
                          for k in range(1, leftover + 1)]
            else:
                top_up = np.argsort(exact - shares)[::-1][:leftover]
            shares[top_up] += 1

        shares[0] += self.initial_queries
        return shares.tolist()

    def save_checkpoint(self, checkpoint_dir: str, iteration: int) -> None:
        ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{iteration:04d}")
        os.makedirs(ckpt_path, exist_ok=True)
        th.save(self.reward_model.state_dict(), os.path.join(ckpt_path, "reward_model.pt"))
        self.trajectory_generator.agent.save(os.path.join(ckpt_path, "agent"))
        self._save_checkpoint_extras(ckpt_path, iteration)
        print(f"  checkpoint saved in {ckpt_path}")

    def _save_checkpoint_extras(self, ckpt_path: str, iteration: int) -> None:
        """Hook: persist algorithm-specific state next to the standard checkpoint files."""

    def log_iteration(self, t0: float) -> None:
        """Log iteration-level scalars and flush the logger.

        All component metrics (rollout, fragmenter, gatherer, reward model,
        agent training) are already recorded by the components themselves.
        This method only adds the aggregated scalars that only the base loop knows.
        """
        t_total = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.logger.record("iterations",                  self.iteration)
        self.logger.record("agent/time/total_timesteps",  self.agent.num_timesteps)
        self.logger.record("time/total",                  t_total)
        self.logger.record_sum("time/loggings",           time.perf_counter() - t0)
        self.logger.dump()

    def sample_rollout(self, agent_steps: int, exploration_steps: int = 0):
        """Collect trajectories via the trajectory generator."""

        t0 = time.perf_counter()
        trajectories = self.trajectory_generator.sample(agent_steps, exploration_steps)
        t_sample_rollout = time.perf_counter() - t0

        t0 = time.perf_counter()
        true_rewards  = [traj.total_reward()              for traj in trajectories]
        model_rewards = self._score_trajectories(trajectories)
        lengths       = [len(traj)                        for traj in trajectories]

        self.logger.record("rollout/mean_true_reward",  float(np.mean(true_rewards)))
        self.logger.record("rollout/mean_model_reward", float(np.mean(model_rewards)))
        self.logger.record("rollout/mean_length",       float(np.mean(lengths)))
        self.logger.record("rollout/n_trajectories",    len(trajectories))
        self.logger.record("rollout/total_transitions", int(np.sum(lengths)))
        self._log_action_boundaries(trajectories)

        self.logger.record("time/sample_rollout",       t_sample_rollout)
        self.logger.record_sum("time/loggings",         time.perf_counter() - t0)

        return trajectories

    def _log_action_boundaries(self, trajectories) -> None:
        """Log the fraction of rollout actions saturating the Box action space (no-op otherwise)."""
        action_space = self.env.action_space
        if not hasattr(action_space, "low") or not hasattr(action_space, "high"):
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

    def _score_trajectories(self, trajectories) -> List[float]:
        """Predicted (agent-facing) return of each trajectory, via one batched predict."""
        if not trajectories:
            return []
        lengths = [len(traj) for traj in trajectories]
        parts = [stacked_transitions(traj) for traj in trajectories]
        obs, acts, next_status, done = (
            th.cat([p[i] for p in parts]).numpy() for i in range(4)
        )
        rewards = self.reward_model.predict(obs, acts, next_status, done)
        boundaries = np.cumsum(lengths)[:-1]
        return [float(chunk.sum()) for chunk in np.split(rewards, boundaries)]

    def train_agent(self, steps: int, log_interval: int) -> None:
        """Train the agent for ``steps`` timesteps via the trajectory generator."""

        t0 = time.perf_counter()
        self.trajectory_generator.train(steps=steps, log_interval=log_interval)
        t_train_agent = time.perf_counter() - t0

        self.logger.record("time/train_agent", t_train_agent)

