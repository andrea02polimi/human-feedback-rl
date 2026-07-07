"""
Shared base for reward-learning algorithms (preference-based and demo-based).

Owns everything both algorithm families need: the learned reward model, the
trajectory generator (rollouts + agent updates on predicted rewards), rollout
and validation logging, query scheduling, and checkpointing. Each concrete
algorithm implements its own ``train()`` loop on top of these utilities.
"""

import os
import time
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch as th
from scipy.stats import kendalltau

from human_feedback_rl.common import status
from human_feedback_rl.common.base_algorithm import BaseAlgorithm
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
    """
    Shared base for reward-learning algorithms.

    Extends ``BaseAlgorithm`` with:
      * A learned reward model driving the agent's rewards.
      * A ``TrajectoryGeneratorFromAgent`` for rollouts and agent updates.
      * Rollout sampling/logging, reward-model validation logging, query
        scheduling, per-member reward training, and checkpointing.

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
        train_comparison_frac: float = 0.7,
        fragment_length: int = 1,
        initial_queries: int = 0,
        exploration_frac: float = 0.0,
        exploration_eps: float = 0.5,
        query_schedule: Union[str, Callable[[float], float]] = "constant",
        temperature: float = 1,
        rng: Optional[np.random.Generator] = None,
        log_folder: Optional[str] = None,
        output_formats: Optional[List] = None,
        debug_dataset: Optional[Dict] = None,
        sampling_venv=None,
        agent_log_timestep_interval: Optional[int] = None,
    ):
        super().__init__(env, agent, rng, log_folder=log_folder, output_formats=output_formats)

        self.fragment_length          = fragment_length
        self.initial_queries          = initial_queries
        self.train_comparison_frac    = train_comparison_frac
        self.exploration_frac         = exploration_frac
        self.temperature              = temperature
        self.iteration                = 0
        self.trajectories             = []
        self.debug_dataset            = debug_dataset or {}

        self.query_schedule      = QUERY_SCHEDULES[query_schedule]
        self.query_schedule_name = query_schedule

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
        """Return one query count per iteration following the configured schedule.

        The list has exactly ``n_iterations`` entries summing to ``total_queries``
        (largest-remainder rounding); ``initial_queries`` are added to iteration 0.
        """
        if n_iterations <= 0:
            return []
        remaining = max(total_queries - self.initial_queries, 0)

        t_vec = np.linspace(0, 1, n_iterations)
        weights = np.array([self.query_schedule(t) for t in t_vec])
        exact = weights / weights.sum() * remaining

        # Largest-remainder rounding: floor everything, then hand the leftover
        # queries to the iterations with the largest fractional parts.
        shares = np.floor(exact).astype(int)
        leftover = remaining - int(shares.sum())
        if leftover > 0:
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

    def _run_reward_inference(self, transitions):
        """Run the reward model in eval mode; return arrays of true/pred rewards, status, done."""
        self.reward_model.eval()
        with th.no_grad():
            true_rewards = np.array([t.true_reward  for t in transitions], dtype=np.float32)
            obs          = np.array([t.observation  for t in transitions], dtype=np.float32)
            acts         = np.array([t.action       for t in transitions], dtype=np.float32)
            status       = np.array([t.next_status  for t in transitions], dtype=np.float32)
            done         = np.array([float(t.done)  for t in transitions], dtype=np.float32)
            pred_rewards = self.reward_model.predict(obs, acts, status, done)
        self.reward_model.train()
        return true_rewards, pred_rewards, status

    def _normalize_predictions(
        self,
        pred_rewards: np.ndarray,
        true_rewards: np.ndarray,
        status: np.ndarray,
        norm_on_running: bool = True,
        match_mean: bool = True,
        match_std: bool = False,
    ) -> np.ndarray:
        """Shift/scale pred_rewards to align with true_rewards statistics on running steps."""
        pred_rewards = pred_rewards * self.temperature

        norm_mask = np.ones(len(pred_rewards), dtype=bool)
        if norm_on_running:
            norm_mask = status[:, self.STATUS_RUNNING] == 1
            if not norm_mask.any():
                norm_mask = np.ones(len(pred_rewards), dtype=bool)

        true_mean = np.mean(true_rewards[norm_mask])
        pred_mean = np.mean(pred_rewards[norm_mask])

        if match_mean and match_std:
            true_std = np.std(true_rewards[norm_mask])
            pred_std = np.std(pred_rewards[norm_mask])
            return (pred_rewards - pred_mean) / max(pred_std, 1e-8) * true_std + true_mean
        elif match_mean:
            return pred_rewards - pred_mean + true_mean
        return pred_rewards

    def log_reward_model_validation(self, transitions, log_class: str) -> None:
        """Log per-outcome MAE (and Kendall tau on running steps) of normalized predictions.

        Outcomes absent from ``transitions`` are skipped rather than logged as NaN.
        """
        true_rewards, pred_rewards, status_onehot = self._run_reward_inference(transitions)
        pred_rewards_norm = self._normalize_predictions(pred_rewards, true_rewards, status_onehot)

        outcome_indices = {
            "arrived": self.STATUS_ARRIVED,
            "collided": self.STATUS_COLLIDED,
            "offroad": self.STATUS_OFFROAD,
            "timeout": self.STATUS_TIMEOUT,
            "running": self.STATUS_RUNNING,
        }
        for name, idx in outcome_indices.items():
            mask = status_onehot[:, idx] == 1
            if mask.any():
                mae = np.mean(np.abs(pred_rewards_norm[mask] - true_rewards[mask]))
                self.logger.record(f"{log_class}/mae_{name}", mae)

        running_mask = status_onehot[:, self.STATUS_RUNNING] == 1
        if running_mask.sum() >= 2:
            kt_running, _ = kendalltau(true_rewards[running_mask], pred_rewards_norm[running_mask])
            self.logger.record(f"{log_class}/kendall_running", kt_running)

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
        model_rewards = [self._score_trajectory(traj)     for traj in trajectories]
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

    def _score_trajectory(self, traj: Trajectory) -> float:
        obs         = np.array([t.observation for t in traj])
        acts        = np.array([t.action for t in traj])
        next_status = np.array([t.next_status for t in traj])
        done        = np.array([float(t.done) for t in traj])
        return self.reward_model.predict(obs, acts, next_status, done).sum()


    def train_agent(self, steps: int, log_interval: int) -> None:
        """Train the agent for ``steps`` timesteps via the trajectory generator."""
        
        t0 = time.perf_counter()
        self.trajectory_generator.train(steps=steps, log_interval=log_interval)
        t_train_agent = time.perf_counter() - t0
        
        self.logger.record("time/train_agent", t_train_agent)

