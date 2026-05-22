"""
Base class for RLHF reward-learning algorithms.

Implements the outer training loop (schedule → collect → train reward model →
train agent → log → checkpoint) using the Template Method pattern inspired by
Stable-Baselines3's BaseAlgorithm.  Algorithm-specific steps are left as
abstract hooks that subclasses must fill in.
"""

import os
import random
import time
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch as th
from scipy.stats import kendalltau, pearsonr, spearmanr
from stable_baselines3.common.vec_env import VecEnv

from human_feedback_rl.common.base_algorithm import BaseAlgorithm
from human_feedback_rl.common.fragmenters import RandomSingleFragmenter
from human_feedback_rl.common.loggers import ExcludeFormatLogger, PrefixedLogger
from human_feedback_rl.common.reward_nets import RewardEnsemble, RewardNet, SumoRewardNet
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
    Template-method base for RLHF reward-learning algorithms.

    Extends ``BaseAlgorithm`` with:
      * A learned reward ensemble and its Adam optimizers.
      * A ``TrajectoryGeneratorFromAgent`` for rollouts and PPO updates.
      * The outer training loop: query schedule → rollout → feedback collection
        → reward model training → agent training → logging → checkpointing.
      * Reward-model correlation logging against ground-truth returns.

    Subclasses must implement the four abstract hooks below and may override
    the optional hooks to inject extra behaviour without rewriting the loop.

    Abstract hooks (must implement):
        collect_feedback        – fragment trajectories and gather feedback labels.
        push_data               – route (fragment, feedback) pairs into datasets.
        train_reward_model      – gradient updates on the reward model.
        _evaluate_reward_model  – loss + accuracy on a dataset split.

    Optional hooks (may override):
        before_reward_training  – called after data collection, before RM training.

    Class attributes:
        _CORRELATION_SEGMENT_LENGTHS – segment lengths used in rew_model_correlation.
    """

    STATUS_ARRIVED  = 0
    STATUS_COLLIDED = 1
    STATUS_OFFROAD  = 2
    STATUS_TIMEOUT  = 3
    STATUS_RUNNING  = 4
    
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
        rng: Optional[np.random.Generator] = None,
        log_folder: Optional[str] = None,
        output_formats: Optional[List] = None,
        debug_dataset: Optional[Dict] = None,
    ):
        super().__init__(env, agent, rng, log_folder=log_folder, output_formats=output_formats)

        self.fragment_length          = fragment_length
        self.initial_queries          = initial_queries
        self.train_comparison_frac    = train_comparison_frac
        self.exploration_frac         = exploration_frac
        self.iteration                = 0
        self.trajectories             = []
        self.debug_dataset            = debug_dataset or {}

        if fragment_length not in self._CORRELATION_SEGMENT_LENGTHS:
            self._CORRELATION_SEGMENT_LENGTHS = self._CORRELATION_SEGMENT_LENGTHS + (fragment_length,)

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
        )

        # Used exclusively for reward-correlation logging (always single fragments).
        self._single_fragmenter = RandomSingleFragmenter(
            rng=self.rng,
            logger=self.logger,
        )


    # ------------------------------------------------------------------
    # Abstract hooks – subclasses must implement all of these
    # ------------------------------------------------------------------

    @abstractmethod
    def collect_feedback(self, num_queries: int) -> tuple:
        """Fragment trajectories and gather feedback labels.

        Fragmenter and gatherer log their own timing via self.logger.
        Returns:
            (fragments, feedback)
        """
        ...

    @abstractmethod
    def push_data(self, fragments, feedback) -> None:
        """Shuffle then push into dataset_train / dataset_val split by train_comparison_frac."""
        ...

    @abstractmethod
    def train_reward_model(self) -> None:
        """Run one round of reward-model gradient updates and log metrics via self.logger."""
        ...


    # ------------------------------------------------------------------
    # Optional hooks – subclasses may override for extra behaviour
    # ------------------------------------------------------------------

    def before_reward_training(self) -> None:
        """Called after data collection, before reward-model training each iteration.

        Default: logs reward-model correlation against ground-truth returns.
        Override and call super() to add extra steps (e.g. reward normalization).
        """
        
        all_transitions = [t for traj in self.trajectories for t in traj]
        self.log_reward_model_validation(all_transitions, "reward_val/current_rollout")
        
        if self.debug_dataset:
            self.log_reward_model_validation(self.debug_dataset, "reward_val/debug_dataset")


    def before_agent_training(self) -> None:
        """Called after reward-model training, before agent training each iteration.

        Default: no-op. Override to add extra steps (e.g. reward normalization,
        reward-model evaluation, or logging) without rewriting the loop.
        """

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    def build_query_schedule(self, n_iterations: int, total_queries: int) -> List[int]:
        """Return a per-iteration list of query counts following the configured schedule."""
        t_vec = np.linspace(0, 1, n_iterations)
        weights = np.array([self.query_schedule(t) for t in t_vec])
        probs = weights / weights.sum()
        shares = np.round(probs * (total_queries - self.initial_queries)).astype(int)
        return [self.initial_queries] + shares.tolist()

    def save_checkpoint(self, checkpoint_dir: str, iteration: int) -> None:
        ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{iteration:04d}")
        os.makedirs(ckpt_path, exist_ok=True)
        th.save(self.reward_model.state_dict(), os.path.join(ckpt_path, "reward_model.pt"))
        self.trajectory_generator.agent.save(os.path.join(ckpt_path, "agent"))
        print(f"  checkpoint saved in {ckpt_path}")

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
        
        norm_mask = np.ones(len(pred_rewards), dtype=bool)
        if norm_on_running:
            norm_mask = status[:, self.STATUS_RUNNING] == 1
            
        true_mean = np.mean(true_rewards[norm_mask])
        pred_mean = np.mean(pred_rewards[norm_mask])

        if match_mean and match_std:
            true_std = np.std(true_rewards[norm_mask])
            pred_std = np.std(pred_rewards[norm_mask])
            return (pred_rewards - pred_mean) / pred_std * true_std + true_mean
        elif match_mean:
            return pred_rewards - pred_mean + true_mean
        return pred_rewards

    def log_reward_model_validation(self, transitions, log_class: str) -> None:
        true_rewards, pred_rewards, status = self._run_reward_inference(transitions)
        pred_rewards_norm = self._normalize_predictions(pred_rewards, true_rewards, status)
        
        arrived_mask  = status[:, self.STATUS_ARRIVED] == 1
        collided_mask = status[:, self.STATUS_COLLIDED] == 1
        offroad_mask  = status[:, self.STATUS_OFFROAD] == 1
        timeout_mask  = status[:, self.STATUS_TIMEOUT] == 1
        running_mask  = status[:, self.STATUS_RUNNING] == 1

        mae_arrived   = np.mean(np.abs(pred_rewards_norm[arrived_mask] - true_rewards[arrived_mask]))
        mae_collided  = np.mean(np.abs(pred_rewards_norm[collided_mask] - true_rewards[collided_mask]))
        mae_offroad   = np.mean(np.abs(pred_rewards_norm[offroad_mask] - true_rewards[offroad_mask]))
        mae_timeout   = np.mean(np.abs(pred_rewards_norm[timeout_mask] - true_rewards[timeout_mask]))
        mae_running   = np.mean(np.abs(pred_rewards_norm[running_mask] - true_rewards[running_mask]))
        kt_running, _ = kendalltau(true_rewards[running_mask], pred_rewards_norm[running_mask])

        self.logger.record(f"{log_class}/mae_arrived", mae_arrived)
        self.logger.record(f"{log_class}/mae_collided", mae_collided)
        self.logger.record(f"{log_class}/mae_offroad", mae_offroad)
        self.logger.record(f"{log_class}/mae_timeout", mae_timeout)
        self.logger.record(f"{log_class}/mae_running", mae_running)
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
        
        self.logger.record("time/sample_rollout",       t_sample_rollout)
        self.logger.record_sum("time/loggings",         time.perf_counter() - t0)

        return trajectories
    
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

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(
        self,
        total_timesteps: int = 1_000_000,
        total_queries: int = 10_000,
        timesteps_per_iteration: int = 1024,
        log_interval: int = 1,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 10,
    ) -> Any:
        """Run the full alternating reward-learning + agent-training loop."""
        
        self.iteration = 0
        n_iterations = int(total_timesteps / timesteps_per_iteration)
        schedule = self.build_query_schedule(n_iterations, total_queries)

        print("="*100)
        print("RLHF reward learning algorithm")
        print("="*100)
        print("")
        print(f"Query {self.query_schedule_name} schedule: {schedule}")

        for num_queries in schedule:
            t_iter = time.perf_counter()
            print(f"\nIteration {self.iteration}/{len(schedule)-1}")

            if num_queries == 0:
                continue

            # ---- Data collection ----------------------------------------
            exploration_steps = int(self.exploration_frac * timesteps_per_iteration)
            print(f"- Collecting {timesteps_per_iteration} agent + {exploration_steps} exploration transitions")
            self.trajectories = self.sample_rollout(timesteps_per_iteration, exploration_steps)

            print(f"- Collecting {num_queries} feedbacks on the current rollout")
            fragments, feedback = self.collect_feedback(num_queries)
            self.push_data(fragments, feedback)

            # ---- Reward model training ----------------------------------
            self.before_reward_training()

            print(f"- Training reward model")
            self.train_reward_model()

            # ---- Agent training -----------------------------------------
            self.before_agent_training()

            print(f"- Training agent for {timesteps_per_iteration} timesteps")
            self.train_agent(timesteps_per_iteration, log_interval)

            # ---- Logging & checkpointing --------------------------------
            self.log_iteration(t_iter)

            if checkpoint_dir is not None and (self.iteration + 1) % checkpoint_interval == 0:
                self.save_checkpoint(checkpoint_dir, self.iteration + 1)

            self.iteration += 1


        return self.trajectory_generator.agent
