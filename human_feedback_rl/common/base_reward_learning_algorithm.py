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

    _CORRELATION_SEGMENT_LENGTHS: tuple = (1, 5, 20, None)
    _CORRELATION_N_SAMPLES: int = 100

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
        debug_datasets: Optional[Dict] = None,
    ):
        super().__init__(env, agent, rng, log_folder=log_folder, output_formats=output_formats)

        self.fragment_length          = fragment_length
        self.initial_queries          = initial_queries
        self.train_comparison_frac    = train_comparison_frac
        self.exploration_frac         = exploration_frac
        self.iteration                = 0
        self.trajectories             = []
        self.debug_datasets           = debug_datasets or {}

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
        self.log_reward_model_correlations(self.trajectories)


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

    def log_reward_model_correlations(self, trajectories) -> None:
        """Log Spearman-ρ and Pearson-r between model and true returns.

        Uses pre-computed debug datasets when available; otherwise samples
        fragments on-the-fly from the provided trajectories.
        """
        if self.debug_datasets:
            for key, data in self.debug_datasets.items():
                frag_list = [frag for bucket in data.values() for frag in bucket]
                if not frag_list:
                    continue
                self._log_correlations(frag_list, tag=key)
        else:
            for seg_len in self._CORRELATION_SEGMENT_LENGTHS:
                frag_list = self._single_fragmenter(trajectories, seg_len, self._CORRELATION_N_SAMPLES)
                self._log_correlations(frag_list, tag=f"seg{seg_len}")

    def _log_correlations(self, frag_list, tag: str) -> None:
        true_returns = np.array(
            [sum(t.true_reward for t in frag) / len(frag) for frag in frag_list]
        )

        self.reward_model.eval()
        with th.no_grad():
            model_returns = []
            for frag in frag_list:
                obs    = th.tensor(np.array([t.observation  for t in frag]), dtype=th.float32)
                acts   = th.tensor(np.array([t.action       for t in frag]), dtype=th.float32)
                next_s = th.tensor(np.array([t.next_status  for t in frag]), dtype=th.float32)
                done   = th.tensor(np.array([float(t.done)  for t in frag]), dtype=th.float32)
                model_returns.append(self.reward_model(obs, acts, next_s, done).sum().item() / len(frag))
            model_returns = np.array(model_returns)
        self.reward_model.train()

        rho, _ = spearmanr(true_returns, model_returns)
        r,   _ = pearsonr(true_returns, model_returns)

        self.logger.record(f"reward_correlation/spearman_{tag}", rho, exclude="stdout")
        self.logger.record(f"reward_correlation/pearson_{tag}",  r,   exclude="stdout")

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
