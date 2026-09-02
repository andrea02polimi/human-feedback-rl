"""Hybrid reward learning from demonstrations and preferences.

The only reward-learning algorithm here. With both channels it is the hybrid
method; with demo_weight=0 it is the preference-only baseline, and with
total_queries=0 the demonstration-only one.

demo_mode chooses how demonstrations enter: gcl fuses an IRL loss with the
Bradley-Terry loss on one net, preferences turns them into preference pairs
(Ibarz et al. 2018).

With soft labels at a high pref_temperature the BT loss sits at its ln(2) floor
even when learning works. Read reward/acc_pref_train instead.
"""

import os
import time
from typing import Any, Callable, List, Optional, Union

import numpy as np
import torch as th

from human_feedback_rl.algorithms.hybrid.demonstration_losses import (
    VALID_LOSSES,
    RewardLossMixin,
)
from human_feedback_rl.algorithms.hybrid.feedback_collection import FeedbackCollectionMixin
from human_feedback_rl.algorithms.hybrid.gradient_fusion import GradientFusionMixin
from human_feedback_rl.algorithms.hybrid.imitation_metrics import ImitationMetricsMixin
from human_feedback_rl.algorithms.hybrid.reliability_weight import ReliabilityWeightMixin
from human_feedback_rl.algorithms.hybrid.reward_diagnostics import RewardDiagnosticsMixin
from human_feedback_rl.algorithms.hybrid.reward_model_training import RewardModelTrainingMixin
from human_feedback_rl.algorithms.hybrid.reward_training import RewardTrainingMixin
from human_feedback_rl.common.base_reward_learning_algorithm import (
    QUERY_SCHEDULES,
    BaseRewardLearningAlgorithm,
)
from human_feedback_rl.common.datasets import PreferenceDataset
from human_feedback_rl.common.fragmenters import make_pair_fragmenter
from human_feedback_rl.common.gatherers import PreferenceGathererFromReward
from human_feedback_rl.common.reward_nets import make_reward_ensemble
from human_feedback_rl.common.types import Trajectory

def _require(condition, message: str) -> None:
    """Reject a hyperparameter combination the algorithm cannot honour."""
    if not condition:
        raise ValueError(message)


VALID_DEMO_MODES = ("gcl", "preferences")

# How the two gradients become one update.
#   norm_balance            norm-balanced sum
#   alpha_norm_single_adam  unit directions combined by alpha, a SINGLE Adam
#   unit_mean_single_adam   the same, with alpha pinned to 1/2 instead of
#                           estimated: the unweighted counterpart of the above
VALID_GCL_FUSIONS = ("norm_balance", "alpha_norm_single_adam",
                     "unit_mean_single_adam")


class HybridAlgorithm(
    FeedbackCollectionMixin,
    RewardModelTrainingMixin,
    GradientFusionMixin,
    ReliabilityWeightMixin,
    RewardLossMixin,
    RewardTrainingMixin,
    ImitationMetricsMixin,
    RewardDiagnosticsMixin,
    BaseRewardLearningAlgorithm,
):
    """Trains one reward model from preferences and/or demonstrations.

    See the module docstring for the two ``demo_mode`` mechanisms and the
    degenerate single-source baselines. In ``"gcl"`` mode the two losses are
    combined with a norm-balanced sum: the demo gradient is rescaled so its
    norm is ``demo_weight`` times the preference gradient's norm before the
    two are added.
    """

    VALID_LOSSES = VALID_LOSSES

    def __init__(
        self,
        env,
        agent,
        expert_trajectories: List[Trajectory],
        loss_type: str = "demo_2",
        lr_rew: float = 0.001,
        gradient_steps_rew: int = 10,
        batch_size_expert: int = 32,
        batch_size_model: int = 64,
        batch_size_pref: int = 32,
        l2_rew: float = 0.0001,
        pref_temperature: float = 20.0,
        preference_fragment_length: int = 1,
        fragmenter_type: str = "random",
        labels_type: str = "binary",
        comparison_queue_size: int = 1_000_000,
        total_queries: int = 10_000,
        initial_queries: int = 0,
        query_schedule: Union[str, Callable[[float], float]] = "constant",
        demo_mode: str = "gcl",
        demo_weight: float = 1.0,
        max_balance_scale: float = 100.0,
        balance_eps: float = 1e-8,
        gcl_fusion: str = "norm_balance",
        alpha_eps: float = 1e-8,
        label_smoothing: float = 0.0,
        bootstrap_comparisons: Optional[bool] = None,
        demo_pref_pairs_per_iteration: int = 64,
        demo_pref_batch_fraction: float = 0.5,
        initial_agent_timesteps: int = 0,
        exploration_frac: float = 0.0,
        exploration_eps: float = 0.5,
        reward_model_kwargs: Optional[dict] = None,
        rng: Optional[np.random.Generator] = None,
        log_folder: Optional[str] = None,
        output_formats: Optional[List] = None,
        debug_dataset: Optional[dict] = None,
        rollout_env=None,
        relabel_rewards: bool = True,
        normalize_agent_reward: bool = True,
        agent_log_timestep_interval: Optional[int] = None,
    ):
        _require(expert_trajectories,
                 "expert_trajectories must be a non-empty list of Trajectory objects.")
        _require(loss_type in self.VALID_LOSSES,
                 f"loss_type must be one of {self.VALID_LOSSES}, got {loss_type!r}.")
        _require(gradient_steps_rew > 0, "gradient_steps_rew must be positive.")
        _require(batch_size_expert > 0 and batch_size_model > 0,
                 "Reward-model batch sizes must be positive.")
        _require(batch_size_pref > 0, "batch_size_pref must be positive.")
        _require(preference_fragment_length > 0,
                 "preference_fragment_length must be positive.")
        _require(demo_weight >= 0, "demo_weight must be non-negative.")
        _require(max_balance_scale > 0 and balance_eps > 0,
                 "max_balance_scale and balance_eps must be positive.")
        _require(pref_temperature > 0, "pref_temperature must be positive.")
        _require(demo_mode in VALID_DEMO_MODES,
                 f"demo_mode must be one of {VALID_DEMO_MODES}, got {demo_mode!r}.")
        _require(gcl_fusion in VALID_GCL_FUSIONS,
                 f"gcl_fusion must be one of {VALID_GCL_FUSIONS}, got {gcl_fusion!r}.")
        _require(alpha_eps > 0, "alpha_eps must be positive.")
        _require(0.0 <= label_smoothing < 1.0, "label_smoothing must be in [0, 1).")
        _require(demo_pref_pairs_per_iteration >= 0,
                 "demo_pref_pairs_per_iteration must be non-negative.")
        _require(0 <= demo_pref_batch_fraction <= 1,
                 "demo_pref_batch_fraction must be in [0, 1].")

        reward_model = make_reward_ensemble(env, **(reward_model_kwargs or {}))

        super().__init__(
            env=env,
            agent=agent,
            reward_model=reward_model,
            exploration_frac=exploration_frac,
            exploration_eps=exploration_eps,
            rng=rng,
            log_folder=log_folder,
            output_formats=output_formats,
            debug_dataset=debug_dataset,
            sampling_venv=rollout_env,
            agent_log_timestep_interval=agent_log_timestep_interval,
        )

        self.expert_trajectories = list(expert_trajectories)
        self.loss_type = loss_type
        self.gradient_steps_rew = gradient_steps_rew
        self.batch_size_expert = batch_size_expert
        self.batch_size_model = batch_size_model
        self.initial_agent_timesteps = initial_agent_timesteps
        self.relabel_rewards = relabel_rewards
        self.normalize_agent_reward = normalize_agent_reward
        self._debug_rng = np.random.default_rng(0)
        self._debug_trajectories = self._split_into_trajectories(self.debug_dataset)

        replay_buffer = getattr(agent, "replay_buffer", None)
        if replay_buffer is not None:
            if hasattr(replay_buffer, "set_reward_model"):
                replay_buffer.set_reward_model(self.reward_model)
                replay_buffer.set_relabel_rewards(relabel_rewards)
            elif relabel_rewards:
                raise ValueError("relabel_rewards=True requires RewardRelabelReplayBuffer.")

        self.optimizers = [
            th.optim.Adam(member.parameters(), lr=lr_rew, weight_decay=l2_rew)
            for member in self.reward_model.members
        ]

        self.batch_size_pref = batch_size_pref
        self.preference_fragment_length = int(preference_fragment_length)
        self.total_queries = total_queries
        self.initial_queries = initial_queries
        self.query_schedule_name = query_schedule if isinstance(query_schedule, str) else "callable"
        if isinstance(query_schedule, str):
            if query_schedule not in QUERY_SCHEDULES:
                raise ValueError(f"Unknown query_schedule: {query_schedule!r}.")
            self.query_schedule = QUERY_SCHEDULES[query_schedule]
        else:
            self.query_schedule = query_schedule

        self.demo_mode = demo_mode
        self.demo_weight = float(demo_weight)
        self.max_balance_scale = float(max_balance_scale)
        self.balance_eps = float(balance_eps)
        self.gcl_fusion = gcl_fusion
        self.alpha_eps = float(alpha_eps)
        self.label_smoothing = float(label_smoothing)
        # None lets the number of members decide (the bootstrap is there to
        # decorrelate them). True/False force it, and exist to reproduce an
        # earlier configuration: resampling used to be unconditional, so
        # single-member runs had it too.
        self.bootstrap_comparisons = bootstrap_comparisons
        self.labels_type = labels_type
        # This iteration's alpha, one per member (id -> AlphaEstimate).
        self._alpha_current = {}
        # Separate RNG for the diagnostics: drawing the rollout for the
        # estimate must not consume the state the training draws from.
        self._grad_probe_rng = np.random.default_rng(12345)
        self._rng_query, self._rng_oracle, self._rng_train = self._split_rng()
        # Diagnostic counters for duplicated comparisons (see
        # _count_duplicate_comparisons): the threshold counts stored items, not
        # distinct comparisons, and this says how far the two drift apart.
        self._seen_pairs = set()
        self._seen_fragments = set()
        self._dup_pairs = 0
        self._dup_self_pairs = 0
        self._dup_fragments = 0
        self.demo_pref_pairs_per_iteration = int(demo_pref_pairs_per_iteration)
        self.demo_pref_batch_fraction = float(demo_pref_batch_fraction)

        self.fragmenter = make_pair_fragmenter(
            fragmenter_type, rng=self._rng_query, logger=self.logger,
            reward_ensemble=self.reward_model
        )
        # Oracle label softness is a property of the (synthetic) annotator,
        # NOT of the demo IRL loss: it gets its own temperature.
        self.preference_gatherer = PreferenceGathererFromReward(
            logger=self.logger,
            labels_type=labels_type,
            temperature=pref_temperature,
            rng=self._rng_oracle,
        )
        self.dataset_train = PreferenceDataset(
            queue_size=comparison_queue_size, rng=self._rng_train)
        # Expert-vs-agent pairs (demo_mode="preferences" only).
        self.dataset_demo_prefs_train = PreferenceDataset(
            queue_size=comparison_queue_size, rng=self._rng_train)

    def _split_rng(self):
        """Three independent streams, spawned from the master seed.

            query    which fragments get compared
            oracle   oracle labels, Bernoulli draws included
            train    preference and demonstration minibatches, bootstrap

        Sharing one RNG made the feedback depend on how many gradient steps the reward
        model took, which put the optimizer settings inside the comparison.
        """
        seed_seq = getattr(self.rng.bit_generator, "seed_seq", None)
        if seed_seq is None:      # Generator built without a SeedSequence
            seed_seq = np.random.SeedSequence(int(self.rng.integers(0, 2**63)))
        return tuple(np.random.default_rng(s) for s in seed_seq.spawn(3))

    def train(
        self,
        total_timesteps: int = 1_000_000,
        timesteps_per_iteration: int = 1024,
        log_interval: int = 1,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 10,
        scatter_interval: Optional[int] = None,
    ) -> Any:
        """Run hybrid reward learning and agent training."""
        if scatter_interval is None:
            scatter_interval = 10
        if scatter_interval < 0:
            raise ValueError("scatter_interval must be non-negative.")

        n_iterations = int(total_timesteps / timesteps_per_iteration)
        # The query budget sits with its siblings (initial_queries,
        # query_schedule) among the algorithm kwargs. It used to be a train()
        # parameter as well, which quietly won when passed: two places for one
        # number, and no way to notice you had set only one.
        schedule = self.build_query_schedule(n_iterations, self.total_queries)
        self._n_training_iterations = n_iterations

        if self.initial_agent_timesteps > 0:
            print(f"- Collecting {self.initial_agent_timesteps} bootstrap transitions")
            self.trajectories = self.sample_rollout(self.initial_agent_timesteps)
            bootstrap_queries = self.initial_queries
            self._collect_feedback(bootstrap_queries)
            if schedule:
                schedule[0] = max(schedule[0] - bootstrap_queries, 0)
            print("- Bootstrapping reward model")
            self._train_reward_model()
            self._update_agent_reward_normalization()
            self._refresh_replay_relabel_cache()
            print(f"- Pre-warming agent for {self.initial_agent_timesteps} timesteps on learned reward")
            self.train_agent(self.initial_agent_timesteps, log_interval)

        for iteration, num_queries in enumerate(schedule):
            self.iteration = iteration
            t_iter = time.perf_counter()
            print(f"\nIteration {iteration}/{n_iterations - 1}")

            exploration_steps = int(self.exploration_frac * timesteps_per_iteration)
            print(
                f"- Collecting {timesteps_per_iteration} agent + "
                f"{exploration_steps} exploration transitions"
            )
            self.trajectories = self.sample_rollout(timesteps_per_iteration, exploration_steps)
            self._collect_feedback(num_queries)

            self._log_expert_imitation_errors()
            all_transitions = [transition for traj in self.trajectories for transition in traj]
            self._log_validation_snapshot(all_transitions, "pre_update")

            print("- Training hybrid reward model")
            self._train_reward_model()
            self._update_agent_reward_normalization()
            self._log_validation_snapshot(all_transitions, "post_update")
            self._log_outcome_returns()
            self._log_replay_reward_staleness()

            should_log_scatter = self.debug_dataset and scatter_interval > 0 and (
                iteration % scatter_interval == 0 or iteration == n_iterations - 1
            )
            if should_log_scatter:
                self._log_return_scatter(
                    self._debug_trajectories,
                    "reward_val/debug_dataset/post_update",
                    iteration,
                )

            self._refresh_replay_relabel_cache()
            print(f"- Training agent for {timesteps_per_iteration} timesteps")
            self.train_agent(timesteps_per_iteration, log_interval)

            self.log_iteration(t_iter)
            if checkpoint_dir is not None and (iteration + 1) % checkpoint_interval == 0:
                self.save_checkpoint(checkpoint_dir, iteration + 1)

        return self.trajectory_generator.agent

    def _refresh_replay_relabel_cache(self) -> None:
        """Relabel the replay buffer once per iteration (the model is frozen during learn).

        Must run after ``_update_agent_reward_normalization``: cached rewards
        use the final normalization statistics for this iteration.
        """
        if not self.relabel_rewards:
            return
        replay_buffer = getattr(self.agent, "replay_buffer", None)
        if replay_buffer is not None and hasattr(replay_buffer, "refresh_relabel_cache"):
            replay_buffer.refresh_relabel_cache()

    def _save_checkpoint_extras(self, ckpt_path: str, iteration: int) -> None:
        """Persist reward-training state, the replay buffer and the datasets."""
        th.save(
            {
                "iteration": iteration,
                "loss_type": self.loss_type,
                "relabel_rewards": self.relabel_rewards,
                "normalize_agent_reward": self.normalize_agent_reward,
                "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
            },
            os.path.join(ckpt_path, "reward_training.pt"),
        )
        agent = self.trajectory_generator.agent
        if hasattr(agent, "save_replay_buffer"):
            agent.save_replay_buffer(os.path.join(ckpt_path, "replay_buffer.pkl"))
        th.save(
            {
                "iteration": iteration,
                "demo_mode": self.demo_mode,
                "demo_weight": self.demo_weight,
                "preference_fragment_length": self.preference_fragment_length,
                "dataset_train": self.dataset_train,
                "dataset_demo_prefs_train": self.dataset_demo_prefs_train,
            },
            os.path.join(ckpt_path, "hybrid_training.pt"),
        )
