"""Demonstration-based reward learning via MaxEnt IRL.

``DemoAlgorithm`` is the public facade and owns the alternating training loop.
Loss, reward-training, imitation-metric, and diagnostic methods live in the
focused mixins under :mod:`human_feedback_rl.algorithms.demo`; rollout,
checkpointing and validation plumbing comes from the shared
:class:`BaseRewardLearningAlgorithm`.

Historical losses are preserved as separate configuration choices:

    maxent             historical model-only partition surrogate
    maxent_2           historical expert+model partition surrogate
    demo / demo_loss    historical difference-of-means loss
    maxent_corrected   importance-corrected MaxEnt negative log-likelihood
    demo_corrected     bounded ranking loss on mean trajectory rewards
"""

import os
import time
from typing import Any, List, Optional

import numpy as np
import torch as th

from human_feedback_rl.algorithms.demo.imitation_metrics import ImitationMetricsMixin
from human_feedback_rl.algorithms.demo.losses import (
    VALID_LOSSES as DEMO_VALID_LOSSES,
    RewardLossMixin,
)
from human_feedback_rl.algorithms.demo.reward_diagnostics import RewardDiagnosticsMixin
from human_feedback_rl.algorithms.demo.reward_training import RewardTrainingMixin
from human_feedback_rl.common.base_reward_learning_algorithm import BaseRewardLearningAlgorithm
from human_feedback_rl.common.reward_nets import make_reward_ensemble
from human_feedback_rl.common.types import Trajectory


class DemoAlgorithm(
    RewardLossMixin,
    RewardTrainingMixin,
    ImitationMetricsMixin,
    RewardDiagnosticsMixin,
    BaseRewardLearningAlgorithm,
):
    """Alternating reward-learning (MaxEnt IRL) and agent-training loop."""

    VALID_LOSSES = DEMO_VALID_LOSSES

    def __init__(
        self,
        env,
        agent,
        expert_trajectories: List[Trajectory],
        loss_type: str = "maxent",
        lr_rew: float = 0.001,
        gradient_steps_rew: int = 10,
        batch_size_expert: int = 32,
        batch_size_model: int = 64,
        l2_rew: float = 0.01,
        temperature: float = 1.0,
        fragment_length: Optional[int] = None,
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
        if not expert_trajectories:
            raise ValueError("expert_trajectories must be a non-empty list of Trajectory objects.")
        if loss_type not in self.VALID_LOSSES:
            raise ValueError(f"loss_type must be one of {self.VALID_LOSSES}, got {loss_type!r}.")
        if loss_type == "maxent_corrected" and rollout_env is None:
            raise ValueError(
                "maxent_corrected requires a dedicated rollout_env so trajectories come from "
                "one fixed proposal policy and do not desynchronize the SB3 training env."
            )
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if fragment_length is not None and (
            isinstance(fragment_length, (bool, np.bool_))
            or not isinstance(fragment_length, (int, np.integer))
            or fragment_length <= 0
        ):
            raise ValueError("fragment_length must be a positive integer or None.")
        if gradient_steps_rew <= 0:
            raise ValueError("gradient_steps_rew must be positive.")
        if batch_size_expert <= 0 or batch_size_model <= 0:
            raise ValueError("Reward-model batch sizes must be positive.")

        reward_model = make_reward_ensemble(env, **(reward_model_kwargs or {}))

        super().__init__(
            env=env,
            agent=agent,
            reward_model=reward_model,
            fragment_length=None if fragment_length is None else int(fragment_length),
            exploration_frac=exploration_frac,
            exploration_eps=exploration_eps,
            temperature=temperature,
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
        # Per-gradient-step diagnostics for maxent_corrected (populated by the loss).
        self._maxent_corrected_steps = []
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

    def train(
        self,
        total_timesteps: int = 1_000_000,
        timesteps_per_iteration: int = 1024,
        log_interval: int = 1,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 10,
        scatter_interval: Optional[int] = None,
    ) -> Any:
        """Run the full alternating reward-learning and agent-training loop."""
        if scatter_interval is None:
            scatter_interval = 10
        if scatter_interval < 0:
            raise ValueError("scatter_interval must be non-negative.")
        n_iterations = int(total_timesteps / timesteps_per_iteration)

        if self.initial_agent_timesteps > 0:
            print(f"- Collecting {self.initial_agent_timesteps} bootstrap transitions")
            self.trajectories = self.sample_rollout(self.initial_agent_timesteps)
            print("- Bootstrapping reward model")
            self._train_reward_model()
            self._update_agent_reward_normalization()
            print(f"- Pre-warming agent for {self.initial_agent_timesteps} timesteps on learned reward")
            self.train_agent(self.initial_agent_timesteps, log_interval)

        for iteration in range(n_iterations):
            self.iteration = iteration
            t_iter = time.perf_counter()
            print(f"\nIteration {iteration}/{n_iterations - 1}")

            exploration_steps = int(self.exploration_frac * timesteps_per_iteration)
            print(
                f"- Collecting {timesteps_per_iteration} agent + "
                f"{exploration_steps} exploration transitions"
            )
            self.trajectories = self.sample_rollout(
                timesteps_per_iteration, exploration_steps
            )

            self._log_expert_imitation_errors()

            all_transitions = [transition for traj in self.trajectories for transition in traj]
            self._log_validation_snapshot(all_transitions, "pre_update")

            print("- Training reward model")
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

            print(f"- Training agent for {timesteps_per_iteration} timesteps")
            self.train_agent(timesteps_per_iteration, log_interval)

            self.log_iteration(t_iter)
            if checkpoint_dir is not None and (iteration + 1) % checkpoint_interval == 0:
                self.save_checkpoint(checkpoint_dir, iteration + 1)

        return self.trajectory_generator.agent

    def _save_checkpoint_extras(self, ckpt_path: str, iteration: int) -> None:
        """Persist reward-training state and the replay buffer next to the checkpoint."""
        th.save(
            {
                "iteration": iteration,
                "loss_type": self.loss_type,
                "temperature": self.temperature,
                "relabel_rewards": self.relabel_rewards,
                "normalize_agent_reward": self.normalize_agent_reward,
                "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
            },
            os.path.join(ckpt_path, "reward_training.pt"),
        )
        agent = self.trajectory_generator.agent
        if hasattr(agent, "save_replay_buffer"):
            agent.save_replay_buffer(os.path.join(ckpt_path, "replay_buffer.pkl"))
