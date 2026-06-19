"""Demonstration-based reward learning via MaxEnt IRL.

A fixed set of expert trajectories is supplied at construction. Each iteration the
agent's fresh rollout serves as the model trajectories, the reward model is updated
against the expert set, and the agent is trained on the updated reward.

Historical losses are kept for reproducibility, with corrected variants available
as separate configuration choices:

    maxent             historical model-only partition surrogate
    maxent_2           historical expert+model partition surrogate
    demo / demo_loss    historical difference-of-means loss
    maxent_corrected   importance-corrected MaxEnt negative log-likelihood
    demo_corrected     bounded ranking loss on mean trajectory rewards

For MaxEnt, model trajectories retain their sampling log-probability so the
partition estimate can correct for the current policy proposal distribution.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Optional

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from scipy.stats import spearmanr

from human_feedback_rl.common.base_algorithm import BaseAlgorithm
from human_feedback_rl.common.loggers import ExcludeFormatLogger, PrefixedLogger
from human_feedback_rl.common.reward_nets import make_reward_ensemble
from human_feedback_rl.common.trajectory_generators import (
    TrajectoryGeneratorFromAgent,
    policy_action_log_probs,
)
from human_feedback_rl.common.types import Trajectory


class DemoAlgorithm(BaseAlgorithm):
    """Alternating reward-learning (MaxEnt IRL) and agent-training loop."""

    STATUS_ARRIVED  = 0
    STATUS_COLLIDED = 1
    STATUS_OFFROAD  = 2
    STATUS_TIMEOUT  = 3
    STATUS_RUNNING  = 4

    VALID_LOSSES = (
        "maxent",
        "maxent_2",
        "demo",
        "demo_loss",
        "maxent_corrected",
        "demo_corrected",
    )

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
        if gradient_steps_rew <= 0:
            raise ValueError("gradient_steps_rew must be positive.")
        if batch_size_expert <= 0 or batch_size_model <= 0:
            raise ValueError("Reward-model batch sizes must be positive.")

        super().__init__(env, agent, rng, log_folder=log_folder, output_formats=output_formats)

        self.expert_trajectories     = list(expert_trajectories)
        self.loss_type               = loss_type
        self.gradient_steps_rew      = gradient_steps_rew
        self.batch_size_expert       = batch_size_expert
        self.batch_size_model        = batch_size_model
        self.initial_agent_timesteps = initial_agent_timesteps
        self.exploration_frac        = exploration_frac
        self.temperature             = temperature
        self.relabel_rewards         = relabel_rewards
        self.trajectories            = []
        self.debug_dataset           = debug_dataset or {}
        self._debug_rng              = np.random.default_rng(0)

        # The debug dataset is a flat, sequentially-ordered list of transitions;
        # split it on `done` so we can compute return-level ranking metrics on it.
        self._debug_trajectories     = self._split_into_trajectories(self.debug_dataset)

        self.reward_model = make_reward_ensemble(env, **(reward_model_kwargs or {}))

        agent.set_logger(ExcludeFormatLogger(PrefixedLogger(self.logger, "agent"), exclude="stdout"))
        self.trajectory_generator = TrajectoryGeneratorFromAgent(
            venv=env,
            agent=agent,
            reward_model=self.reward_model,
            exploration_eps=exploration_eps,
            rng=self.rng,
            logger=self.logger,
            sampling_venv=rollout_env,
        )

        replay_buffer = getattr(agent, "replay_buffer", None)
        if replay_buffer is not None:
            if hasattr(replay_buffer, "set_reward_model"):
                replay_buffer.set_reward_model(self.reward_model)
                replay_buffer.set_relabel_rewards(relabel_rewards)
            elif relabel_rewards:
                raise ValueError(
                    "relabel_rewards=True requires RewardRelabelReplayBuffer."
                )

        self.optimizers = [
            th.optim.Adam(m.parameters(), lr=lr_rew, weight_decay=l2_rew)
            for m in self.reward_model.members
        ]

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(
        self,
        total_timesteps: int = 1_000_000,
        timesteps_per_iteration: int = 1024,
        log_interval: int = 1,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 10,
    ) -> Any:
        """Run the full alternating reward-learning + agent-training loop."""
        n_iterations = int(total_timesteps / timesteps_per_iteration)

        # Bootstrap the reward before the agent sees it. This avoids optimizing
        # an initially random reward and poisoning an off-policy replay buffer.
        if self.initial_agent_timesteps > 0:
            print(f"- Collecting {self.initial_agent_timesteps} bootstrap transitions")
            self.trajectories = self._sample_rollout(self.initial_agent_timesteps)
            print("- Bootstrapping reward model")
            self._train_reward_model()
            print(f"- Pre-warming agent for {self.initial_agent_timesteps} timesteps on learned reward")
            self._train_agent(self.initial_agent_timesteps, log_interval)

        for iteration in range(n_iterations):
            t_iter = time.perf_counter()
            print(f"\nIteration {iteration}/{n_iterations - 1}")

            exploration_steps = int(self.exploration_frac * timesteps_per_iteration)
            print(f"- Collecting {timesteps_per_iteration} agent + {exploration_steps} exploration transitions")
            self.trajectories = self._sample_rollout(timesteps_per_iteration, exploration_steps)

            all_transitions = [t for traj in self.trajectories for t in traj]
            self._log_validation_snapshot(all_transitions, "pre_update")

            print("- Training reward model")
            self._train_reward_model()
            self._log_validation_snapshot(all_transitions, "post_update")

            self._log_replay_reward_staleness()

            print(f"- Training agent for {timesteps_per_iteration} timesteps")
            self._train_agent(timesteps_per_iteration, log_interval)

            self._log_iteration(t_iter, iteration)
            if checkpoint_dir is not None and (iteration + 1) % checkpoint_interval == 0:
                self._save_checkpoint(checkpoint_dir, iteration + 1)

        return self.trajectory_generator.agent

    # ------------------------------------------------------------------
    # Reward model training (MaxEnt IRL)
    # ------------------------------------------------------------------

    @staticmethod
    def _grad_norm(module) -> float:
        """Total L2 norm of the module's current parameter gradients."""
        squared_norms = [
            p.grad.pow(2).sum() for p in module.parameters() if p.grad is not None
        ]
        if not squared_norms:
            return 0.0
        return float(th.sqrt(th.stack(squared_norms).sum()))

    def _train_reward_model(self) -> None:
        if not self.trajectories:
            return

        t0 = time.perf_counter()

        def train_member(member, optimizer):
            member.train()
            norms = []
            for _ in range(self.gradient_steps_rew):
                loss = self._reward_loss(member)
                if not th.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite reward loss for loss_type={self.loss_type}: {loss.item()}"
                    )
                optimizer.zero_grad()
                loss.backward()
                grad_norm = self._grad_norm(member)
                if not np.isfinite(grad_norm):
                    raise FloatingPointError(
                        f"Non-finite reward gradient norm for loss_type={self.loss_type}."
                    )
                norms.append(grad_norm)
                optimizer.step()
            return norms

        all_norms = []
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = [
                executor.submit(train_member, member, optimizer)
                for member, optimizer in zip(self.reward_model.members, self.optimizers)
            ]
            for future in as_completed(futures):
                all_norms.extend(future.result())

        t_train = time.perf_counter() - t0

        t0 = time.perf_counter()
        self._log_reward_loss_diagnostics()
        self.logger.record("reward/grad_norm",        float(np.mean(all_norms)), exclude="stdout")
        self.logger.record("reward/grad_norm_max",    float(np.max(all_norms)), exclude="stdout")
        self.logger.record("reward/weight_norm",      self._param_norm(self.reward_model), exclude="stdout")
        self.logger.record("time/train_reward_model", t_train)
        self.logger.record_sum("time/loggings",       time.perf_counter() - t0)

    @staticmethod
    def _param_norm(module) -> float:
        """Total L2 norm of model parameters."""
        return float(th.sqrt(sum(p.detach().pow(2).sum() for p in module.parameters())))

    def _diagnostic_returns(self, member):
        """Deterministic expert/model trajectory returns for reward debugging."""
        expert_trajs = self._even_subset(self.expert_trajectories, self.batch_size_expert)
        model_trajs = self._even_subset(self.trajectories, self.batch_size_model)
        expert_returns = th.stack([self._traj_sum_reward(member, traj) for traj in expert_trajs])
        model_returns = th.stack([self._traj_sum_reward(member, traj) for traj in model_trajs])
        return expert_returns, model_returns, expert_trajs, model_trajs

    @staticmethod
    def _even_subset(items, max_items: int):
        """Return a deterministic, evenly-spaced subset without advancing RNG state."""
        n_items = len(items)
        if max_items <= 0:
            max_items = 1
        if n_items <= max_items:
            return list(items)
        indices = np.linspace(0, n_items - 1, max_items, dtype=int)
        return [items[i] for i in indices]

    def _log_reward_loss_diagnostics(self) -> None:
        """Log loss-specific reward diagnostics that are meaningful for IRL."""
        if not self.trajectories:
            return

        self.reward_model.eval()
        with th.no_grad():
            expert_returns, model_returns, expert_trajs, model_trajs = self._diagnostic_returns(
                self.reward_model
            )
            expert_term = -expert_returns.mean()
            model_mean = model_returns.mean()

            all_returns = th.cat([expert_returns, model_returns], dim=0)
            if not th.isfinite(all_returns).all():
                raise FloatingPointError("Non-finite trajectory returns in reward diagnostics.")
            margin = expert_returns.mean() - model_mean
            abs_mean = all_returns.abs().mean()

            self.logger.record("reward/expert_return_mean", float(expert_returns.mean()), exclude="stdout")
            self.logger.record("reward/model_return_mean",  float(model_mean), exclude="stdout")
            self.logger.record("reward/expert_model_margin", float(margin), exclude="stdout")
            self.logger.record("reward/return_std",         float(all_returns.std(unbiased=False)), exclude="stdout")
            self.logger.record("reward/return_abs_mean",    float(abs_mean), exclude="stdout")
            self.logger.record("reward/return_min",         float(all_returns.min()), exclude="stdout")
            self.logger.record("reward/return_max",         float(all_returns.max()), exclude="stdout")

            if self.loss_type in ("demo", "demo_loss"):
                loss = expert_term + model_mean
                self.logger.record("reward/loss",            float(loss), exclude="stdout")
                self.logger.record("reward/demo_margin",     float(margin), exclude="stdout")
                self.logger.record("reward/demo_scale_std",  float(all_returns.std(unbiased=False)), exclude="stdout")
                self.logger.record("reward/demo_scale_abs",  float(abs_mean), exclude="stdout")
                self.reward_model.train()
                return

            if self.loss_type == "demo_corrected":
                demo_margin = self._demo_corrected_margins(
                    expert_returns, model_returns, expert_trajs, model_trajs
                )
                loss = F.softplus(-demo_margin / self.temperature).mean()
                self.logger.record("reward/loss",                    float(loss), exclude="stdout")
                self.logger.record("reward/demo_corrected_margin",   float(demo_margin.mean()), exclude="stdout")
                self.logger.record("reward/demo_corrected_scale_std", float(all_returns.std(unbiased=False)), exclude="stdout")
                self.reward_model.train()
                return

            if self.loss_type == "maxent_2":
                partition = th.logsumexp(all_returns, dim=0) - np.log(len(all_returns))
                weights = th.softmax(all_returns, dim=0)
                expert_mass = weights[:len(expert_returns)].sum()
                model_mass = weights[len(expert_returns):].sum()
                top1_weight, top1_idx = weights.max(dim=0)
                ess = 1.0 / weights.pow(2).sum()
                self.logger.record("reward/loss",                          float(expert_term + partition), exclude="stdout")
                self.logger.record("reward/maxent2_partition_all",         float(partition), exclude="stdout")
                self.logger.record("reward/maxent2_expert_softmax_mass",   float(expert_mass), exclude="stdout")
                self.logger.record("reward/maxent2_model_softmax_mass",    float(model_mass), exclude="stdout")
                self.logger.record("reward/maxent2_top1_softmax_weight",   float(top1_weight), exclude="stdout")
                self.logger.record("reward/maxent2_top1_is_expert",        float((top1_idx < len(expert_returns)).item()), exclude="stdout")
                self.logger.record("reward/maxent2_effective_sample_size", float(ess), exclude="stdout")
                self._log_ess_fraction("reward/maxent2", ess, len(weights))
                self.reward_model.train()
                return

            log_q = None
            partition_logits = model_returns
            log_prefix = "reward/maxent"
            loss_expert_term = expert_term
            if self.loss_type == "maxent_corrected":
                log_q = th.tensor(
                    [self._traj_log_policy_prob(traj) for traj in model_trajs],
                    dtype=th.float32,
                )
                partition_logits = model_returns / self.temperature - log_q
                log_prefix = "reward/maxent_corrected"
                loss_expert_term = expert_term / self.temperature

            partition = th.logsumexp(partition_logits, dim=0) - np.log(len(model_returns))
            weights = th.softmax(partition_logits, dim=0)
            top_values = th.topk(weights, k=min(5, len(weights))).values
            ess = 1.0 / weights.pow(2).sum()
            loss = loss_expert_term + partition

            self.logger.record("reward/loss",                         float(loss), exclude="stdout")
            self.logger.record(f"{log_prefix}_partition_model",       float(partition), exclude="stdout")
            self.logger.record(f"{log_prefix}_top1_softmax_weight",   float(weights.max()), exclude="stdout")
            self.logger.record(f"{log_prefix}_top5_softmax_mass",     float(top_values.sum()), exclude="stdout")
            self.logger.record(f"{log_prefix}_effective_sample_size", float(ess), exclude="stdout")
            self._log_ess_fraction(log_prefix, ess, len(weights))
            if log_q is not None:
                self.logger.record(f"{log_prefix}_log_q_mean", float(log_q.mean()), exclude="stdout")
        self.reward_model.train()

    def _log_ess_fraction(self, log_prefix: str, ess: th.Tensor, n_items: int) -> None:
        ess_fraction = float(ess) / n_items
        self.logger.record(
            f"{log_prefix}_effective_sample_fraction", ess_fraction, exclude="stdout"
        )
        if ess_fraction < 0.1:
            self.logger.warn(
                f"Low effective sample fraction for {self.loss_type}: {ess_fraction:.3f}"
            )

    def _sample_returns(self, member):
        """Sample trajectories and compute differentiable reward returns."""
        n_e     = min(self.batch_size_expert, len(self.expert_trajectories))
        exp_idx = self.rng.choice(len(self.expert_trajectories), size=n_e, replace=False)
        expert_trajs = [self.expert_trajectories[i] for i in exp_idx]
        expert_returns = th.stack([
            self._traj_sum_reward(member, traj) for traj in expert_trajs
        ])

        n_m       = min(self.batch_size_model, len(self.trajectories))
        model_idx = self.rng.choice(len(self.trajectories), size=n_m, replace=False)
        model_trajs = [self.trajectories[i] for i in model_idx]
        model_returns = th.stack([
            self._traj_sum_reward(member, traj) for traj in model_trajs
        ])
        return expert_returns, model_returns, expert_trajs, model_trajs

    def _reward_loss(self, member) -> th.Tensor:
        """IRL loss selected by ``self.loss_type``.

        Historical losses preserve their original formulas. Corrected variants
        add proposal correction or a bounded, length-normalized ranking objective.
        """
        expert_returns, model_returns, expert_trajs, model_trajs = self._sample_returns(member)

        expert_term = -expert_returns.mean()
        if self.loss_type in ("demo", "demo_loss"):
            return expert_term + model_returns.mean()

        if self.loss_type == "demo_corrected":
            margins = self._demo_corrected_margins(
                expert_returns, model_returns, expert_trajs, model_trajs
            )
            return F.softplus(-margins / self.temperature).mean()

        if self.loss_type == "maxent_2":
            all_returns = th.cat([model_returns, expert_returns], dim=0)
            return expert_term + th.logsumexp(all_returns, dim=0) - np.log(len(all_returns))

        if self.loss_type == "maxent":
            return expert_term + th.logsumexp(model_returns, dim=0) - np.log(len(model_returns))

        log_q = th.tensor(
            [self._traj_log_policy_prob(traj) for traj in model_trajs], dtype=th.float32
        )
        corrected_logits = model_returns / self.temperature - log_q
        partition = th.logsumexp(corrected_logits, dim=0) - np.log(len(model_returns))
        return -expert_returns.mean() / self.temperature + partition

    @staticmethod
    def _demo_corrected_margins(expert_returns, model_returns, expert_trajs, model_trajs):
        n_pairs = min(len(expert_returns), len(model_returns))
        expert_scores = th.stack([
            expert_returns[i] / len(expert_trajs[i]) for i in range(n_pairs)
        ])
        model_scores = th.stack([
            model_returns[i] / len(model_trajs[i]) for i in range(n_pairs)
        ])
        return expert_scores - model_scores

    def _traj_log_policy_prob(self, traj: Trajectory) -> float:
        stored = [getattr(t, "log_policy_prob", None) for t in traj]
        if all(value is not None for value in stored):
            return float(sum(stored))

        obs = np.asarray([t.observation for t in traj], dtype=np.float32)
        actions = np.asarray([t.action for t in traj])
        return float(policy_action_log_probs(self.agent, obs, actions).sum())

    def _traj_sum_reward(self, member, traj: Trajectory) -> th.Tensor:
        """Sum of per-step rewards over a trajectory (supports gradients)."""
        obs         = th.tensor(np.array([t.observation for t in traj]),  dtype=th.float32)
        actions     = th.tensor(np.array([t.action      for t in traj]),  dtype=th.float32)
        next_status = th.tensor(np.array([t.next_status for t in traj]),  dtype=th.float32)
        done        = th.tensor(np.array([float(t.done) for t in traj]),  dtype=th.float32)
        return member(obs, actions, next_status, done).sum()

    def _log_replay_reward_staleness(self, batch_size: int = 2048) -> None:
        """Compare stored and current rewards on actual replay-buffer entries."""
        replay_buffer = getattr(self.agent, "replay_buffer", None)
        if replay_buffer is None or not hasattr(replay_buffer, "sample_reward_staleness"):
            return

        batch = replay_buffer.sample_reward_staleness(batch_size, self._debug_rng)
        if batch is None:
            return

        stored_rewards, current_rewards = batch
        delta = current_rewards - stored_rewards
        abs_delta = np.abs(delta)
        stored_std = float(np.std(stored_rewards))
        current_std = float(np.std(current_rewards))
        denom = current_std if current_std > 1e-8 else 1.0

        self.logger.record("replay_relabel_debug/sample_size",          len(stored_rewards), exclude="stdout")
        self.logger.record("replay_relabel_debug/stored_reward_mean",   float(np.mean(stored_rewards)), exclude="stdout")
        self.logger.record("replay_relabel_debug/current_reward_mean",  float(np.mean(current_rewards)), exclude="stdout")
        self.logger.record("replay_relabel_debug/stored_reward_std",    stored_std, exclude="stdout")
        self.logger.record("replay_relabel_debug/current_reward_std",   current_std, exclude="stdout")
        self.logger.record("replay_relabel_debug/delta_mean",           float(np.mean(delta)), exclude="stdout")
        self.logger.record("replay_relabel_debug/delta_std",            float(np.std(delta)), exclude="stdout")
        self.logger.record("replay_relabel_debug/delta_abs_mean",       float(np.mean(abs_delta)), exclude="stdout")
        self.logger.record("replay_relabel_debug/delta_abs_p95",        float(np.percentile(abs_delta, 95)), exclude="stdout")
        self.logger.record("replay_relabel_debug/staleness_ratio",      float(np.mean(abs_delta) / denom), exclude="stdout")
        self.logger.record("replay_relabel_debug/sign_flip_frac",       float(np.mean(np.sign(stored_rewards) != np.sign(current_rewards))), exclude="stdout")
        relabel_enabled = float(getattr(replay_buffer, "relabel_rewards", False))
        self.logger.record("replay_relabel_debug/relabel_enabled",       relabel_enabled, exclude="stdout")
        self.logger.record("replay_relabel_debug/critic_uses_current_reward", relabel_enabled, exclude="stdout")

        if stored_std > 1e-8 and current_std > 1e-8:
            corr = np.corrcoef(stored_rewards, current_rewards)[0, 1]
            self.logger.record("replay_relabel_debug/stored_current_corr", float(corr), exclude="stdout")

    # ------------------------------------------------------------------
    # Rollout collection and agent training
    # ------------------------------------------------------------------

    def _sample_rollout(self, agent_steps: int, exploration_steps: int = 0) -> list:
        t0 = time.perf_counter()
        trajectories = self.trajectory_generator.sample(agent_steps, exploration_steps)
        t_sample = time.perf_counter() - t0

        t0 = time.perf_counter()
        true_rewards  = [traj.total_reward()          for traj in trajectories]
        model_rewards = [self._score_trajectory(traj) for traj in trajectories]
        lengths       = [len(traj)                    for traj in trajectories]

        self.logger.record("rollout/mean_true_reward",  float(np.mean(true_rewards)))
        self.logger.record("rollout/mean_model_reward", float(np.mean(model_rewards)))
        self.logger.record("rollout/mean_length",       float(np.mean(lengths)))
        self._log_action_boundaries(trajectories)
        self.logger.record("time/sample_rollout",       t_sample)
        self.logger.record_sum("time/loggings",         time.perf_counter() - t0)

        return trajectories

    def _log_action_boundaries(self, trajectories) -> None:
        """Log how often continuous actions land exactly on environment bounds."""
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
        obs    = np.array([t.observation for t in traj])
        acts   = np.array([t.action      for t in traj])
        status = np.array([t.next_status for t in traj])
        done   = np.array([float(t.done) for t in traj])
        return self.reward_model.predict(obs, acts, status, done).sum()

    def _train_agent(self, steps: int, log_interval: int) -> None:
        t0 = time.perf_counter()
        self.trajectory_generator.train(steps=steps, log_interval=log_interval)
        self.logger.record("time/train_agent", time.perf_counter() - t0)

    # ------------------------------------------------------------------
    # Reward model validation logging
    # ------------------------------------------------------------------

    def _log_validation_snapshot(self, rollout_transitions, stage: str) -> None:
        rollout_prefix = f"reward_val/current_rollout/{stage}"
        self._log_reward_validation(rollout_transitions, rollout_prefix)
        self._log_return_ranking(self.trajectories, rollout_prefix)

        if self.debug_dataset:
            debug_prefix = f"reward_val/debug_dataset/{stage}"
            self._log_reward_validation(self.debug_dataset, debug_prefix)
            self._log_return_ranking(self._debug_trajectories, debug_prefix)

    def _log_reward_validation(self, transitions, log_class: str) -> None:
        """Log reward scale, status shortcuts, and ensemble disagreement."""
        _, pred_rewards, pred_std, status = self._run_reward_inference(transitions)

        arrived_mask  = status[:, self.STATUS_ARRIVED]  == 1
        collided_mask = status[:, self.STATUS_COLLIDED] == 1
        offroad_mask  = status[:, self.STATUS_OFFROAD]  == 1
        timeout_mask  = status[:, self.STATUS_TIMEOUT]  == 1
        running_mask  = status[:, self.STATUS_RUNNING]  == 1

        self.logger.record(f"{log_class}/reward_mean", float(np.mean(pred_rewards)))
        self.logger.record(f"{log_class}/reward_std",  float(np.std(pred_rewards)))
        self.logger.record(f"{log_class}/reward_min",  float(np.min(pred_rewards)))
        self.logger.record(f"{log_class}/reward_max",  float(np.max(pred_rewards)))

        running_mean = self._record_masked_mean(
            f"{log_class}/reward_running", pred_rewards, running_mask
        )
        arrived_mean = self._record_masked_mean(
            f"{log_class}/reward_arrived", pred_rewards, arrived_mask
        )
        collided_mean = self._record_masked_mean(
            f"{log_class}/reward_collided", pred_rewards, collided_mask
        )
        self._record_masked_mean(f"{log_class}/reward_offroad",  pred_rewards, offroad_mask)
        self._record_masked_mean(f"{log_class}/reward_timeout",  pred_rewards, timeout_mask)

        if arrived_mean is not None and collided_mean is not None:
            self.logger.record(
                f"{log_class}/gap_arrived_collided", arrived_mean - collided_mean
            )
        if arrived_mean is not None and running_mean is not None:
            self.logger.record(
                f"{log_class}/gap_arrived_running", arrived_mean - running_mean
            )

        # Rising ensemble disagreement signals inputs that are becoming OOD for the reward model.
        self.logger.record(f"{log_class}/ensemble_std", float(np.mean(pred_std)))
        self._record_masked_mean(f"{log_class}/ensemble_std_running", pred_std, running_mask)

    def _record_masked_mean(
        self, key: str, values: np.ndarray, mask: np.ndarray
    ) -> Optional[float]:
        if np.any(mask):
            mean = float(np.mean(values[mask]))
            self.logger.record(key, mean)
            return mean
        return None

    def _run_reward_inference(self, transitions):
        """Run the reward model in eval+no_grad mode; return (true, pred_mean, pred_std, status)."""
        self.reward_model.eval()
        with th.no_grad():
            true_rewards = np.array([t.true_reward for t in transitions], dtype=np.float32)
            obs          = np.array([t.observation  for t in transitions], dtype=np.float32)
            acts         = np.array([t.action       for t in transitions], dtype=np.float32)
            status       = np.array([t.next_status  for t in transitions], dtype=np.float32)
            done         = np.array([float(t.done)  for t in transitions], dtype=np.float32)
            pred_rewards = self.reward_model.predict(obs, acts, status, done)
            # Per-member predictions (N, n_members); std across members = disagreement.
            pred_std     = self.reward_model.predict_all(obs, acts, status, done).std(axis=1)
            if not np.isfinite(pred_rewards).all() or not np.isfinite(pred_std).all():
                raise FloatingPointError("Non-finite reward-model validation predictions.")
        self.reward_model.train()
        return true_rewards, pred_rewards, pred_std, status

    @staticmethod
    def _split_into_trajectories(transitions) -> List[Trajectory]:
        """Split a flat, sequentially-ordered transition list into trajectories on `done`."""
        trajectories, current = [], Trajectory()
        for t in transitions:
            current.add_transition(t)
            if t.done:
                trajectories.append(current)
                current = Trajectory()
        if len(current) > 0:
            trajectories.append(current)
        return trajectories

    def _log_return_ranking(self, trajectories, log_class: str) -> None:
        """Spearman rho between predicted and true trajectory returns (ranking quality)."""
        if len(trajectories) < 2:
            return
        true_returns, pred_returns = [], []
        for traj in trajectories:
            _, pred_rewards, _, _ = self._run_reward_inference(traj)
            true_returns.append(float(sum(t.true_reward for t in traj)))
            pred_returns.append(float(pred_rewards.sum()))
        rho, _ = spearmanr(true_returns, pred_returns)
        is_defined = float(np.isfinite(rho))
        self.logger.record(f"{log_class}/spearman_returns_defined", is_defined)
        if is_defined:
            self.logger.record(f"{log_class}/spearman_returns", float(rho))

    # ------------------------------------------------------------------
    # Logging and checkpointing
    # ------------------------------------------------------------------

    def _log_iteration(self, t_iter: float, iteration: int) -> None:
        t_log = time.perf_counter()
        self.logger.record("iterations",                 iteration)
        self.logger.record("agent/time/total_timesteps", self.agent.num_timesteps)
        self.logger.record("time/total",                 time.perf_counter() - t_iter)
        self.logger.record_sum("time/loggings",          time.perf_counter() - t_log)
        self.logger.dump()

    def _save_checkpoint(self, checkpoint_dir: str, iteration: int) -> None:
        ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{iteration:04d}")
        os.makedirs(ckpt_path, exist_ok=True)
        th.save(self.reward_model.state_dict(), os.path.join(ckpt_path, "reward_model.pt"))
        th.save(
            {
                "iteration": iteration,
                "loss_type": self.loss_type,
                "temperature": self.temperature,
                "relabel_rewards": self.relabel_rewards,
                "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
            },
            os.path.join(ckpt_path, "reward_training.pt"),
        )
        self.trajectory_generator.agent.save(os.path.join(ckpt_path, "agent"))
        if hasattr(self.trajectory_generator.agent, "save_replay_buffer"):
            self.trajectory_generator.agent.save_replay_buffer(
                os.path.join(ckpt_path, "replay_buffer.pkl")
            )
        print(f"  checkpoint saved in {ckpt_path}")
