"""Guided Cost Learning for SUMO.

This implementation follows the maximum-entropy IOC objective from
Finn, Levine and Abbeel (2016), while replacing the paper's local
linear-Gaussian policy optimizer with an SB3 policy optimizer suitable for the
existing SUMO codebase.
"""

from __future__ import annotations

import math
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch as th

from human_feedback_rl.common.base_algorithm import BaseAlgorithm
from human_feedback_rl.common.gcl import (
    GuidedCostNet,
    StepGaussianTrajectoryDistribution,
    finite_mean,
    flatten_trajectories,
    fusion_log_prob,
    local_constant_rate_regularizer,
    minibatch,
    monotonic_regularizer,
    reduce_trajectory_costs,
    trajectory_step_costs,
)
from human_feedback_rl.common.loggers import ExcludeFormatLogger, PrefixedLogger
from human_feedback_rl.common.trajectory_generators import TrajectoryGeneratorFromAgent
from human_feedback_rl.common.types import Trajectory


@dataclass
class CostTrainingStats:
    loss: float
    ioc_loss: float
    demo_cost: float
    sample_cost: float
    log_partition: float
    lcr_regularizer: float
    monotonic_regularizer: float
    effective_sample_fraction: float
    top1_importance_weight: float
    expert_softmax_mass: float
    raw_log_term_span: float
    partition_log_term_span: float
    partition_clipped_fraction: float
    partition_temperature: float
    grad_norm: float


class GuidedCostLearning(BaseAlgorithm):
    """Sample-based maximum-entropy IOC with SB3-guided sampling.

    The algorithm stores all sampled trajectories in a replay-style background
    set, updates the cost with the sample-based MaxEnt IOC likelihood, then
    trains the policy on the negative learned cost. The learned policy therefore
    plays the role of the adaptive sampler ``q_k`` in Algorithm 1 of the paper.
    """

    def __init__(
        self,
        env,
        agent,
        expert_trajectories: Sequence[Trajectory],
        cost_model_kwargs: Optional[dict] = None,
        lr_cost: float = 3e-4,
        weight_decay_cost: float = 1e-4,
        cost_gradient_steps: int = 100,
        demo_batch_size: int = 16,
        sample_batch_size: int = 16,
        trajectory_cost_reduction: str = "sum",
        lcr_reg_weight: float = 1e-3,
        monotonic_reg_weight: float = 1e-4,
        monotonic_margin: float = 1.0,
        importance_mode: str = "fusion_gaussian",
        gaussian_regularization: float = 1e-3,
        max_gaussian_transitions: int = 50_000,
        max_fusion_distributions: int = 12,
        max_sample_trajectories: int = 2_000,
        max_log_weight: Optional[float] = 80.0,
        partition_temperature: float = 1.0,
        max_partition_log_term_span: Optional[float] = None,
        exploration_frac: float = 0.0,
        exploration_eps: float = 0.2,
        expert_rmse_transitions: int = 4096,
        grad_clip_norm: Optional[float] = 10.0,
        rng: Optional[np.random.Generator] = None,
        log_folder: Optional[str] = None,
        output_formats: Optional[List] = None,
    ):
        if not expert_trajectories:
            raise ValueError("GuidedCostLearning requires at least one expert trajectory.")

        super().__init__(env, agent, rng, log_folder=log_folder, output_formats=output_formats)

        self.expert_trajectories = list(expert_trajectories)
        self.cost_model = GuidedCostNet(
            env.observation_space,
            env.action_space,
            **(cost_model_kwargs or {}),
        )
        self.cost_optimizer = th.optim.Adam(
            self.cost_model.parameters(),
            lr=float(lr_cost),
            weight_decay=float(weight_decay_cost),
        )

        self.cost_gradient_steps = int(cost_gradient_steps)
        self.demo_batch_size = int(demo_batch_size)
        self.sample_batch_size = int(sample_batch_size)
        self.trajectory_cost_reduction = trajectory_cost_reduction
        self.lcr_reg_weight = float(lcr_reg_weight)
        self.monotonic_reg_weight = float(monotonic_reg_weight)
        self.monotonic_margin = float(monotonic_margin)
        self.importance_mode = importance_mode
        self.gaussian_regularization = float(gaussian_regularization)
        self.max_gaussian_transitions = int(max_gaussian_transitions)
        self.max_fusion_distributions = int(max_fusion_distributions)
        self.max_sample_trajectories = int(max_sample_trajectories)
        self.max_log_weight = None if max_log_weight is None else float(max_log_weight)
        self.partition_temperature = float(partition_temperature)
        if self.partition_temperature <= 0.0:
            raise ValueError("partition_temperature must be greater than zero.")
        self.max_partition_log_term_span = (
            None
            if max_partition_log_term_span is None
            else float(max_partition_log_term_span)
        )
        if (
            self.max_partition_log_term_span is not None
            and self.max_partition_log_term_span <= 0.0
        ):
            raise ValueError("max_partition_log_term_span must be greater than zero.")
        self.exploration_frac = float(exploration_frac)
        self.exploration_eps = float(exploration_eps)
        self.expert_rmse_transitions = int(expert_rmse_transitions)
        self.grad_clip_norm = None if grad_clip_norm is None else float(grad_clip_norm)
        self.iteration = 0

        self.sample_trajectories: List[Trajectory] = []
        self.sample_distributions: List[StepGaussianTrajectoryDistribution] = []
        self.demo_distribution = StepGaussianTrajectoryDistribution.fit(
            self.expert_trajectories,
            regularization=self.gaussian_regularization,
            max_transitions=self.max_gaussian_transitions,
            rng=self.rng,
            name="expert_demo",
        )

        self.agent_logger = PrefixedLogger(self.logger, "agent")
        self.agent.set_logger(ExcludeFormatLogger(self.agent_logger, exclude="stdout"))
        self.trajectory_generator = TrajectoryGeneratorFromAgent(
            venv=env,
            agent=agent,
            reward_model=self.cost_model,
            exploration_eps=self.exploration_eps,
            rng=self.rng,
            logger=self.logger,
        )

    # ------------------------------------------------------------------
    # Training pieces
    # ------------------------------------------------------------------

    def collect_policy_samples(self, rollout_steps: int) -> List[Trajectory]:
        t0 = time.perf_counter()
        exploration_steps = int(self.exploration_frac * rollout_steps)
        trajectories = list(
            self.trajectory_generator.sample(
                agent_steps=int(rollout_steps),
                exploration_steps=exploration_steps,
            )
        )
        self._append_policy_samples(trajectories)
        self._log_rollout_stats(trajectories, time.perf_counter() - t0)
        return trajectories

    def _append_policy_samples(self, trajectories: Sequence[Trajectory]) -> None:
        if not trajectories:
            return

        self.sample_trajectories.extend(trajectories)
        if self.max_sample_trajectories > 0:
            self.sample_trajectories = self.sample_trajectories[-self.max_sample_trajectories :]

        dist = StepGaussianTrajectoryDistribution.fit(
            trajectories,
            regularization=self.gaussian_regularization,
            max_transitions=self.max_gaussian_transitions,
            rng=self.rng,
            name=f"policy_iter_{self.iteration:04d}",
        )
        self.sample_distributions.append(dist)
        if self.max_fusion_distributions > 0:
            self.sample_distributions = self.sample_distributions[-self.max_fusion_distributions :]

    def train_cost_model(self) -> CostTrainingStats:
        if not self.sample_trajectories:
            raise RuntimeError("No policy samples available for the GCL partition estimate.")

        last_stats = None
        for _ in range(self.cost_gradient_steps):
            demo_batch = minibatch(
                self.expert_trajectories,
                self.demo_batch_size,
                self.rng,
                replace=len(self.expert_trajectories) < self.demo_batch_size,
            )
            sample_batch = minibatch(
                self.sample_trajectories,
                self.sample_batch_size,
                self.rng,
                replace=len(self.sample_trajectories) < self.sample_batch_size,
            )

            loss, diagnostics = self._cost_loss(demo_batch, sample_batch)
            self.cost_optimizer.zero_grad()
            loss.backward()
            if self.grad_clip_norm is not None:
                grad_norm = th.nn.utils.clip_grad_norm_(
                    self.cost_model.parameters(),
                    self.grad_clip_norm,
                )
            else:
                grad_norm = self._grad_norm()
            self.cost_optimizer.step()
            diagnostics["grad_norm"] = float(grad_norm)
            last_stats = diagnostics

        assert last_stats is not None
        stats = CostTrainingStats(**last_stats)
        self._log_cost_stats(stats)
        return stats

    def _cost_loss(
        self,
        demo_batch: Sequence[Trajectory],
        sample_batch: Sequence[Trajectory],
    ) -> tuple[th.Tensor, Dict[str, float]]:
        background_batch = list(sample_batch) + list(demo_batch)

        demo_step_costs = trajectory_step_costs(self.cost_model, demo_batch)
        background_step_costs = trajectory_step_costs(self.cost_model, background_batch)
        sample_step_costs = (
            trajectory_step_costs(self.cost_model, sample_batch) if sample_batch else []
        )

        demo_costs = reduce_trajectory_costs(
            demo_step_costs,
            reduction=self.trajectory_cost_reduction,
        )
        background_costs = reduce_trajectory_costs(
            background_step_costs,
            reduction=self.trajectory_cost_reduction,
        )

        log_q = th.as_tensor(
            [self._trajectory_log_q(traj) for traj in background_batch],
            dtype=background_costs.dtype,
            device=background_costs.device,
        )
        log_weights = -log_q
        if self.max_log_weight is not None:
            log_weights = th.clamp(log_weights, max=self.max_log_weight)

        raw_log_terms = log_weights - background_costs
        log_terms, partition_diagnostics = self._stabilize_partition_terms(raw_log_terms)
        tempered_log_terms = log_terms / self.partition_temperature
        log_partition = self.partition_temperature * (
            th.logsumexp(tempered_log_terms, dim=0) - math.log(len(background_batch))
        )
        ioc_loss = demo_costs.mean() + log_partition

        lcr = local_constant_rate_regularizer(background_step_costs)
        mono = monotonic_regularizer(background_step_costs, margin=self.monotonic_margin)
        loss = ioc_loss + self.lcr_reg_weight * lcr + self.monotonic_reg_weight * mono

        with th.no_grad():
            soft = th.softmax(tempered_log_terms, dim=0)
            effective_fraction = 1.0 / (len(background_batch) * soft.pow(2).sum())
            expert_start = len(sample_batch)
            expert_mass = soft[expert_start:].sum() if demo_batch else th.tensor(0.0)
            sample_cost = (
                reduce_trajectory_costs(
                    sample_step_costs,
                    reduction=self.trajectory_cost_reduction,
                ).mean()
                if sample_step_costs
                else th.tensor(float("nan"), device=loss.device)
            )

        diagnostics = {
            "loss": float(loss.detach().cpu()),
            "ioc_loss": float(ioc_loss.detach().cpu()),
            "demo_cost": float(demo_costs.mean().detach().cpu()),
            "sample_cost": float(sample_cost.detach().cpu()),
            "log_partition": float(log_partition.detach().cpu()),
            "lcr_regularizer": float(lcr.detach().cpu()),
            "monotonic_regularizer": float(mono.detach().cpu()),
            "effective_sample_fraction": float(effective_fraction.detach().cpu()),
            "top1_importance_weight": float(soft.max().detach().cpu()),
            "expert_softmax_mass": float(expert_mass.detach().cpu()),
            **partition_diagnostics,
            "grad_norm": 0.0,
        }
        return loss, diagnostics

    def _stabilize_partition_terms(
        self,
        raw_log_terms: th.Tensor,
    ) -> tuple[th.Tensor, Dict[str, float]]:
        """Optionally keep the sample-based partition estimate from collapsing.

        The exact GCL estimator is recovered with the default configuration:
        no span clipping and ``partition_temperature=1``. SUMO runs can produce
        very peaky log-terms because the empirical Gaussian sampler is only an
        approximation of the true trajectory distribution; a bounded span plus
        a mild temperature makes the cost-model gradient less single-sample
        dominated.
        """
        log_terms = raw_log_terms
        clipped_fraction = 0.0
        if self.max_partition_log_term_span is not None and raw_log_terms.numel() > 0:
            floor = raw_log_terms.detach().max() - self.max_partition_log_term_span
            clipped = raw_log_terms < floor
            clipped_fraction = float(clipped.float().mean().detach().cpu())
            log_terms = th.maximum(raw_log_terms, floor)

        with th.no_grad():
            raw_span = raw_log_terms.max() - raw_log_terms.min()
            stabilized_span = log_terms.max() - log_terms.min()
            diagnostics = {
                "raw_log_term_span": float(raw_span.detach().cpu()),
                "partition_log_term_span": float(stabilized_span.detach().cpu()),
                "partition_clipped_fraction": clipped_fraction,
                "partition_temperature": float(self.partition_temperature),
            }
        return log_terms, diagnostics

    def _trajectory_log_q(self, trajectory: Trajectory) -> float:
        if self.importance_mode == "uniform":
            return 0.0

        if self.importance_mode == "behavior_policy":
            flat = flatten_trajectories([trajectory])
            logp = flat.log_policy_probs[0]
            if logp is not None and np.isfinite(logp):
                return float(logp)

        if self.importance_mode not in {"fusion_gaussian", "behavior_policy"}:
            raise ValueError(f"Unknown importance_mode: {self.importance_mode!r}")

        distributions = [self.demo_distribution, *self.sample_distributions]
        return fusion_log_prob(trajectory, distributions)

    def train_policy(self, steps: int, log_interval: int) -> Optional[float]:
        t0 = time.perf_counter()
        self.trajectory_generator.train(steps=int(steps), log_interval=int(log_interval))
        self.logger.record("time/train_policy", time.perf_counter() - t0)
        if getattr(self.agent_logger, "_data", []):
            self.agent_logger.dump()

        policy_loss = self._latest_policy_loss()
        self.logger.record(
            "policy/loss",
            policy_loss if policy_loss is not None else float("nan"),
        )
        self.logger.record("policy/train_steps", int(steps))
        return policy_loss

    # ------------------------------------------------------------------
    # Evaluation and diagnostics
    # ------------------------------------------------------------------

    def evaluate_policy(self, n_episodes: int) -> Dict[str, float]:
        if n_episodes <= 0:
            return {}

        venv = self.trajectory_generator.sampling_venv
        buffer = self.trajectory_generator.buffering_wrapper
        buffer.pop_finished_trajectories()

        obs = venv.reset()
        state = None
        episode_starts = np.ones(venv.num_envs, dtype=bool)
        fast_returns = []
        comfort_returns = []
        lengths = []
        successes = []
        collisions = []
        offroads = []
        timeouts = []

        while len(fast_returns) < n_episodes:
            actions, state = self.agent.predict(
                obs,
                state=state,
                episode_start=episode_starts,
                deterministic=True,
            )
            obs, _, dones, infos = venv.step(actions)
            for env_idx, done in enumerate(dones):
                if not done or len(fast_returns) >= n_episodes:
                    continue
                info = infos[env_idx]
                episode_metrics = info.get("metrics", {}).get("episode", {})
                fast_returns.append(
                    float(episode_metrics.get("rewards/ep_fast_return", np.nan))
                )
                comfort_returns.append(
                    float(episode_metrics.get("rewards/ep_comfort_return", np.nan))
                )
                lengths.append(float(info.get("step", 0)))
                status = info.get("ego_status", "running")
                successes.append(float(self._status_equals(status, "arrived")))
                collisions.append(float(self._status_equals(status, "collided")))
                offroads.append(float(self._status_equals(status, "offroad")))
                timeouts.append(float(self._status_equals(status, "timeout")))
            episode_starts = dones

        buffer.pop_finished_trajectories()
        self._sync_agent_observation(obs, episode_starts)

        metrics = {
            "eval/fast_return": finite_mean(fast_returns),
            "eval/comfort_return": finite_mean(comfort_returns),
            "eval/mean_ep_length": finite_mean(lengths),
            "eval/success_rate": finite_mean(successes),
            "eval/collision_rate": finite_mean(collisions),
            "eval/offroad_rate": finite_mean(offroads),
            "eval/timeout_rate": finite_mean(timeouts),
        }
        for key, value in metrics.items():
            self.logger.record(key, value)
        return metrics

    def expert_action_rmse(self) -> float:
        if self.expert_rmse_transitions <= 0:
            return float("nan")

        observations = []
        expert_actions = []
        attempts = 0
        while len(observations) < self.expert_rmse_transitions and attempts < 10 * self.expert_rmse_transitions:
            attempts += 1
            traj = self.expert_trajectories[int(self.rng.integers(len(self.expert_trajectories)))]
            if not traj:
                continue
            transition = traj[int(self.rng.integers(len(traj)))]
            observations.append(np.asarray(transition.observation, dtype=np.float32))
            expert_actions.append(np.asarray(transition.action, dtype=np.float32))

        if not observations:
            return float("nan")

        obs = np.stack(observations).astype(np.float32)
        expert = np.stack(expert_actions).astype(np.float32).reshape(len(obs), -1)
        actions, _ = self.agent.predict(obs, deterministic=True)
        policy_actions = np.asarray(actions, dtype=np.float32).reshape(len(obs), -1)
        return float(np.sqrt(np.mean(np.square(policy_actions - expert))))

    def _log_rollout_stats(self, trajectories: Sequence[Trajectory], elapsed: float) -> None:
        true_returns = [traj.total_reward() for traj in trajectories]
        lengths = [len(traj) for traj in trajectories]
        self.logger.record("rollout/mean_true_reward", finite_mean(true_returns))
        self.logger.record("rollout/mean_length", finite_mean(lengths))
        self.logger.record("rollout/n_trajectories", len(trajectories))
        self.logger.record("rollout/total_transitions", int(np.sum(lengths)))
        self.logger.record("gcl/sample_buffer_trajectories", len(self.sample_trajectories))
        self.logger.record("gcl/fusion_distributions", len(self.sample_distributions) + 1)
        self.logger.record("time/collect_policy_samples", elapsed)

    def _log_cost_stats(self, stats: CostTrainingStats) -> None:
        self.logger.record("cost_model/loss", stats.loss)
        self.logger.record("cost_model/ioc_loss", stats.ioc_loss)
        self.logger.record("cost_model/demo_cost", stats.demo_cost)
        self.logger.record("cost_model/sample_cost", stats.sample_cost)
        self.logger.record("cost_model/log_partition", stats.log_partition)
        self.logger.record("cost_model/lcr_regularizer", stats.lcr_regularizer)
        self.logger.record("cost_model/monotonic_regularizer", stats.monotonic_regularizer)
        self.logger.record("cost_model/grad_norm", stats.grad_norm)
        self.logger.record(
            "cost_model/effective_sample_fraction",
            stats.effective_sample_fraction,
        )
        self.logger.record("cost_model/top1_importance_weight", stats.top1_importance_weight)
        self.logger.record("cost_model/expert_softmax_mass", stats.expert_softmax_mass)
        self.logger.record("cost_model/raw_log_term_span", stats.raw_log_term_span)
        self.logger.record("cost_model/partition_log_term_span", stats.partition_log_term_span)
        self.logger.record(
            "cost_model/partition_clipped_fraction",
            stats.partition_clipped_fraction,
        )
        self.logger.record("cost_model/partition_temperature", stats.partition_temperature)

    def _latest_policy_loss(self) -> Optional[float]:
        values = dict(getattr(self.agent_logger, "last_values", {}))
        for _, key, value, _ in getattr(self.agent_logger, "_data", []):
            values[key] = value
        candidate_keys = (
            "agent/train/actor_loss",
            "agent/train/policy_gradient_loss",
            "agent/train/loss",
        )
        for key in candidate_keys:
            if key in values:
                value = values[key]
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None
        return None

    def _grad_norm(self) -> th.Tensor:
        norms = []
        for param in self.cost_model.parameters():
            if param.grad is not None:
                norms.append(param.grad.detach().norm(2))
        if not norms:
            return th.tensor(0.0)
        return th.norm(th.stack(norms), 2)

    @staticmethod
    def _status_equals(status: Any, expected: str) -> bool:
        if hasattr(status, "value"):
            status = status.value
        return str(status) == expected

    def _sync_agent_observation(self, obs: np.ndarray, dones: np.ndarray) -> None:
        if hasattr(self.agent, "_last_obs"):
            self.agent._last_obs = obs
        if hasattr(self.agent, "_last_episode_starts"):
            self.agent._last_episode_starts = dones
        if getattr(self.agent, "_vec_normalize_env", None) is not None:
            self.agent._last_original_obs = self.agent._vec_normalize_env.get_original_obs()

    # ------------------------------------------------------------------
    # Checkpointing and public loop
    # ------------------------------------------------------------------

    def save_checkpoint(self, checkpoint_dir: str | os.PathLike, iteration: int) -> Path:
        ckpt_path = Path(checkpoint_dir) / f"checkpoint_{iteration:04d}"
        ckpt_path.mkdir(parents=True, exist_ok=True)
        th.save(self.cost_model.state_dict(), ckpt_path / "cost_model.pt")
        th.save(self.cost_optimizer.state_dict(), ckpt_path / "cost_optimizer.pt")
        self.trajectory_generator.agent.save(str(ckpt_path / "agent"))
        metadata = {
            "iteration": iteration,
            "importance_mode": self.importance_mode,
            "sample_trajectories": len(self.sample_trajectories),
            "sample_distributions": [dist.name for dist in self.sample_distributions],
        }
        with (ckpt_path / "metadata.pkl").open("wb") as f:
            pickle.dump(metadata, f)
        return ckpt_path

    def train(
        self,
        n_iterations: int = 100,
        rollout_steps_per_iteration: int = 4096,
        policy_steps_per_iteration: int = 4096,
        log_interval: int = 1,
        eval_episodes: int = 5,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 10,
    ) -> Any:
        print("=" * 100)
        print("Guided Cost Learning")
        print("=" * 100)
        print(f"Expert trajectories: {len(self.expert_trajectories)}")
        print(f"Importance mode: {self.importance_mode}")

        for iteration in range(int(n_iterations)):
            self.iteration = iteration
            t_iter = time.perf_counter()
            print(f"\nIteration {iteration + 1}/{n_iterations}")

            print(f"- Sampling {rollout_steps_per_iteration} policy transitions")
            self.collect_policy_samples(rollout_steps_per_iteration)

            print(f"- Training cost model for {self.cost_gradient_steps} gradient steps")
            self.train_cost_model()

            print(f"- Training policy for {policy_steps_per_iteration} timesteps")
            self.train_policy(policy_steps_per_iteration, log_interval)

            print(f"- Evaluating policy on {eval_episodes} episodes")
            self.evaluate_policy(eval_episodes)
            rmse = self.expert_action_rmse()
            self.logger.record("imitation/expert_action_rmse", rmse)
            self.logger.record("iterations", iteration + 1)
            self.logger.record("agent/time/total_timesteps", self.agent.num_timesteps)
            self.logger.record("time/iteration", time.perf_counter() - t_iter)
            self.logger.dump()

            if (
                checkpoint_dir is not None
                and checkpoint_interval > 0
                and (iteration + 1) % checkpoint_interval == 0
            ):
                ckpt = self.save_checkpoint(checkpoint_dir, iteration + 1)
                print(f"  checkpoint saved in {ckpt}")

        return self.trajectory_generator.agent
