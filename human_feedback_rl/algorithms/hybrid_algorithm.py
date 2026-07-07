"""Hybrid reward learning from demonstrations and preferences."""

import os
import time
from typing import Any, Callable, List, Optional, Union

import numpy as np
import torch as th

from human_feedback_rl.algorithms.demo_algorithm import DemoAlgorithm
from human_feedback_rl.common.base_reward_learning_algorithm import QUERY_SCHEDULES
from human_feedback_rl.common.batching import fragment_avg_rewards
from human_feedback_rl.common.datasets import PreferenceDataset
from human_feedback_rl.common.fragmenters import HighVariancePairFragmenter, RandomPairFragmenter
from human_feedback_rl.common.gatherers import PreferenceGathererFromReward
from human_feedback_rl.common.losses import (
    bradley_terry_probs,
    preference_accuracy,
    preference_nll,
)
from human_feedback_rl.common.types import Trajectory


class HybridAlgorithm(DemoAlgorithm):
    """Train one reward model with a BT preference loss and a demo IRL loss.

    The demo loss is gradient-balanced against the BT loss at every reward-model
    step, so ``demo_weight`` means "desired demo gradient strength relative to
    the preference gradient" rather than a raw loss multiplier.
    """

    def __init__(
        self,
        env,
        agent,
        expert_trajectories: List[Trajectory],
        loss_type: str = "maxent_2",
        lr_rew: float = 0.001,
        gradient_steps_rew: int = 10,
        batch_size_expert: int = 32,
        batch_size_model: int = 64,
        batch_size_pref: int = 32,
        l2_rew: float = 0.01,
        temperature: float = 1.0,
        fragment_length: Optional[int] = None,
        preference_fragment_length: int = 1,
        fragmenter_type: str = "random",
        labels_type: str = "binary",
        comparison_queue_size: int = 1_000_000,
        train_comparison_frac: float = 0.8,
        total_queries: int = 10_000,
        initial_queries: int = 0,
        query_schedule: Union[str, Callable[[float], float]] = "constant",
        demo_weight: float = 1.0,
        max_balance_scale: float = 100.0,
        balance_eps: float = 1e-8,
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
        if batch_size_pref <= 0:
            raise ValueError("batch_size_pref must be positive.")
        if preference_fragment_length <= 0:
            raise ValueError("preference_fragment_length must be positive.")
        if not 0 < train_comparison_frac < 1:
            raise ValueError("train_comparison_frac must be in (0, 1).")
        if demo_weight < 0:
            raise ValueError("demo_weight must be non-negative.")
        if max_balance_scale <= 0 or balance_eps <= 0:
            raise ValueError("max_balance_scale and balance_eps must be positive.")

        super().__init__(
            env=env,
            agent=agent,
            expert_trajectories=expert_trajectories,
            loss_type=loss_type,
            lr_rew=lr_rew,
            gradient_steps_rew=gradient_steps_rew,
            batch_size_expert=batch_size_expert,
            batch_size_model=batch_size_model,
            l2_rew=l2_rew,
            temperature=temperature,
            fragment_length=fragment_length,
            initial_agent_timesteps=initial_agent_timesteps,
            exploration_frac=exploration_frac,
            exploration_eps=exploration_eps,
            reward_model_kwargs=reward_model_kwargs,
            rng=rng,
            log_folder=log_folder,
            output_formats=output_formats,
            debug_dataset=debug_dataset,
            rollout_env=rollout_env,
            relabel_rewards=relabel_rewards,
            normalize_agent_reward=normalize_agent_reward,
            agent_log_timestep_interval=agent_log_timestep_interval,
        )

        self.batch_size_pref = batch_size_pref
        self.preference_fragment_length = int(preference_fragment_length)
        self.train_comparison_frac = train_comparison_frac
        self.total_queries = total_queries
        self.initial_queries = initial_queries
        self.query_schedule_name = query_schedule if isinstance(query_schedule, str) else "callable"
        if isinstance(query_schedule, str):
            if query_schedule not in QUERY_SCHEDULES:
                raise ValueError(f"Unknown query_schedule: {query_schedule!r}.")
            self.query_schedule = QUERY_SCHEDULES[query_schedule]
        else:
            self.query_schedule = query_schedule

        self.demo_weight = float(demo_weight)
        self.max_balance_scale = float(max_balance_scale)
        self.balance_eps = float(balance_eps)

        self.fragmenter = self._make_fragmenter(fragmenter_type)
        self.preference_gatherer = PreferenceGathererFromReward(
            logger=self.logger,
            labels_type=labels_type,
            temperature=temperature,
            rng=self.rng,
        )
        self.dataset_train = PreferenceDataset(queue_size=comparison_queue_size, rng=self.rng)
        self.dataset_val = PreferenceDataset(queue_size=comparison_queue_size, rng=self.rng)

    def train(
        self,
        total_timesteps: int = 1_000_000,
        timesteps_per_iteration: int = 1024,
        log_interval: int = 1,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 10,
        scatter_interval: Optional[int] = None,
        total_queries: Optional[int] = None,
    ) -> Any:
        """Run hybrid reward learning and agent training."""
        if scatter_interval is None:
            scatter_interval = 10
        if scatter_interval < 0:
            raise ValueError("scatter_interval must be non-negative.")

        n_iterations = int(total_timesteps / timesteps_per_iteration)
        query_budget = self.total_queries if total_queries is None else total_queries
        schedule = self.build_query_schedule(n_iterations, query_budget)

        if self.initial_agent_timesteps > 0:
            print(f"- Collecting {self.initial_agent_timesteps} bootstrap transitions")
            self.trajectories = self.sample_rollout(self.initial_agent_timesteps)
            bootstrap_queries = self.initial_queries
            self._collect_preference_feedback(bootstrap_queries)
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
            self._collect_preference_feedback(num_queries)

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

    def _collect_preference_feedback(self, num_queries: int) -> None:
        if num_queries <= 0:
            return
        fragments = self.fragmenter(
            self.trajectories,
            self.preference_fragment_length,
            num_queries,
        )
        preferences = self.preference_gatherer(fragments)
        idx = self.rng.permutation(len(fragments))
        n_train = int(self.train_comparison_frac * len(fragments))
        if fragments:
            n_train = min(len(fragments), max(1, n_train))
        fragments = [fragments[i] for i in idx]
        preferences = [preferences[i] for i in idx]
        self.dataset_train.push(fragments[:n_train], preferences[:n_train])
        self.dataset_val.push(fragments[n_train:], preferences[n_train:])
        self.logger.record("dataset/n_train", len(self.dataset_train), exclude="stdout")
        self.logger.record("dataset/n_val", len(self.dataset_val), exclude="stdout")

    def _train_reward_model(self) -> None:
        if not self.trajectories or len(self.dataset_train) == 0:
            return

        self._maxent_corrected_steps = []
        t0 = time.perf_counter()

        def member_step(member, optimizer):
            member.train()
            boot_dataset = self.dataset_train.bootstrap()
            stats = []
            for _ in range(self.gradient_steps_rew):
                pref_loss = self._preference_loss(member, boot_dataset.sample(self.batch_size_pref))
                demo_loss = self._reward_loss(member)
                scale, pref_norm, demo_norm = self._balance_scale(member, pref_loss, demo_loss)

                optimizer.zero_grad()
                loss = pref_loss + scale * demo_loss
                loss.backward()
                grad_norm = self._grad_norm(member)
                optimizer.step()

                stats.append(
                    (
                        float(pref_loss.detach()),
                        float(demo_loss.detach()),
                        scale,
                        pref_norm,
                        demo_norm,
                        grad_norm,
                    )
                )
            return stats

        all_stats = [s for stats in self.train_reward_members(member_step) for s in stats]
        t_train = time.perf_counter() - t0

        pref_losses, demo_losses, scales, pref_norms, demo_norms, grad_norms = zip(*all_stats)
        self._log_reward_loss_diagnostics()
        self._log_maxent_corrected_step_diagnostics()
        self._log_preference_diagnostics()
        self.logger.record("reward/hybrid_pref_loss", float(np.mean(pref_losses)), exclude="stdout")
        self.logger.record("reward/hybrid_demo_loss", float(np.mean(demo_losses)), exclude="stdout")
        self.logger.record("reward/hybrid_demo_scale", float(np.mean(scales)), exclude="stdout")
        self.logger.record("reward/grad_norm_pref", float(np.mean(pref_norms)), exclude="stdout")
        self.logger.record("reward/grad_norm_demo", float(np.mean(demo_norms)), exclude="stdout")
        self.logger.record(
            "reward/grad_norm_demo_pref_ratio",
            float(np.mean(demo_norms) / (np.mean(pref_norms) + self.balance_eps)),
            exclude="stdout",
        )
        self.logger.record("reward/grad_norm", float(np.mean(grad_norms)), exclude="stdout")
        self.logger.record("reward/grad_norm_max", float(np.max(grad_norms)), exclude="stdout")
        self.logger.record("reward/weight_norm", self._param_norm(self.reward_model), exclude="stdout")
        self.logger.record("time/train_reward_model", t_train)

    def _balance_scale(self, member, pref_loss, demo_loss) -> tuple[float, float, float]:
        optimizer_params = list(member.parameters())
        for p in optimizer_params:
            p.grad = None
        pref_loss.backward(retain_graph=True)
        pref_norm = self._grad_norm(member)

        for p in optimizer_params:
            p.grad = None
        demo_loss.backward(retain_graph=True)
        demo_norm = self._grad_norm(member)

        for p in optimizer_params:
            p.grad = None
        raw_scale = self.demo_weight * pref_norm / (demo_norm + self.balance_eps)
        scale = min(raw_scale, self.max_balance_scale)
        return float(scale), float(pref_norm), float(demo_norm)

    def _preference_loss(self, member, batch) -> th.Tensor:
        r1 = fragment_avg_rewards(member, [p.frag1 for p in batch.fragment_pairs])
        r2 = fragment_avg_rewards(member, [p.frag2 for p in batch.fragment_pairs])
        labels = th.tensor([[p.pref1, p.pref2] for p in batch.preferences], dtype=th.float32)
        return preference_nll(bradley_terry_probs(r1, r2), labels)

    def _log_preference_diagnostics(self) -> None:
        train_loss, train_acc = self._evaluate_preference_model(self.dataset_train.get_all())
        val_loss, val_acc = self._evaluate_preference_model(self.dataset_val.get_all())
        self.logger.record("reward/loss_pref_train", train_loss, exclude="stdout")
        self.logger.record("reward/loss_pref_val", val_loss, exclude="stdout")
        self.logger.record("reward/acc_pref_train", train_acc, exclude="stdout")
        self.logger.record("reward/acc_pref_val", val_acc, exclude="stdout")

    def _evaluate_preference_model(self, data) -> tuple[float, float]:
        if not data.fragment_pairs:
            return float("nan"), float("nan")
        self.reward_model.eval()
        with th.no_grad():
            r1 = fragment_avg_rewards(self.reward_model, [p.frag1 for p in data.fragment_pairs])
            r2 = fragment_avg_rewards(self.reward_model, [p.frag2 for p in data.fragment_pairs])
            probs = bradley_terry_probs(r1, r2)
            labels = th.tensor([[p.pref1, p.pref2] for p in data.preferences], dtype=th.float32)
            loss = preference_nll(probs, labels)
            acc = preference_accuracy(probs, labels)
        self.reward_model.train()
        return float(loss), float(acc)

    def _make_fragmenter(self, fragmenter_type: str):
        if fragmenter_type == "active":
            return HighVariancePairFragmenter(
                reward_ensemble=self.reward_model,
                oversample=5,
                logger=self.logger,
                rng=self.rng,
            )
        if fragmenter_type == "random":
            return RandomPairFragmenter(logger=self.logger, rng=self.rng)
        raise ValueError(f"Unknown fragmenter_type: {fragmenter_type!r}")

    def _save_checkpoint_extras(self, ckpt_path: str, iteration: int) -> None:
        super()._save_checkpoint_extras(ckpt_path, iteration)
        th.save(
            {
                "iteration": iteration,
                "demo_weight": self.demo_weight,
                "preference_fragment_length": self.preference_fragment_length,
                "dataset_train": self.dataset_train,
                "dataset_val": self.dataset_val,
            },
            os.path.join(ckpt_path, "hybrid_training.pt"),
        )
