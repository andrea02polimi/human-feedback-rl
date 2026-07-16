"""Hybrid reward learning from demonstrations and preferences.

Two integration mechanisms for the demonstration signal (``demo_mode``):

* ``"gcl"`` — the demo IRL loss (MaxEnt/GCL family, from ``DemoAlgorithm``) is
  combined with the Bradley-Terry preference loss on one shared reward net.
  The demo gradient is norm-balanced against the preference gradient
  (``demo_weight`` = desired demo/preference gradient-strength ratio).
* ``"preferences"`` — demonstrations enter as preference pairs (expert
  fragment preferred over agent fragment, Ibarz et al. 2018): a single
  Bradley-Terry objective on mixed batches, no scale conflict by
  construction.

Health metrics: with soft oracle labels at high ``pref_temperature`` the BT
loss sits at its ln(2) cross-entropy floor even when learning succeeds —
watch ``reward/acc_pref_val`` and ``reward_val/.../pred_true/pearson_*``
instead. Keep ``l2_rew`` at 1e-4: Adam weight decay at 1e-2 overwhelms the BT
gradient and collapses the reward net (diagnosed 2026-07-05).
"""

import os
import time
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch as th

from human_feedback_rl.algorithms.demo_algorithm import DemoAlgorithm
from human_feedback_rl.common.base_reward_learning_algorithm import QUERY_SCHEDULES
from human_feedback_rl.common.batching import fragment_avg_rewards
from human_feedback_rl.common.datasets import PreferenceBatch, PreferenceDataset
from human_feedback_rl.common.fragmenters import make_pair_fragmenter
from human_feedback_rl.common.gatherers import PreferenceGathererFromReward
from human_feedback_rl.common.losses import (
    bradley_terry_probs,
    evaluate_preference_batch,
    preference_labels_tensor,
    preference_nll,
)
from human_feedback_rl.common.types import Preference, Trajectory

VALID_DEMO_MODES = ("gcl", "preferences")


class HybridAlgorithm(DemoAlgorithm):
    """Trains one reward model from preferences and demonstrations.

    See the module docstring for the two ``demo_mode`` mechanisms. In
    ``"gcl"`` mode the two losses are combined with a norm-balanced sum:
    the demo gradient is rescaled so its norm is ``demo_weight`` times the
    preference gradient's norm before the two are added.
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
        l2_rew: float = 0.0001,
        temperature: float = 1.0,
        pref_temperature: float = 20.0,
        fragment_length: Optional[int] = None,
        preference_fragment_length: int = 1,
        fragmenter_type: str = "random",
        labels_type: str = "binary",
        comparison_queue_size: int = 1_000_000,
        train_comparison_frac: float = 0.8,
        total_queries: int = 10_000,
        initial_queries: int = 0,
        query_schedule: Union[str, Callable[[float], float]] = "constant",
        demo_mode: str = "gcl",
        demo_weight: float = 1.0,
        max_balance_scale: float = 100.0,
        balance_eps: float = 1e-8,
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
        if pref_temperature <= 0:
            raise ValueError("pref_temperature must be positive.")
        if demo_mode not in VALID_DEMO_MODES:
            raise ValueError(f"demo_mode must be one of {VALID_DEMO_MODES}, got {demo_mode!r}.")
        if demo_pref_pairs_per_iteration < 0:
            raise ValueError("demo_pref_pairs_per_iteration must be non-negative.")
        if not 0 <= demo_pref_batch_fraction <= 1:
            raise ValueError("demo_pref_batch_fraction must be in [0, 1].")

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

        self.demo_mode = demo_mode
        self.demo_weight = float(demo_weight)
        self.max_balance_scale = float(max_balance_scale)
        self.balance_eps = float(balance_eps)
        self.demo_pref_pairs_per_iteration = int(demo_pref_pairs_per_iteration)
        self.demo_pref_batch_fraction = float(demo_pref_batch_fraction)

        self.fragmenter = make_pair_fragmenter(
            fragmenter_type, rng=self.rng, logger=self.logger, reward_ensemble=self.reward_model
        )
        # Oracle label softness is a property of the (synthetic) annotator,
        # NOT of the demo IRL loss: it gets its own temperature.
        self.preference_gatherer = PreferenceGathererFromReward(
            logger=self.logger,
            labels_type=labels_type,
            temperature=pref_temperature,
            rng=self.rng,
        )
        self.dataset_train = PreferenceDataset(queue_size=comparison_queue_size, rng=self.rng)
        self.dataset_val = PreferenceDataset(queue_size=comparison_queue_size, rng=self.rng)
        # Expert-vs-agent pairs (demo_mode="preferences" only).
        self.dataset_demo_prefs_train = PreferenceDataset(queue_size=comparison_queue_size, rng=self.rng)
        self.dataset_demo_prefs_val = PreferenceDataset(queue_size=comparison_queue_size, rng=self.rng)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Feedback collection
    # ------------------------------------------------------------------

    def _collect_feedback(self, num_queries: int) -> None:
        self._collect_preference_feedback(num_queries)
        if self.demo_mode == "preferences":
            self._collect_demo_preference_pairs(self.demo_pref_pairs_per_iteration)

    def _collect_preference_feedback(self, num_queries: int) -> None:
        if num_queries <= 0:
            return
        fragments = self.fragmenter(
            self.trajectories,
            self.preference_fragment_length,
            num_queries,
        )
        preferences = self.preference_gatherer(fragments)
        self._push_split(self.dataset_train, self.dataset_val, fragments, preferences)
        self.logger.record("dataset/n_train", len(self.dataset_train), exclude="stdout")
        self.logger.record("dataset/n_val", len(self.dataset_val), exclude="stdout")

    def _collect_demo_preference_pairs(self, num_pairs: int) -> None:
        """Expert fragment > agent fragment pairs (Ibarz et al. 2018).

        The expert is preferred by assumption — no reward signal is used.
        """
        if num_pairs <= 0 or not self.trajectories:
            return
        from human_feedback_rl.common.types import FragmentPair

        expert_frags = self._single_fragmenter(
            self.expert_trajectories, self.preference_fragment_length, num_pairs
        )
        agent_frags = self._single_fragmenter(
            self.trajectories, self.preference_fragment_length, num_pairs
        )
        pairs = [FragmentPair(e, a) for e, a in zip(expert_frags, agent_frags)]
        preferences = [Preference(1.0, 0.0) for _ in pairs]
        self._push_split(
            self.dataset_demo_prefs_train, self.dataset_demo_prefs_val, pairs, preferences
        )
        self.logger.record(
            "dataset/n_demo_prefs_train", len(self.dataset_demo_prefs_train), exclude="stdout"
        )

    def _push_split(self, train_ds, val_ds, fragments, preferences) -> None:
        """Shuffle and split into train/val by ``train_comparison_frac``."""
        if not fragments:
            return
        idx = self.rng.permutation(len(fragments))
        fragments = [fragments[i] for i in idx]
        preferences = [preferences[i] for i in idx]
        n_train = min(len(fragments), max(1, int(self.train_comparison_frac * len(fragments))))
        train_ds.push(fragments[:n_train], preferences[:n_train])
        val_ds.push(fragments[n_train:], preferences[n_train:])

    # ------------------------------------------------------------------
    # Reward-model training
    # ------------------------------------------------------------------

    def _train_reward_model(self) -> None:
        if not self.trajectories:
            return
        if self.demo_mode == "preferences":
            self._train_reward_model_pure_preferences()
        else:
            self._train_reward_model_gcl()

    def _train_reward_model_gcl(self) -> None:
        """BT + GCL on the shared net with norm-balanced gradient fusion.

        Degenerate cases stay well-defined: with an empty preference dataset
        only the demo loss trains (demo-only arm); with ``demo_weight == 0``
        only the preference loss trains (pref-only arm).
        """
        has_prefs = len(self.dataset_train) > 0
        demo_weight = self.demo_weight
        if not has_prefs and demo_weight == 0.0:
            return

        self._maxent_corrected_steps = []
        self.logger.record("reward/demo_weight", demo_weight, exclude="stdout")
        t0 = time.perf_counter()

        def member_step(member, optimizer):
            member.train()
            boot_dataset = self.dataset_train.bootstrap() if has_prefs else None
            stats = []
            for _ in range(self.gradient_steps_rew):
                pref_loss = (
                    self._preference_loss(member, boot_dataset.sample(self.batch_size_pref))
                    if boot_dataset is not None
                    else None
                )
                demo_loss = self._reward_loss(member) if demo_weight > 0.0 else None
                stats.append(self._reward_step(member, optimizer, pref_loss, demo_loss))
            return stats

        all_stats = [s for stats in self.train_reward_members(member_step) for s in stats]
        t_train = time.perf_counter() - t0

        self._log_reward_loss_diagnostics()
        self._log_maxent_corrected_step_diagnostics()
        self._log_preference_diagnostics()
        self._log_hybrid_step_stats(all_stats)
        self.logger.record("reward/weight_norm", self._param_norm(self.reward_model), exclude="stdout")
        self.logger.record("time/train_reward_model", t_train)

    def _train_reward_model_pure_preferences(self) -> None:
        """Single BT objective on mixed oracle + expert-vs-agent batches."""
        has_oracle = len(self.dataset_train) > 0
        has_demo = len(self.dataset_demo_prefs_train) > 0
        if not has_oracle and not has_demo:
            return

        t0 = time.perf_counter()
        n_demo = round(self.batch_size_pref * self.demo_pref_batch_fraction)
        if not has_demo:
            n_demo = 0
        if not has_oracle:
            n_demo = self.batch_size_pref
        n_oracle = self.batch_size_pref - n_demo

        def member_step(member, optimizer):
            member.train()
            boot_oracle = self.dataset_train.bootstrap() if n_oracle else None
            boot_demo = self.dataset_demo_prefs_train.bootstrap() if n_demo else None
            losses = []
            for _ in range(self.gradient_steps_rew):
                parts = []
                if boot_oracle is not None:
                    parts.append(boot_oracle.sample(n_oracle))
                if boot_demo is not None:
                    parts.append(boot_demo.sample(n_demo))
                batch = PreferenceBatch(
                    [pair for part in parts for pair in part.fragment_pairs],
                    [pref for part in parts for pref in part.preferences],
                )
                loss = self._preference_loss(member, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            return losses

        all_losses = [l for losses in self.train_reward_members(member_step) for l in losses]
        t_train = time.perf_counter() - t0

        self._log_preference_diagnostics()
        self.logger.record("reward/hybrid_pref_loss", float(np.mean(all_losses)), exclude="stdout")
        self.logger.record("reward/weight_norm", self._param_norm(self.reward_model), exclude="stdout")
        self.logger.record("time/train_reward_model", t_train)

    # ------------------------------------------------------------------
    # Balanced gradient step
    # ------------------------------------------------------------------

    def _reward_step(self, member, optimizer, pref_loss, demo_loss) -> Dict[str, float]:
        """One optimizer step from the preference and/or demo losses.

        With both losses present, the two gradients are computed separately,
        the demo gradient is norm-balanced to ``demo_weight`` times the
        preference gradient, and the composed gradient (an exact per-parameter
        linear combination ``g_pref + scale * g_demo``) is written back before
        ``step()``.
        """
        nan = float("nan")
        stats = {
            "pref_loss": nan, "demo_loss": nan, "scale": nan, "pref_norm": nan,
            "demo_norm": nan, "grad_norm": nan,
        }

        if pref_loss is None and demo_loss is None:
            return stats

        if demo_loss is None:  # pref-only arm
            optimizer.zero_grad()
            pref_loss.backward()
            stats.update(pref_loss=float(pref_loss.detach()), grad_norm=self._grad_norm(member))
            optimizer.step()
            return stats
        if pref_loss is None:  # demo-only arm
            optimizer.zero_grad()
            demo_loss.backward()
            stats.update(demo_loss=float(demo_loss.detach()), grad_norm=self._grad_norm(member))
            optimizer.step()
            return stats

        params = list(member.parameters())
        optimizer.zero_grad()
        pref_loss.backward()
        g_pref = [None if p.grad is None else p.grad.detach().clone() for p in params]
        optimizer.zero_grad()
        demo_loss.backward()
        g_demo = [None if p.grad is None else p.grad.detach().clone() for p in params]

        flat_pref = self._flatten(g_pref, params)
        flat_demo = self._flatten(g_demo, params)
        pref_norm = float(flat_pref.norm())
        demo_norm = float(flat_demo.norm())

        scale = min(
            self.demo_weight * pref_norm / (demo_norm + self.balance_eps),
            self.max_balance_scale,
        )

        for p, gp, gd in zip(params, g_pref, g_demo):
            if gp is None and gd is None:
                p.grad = None
                continue
            grad = th.zeros_like(p)
            if gp is not None:
                grad += gp
            if gd is not None:
                grad += scale * gd
            p.grad = grad

        grad_norm = self._grad_norm(member)
        optimizer.step()

        stats.update(
            pref_loss=float(pref_loss.detach()),
            demo_loss=float(demo_loss.detach()),
            scale=scale, pref_norm=pref_norm, demo_norm=demo_norm, grad_norm=grad_norm,
        )
        return stats

    @staticmethod
    def _flatten(grads, params) -> th.Tensor:
        parts = [
            g.reshape(-1) if g is not None else th.zeros(p.numel())
            for g, p in zip(grads, params)
        ]
        return th.cat(parts) if parts else th.zeros(0)

    def _log_hybrid_step_stats(self, all_stats: List[Dict[str, float]]) -> None:
        def nanmean(key):
            values = np.asarray([s[key] for s in all_stats], dtype=float)
            finite = values[np.isfinite(values)]
            return float(finite.mean()) if finite.size else None

        pairs = {
            "reward/hybrid_pref_loss": "pref_loss",
            "reward/hybrid_demo_loss": "demo_loss",
            "reward/hybrid_demo_scale": "scale",
            "reward/grad_norm_pref": "pref_norm",
            "reward/grad_norm_demo": "demo_norm",
            "reward/grad_norm": "grad_norm",
        }
        for log_key, stat_key in pairs.items():
            value = nanmean(stat_key)
            if value is not None:
                self.logger.record(log_key, value, exclude="stdout")

        norms = {k: nanmean(k2) for k, k2 in (("pref", "pref_norm"), ("demo", "demo_norm"))}
        if norms["pref"] is not None and norms["demo"] is not None:
            self.logger.record(
                "reward/grad_norm_demo_pref_ratio",
                norms["demo"] / (norms["pref"] + self.balance_eps),
                exclude="stdout",
            )
        grad_norms = [s["grad_norm"] for s in all_stats if np.isfinite(s["grad_norm"])]
        if grad_norms:
            self.logger.record("reward/grad_norm_max", float(np.max(grad_norms)), exclude="stdout")

    # ------------------------------------------------------------------
    # Preference loss / diagnostics
    # ------------------------------------------------------------------

    def _preference_loss(self, member, batch) -> th.Tensor:
        r1 = fragment_avg_rewards(member, [p.frag1 for p in batch.fragment_pairs])
        r2 = fragment_avg_rewards(member, [p.frag2 for p in batch.fragment_pairs])
        return preference_nll(
            bradley_terry_probs(r1, r2), preference_labels_tensor(batch.preferences)
        )

    def _log_preference_diagnostics(self) -> None:
        train_loss, train_acc = evaluate_preference_batch(self.reward_model, self.dataset_train.get_all())
        val_loss, val_acc = evaluate_preference_batch(self.reward_model, self.dataset_val.get_all())
        self.logger.record("reward/loss_pref_train", train_loss, exclude="stdout")
        self.logger.record("reward/loss_pref_val", val_loss, exclude="stdout")
        self.logger.record("reward/acc_pref_train", train_acc, exclude="stdout")
        self.logger.record("reward/acc_pref_val", val_acc, exclude="stdout")
        if self.demo_mode == "preferences" and len(self.dataset_demo_prefs_val):
            _, demo_acc = evaluate_preference_batch(
                self.reward_model, self.dataset_demo_prefs_val.get_all()
            )
            self.logger.record("reward/acc_demo_pref_val", demo_acc, exclude="stdout")

    def _save_checkpoint_extras(self, ckpt_path: str, iteration: int) -> None:
        super()._save_checkpoint_extras(ckpt_path, iteration)
        th.save(
            {
                "iteration": iteration,
                "demo_mode": self.demo_mode,
                "demo_weight": self.demo_weight,
                "preference_fragment_length": self.preference_fragment_length,
                "dataset_train": self.dataset_train,
                "dataset_val": self.dataset_val,
                "dataset_demo_prefs_train": self.dataset_demo_prefs_train,
                "dataset_demo_prefs_val": self.dataset_demo_prefs_val,
            },
            os.path.join(ckpt_path, "hybrid_training.pt"),
        )
