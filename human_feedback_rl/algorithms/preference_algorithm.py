import time
from typing import Any, Callable, List, Optional, Union

import numpy as np
import torch as th

from human_feedback_rl.common import status
from human_feedback_rl.common.base_reward_learning_algorithm import BaseRewardLearningAlgorithm
from human_feedback_rl.common.batching import fragment_avg_rewards, stacked_transitions
from human_feedback_rl.common.datasets import PreferenceDataset
from human_feedback_rl.common.fragmenters import make_pair_fragmenter
from human_feedback_rl.common.gatherers import PreferenceGathererFromReward
from human_feedback_rl.common.losses import (
    bradley_terry_probs,
    evaluate_preference_batch,
    preference_labels_tensor,
    preference_nll,
)
from human_feedback_rl.common.reward_nets import make_reward_ensemble


class PreferenceAlgorithm(BaseRewardLearningAlgorithm):
    """
    Preference-based reward learning following Christiano et al. (2017).

    Human (or synthetic) preferences over trajectory-fragment pairs train an
    ensemble reward model with the Bradley-Terry loss; the agent trains on the
    predicted rewards in an alternating outer loop driven by a query schedule.
    """

    def __init__(
        self,
        env,
        agent,
        lr_rew: float = 0.001,
        gradient_steps_rew: int = 10,
        batch_size_rew: int = 32,
        l2_rew: float = 0.01,
        fragmenter_type: str = "random",
        comparison_queue_size: int = 1_000_000,
        labels_type: str = "binary",
        train_comparison_frac: float = 0.7,
        fragment_length: int = 1,
        temperature: float = 1,
        initial_queries: int = 0,
        exploration_frac: float = 0.0,
        exploration_eps: float = 0.5,
        query_schedule: Union[str, Callable[[float], float]] = "constant",
        reward_model_kwargs: Optional[dict] = None,
        rng: Optional[np.random.Generator] = None,
        log_folder: Optional[str] = None,
        output_formats: Optional[List] = None,
        debug_dataset: Optional[dict] = None,
        agent_log_timestep_interval: Optional[int] = None,
    ):
        reward_model = make_reward_ensemble(env, **(reward_model_kwargs or {}))

        super().__init__(
            env=env,
            agent=agent,
            reward_model=reward_model,
            train_comparison_frac=train_comparison_frac,
            fragment_length=fragment_length,
            initial_queries=initial_queries,
            exploration_frac=exploration_frac,
            exploration_eps=exploration_eps,
            query_schedule=query_schedule,
            temperature=temperature,
            rng=rng,
            log_folder=log_folder,
            output_formats=output_formats,
            debug_dataset=debug_dataset,
            agent_log_timestep_interval=agent_log_timestep_interval,
        )

        self.gradient_steps_rew  = gradient_steps_rew
        self.batch_size_rew      = batch_size_rew

        self.fragmenter = make_pair_fragmenter(
            fragmenter_type, rng=self.rng, logger=self.logger, reward_ensemble=self.reward_model
        )
        self.preference_gatherer = PreferenceGathererFromReward(
            logger=self.logger, labels_type=labels_type, temperature=temperature, rng=self.rng
        )
        self.dataset_train       = PreferenceDataset(queue_size=comparison_queue_size, rng=self.rng)
        self.dataset_val         = PreferenceDataset(queue_size=comparison_queue_size, rng=self.rng)

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
        total_queries: int = 10_000,
        timesteps_per_iteration: int = 1024,
        log_interval: int = 1,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 10,
    ) -> Any:
        """Run the alternating preference-collection / reward-learning / agent-training loop."""

        self.iteration = 0
        n_iterations = int(total_timesteps / timesteps_per_iteration)
        schedule = self.build_query_schedule(n_iterations, total_queries)

        print("=" * 100)
        print("Preference-based reward learning (Christiano et al. 2017)")
        print("=" * 100)
        print("")
        print(f"Query {self.query_schedule_name} schedule: {schedule}")

        for num_queries in schedule:
            t_iter = time.perf_counter()
            print(f"\nIteration {self.iteration}/{len(schedule) - 1}")

            # ---- Data collection ----------------------------------------
            exploration_steps = int(self.exploration_frac * timesteps_per_iteration)
            print(f"- Collecting {timesteps_per_iteration} agent + {exploration_steps} exploration transitions")
            self.trajectories = self.sample_rollout(timesteps_per_iteration, exploration_steps)

            # ---- Feedback collection & reward model training -------------
            # Iterations without new queries skip feedback collection and RM
            # retraining (no new data), but the agent still trains below.
            if num_queries > 0:
                print(f"- Collecting {num_queries} feedbacks on the current rollout")
                fragments, feedback = self.collect_feedback(num_queries)
                self.push_data(fragments, feedback)

                self.before_reward_training()

                print("- Training reward model")
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

    # ------------------------------------------------------------------
    # Loop steps
    # ------------------------------------------------------------------

    def collect_feedback(self, num_queries: int) -> tuple:
        """Fragment the current rollout and label fragment pairs via the gatherer."""
        t0 = time.perf_counter()
        fragment_pairs = self.fragmenter(self.trajectories, self.fragment_length, num_queries)
        preferences = self.preference_gatherer(fragment_pairs)
        t_collect_feedback = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.logger.record("time/collect_feedback", t_collect_feedback)
        self.logger.record_sum("time/loggings", time.perf_counter() - t0)

        return fragment_pairs, preferences

    def push_data(self, fragments, feedback) -> None:
        """Shuffle and split into train/val datasets by ``train_comparison_frac``."""
        t0 = time.perf_counter()
        idx = self.rng.permutation(len(fragments))
        fragments = [fragments[i] for i in idx]
        feedback  = [feedback[i]  for i in idx]
        n_train = int(self.train_comparison_frac * len(fragments))
        self.dataset_train.push(fragments[:n_train], feedback[:n_train])
        self.dataset_val.push(fragments[n_train:],   feedback[n_train:])
        t_push_data = time.perf_counter() - t0

        t0 = time.perf_counter()
        pct = self._fragment_status_pct(self.dataset_train.get_all().fragment_pairs)
        self.logger.record("dataset/n_train",              len(self.dataset_train))
        self.logger.record("dataset/n_val",                len(self.dataset_val))
        self.logger.record("dataset/train_collisions",     pct["collided"])
        self.logger.record("dataset/train_arrives",        pct["arrived"])
        self.logger.record("dataset/train_timeouts",       pct["timeout"])
        self.logger.record("dataset/train_offroads",       pct["offroad"])
        self.logger.record("dataset/train_only_running",   pct["only_running"])
        self.logger.record("time/push_data",               t_push_data)
        self.logger.record_sum("time/loggings",            time.perf_counter() - t0)

    def before_reward_training(self) -> None:
        """Log reward-model validation metrics on the current rollout (and debug dataset)."""
        all_transitions = [t for traj in self.trajectories for t in traj]
        self.log_reward_model_validation(all_transitions, "reward_val/current_rollout")

        if self.debug_dataset:
            self.log_reward_model_validation(self.debug_dataset, "reward_val/debug_dataset")

    def train_reward_model(self) -> None:
        """Train each ensemble member on its own bootstrap of the preference dataset."""
        t0 = time.perf_counter()

        def member_step(member, optimizer):
            member.train()
            boot_dataset = self.dataset_train.bootstrap()
            for _ in range(self.gradient_steps_rew):
                batch = boot_dataset.sample(self.batch_size_rew)

                r1 = fragment_avg_rewards(member, [p.frag1 for p in batch.fragment_pairs])
                r2 = fragment_avg_rewards(member, [p.frag2 for p in batch.fragment_pairs])
                bt_probs = bradley_terry_probs(r1, r2)
                loss = preference_nll(bt_probs, preference_labels_tensor(batch.preferences))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.train_reward_members(member_step)
        t_train = time.perf_counter() - t0

        t0 = time.perf_counter()
        loss_train, acc_train = self._evaluate_reward_model(self.dataset_train.get_all())
        loss_val,   acc_val   = self._evaluate_reward_model(self.dataset_val.get_all())

        self.logger.record("reward/loss_all_train",     loss_train,              exclude="stdout")
        self.logger.record("reward/loss_all_val",       loss_val,                exclude="stdout")
        self.logger.record("reward/loss_all_gap",       loss_train - loss_val,   exclude="stdout")
        self.logger.record("reward/acc_all_train",      acc_train,               exclude="stdout")
        self.logger.record("reward/acc_all_val",        acc_val,                 exclude="stdout")
        self.logger.record("reward/acc_all_gap",        acc_train - acc_val,     exclude="stdout")
        self.logger.record("time/train_reward_model",   t_train)
        self.logger.record_sum("time/loggings",         time.perf_counter() - t0)

    def _evaluate_reward_model(self, data) -> tuple:
        """Return (loss, accuracy) of the full ensemble on a preference batch."""
        return evaluate_preference_batch(self.reward_model, data)

    def before_agent_training(self):
        """Center the agent-facing reward on the current rollout's mean raw reward."""
        all_transitions = [t for traj in self.trajectories for t in traj]
        if not all_transitions:
            return
        obs         = np.array([t.observation for t in all_transitions])
        acts        = np.array([t.action      for t in all_transitions])
        next_status = np.array([t.next_status for t in all_transitions])
        done        = np.array([float(t.done) for t in all_transitions])

        raw = self.reward_model.predict_unnormalized(obs, acts, next_status, done)
        self.reward_model.set_mean(raw.mean())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fragment_status_pct(fragment_pairs) -> dict:
        STATUS_IDX = {
            "arrived": status.STATUS_ARRIVED,
            "collided": status.STATUS_COLLIDED,
            "offroad": status.STATUS_OFFROAD,
            "timeout": status.STATUS_TIMEOUT,
        }
        frags = [fp.frag1 for fp in fragment_pairs] + [fp.frag2 for fp in fragment_pairs]
        n = len(frags)
        if n == 0:
            return {k: 0.0 for k in (*STATUS_IDX, "only_running")}

        # (n_frags, STATUS_DIM) booleans: does the fragment contain each status?
        has_status = np.stack([
            stacked_transitions(f)[2].numpy().any(axis=0) for f in frags
        ])
        counts = {k: int(has_status[:, i].sum()) for k, i in STATUS_IDX.items()}
        terminal_indices = list(STATUS_IDX.values())
        counts["only_running"] = int((~has_status[:, terminal_indices].any(axis=1)).sum())
        return {k: v / n for k, v in counts.items()}

