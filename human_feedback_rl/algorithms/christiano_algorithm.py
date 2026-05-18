import time
from typing import Any, Callable, List, Optional, Union

import numpy as np
import torch as th

from human_feedback_rl.common.base_reward_learning_algorithm import BaseRewardLearningAlgorithm
from human_feedback_rl.common.reward_nets import make_reward_ensemble, NormalizedRewardNet
from human_feedback_rl.common.datasets import PreferenceDataset
from human_feedback_rl.common.fragmenters import HighVariancePairFragmenter, RandomPairFragmenter
from human_feedback_rl.common.gatherers import PreferenceGathererFromReward
from human_feedback_rl.common.bradley_terry import BradleyTerry


class ChristianoAlgorithm(BaseRewardLearningAlgorithm):
    """
    Preference-based reward learning following Christiano et al. (2017).

    Human (or synthetic) preferences over trajectory pairs are used to train
    an ensemble reward model with the Bradley-Terry loss.  The model is updated
    in the inner loop while a policy is trained with PPO in the outer loop.
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
        hard_labels: bool = True,
        train_comparison_frac: float = 0.7,
        fragment_length: int = 1,
        initial_queries: int = 0,
        exploration_frac: float = 0.0,
        exploration_eps: float = 0.5,
        query_schedule: Union[str, Callable[[float], float]] = "constant",
        reward_model_kwargs: Optional[dict] = None,
        rng: Optional[np.random.Generator] = None,
        log_folder: Optional[str] = None,
        output_formats: Optional[List] = None,
        debug_datasets: Optional[dict] = None,
    ):
        reward_model = NormalizedRewardNet(make_reward_ensemble(env, **(reward_model_kwargs or {})))

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
            rng=rng,
            log_folder=log_folder,
            output_formats=output_formats,
            debug_datasets=debug_datasets,
        )

        self.gradient_steps_rew  = gradient_steps_rew
        self.batch_size_rew      = batch_size_rew

        self.fragmenter          = self._make_fragmenter(fragmenter_type)
        self.preference_gatherer = PreferenceGathererFromReward(logger=self.logger, hard_labels=hard_labels)
        self.dataset_train       = PreferenceDataset(queue_size=comparison_queue_size, rng=self.rng)
        self.dataset_val         = PreferenceDataset(queue_size=comparison_queue_size, rng=self.rng)

        self.optimizers    = [
            th.optim.Adam(m.parameters(), lr=lr_rew, weight_decay=l2_rew)
            for m in self.reward_model.members
        ]


    # ------------------------------------------------------------------
    # Abstract hook implementations
    # ------------------------------------------------------------------

    def collect_feedback(self, num_queries):

        t0 = time.perf_counter()
        fragment_pairs = self.fragmenter(self.trajectories, self.fragment_length, num_queries)
        preferences = self.preference_gatherer(fragment_pairs)
        t_collect_feedback = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.logger.record("time/collect_feedback", t_collect_feedback)
        self.logger.record_sum("time/loggings", time.perf_counter() - t0)


        return fragment_pairs, preferences

    def push_data(self, fragments, feedback) -> None:

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


    def train_reward_model(self) -> None:
        t0 = time.perf_counter()

        for member, optimizer in zip(self.reward_model.members, self.optimizers):
            member.train()
            boot_dataset = self.dataset_train.bootstrap()
            for _ in range(self.gradient_steps_rew):
                batch = boot_dataset.sample(self.batch_size_rew)
                r1 = th.stack([member.fragment_avg_reward(p.frag1) for p in batch.fragment_pairs])
                r2 = th.stack([member.fragment_avg_reward(p.frag2) for p in batch.fragment_pairs])
                bt_probs = BradleyTerry(r1, r2)
                labels   = th.tensor(
                    [[p.pref1, p.pref2] for p in batch.preferences], dtype=th.float32
                )
                loss = -(labels * bt_probs.clamp(min=1e-7).log()).sum(dim=1).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

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
        self.reward_model.eval()
        with th.no_grad():
            r1 = th.stack([self.reward_model.fragment_avg_reward(p.frag1) for p in data.fragment_pairs])
            r2 = th.stack([self.reward_model.fragment_avg_reward(p.frag2) for p in data.fragment_pairs])
            bt_probs = BradleyTerry(r1, r2)
        self.reward_model.train()

        labels = th.tensor([[p.pref1, p.pref2] for p in data.preferences], dtype=th.float32)
        loss = -(labels * bt_probs.clamp(min=1e-7).log()).sum(dim=1).mean().item()
        acc  = (bt_probs.argmax(dim=1) == labels.argmax(dim=1)).float().mean().item()
        return loss, acc


    def before_agent_training(self):
        all_transitions = [t for traj in self.trajectories for t in traj]
        if not all_transitions:
            return
        obs    = np.array([t.observation for t in all_transitions])
        acts   = np.array([t.action      for t in all_transitions])
        status = np.array([t.next_status for t in all_transitions])
        done   = np.array([float(t.done) for t in all_transitions])
        raw = self.reward_model.predict_unnormalized(obs, acts, status, done)
        self.reward_model.set_mean(float(raw.mean()))
        self.reward_model.set_std(float(raw.std()))


    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fragment_status_pct(fragment_pairs) -> dict:
        # next_status is 7-dim one-hot: [arrived, collided, off_road, timeout, running, teleported, removed_unknown]
        STATUS_IDX = {"arrived": 0, "collided": 1, "offroad": 2, "timeout": 3}
        frags = [fp.frag1 for fp in fragment_pairs] + [fp.frag2 for fp in fragment_pairs]
        n = len(frags)
        if n == 0:
            return {k: 0.0 for k in (*STATUS_IDX, "only_running")}

        def has_status(frag, idx):
            return any(t.next_status is not None and t.next_status[idx] for t in frag)

        counts = {k: sum(has_status(f, i) for f in frags) for k, i in STATUS_IDX.items()}
        counts["only_running"] = sum(
            not any(has_status(f, i) for i in STATUS_IDX.values()) for f in frags
        )
        return {k: v / n for k, v in counts.items()}

    def _make_fragmenter(self, fragmenter_type: str):
        if fragmenter_type == "active":
            return HighVariancePairFragmenter(
                reward_ensemble=self.reward_model,
                oversample=5,
                logger=self.logger,
                rng=self.rng,
            )
        elif fragmenter_type == "random":
            return RandomPairFragmenter(
                logger=self.logger,
                rng=self.rng,
            )
        else:
            raise ValueError(f"Unknown fragmenter_type: {fragmenter_type!r}")

    # ------------------------------------------------------------------
    # Public API
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
        return super().train(
            total_timesteps=total_timesteps,
            total_queries=total_queries,
            timesteps_per_iteration=timesteps_per_iteration,
            log_interval=log_interval,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=checkpoint_interval,
        )

