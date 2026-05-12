from dataclasses import dataclass
import math
import random
import time
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from human_feedback_rl.common.datasets import PreferenceDataset
from human_feedback_rl.common.fragmenters import RandomFragmenter
from human_feedback_rl.common.reward_nets import RewardEnsemble, SumoSimpleRewardNet
from human_feedback_rl.common.trajectory_generators import TrajectoryGeneratorFromAgent
from human_feedback_rl.common.preference_models import PreferenceModelFromReward
from human_feedback_rl.common.gatherers import PreferenceGathererFromReward

from human_feedback_rl.common.loggers import WandbWriter, PrefixedLogger, ExcludedLogger, FilteredHumanOutput
from stable_baselines3.common.logger import Logger
from scipy.stats import kendalltau, spearmanr
import sys

import torch as th


@dataclass
class RewMetrics:
    loss_all_train: float
    loss_all_val: float
    acc_all_train: float
    acc_all_val: float
    time_train_rew: float   # seconds spent on gradient steps
    time_logs_rew: float    # seconds spent computing eval metrics

def make_reward_ensemble(
    venv: VecEnv,
    n_ensembles: int = 3,
    net_arch: list[int] = [128,128],
    activation_fn: str = "tanh",
) -> RewardEnsemble:
    obs_space = venv.observation_space
    act_space = venv.action_space

    members = [
        SumoSimpleRewardNet(obs_space, act_space, net_arch=net_arch, activation_fn=activation_fn)
        for _ in range(n_ensembles)
    ]

    return RewardEnsemble(obs_space, act_space, members)


QUERY_SCHEDULES: Dict[str, Callable[[float], float]] = {
    "constant": lambda t: 1.0,
    "hyperbolic": lambda t: 1.0 / (1.0 + t),
    "inverse_quadratic": lambda t: 1.0 / (1.0 + t**2),
}


class ChristianoAlgorithm:
    def __init__(
        self,
        env,
        agent,
        lr_rew: float = 0.001,
        batch_size_rew: int = 100,
        n_epochs_rew: int = 5,
        n_ensembles_rew: int = 2,
        net_arch_rew: list[int] = [128, 128],
        decay_rew: float = 0.0,
        l2_rew: float = 0.01,
        train_comparison_frac: int = 0.7,
        fragment_length: int = 1,
        transition_oversampling: int = 3,
        initial_comparison_frac: float = 0,
        initial_epoch_multiplier: int = 2,
        query_schedule: Union[str, Callable[[float], float]] = "constant",
        comparison_queue_size: int = 1_000_000,
        rng: Optional[np.random.Generator] = None,
    ):
        self.batch_size_rew = batch_size_rew
        self.n_epochs_rew = n_epochs_rew
        self.decay_rew = decay_rew
        self.l2_rew = l2_rew
        self.fragment_length = fragment_length
        self.initial_comparison_frac = initial_comparison_frac
        self.initial_epoch_multiplier = initial_epoch_multiplier
        self.transition_oversampling = transition_oversampling
        self.train_comparison_frac = train_comparison_frac
        self._iteration = 0
        self._gradient_steps_rew = 0
        self.rng = rng

        self.logger = Logger(
            folder=None,
            output_formats=[
                FilteredHumanOutput(),
                WandbWriter(),
            ]
        )
            
        self.query_schedule = QUERY_SCHEDULES[query_schedule]
        self.query_schedule_name = query_schedule

        self.reward_model = make_reward_ensemble(env, n_ensembles=n_ensembles_rew, net_arch=net_arch_rew)

        self.preference_model = PreferenceModelFromReward(self.reward_model)

        self.optimizers = [
            th.optim.Adam(member.parameters(), lr=lr_rew, weight_decay=self.l2_rew)
            for member in self.reward_model.members
        ]

        agent.set_logger(ExcludedLogger(PrefixedLogger(self.logger, "agent"), exclude="stdout"))

        self.trajectory_generator = TrajectoryGeneratorFromAgent(
            agent=agent,
            reward_model=self.reward_model,
            venv=env,
            rng=rng if rng is not None else np.random.default_rng(),
            logger=PrefixedLogger(self.logger, "rollout"),
        )

        self.fragmenter = RandomFragmenter(
            logger=PrefixedLogger(self.logger, "fragmenter"),
            rng=rng if rng is not None else np.random.default_rng(),
        )

        self.dataset_train = PreferenceDataset(queue_size=comparison_queue_size)
        self.dataset_val = PreferenceDataset(queue_size=comparison_queue_size)

        self.preference_gatherer = PreferenceGathererFromReward()



    def train(self,
            total_timesteps: int = 1_000_000,
            timesteps_per_iteration: int = 1024,
            comparisons_per_iteration: int = 10,
            log_interval: int = 1,
        ) -> Any:

        n_iterations = int(total_timesteps / timesteps_per_iteration)

        total_comparisons = comparisons_per_iteration * n_iterations
        initial_comparisons = int(total_comparisons * self.initial_comparison_frac)
        total_comparisons = total_comparisons - initial_comparisons

        # Compute the number of comparisons to request at each iteration in advance.
        t_vec = np.linspace(0, 1, n_iterations)
        weights = np.array([self.query_schedule(t) for t in t_vec])
        probs = weights / weights.sum()
        shares = np.round(probs * total_comparisons).astype(int)
        schedule = [initial_comparisons] + shares.tolist()
        print(f"- Query {self.query_schedule_name} schedule: {schedule}")

        for i, num_pairs in enumerate(schedule):
            t0 = time.perf_counter()

            print(f"\nIteration {i+1}/{len(schedule)}")

            if num_pairs == 0:
                continue

            ##########################
            # Gather new preferences #
            ##########################
            num_steps = math.ceil(2 * self.transition_oversampling * num_pairs * self.fragment_length)

            print(f"- Collecting {2 * num_pairs} fragments (2x{self.transition_oversampling}x{num_pairs}x{self.fragment_length}={num_steps} transitions)")
            trajectories, rollout_logs_metrics = self.trajectory_generator.sample(num_steps)

            print(f"- Creating fragment pairs (from {len(trajectories)} trajectories and {np.sum(rollout_logs_metrics.lengths)} total transitions)")
            fragments, frag_metrics = self.fragmenter(trajectories, self.fragment_length, num_pairs)

            print("- Gathering preferences")
            preferences, gath_metrics = self.preference_gatherer(fragments)

            self._push_train_and_val(fragments, preferences, i)
            print(f"- Train dataset now contains {len(self.dataset_train)} comparisons | Val dataset now contains {len(self.dataset_val)} comparisons")


            ##########################
            # Train the reward model #
            ##########################

            t0_val = time.perf_counter()
            self.rew_model_correlation(trajectories)
            time_rew_model_correlation = time.perf_counter() - t0_val

            epoch_multiplier = 1.0
            if i == 0:
                epoch_multiplier = self.initial_epoch_multiplier

            print(f"- Training reward model for {int(epoch_multiplier*self.n_epochs_rew)} epochs ({len(self.dataset_train)} comparisons/member, bootstrapped)")
            rew_logs_metrics = self.train_reward_model(epoch_multiplier)


            ###################
            # Train the agent #
            ###################
            print(f"- Training agent for {timesteps_per_iteration} timesteps")
            train_logs_metrics = self.trajectory_generator.train(steps=timesteps_per_iteration, log_interval=log_interval)


            ############
            # Loggings #
            ############

            self.logger.record("rollout/mean_true_reward", rollout_logs_metrics.mean_true_reward)
            self.logger.record("rollout/mean_model_reward", rollout_logs_metrics.mean_model_reward)
            self.logger.record("rollout/mean_length", rollout_logs_metrics.mean_length)
            self.logger.record("rollout/n_trajectories", len(trajectories))
            self.logger.record("rollout/total_transitions", np.sum(rollout_logs_metrics.lengths))

            self.logger.record("reward/loss_all_train", rew_logs_metrics.loss_all_train, exclude="stdout")
            self.logger.record("reward/loss_all_val", rew_logs_metrics.loss_all_val, exclude="stdout")
            self.logger.record("reward/loss_all_gap", rew_logs_metrics.loss_all_train - rew_logs_metrics.loss_all_val, exclude="stdout")

            self.logger.record("reward/acc_all_train", rew_logs_metrics.acc_all_train, exclude="stdout")
            self.logger.record("reward/acc_all_val", rew_logs_metrics.acc_all_val, exclude="stdout")
            self.logger.record("reward/acc_all_gap", rew_logs_metrics.acc_all_train - rew_logs_metrics.acc_all_val, exclude="stdout")

            self.logger.record("agent/time/current_timesteps", train_logs_metrics.current_timesteps)

            self.logger.record("iterations", i)
            self.logger.record("reward/total_gradient_steps", self._gradient_steps_rew)
            self.logger.record("agent/time/total_timesteps", train_logs_metrics.total_timesteps)

            self.logger.record("time/train_rew", rew_logs_metrics.time_train_rew)
            self.logger.record("time/extra_rollout", rollout_logs_metrics.time_sample) 
            self.logger.record("time/fragmenter", frag_metrics.time_fragmenter)
            self.logger.record("time/gatherer", gath_metrics.time_gatherer)
            self.logger.record("time/train_agent", train_logs_metrics.time_train)
            self.logger.record("time/loggings", rew_logs_metrics.time_logs_rew + time_rew_model_correlation)
            self.logger.record("time/this_iteration", time.perf_counter() - t0)
            self.logger.dump()

            self._iteration += 1

        return self.trajectory_generator.agent


    def train_reward_model(self, epoch_multiplier: float = 1.0):

        t0_train = time.perf_counter()
        n_epochs = max(1, int(round(self.n_epochs_rew * epoch_multiplier)))

        member_pref_models = [PreferenceModelFromReward(m) for m in self.reward_model.members]
        member_boot_datasets = [self.dataset_train.bootstrap() for m in self.reward_model.members]

        for _ in range(n_epochs):
            for optimizer, pref_model, boot_dataset in zip(self.optimizers, member_pref_models, member_boot_datasets):

                for batch in boot_dataset.get(self.batch_size_rew):
                    bt_probs = pref_model(batch.fragment_pairs)
                    labels = th.tensor([[p.pref1, p.pref2] for p in batch.preferences], dtype=th.float32)

                    t = th.tensor(batch.timestamps, dtype=th.float32)
                    t_normalized = (t - t.min()) / (t.max() - t.min() + 1e-8)
                    weights = th.exp(self.decay_rew * t_normalized)
                    weights = weights / weights.sum()

                    per_pair_loss = -(labels * bt_probs.clamp(min=1e-7).log()).sum(dim=1)
                    # per_pair_loss = th.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='none')
                    loss = (weights * per_pair_loss).sum()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    self._gradient_steps_rew += 1

        time_train_rew = time.perf_counter() - t0_train


        t0_logs = time.perf_counter()
        loss_all_train, acc_all_train = self._evaluate_reward_model(self.dataset_train.get_all())
        loss_all_val, acc_all_val = self._evaluate_reward_model(self.dataset_val.get_all())
        time_logs_rew = time.perf_counter() - t0_logs

        return RewMetrics(
            loss_all_train=loss_all_train,
            loss_all_val=loss_all_val,

            acc_all_train=acc_all_train,
            acc_all_val=acc_all_val,

            time_train_rew=time_train_rew,
            time_logs_rew=time_logs_rew,
        )

    
    def _evaluate_reward_model(self, data) -> tuple[float, float]:
        fragment_pairs, preferences, timestamps = data.fragment_pairs, data.preferences, data.timestamps

        t = th.tensor(timestamps, dtype=th.float32)
        t_normalized = (t - t.min()) / (t.max() - t.min() + 1e-8)
        weights = th.exp(self.decay_rew * t_normalized)
        weights = weights / weights.sum()
        
        self.preference_model.eval()
        with th.no_grad():
            bt_probs = self.preference_model(list(fragment_pairs))  # (N, 2)
        self.preference_model.train()

        labels = th.tensor(
            [[p.pref1, p.pref2] for p in preferences],
            dtype=th.float32
        )

        per_pair_loss = -(labels * bt_probs.clamp(min=1e-7).log()).sum(dim=1)
        loss = (weights * per_pair_loss).sum().item()

        acc  = (bt_probs.argmax(dim=1) == labels.argmax(dim=1)).float().mean().item()

        return loss, acc
    

    def _push_train_and_val(self, fragments, preferences, i):
        # Split fragments and preferences together to keep them aligned
        pairs = list(zip(fragments, preferences))
        random.shuffle(pairs)
        n_train = round(len(pairs) * self.train_comparison_frac)

        train_pairs = pairs[:n_train]
        val_pairs   = pairs[n_train:]

        if train_pairs:
            train_fragments, train_preferences = zip(*train_pairs)
            self.dataset_train.push(list(train_fragments), list(train_preferences), i)

        if val_pairs:
            val_fragments, val_preferences = zip(*val_pairs)
            self.dataset_val.push(list(val_fragments), list(val_preferences), i)
        
    
    def rew_model_correlation(self, trajectories):

        n_samples = 100
        segment_lengths = [1, 2, 5, 10, 20]

        for seg_len in segment_lengths:
            frag_pairs, _ = self.fragmenter(trajectories, seg_len, n_samples)
            frag_list = [f for pair in frag_pairs for f in [pair.frag1, pair.frag2]]

            true_returns = np.array([sum(t.true_reward for t in frag) / len(frag) for frag in frag_list])

            self.reward_model.eval()
            with th.no_grad():
                model_returns = np.array([
                    self.preference_model._sum_rewards(frag).item()
                    for frag in frag_list
                ])
            self.reward_model.train()

            tau, _ = kendalltau(true_returns, model_returns)
            rho, _ = spearmanr(true_returns, model_returns)

            self.logger.record(f"reward_correlation/kendall_tau_seg{seg_len}", tau, exclude="stdout")
            self.logger.record(f"reward_correlation/spearman_rho_seg{seg_len}", rho, exclude="stdout")


