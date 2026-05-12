from dataclasses import dataclass
import math
import random
import time
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from human_feedback_rl.common.datasets import DemonstrationDataset
from human_feedback_rl.common.fragmenters import RandomSingleFragmenter
from human_feedback_rl.common.reward_nets import RewardEnsemble, SumoSimpleRewardNet
from human_feedback_rl.common.trajectory_generators import TrajectoryGeneratorFromAgent
from human_feedback_rl.common.preference_models import PreferenceModelFromReward
from human_feedback_rl.common.gatherers import DemoGathererFromExpert

from human_feedback_rl.common.loggers import WandbWriter, PrefixedLogger, ExcludedLogger, FilteredHumanOutput
from stable_baselines3.common.logger import Logger
from scipy.stats import kendalltau, spearmanr, pearsonr

import os
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


class ZhangAlgorithm:
    def __init__(
        self,
        env,
        agent,
        expert,
        lr_rew: float = 0.001,
        batch_size_rew: int = 100,
        n_epochs_rew: int = 5,
        n_ensembles_rew: int = 2,
        net_arch_rew: list[int] = [128, 128],
        decay_rew: float = 0.0,
        l2_rew: float = 0.01,
        train_comparison_frac: float = 0.7,
        fragment_length: int = 1,
        transition_oversampling: int = 3,
        initial_comparisons: int = 0,
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
        self.initial_comparisons = initial_comparisons
        self.initial_epoch_multiplier = initial_epoch_multiplier
        self.transition_oversampling = transition_oversampling
        self.train_comparison_frac = train_comparison_frac
        self._iteration = 0
        self._gradient_steps_rew = 0
        self.rng = rng
        self.venv = env
        self.expert = expert

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

        # Used only for reward-correlation logging (ensemble forward pass).
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

        self.fragmenter = RandomSingleFragmenter(
            logger=PrefixedLogger(self.logger, "fragmenter"),
            rng=rng if rng is not None else np.random.default_rng(),
        )

        self.dataset_train = DemonstrationDataset(queue_size=comparison_queue_size)
        self.dataset_val = DemonstrationDataset(queue_size=comparison_queue_size)

        self.demo_gatherer = DemoGathererFromExpert(expert)


    def _save_checkpoint(self, checkpoint_dir: str, iteration: int):
        ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{iteration:04d}")
        os.makedirs(ckpt_path, exist_ok=True)
        th.save(self.reward_model.state_dict(), os.path.join(ckpt_path, "reward_model.pt"))
        self.trajectory_generator.agent.save(os.path.join(ckpt_path, "agent"))
        print(f"- Checkpoint saved to {ckpt_path}")


    def train(self,
            total_timesteps: int = 1_000_000,
            timesteps_per_iteration: int = 1024,
            comparisons_per_iteration: int = 10,
            log_interval: int = 1,
            checkpoint_dir: Optional[str] = None,
            checkpoint_interval: int = 10,
        ) -> Any:

        n_iterations = int(total_timesteps / timesteps_per_iteration)

        total_comparisons = comparisons_per_iteration * n_iterations

        # Compute the number of fragments to request at each iteration in advance.
        t_vec = np.linspace(0, 1, n_iterations)
        weights = np.array([self.query_schedule(t) for t in t_vec])
        probs = weights / weights.sum()
        shares = np.round(probs * total_comparisons).astype(int)
        schedule = [self.initial_comparisons] + shares.tolist()
        print(f"- Query {self.query_schedule_name} schedule: {schedule}")

        for i, num_fragments in enumerate(schedule):
            t0 = time.perf_counter()

            print(f"\nIteration {i}/{len(schedule)-1}")

            if num_fragments == 0:
                continue

            ##########################
            # Gather demonstrations  #
            ##########################
            num_steps = math.ceil(self.transition_oversampling * num_fragments * self.fragment_length)

            print(f"- Collecting {num_fragments} fragments ({self.transition_oversampling}x{num_fragments}x{self.fragment_length}={num_steps} transitions)")
            trajectories, rollout_logs_metrics = self.trajectory_generator.sample(num_steps)

            print(f"- Creating fragments (from {len(trajectories)} trajectories and {np.sum(rollout_logs_metrics.lengths)} total transitions)")
            fragments, frag_metrics = self.fragmenter(trajectories, self.fragment_length, num_fragments)

            print("- Gathering demonstrations")
            fragments_expert, gath_metrics = self.demo_gatherer(fragments)

            self._push_train_and_val(fragments, fragments_expert, i)
            print(f"- Train dataset now contains {len(self.dataset_train)} fragments | Val dataset now contains {len(self.dataset_val)} fragments")


            ##########################
            # Train the reward model #
            ##########################

            # Update per-step normalization stats from the current rollout
            # before training so both RM and PPO see rewards at consistent scale.
            self._update_reward_norm(trajectories)

            t0_val = time.perf_counter()
            self.rew_model_correlation(trajectories)
            time_rew_model_correlation = time.perf_counter() - t0_val

            epoch_multiplier = 1.0
            if i == 0:
                epoch_multiplier = self.initial_epoch_multiplier

            print(f"- Training reward model for {int(epoch_multiplier*self.n_epochs_rew)} epochs ({len(self.dataset_train)} fragments/member, bootstrapped)")
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

            if checkpoint_dir is not None and (i + 1) % checkpoint_interval == 0:
                self._save_checkpoint(checkpoint_dir, i + 1)

            self._iteration += 1

        return self.trajectory_generator.agent


    def _update_reward_norm(self, trajectories) -> tuple[float, float]:
        """Compute per-step normalization stats from the rollout and inject them
        into the reward model.

        Stats are derived from raw (unnormalized) ensemble outputs so they are
        independent of any previously set normalization. After this call,
        reward_model.predict() returns (raw - mean) / std, which is what both
        the RM training data and PPO rollouts will see for the rest of this
        iteration. Stats stay frozen until the next call.
        """
        all_rewards: list[np.ndarray] = []
        self.reward_model.eval()
        with th.no_grad():
            for traj in trajectories:
                obs  = np.array([t.observation  for t in traj], dtype=np.float32)
                acts = np.array([t.action       for t in traj], dtype=np.float32)
                ns   = np.array([t.next_status  for t in traj], dtype=np.float32)
                dn   = np.array([float(t.done)  for t in traj], dtype=np.float32)
                all_rewards.append(self.reward_model.predict_unnormalized(obs, acts, ns, dn))
        self.reward_model.train()

        flat = np.concatenate(all_rewards)
        mean = float(np.mean(flat))
        std  = max(float(np.std(flat)), 1e-8)

        self.reward_model.set_mean(mean)
        self.reward_model.set_std(std)

        self.logger.record("reward/norm_mean", mean, exclude="stdout")
        self.logger.record("reward/norm_std",  std,  exclude="stdout")

        return mean, std


    def train_reward_model(self, epoch_multiplier: float = 1.0) -> RewMetrics:

        t0_train = time.perf_counter()
        n_epochs = max(1, int(round(self.n_epochs_rew * epoch_multiplier)))

        member_boot_datasets = [self.dataset_train.bootstrap() for _ in self.reward_model.members]

        for _ in range(n_epochs):
            for optimizer, rew_model_member, boot_dataset in zip(self.optimizers, self.reward_model.members, member_boot_datasets):

                for batch in boot_dataset.get(self.batch_size_rew):

                    timestamps = th.tensor(batch.timestamps, dtype=th.float32)
                    ts_norm = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min() + 1e-8)
                    weights = th.exp(self.decay_rew * ts_norm)
                    weights = weights / weights.sum()

                    predicted_returns = []
                    for frag in batch.expert_fragments:
                        obs         = th.tensor(np.array([tr.observation  for tr in frag]), dtype=th.float32)
                        actions     = th.tensor(np.array([tr.action       for tr in frag]), dtype=th.float32)
                        next_status = th.tensor(np.array([tr.next_status  for tr in frag]), dtype=th.float32)
                        done        = th.tensor(np.array([float(tr.done)  for tr in frag]), dtype=th.float32)
                        return_frag = rew_model_member(obs, actions, next_status, done).sum() / len(frag)
                        predicted_returns.append(return_frag)

                    returns = th.stack(predicted_returns)                    # (B,)
                    # Loss: -log(sigmoid(R)) — reward model should assign high return to expert fragments.
                    per_frag_loss = -th.log(th.sigmoid(returns).clamp(min=1e-7))
                    loss = (weights * per_frag_loss).sum()

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
        timestamps = th.tensor(data.timestamps, dtype=th.float32)
        ts_norm = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min() + 1e-8)
        weights = th.exp(self.decay_rew * ts_norm)
        weights = weights / weights.sum()

        self.reward_model.eval()
        predicted_returns = []
        with th.no_grad():
            for frag in data.expert_fragments:
                obs         = th.tensor(np.array([tr.observation  for tr in frag]), dtype=th.float32)
                actions     = th.tensor(np.array([tr.action       for tr in frag]), dtype=th.float32)
                next_status = th.tensor(np.array([tr.next_status  for tr in frag]), dtype=th.float32)
                done        = th.tensor(np.array([float(tr.done)  for tr in frag]), dtype=th.float32)
                R = self.reward_model(obs, actions, next_status, done).sum() / len(frag)
                predicted_returns.append(R)
        self.reward_model.train()

        returns = th.stack(predicted_returns)
        per_frag_loss = -th.log(th.sigmoid(returns).clamp(min=1e-7))
        loss = (weights * per_frag_loss).sum().item()
        # Accuracy: fraction of expert fragments the model assigns positive return to.
        acc = (returns > 0).float().mean().item()

        return loss, acc


    def _push_train_and_val(self, fragments, fragments_expert, i):
        pairs = list(zip(fragments, fragments_expert))
        random.shuffle(pairs)
        n_train = round(len(pairs) * self.train_comparison_frac)

        train_pairs = pairs[:n_train]
        val_pairs   = pairs[n_train:]

        if train_pairs:
            train_fragments, train_expert_fragments = zip(*train_pairs)
            self.dataset_train.push(list(train_fragments), list(train_expert_fragments), i)

        if val_pairs:
            val_fragments, val_expert_fragments = zip(*val_pairs)
            self.dataset_val.push(list(val_fragments), list(val_expert_fragments), i)


    def rew_model_correlation(self, trajectories):

        n_samples = 100
        segment_lengths = [1, 5, 20, 10000]

        for seg_len in segment_lengths:
            frag_list, _ = self.fragmenter(trajectories, seg_len, n_samples)

            true_returns = np.array([sum(tr.true_reward for tr in frag) / len(frag) for frag in frag_list])

            self.reward_model.eval()
            with th.no_grad():
                model_returns = np.array([
                    self.preference_model._sum_rewards(frag).item()
                    for frag in frag_list
                ])
            self.reward_model.train()

            tau, _ = kendalltau(true_returns, model_returns)
            rho, _ = spearmanr(true_returns, model_returns)
            r, _ = pearsonr(true_returns, model_returns)

            self.logger.record(f"reward_correlation/kendall_tau_seg{seg_len}", tau, exclude="stdout")
            self.logger.record(f"reward_correlation/spearman_rho_seg{seg_len}", rho, exclude="stdout")
            self.logger.record(f"reward_correlation/pearson_r_seg{seg_len}", r, exclude="stdout")
