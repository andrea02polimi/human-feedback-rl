import time
from typing import Callable, List, Optional, Union

import numpy as np
import torch as th

from human_feedback_rl.common.base_reward_learning_algorithm import BaseRewardLearningAlgorithm, make_reward_model
from human_feedback_rl.common.datasets import DemonstrationDataset
from human_feedback_rl.common.fragmenters import RandomSingleFragmenter
from human_feedback_rl.common.gatherers import DemoGathererFromExpert
from human_feedback_rl.common.reward_nets import NormalizedRewardNet


class ZhangAlgorithm(BaseRewardLearningAlgorithm):
    """
    Demonstration-based reward learning following Zhang et al.

    Expert demonstrations are compared against agent rollouts on the same
    states using a Bradley-Terry margin loss.  The reward model is
    re-normalized from rollout statistics before each training step so that
    both the RM and PPO see rewards at a consistent scale.
    """

    def __init__(
        self,
        env,
        agent,
        expert,
        lr_rew: float = 0.001,
        n_minibatches_rew: int = 5,
        n_epochs_rew: int = 5,
        l2_rew: float = 0.01,
        train_comparison_frac: float = 0.7,
        fragment_length: int = 1,
        initial_comparisons: int = 0,
        query_schedule: Union[str, Callable[[float], float]] = "constant",
        comparison_queue_size: int = 1_000_000,
        reward_model_kwargs: Optional[dict] = None,
        rng: Optional[np.random.Generator] = None,
        log_folder: Optional[str] = None,
        output_formats: Optional[List] = None,
    ):
        self.ensemble       = make_reward_model(env, **(reward_model_kwargs or {}))
        normalized_reward   = NormalizedRewardNet(self.ensemble)

        super().__init__(
            env=env,
            agent=agent,
            reward_model=normalized_reward,
            train_comparison_frac=train_comparison_frac,
            fragment_length=fragment_length,
            initial_queries=initial_comparisons,
            query_schedule=query_schedule,
            rng=rng,
            log_folder=log_folder,
            output_formats=output_formats,
        )

        self.n_epochs_rew      = n_epochs_rew
        self.n_minibatches_rew = n_minibatches_rew
        self._gradient_steps_rew = 0
        self.optimizers = [
            th.optim.Adam(m.parameters(), lr=lr_rew, weight_decay=l2_rew)
            for m in self.ensemble.members
        ]

        self.expert        = expert
        self.fragmenter    = RandomSingleFragmenter(logger=self.logger, rng=self.rng)
        self.demo_gatherer = DemoGathererFromExpert(expert, logger=self.logger)
        self.dataset_train = DemonstrationDataset(queue_size=comparison_queue_size, rng=self.rng)
        self.dataset_val   = DemonstrationDataset(queue_size=comparison_queue_size, rng=self.rng)

    # ------------------------------------------------------------------
    # Abstract hook implementations
    # ------------------------------------------------------------------

    def collect_feedback(self, num_queries):
        print(f"- Creating {num_queries} fragments (from {len(self.trajectories)} trajectories)")
        fragments = self.fragmenter(self.trajectories, self.fragment_length, num_queries)

        print("- Gathering demonstrations")
        fragments_expert = self.demo_gatherer(fragments)

        return fragments, fragments_expert

    def push_data(self, fragments, feedback) -> None:
        idx = self.rng.permutation(len(fragments))
        fragments = [fragments[i] for i in idx]
        feedback  = [feedback[i]  for i in idx]
        n_train = int(self.train_comparison_frac * len(fragments))
        self.dataset_train.push(fragments[:n_train], feedback[:n_train])
        self.dataset_val.push(fragments[n_train:],   feedback[n_train:])


    def before_reward_training(self) -> None:
        """Re-calibrate reward normalization stats, then run correlation logging."""
        self._update_reward_norm(self.trajectories)
        super().before_reward_training()

    def train_reward_model(self, epoch_multiplier: float = 1.0) -> None:
        t0_train = time.perf_counter()
        n_epochs = max(1, int(round(self.n_epochs_rew * epoch_multiplier)))

        member_boot_datasets = [self.dataset_train.bootstrap() for _ in self.ensemble.members]

        for _ in range(n_epochs):
            for optimizer, rew_member, boot_dataset in zip(
                self.optimizers, self.ensemble.members, member_boot_datasets
            ):
                batch_size = max(1, len(boot_dataset) // self.n_minibatches_rew)
                for batch in boot_dataset.get(batch_size):
                    weights = self._temporal_weights(batch.timestamps)

                    expert_returns = self._compute_fragment_returns(rew_member, batch.expert_fragments)
                    agent_returns  = self._compute_fragment_returns(rew_member, batch.fragments)

                    r_expert = th.stack(expert_returns)
                    r_agent  = th.stack(agent_returns)
                    per_frag_loss = -th.log(th.sigmoid(r_expert - r_agent).clamp(min=1e-7))
                    loss = (weights * per_frag_loss).sum()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    self._gradient_steps_rew += 1

        t0_logs = time.perf_counter()
        loss_train, acc_train = self._evaluate_reward_model(self.dataset_train.get_all())
        loss_val,   acc_val   = self._evaluate_reward_model(self.dataset_val.get_all())

        log = self.logger.record
        log("reward/loss_all_train", loss_train,            exclude="stdout")
        log("reward/loss_all_val",   loss_val,              exclude="stdout")
        log("reward/loss_all_gap",   loss_train - loss_val, exclude="stdout")
        log("reward/acc_all_train",  acc_train,             exclude="stdout")
        log("reward/acc_all_val",    acc_val,               exclude="stdout")
        log("reward/acc_all_gap",    acc_train - acc_val,   exclude="stdout")
        log("time/train_rew",        time.perf_counter() - t0_train)
        log("time/reward_eval",      time.perf_counter() - t0_logs)

    def _evaluate_reward_model(self, data) -> tuple:
        weights = self._temporal_weights(data.timestamps)

        self.reward_model.eval()
        with th.no_grad():
            expert_returns = self._compute_fragment_returns(self.reward_model, data.expert_fragments)
            agent_returns  = self._compute_fragment_returns(self.reward_model, data.fragments)
        self.reward_model.train()

        r_expert = th.stack(expert_returns)
        r_agent  = th.stack(agent_returns)
        per_frag_loss = -th.log(th.sigmoid(r_expert - r_agent).clamp(min=1e-7))
        loss = (weights * per_frag_loss).sum().item()
        acc  = (r_expert > r_agent).float().mean().item()

        return loss, acc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_fragment_returns(self, model, fragment_list) -> list:
        """Return a list of per-fragment mean rewards as scalar tensors."""
        returns = []
        for frag in fragment_list:
            obs  = th.tensor(np.array([tr.observation  for tr in frag]), dtype=th.float32)
            acts = th.tensor(np.array([tr.action       for tr in frag]), dtype=th.float32)
            ns   = th.tensor(np.array([tr.next_status  for tr in frag]), dtype=th.float32)
            done = th.tensor(np.array([float(tr.done)  for tr in frag]), dtype=th.float32)
            returns.append(model(obs, acts, ns, done).sum() / len(frag))
        return returns

    def _update_reward_norm(self, trajectories) -> None:
        """Recompute per-step mean/std from unnormalized ensemble outputs and inject
        them into the reward model so RM training and PPO rollouts share the same scale.
        """
        all_rewards: list = []
        self.reward_model.eval()
        with th.no_grad():
            for traj in trajectories:
                obs  = np.array([t.observation  for t in traj], dtype=np.float32)
                acts = np.array([t.action       for t in traj], dtype=np.float32)
                ns   = np.array([t.next_status  for t in traj], dtype=np.float32)
                dn   = np.array([float(t.done)  for t in traj], dtype=np.float32)
                all_rewards.append(self.reward_model.predict(obs, acts, ns, dn))
        self.reward_model.train()

        flat = np.concatenate(all_rewards)
        mean = float(np.mean(flat))
        std  = max(float(np.std(flat)), 1e-8)

        self.normalized_reward.set_mean(mean)
        self.normalized_reward.set_std(std)

        self.logger.record("reward/norm_mean", mean, exclude="stdout")
        self.logger.record("reward/norm_std",  std,  exclude="stdout")
