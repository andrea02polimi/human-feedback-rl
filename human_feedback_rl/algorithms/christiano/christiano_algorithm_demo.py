import math
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import wandb
import torch as th
import torch.nn.functional as F

from human_feedback_rl.common import MainLogger, PrefixWrapper
from human_feedback_rl.common.datasets import DemonstrationDataset
from human_feedback_rl.common.fragmenters import SingleFragmenter
from human_feedback_rl.common.gatherers import DemonstrationGathererFromExpert
from human_feedback_rl.common.reward_nets import make_reward_ensemble
from human_feedback_rl.common.schedules import QUERY_SCHEDULES
from human_feedback_rl.common.trajectory_generators import TrajectoryGeneratorFromAgent
from human_feedback_rl.algorithms.christiano._shared import (
    save_reward_model as _save_reward_model,
    collect_debug_data as _collect_debug_data,
    compute_time_decay_weights,
    build_bootstrap_indices,
)


class ChristianoAlgorithmDemo:
    """
    Variant of Christiano et al. (2017) that uses expert demonstrations instead
    of preference comparisons.

    For each demonstration segment the reward model is trained to maximise:

        log σ( mean_t r̂(o_t, a^expert_t) )

    where a^expert_t is the action the expert would take at observation o_t.
    This pushes predicted rewards higher for (obs, expert_action) pairs.

    An optional regularisation term penalises the squared mean reward to keep
    predicted values centred near zero and prevent explosion.
    """

    def __init__(
        self,
        env,
        agent,
        expert_policy,
        lr_rew: float = 3e-4,
        batch_size_rew: int = 256,
        n_epochs_rew: int = 10,
        n_ensembles_rew: int = 3,
        n_iterations: int = 50,
        fragment_length: int = 1,
        query_schedule: Union[str, Callable[[float], float]] = "constant",
        use_reward_reg: bool = True,
        reward_mean_reg: float = 0.1,
        demo_train_frac: float = 0.8,
        demo_queue_size: int = 1_000_000,
        rng: Optional[np.random.Generator] = None,
    ):
        self.batch_size_rew  = batch_size_rew
        self.n_epochs_rew    = n_epochs_rew
        self.n_iterations    = n_iterations
        self.fragment_length = fragment_length
        self.use_reward_reg  = use_reward_reg
        self.reward_mean_reg = reward_mean_reg
        self.rng             = rng if rng is not None else np.random.default_rng()
        self._iteration      = 0
        self._rm_global_epoch = 0

        self.logger    = MainLogger()
        self.rm_logger = MainLogger()

        if wandb.run is not None:
            wandb.define_metric("rm/*",   step_metric="rm/epoch")
            wandb.define_metric("ppo/*",  step_metric="iteration")
            wandb.define_metric("env/*",  step_metric="iteration")
            wandb.define_metric("hack/*", step_metric="iteration")

        self.query_schedule      = QUERY_SCHEDULES[query_schedule]
        self.query_schedule_name = query_schedule

        self.reward_model = make_reward_ensemble(env, n_ensembles=n_ensembles_rew)

        self.optimizers = [
            th.optim.Adam(member.parameters(), lr=lr_rew, weight_decay=1e-4)
            for member in self.reward_model.members
        ]

        self.trajectory_generator = TrajectoryGeneratorFromAgent(
            agent=agent,
            reward_model=self.reward_model,
            venv=env,
            rng=self.rng,
            logger=self.logger,
        )

        self.fragmenter = SingleFragmenter(
            logger=PrefixWrapper(self.logger, "fragmenter"),
            rng=self.rng,
        )

        self.dataset = DemonstrationDataset(
            train_frac=demo_train_frac,
            queue_size=demo_queue_size,
        )

        self.demo_gatherer = DemonstrationGathererFromExpert(expert_policy=expert_policy)

    # ──────────────────────────────────────────────────────────────────────────
    # Main training loop
    # ──────────────────────────────────────────────────────────────────────────

    def train(self, total_timesteps: int = 100_000, total_demonstrations: int = 1_000):
        t_vec   = np.linspace(0, 1, self.n_iterations)
        weights = np.array([self.query_schedule(t) for t in t_vec])
        probs   = weights / weights.sum()
        schedule = np.round(probs * total_demonstrations).astype(int).tolist()
        print(f"- Demo {self.query_schedule_name} schedule: {schedule}")

        timesteps_per_iteration, extra_timesteps = divmod(total_timesteps, self.n_iterations)

        for i, num_demo in enumerate(schedule):
            self.logger.log(f"\nIteration {i + 1}/{len(schedule)}")

            num_steps = math.ceil(num_demo * self.fragment_length)
            self.logger.log(f"- Collecting {num_demo} demo segments ({num_steps} transitions)")
            trajectories = self.trajectory_generator.sample(num_steps)

            fragments = self.fragmenter(trajectories, self.fragment_length, num_demo)
            demos     = self.demo_gatherer(fragments)

            self.dataset.push(fragments, demos, i)
            self.logger.log(f"- Dataset: {len(self.dataset)} demonstrations")

            self.logger.log(f"- Training reward model for {self.n_epochs_rew} epochs")
            self.train_reward_model(decay=0.01)

            num_steps = timesteps_per_iteration
            if i == self.n_iterations - 1:
                num_steps += extra_timesteps

            self.logger.log(f"- Training agent for {num_steps} timesteps")
            self.trajectory_generator.train(steps=num_steps)

            self.logger.record("iteration", i)
            self.logger.dump()
            self._iteration += 1

        return self.trajectory_generator.agent

    # ──────────────────────────────────────────────────────────────────────────
    # Reward model training
    # ──────────────────────────────────────────────────────────────────────────

    def train_reward_model(self, decay: float = 0.01) -> None:
        train_data = self.dataset.get_train()
        if not train_data:
            return

        all_weights = compute_time_decay_weights(train_data, timestamp_idx=1, decay=decay)

        for member in self.reward_model.members:
            member.train()

        n_train = len(train_data)
        bootstrap_indices = build_bootstrap_indices(self.rng, n_train, len(self.reward_model.members))

        for epoch in range(self.n_epochs_rew):
            epoch_pref_losses: List[float] = []
            epoch_reg_losses:  List[float] = []

            for mi, (member, optimizer) in enumerate(
                zip(self.reward_model.members, self.optimizers)
            ):
                boot_idx = bootstrap_indices[mi]
                perm     = self.rng.permutation(len(boot_idx))

                for start in range(0, len(perm), self.batch_size_rew):
                    batch_idx = boot_idx[perm[start: start + self.batch_size_rew]]

                    pair_batch = [train_data[k] for k in batch_idx]

                    w = all_weights[batch_idx]
                    w = w / w.sum()
                    batch_weights = th.tensor(w, dtype=th.float32)

                    seg_losses:       List[th.Tensor] = []
                    all_step_rewards: List[th.Tensor] = []

                    for (demo, fragment), _ in pair_batch:
                        obs_e = th.tensor(np.array([t.observation for t in demo]), dtype=th.float32)
                        act_e = th.tensor(np.array([t.action for t in demo]), dtype=th.float32)
                        r_e = member(obs_e, act_e)
                        R_e = r_e.mean()

                        obs_a = th.tensor(np.array([t.observation for t in fragment]), dtype=th.float32)
                        act_a = th.tensor(np.array([t.action for t in fragment]), dtype=th.float32)
                        r_a = member(obs_a, act_a)
                        R_a = r_a.mean()

                        seg_losses.append(-F.logsigmoid(R_e - R_a))
                        all_step_rewards.append(r_e)
                        all_step_rewards.append(r_a)

                    seg_losses_t = th.stack(seg_losses)
                    pref_loss    = (batch_weights * seg_losses_t).sum()

                    if self.use_reward_reg:
                        all_r    = th.cat(all_step_rewards)
                        reg_loss = self.reward_mean_reg * all_r.mean().pow(2)
                    else:
                        reg_loss = th.tensor(0.0)

                    loss = pref_loss + reg_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_pref_losses.append(pref_loss.item())
                    epoch_reg_losses.append(reg_loss.item())

            train_loss, train_mean_rew = self._evaluate_demo_loss("train")
            val_loss,   val_mean_rew   = self._evaluate_demo_loss("val")

            mean_pref_loss = float(np.mean(epoch_pref_losses))
            mean_reg_loss  = float(np.mean(epoch_reg_losses))

            if wandb.run is not None:
                val_rewards = self._collect_val_rewards()
                wandb.log(
                    {"rm/reward_histogram": wandb.Histogram(val_rewards.tolist())},
                    commit=False,
                )

            self.rm_logger.record("rm/epoch",             self._rm_global_epoch)
            self.rm_logger.record("rm/train_loss",        train_loss)
            self.rm_logger.record("rm/val_loss",          val_loss)
            self.rm_logger.record("rm/overfit_gap_loss",  val_loss - train_loss)
            self.rm_logger.record("rm/train_mean_reward", train_mean_rew)
            self.rm_logger.record("rm/val_mean_reward",   val_mean_rew)
            self.rm_logger.record("rm/loss_pref",         mean_pref_loss)
            self.rm_logger.record("rm/loss_reg",          mean_reg_loss)
            self.rm_logger.record("rm/loss_total",        mean_pref_loss + mean_reg_loss)
            self.rm_logger.dump()

            self._rm_global_epoch += 1

        for member in self.reward_model.members:
            member.eval()

    # ──────────────────────────────────────────────────────────────────────────
    # Evaluation helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _evaluate_demo_loss(self, split: str) -> Tuple[float, float]:
        """Compute demo loss and mean predicted reward on `split`. Returns (loss, mean_reward)."""
        data = self.dataset.get_train() if split == "train" else self.dataset.get_val()
        if not data:
            return 0.0, 0.0

        self.reward_model.eval()
        losses:    List[float] = []
        mean_rews: List[float] = []

        with th.no_grad():
            for (demo, fragment), _ in data:
                obs  = th.tensor(np.array([t.observation for t in demo]), dtype=th.float32)
                acts = th.tensor(np.array([t.action      for t in demo]), dtype=th.float32)

                r = self.reward_model(obs, acts)
                R = r.mean()

                losses.append(-F.logsigmoid(R).item())
                mean_rews.append(r.mean().item())

        self.reward_model.train()
        return float(np.mean(losses)), float(np.mean(mean_rews))

    def _collect_val_rewards(self) -> np.ndarray:
        """Return flat array of per-step predicted rewards on the val set."""
        val_data = self.dataset.get_val()
        if not val_data:
            return np.array([0.0], dtype=np.float32)

        all_rewards: List[float] = []
        self.reward_model.eval()

        with th.no_grad():
            for (demo, fragment), _ in val_data:
                obs  = th.tensor(np.array([t.observation for t in demo]), dtype=th.float32)
                acts = th.tensor(np.array([t.action      for t in demo]), dtype=th.float32)
                r = self.reward_model(obs, acts)
                all_rewards.extend(r.cpu().numpy().tolist())

        self.reward_model.train()
        return np.array(all_rewards, dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # Serialisation
    # ──────────────────────────────────────────────────────────────────────────

    def save_reward_model(self, path) -> None:
        _save_reward_model(self.reward_model, path)

    def collect_debug_data(self, n_steps: int = 2000) -> dict:
        return _collect_debug_data(self.trajectory_generator, self.reward_model, n_steps)