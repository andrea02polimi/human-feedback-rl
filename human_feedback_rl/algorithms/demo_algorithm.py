"""
DemoAlgorithm: demonstration-based reward learning using MaxEnt IRL.

A fixed set of expert trajectories is passed at initialisation. At each
iteration the agent's current rollout serves as the model trajectories. The
reward model is updated by minimising the MaxEnt IRL loss:

    L = -mean(R_θ(τ^E)) + logsumexp(R_θ(τ^M)) - log(M)

Because τ^M are always fresh samples from the current policy, no importance
sampling correction is needed.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Optional

import numpy as np
import torch as th
from scipy.stats import kendalltau

from human_feedback_rl.common.base_algorithm import BaseAlgorithm
from human_feedback_rl.common.loggers import ExcludeFormatLogger, PrefixedLogger
from human_feedback_rl.common.reward_nets import make_reward_ensemble
from human_feedback_rl.common.trajectory_generators import TrajectoryGeneratorFromAgent
from human_feedback_rl.common.types import Trajectory


class DemoAlgorithm(BaseAlgorithm):
    """
    Demonstration-based reward learning via MaxEnt IRL.

    Expert trajectories are passed once at initialisation and stay fixed for
    the entire training run.  At each iteration the algorithm:

      1. Collects agent trajectories via rollout.
      2. Updates the reward model with:
            L = -mean(R_θ(τ^E)) + logsumexp(R_θ(τ^M)) - log(M)
         sampling mini-batches from the fixed expert set and the current rollout.
      3. Normalises the reward model mean and trains the agent with PPO.
    """

    STATUS_ARRIVED  = 0
    STATUS_COLLIDED = 1
    STATUS_OFFROAD  = 2
    STATUS_TIMEOUT  = 3
    STATUS_RUNNING  = 4

    def __init__(
        self,
        env,
        agent,
        expert_trajectories: List[Trajectory],
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
    ):
        if not expert_trajectories:
            raise ValueError("expert_trajectories must be a non-empty list of Trajectory objects.")

        super().__init__(env, agent, rng, log_folder=log_folder, output_formats=output_formats)

        self.expert_trajectories     = list(expert_trajectories)
        self.gradient_steps_rew      = gradient_steps_rew
        self.batch_size_expert       = batch_size_expert
        self.batch_size_model        = batch_size_model
        self.initial_agent_timesteps = initial_agent_timesteps
        self.exploration_frac        = exploration_frac
        self.temperature             = temperature
        self.trajectories            = []
        self.debug_dataset           = debug_dataset or {}

        self.reward_model = make_reward_ensemble(env, **(reward_model_kwargs or {}))

        agent.set_logger(ExcludeFormatLogger(PrefixedLogger(self.logger, "agent"), exclude="stdout"))
        self.trajectory_generator = TrajectoryGeneratorFromAgent(
            venv=env,
            agent=agent,
            reward_model=self.reward_model,
            exploration_eps=exploration_eps,
            rng=self.rng,
            logger=self.logger,
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

        # Train the agent before the first reward model update so that model
        # trajectories at iteration 0 come from a non-random policy, improving
        # the logsumexp partition function estimate.
        if self.initial_agent_timesteps > 0:
            print(f"- Pre-warming agent for {self.initial_agent_timesteps} timesteps")
            self._train_agent(self.initial_agent_timesteps, log_interval)

        for iteration in range(n_iterations):
            t_iter = time.perf_counter()
            print(f"\nIteration {iteration}/{n_iterations - 1}")

            # ---- Phase 1: Collect agent rollout -------------------------
            exploration_steps = int(self.exploration_frac * timesteps_per_iteration)
            print(f"- Collecting {timesteps_per_iteration} agent + {exploration_steps} exploration transitions")
            self.trajectories = self._sample_rollout(timesteps_per_iteration, exploration_steps)

            self.logger.record("dataset/expert_size", len(self.expert_trajectories))
            self.logger.record("dataset/model_size",  len(self.trajectories))

            # ---- Phase 2: Validate reward model against ground truth ----
            all_transitions = [t for traj in self.trajectories for t in traj]
            self._log_reward_validation(all_transitions, "reward_val/current_rollout")
            if self.debug_dataset:
                self._log_reward_validation(self.debug_dataset, "reward_val/debug_dataset")

            # ---- Phase 3: Update reward model (MaxEnt IRL) --------------
            print("- Training reward model")
            self._train_reward_model()

            # ---- Phase 4: Normalize reward model mean -------------------
            # Centers the reward model output to zero-mean on the current rollout
            # before PPO training so the value function baseline stays well-scaled.
            self._normalize_reward_mean()

            # ---- Phase 5: Train agent with updated reward model ---------
            print(f"- Training agent for {timesteps_per_iteration} timesteps")
            self._train_agent(timesteps_per_iteration, log_interval)

            # ---- Logging and checkpointing ------------------------------
            self._log_iteration(t_iter, iteration)
            if checkpoint_dir is not None and (iteration + 1) % checkpoint_interval == 0:
                self._save_checkpoint(checkpoint_dir, iteration + 1)

        return self.trajectory_generator.agent

    # ------------------------------------------------------------------
    # Reward model training (MaxEnt IRL)
    # ------------------------------------------------------------------

    def _train_reward_model(self) -> None:
        if not self.trajectories:
            return

        t0 = time.perf_counter()

        def train_member(member, optimizer):
            member.train()
            for _ in range(self.gradient_steps_rew):
                loss = self._demo_loss(member)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = [
                executor.submit(train_member, member, optimizer)
                for member, optimizer in zip(self.reward_model.members, self.optimizers)
            ]
            for future in as_completed(futures):
                future.result()

        t_train = time.perf_counter() - t0

        t0 = time.perf_counter()
        loss_val = self._evaluate_reward_model()
        self.logger.record("reward/loss_val",         loss_val, exclude="stdout")
        self.logger.record("time/train_reward_model", t_train)
        self.logger.record_sum("time/loggings",       time.perf_counter() - t0)

    def _evaluate_reward_model(self) -> float:
        """MaxEnt IRL loss on the current expert and model batches (no grad, eval mode)."""
        if not self.trajectories:
            return float("nan")
        self.reward_model.eval()
        with th.no_grad():
            loss = self._demo_loss(self.reward_model).item()
        self.reward_model.train()
        return loss

    def _maxent_loss(self, member) -> th.Tensor:
        """L = -mean(R_θ(τ^E)) + logsumexp(R_θ(τ^M)) - log(M)"""
        n_e     = min(self.batch_size_expert, len(self.expert_trajectories))
        exp_idx = self.rng.choice(len(self.expert_trajectories), size=n_e, replace=False)
        expert_returns = th.stack([
            self._traj_sum_reward(member, self.expert_trajectories[i]) for i in exp_idx
        ])

        n_m       = min(self.batch_size_model, len(self.trajectories))
        model_idx = self.rng.choice(len(self.trajectories), size=n_m, replace=False)
        model_returns = th.stack([
            self._traj_sum_reward(member, self.trajectories[i]) for i in model_idx
        ])

        # logsumexp(returns) - log(n_m) is a numerically stable estimate of
        # log Z_θ, the log partition function over the current policy distribution.
        log_z = th.logsumexp(model_returns, dim=0) - np.log(n_m)

        return -expert_returns.mean() + log_z

    def _demo_loss(self, member) -> th.Tensor:
        """L = -mean(R_θ(τ^E)) + mean(R_θ(τ^A))"""
        n_e     = min(self.batch_size_expert, len(self.expert_trajectories))
        exp_idx = self.rng.choice(len(self.expert_trajectories), size=n_e, replace=False)
        expert_returns = th.stack([
            self._traj_sum_reward(member, self.expert_trajectories[i]) for i in exp_idx
        ])

        n_m       = min(self.batch_size_model, len(self.trajectories))
        model_idx = self.rng.choice(len(self.trajectories), size=n_m, replace=False)
        model_returns = th.stack([
            self._traj_sum_reward(member, self.trajectories[i]) for i in model_idx
        ])

        return -expert_returns.mean() + model_returns.mean()

    def _traj_sum_reward(self, member, traj: Trajectory) -> th.Tensor:
        """Sum of per-step rewards over a trajectory (supports gradients)."""
        obs         = th.tensor(np.array([t.observation for t in traj]),  dtype=th.float32)
        actions     = th.tensor(np.array([t.action      for t in traj]),  dtype=th.float32)
        next_status = th.tensor(np.array([t.next_status for t in traj]),  dtype=th.float32)
        done        = th.tensor(np.array([float(t.done) for t in traj]),  dtype=th.float32)
        return member(obs, actions, next_status, done).sum()

    # ------------------------------------------------------------------
    # Reward model normalisation
    # ------------------------------------------------------------------

    def _normalize_reward_mean(self) -> None:
        """Shift each NormalizedRewardNet's EMA mean so output is zero-mean on current rollout.

        The reward model is double-wrapped: each ensemble member has its own
        NormalizedRewardNet, and the whole ensemble is wrapped again. Both layers
        are updated here. Only the mean is modified; σ stays at its initial value.
        """
        all_transitions = [t for traj in self.trajectories for t in traj]
        if not all_transitions:
            return

        obs    = np.array([t.observation for t in all_transitions])
        acts   = np.array([t.action      for t in all_transitions])
        status = np.array([t.next_status for t in all_transitions])
        done   = np.array([float(t.done) for t in all_transitions])

        for member in self.reward_model.members:
            raw = member.predict_unnormalized(obs, acts, status, done)
            member.set_mean(raw.mean())

        # After centering each member, the outer wrapper should already be near
        # zero-mean, but we update it explicitly to remove any residual bias.
        raw = self.reward_model.predict_unnormalized(obs, acts, status, done)
        self.reward_model.set_mean(raw.mean())

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
        self.logger.record("rollout/n_trajectories",    len(trajectories))
        self.logger.record("rollout/total_transitions", int(np.sum(lengths)))
        self.logger.record("time/sample_rollout",       t_sample)
        self.logger.record_sum("time/loggings",         time.perf_counter() - t0)

        return trajectories

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

    def _log_reward_validation(self, transitions, log_class: str) -> None:
        """Log MAE per terminal-state type and Kendall τ on running steps."""
        true_rewards, pred_rewards, status = self._run_reward_inference(transitions)
        pred_norm = self._align_predictions(pred_rewards, true_rewards, status)

        arrived_mask  = status[:, self.STATUS_ARRIVED]  == 1
        collided_mask = status[:, self.STATUS_COLLIDED] == 1
        offroad_mask  = status[:, self.STATUS_OFFROAD]  == 1
        timeout_mask  = status[:, self.STATUS_TIMEOUT]  == 1
        running_mask  = status[:, self.STATUS_RUNNING]  == 1

        self.logger.record(f"{log_class}/mae_arrived",  np.mean(np.abs(pred_norm[arrived_mask]  - true_rewards[arrived_mask])))
        self.logger.record(f"{log_class}/mae_collided", np.mean(np.abs(pred_norm[collided_mask] - true_rewards[collided_mask])))
        self.logger.record(f"{log_class}/mae_offroad",  np.mean(np.abs(pred_norm[offroad_mask]  - true_rewards[offroad_mask])))
        self.logger.record(f"{log_class}/mae_timeout",  np.mean(np.abs(pred_norm[timeout_mask]  - true_rewards[timeout_mask])))
        self.logger.record(f"{log_class}/mae_running",  np.mean(np.abs(pred_norm[running_mask]  - true_rewards[running_mask])))
        kt_running, _ = kendalltau(true_rewards[running_mask], pred_norm[running_mask])
        self.logger.record(f"{log_class}/kendall_running", kt_running)

    def _run_reward_inference(self, transitions):
        """Run reward model in eval+no_grad mode; return (true, pred, status) arrays."""
        self.reward_model.eval()
        with th.no_grad():
            true_rewards = np.array([t.true_reward for t in transitions], dtype=np.float32)
            obs          = np.array([t.observation  for t in transitions], dtype=np.float32)
            acts         = np.array([t.action       for t in transitions], dtype=np.float32)
            status       = np.array([t.next_status  for t in transitions], dtype=np.float32)
            done         = np.array([float(t.done)  for t in transitions], dtype=np.float32)
            pred_rewards = self.reward_model.predict(obs, acts, status, done)
        self.reward_model.train()
        return true_rewards, pred_rewards, status

    def _align_predictions(
        self,
        pred_rewards: np.ndarray,
        true_rewards: np.ndarray,
        status: np.ndarray,
    ) -> np.ndarray:
        """Shift pred_rewards to match the true-reward mean on running steps (for logging only)."""
        pred_rewards = pred_rewards * self.temperature

        # Use only running steps to compute the shift, so terminal bonuses/penalties
        # don't distort the mean estimate.
        running_mask = status[:, self.STATUS_RUNNING] == 1
        shift = np.mean(true_rewards[running_mask]) - np.mean(pred_rewards[running_mask])
        return pred_rewards + shift

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
        self.trajectory_generator.agent.save(os.path.join(ckpt_path, "agent"))
        print(f"  checkpoint saved in {ckpt_path}")
