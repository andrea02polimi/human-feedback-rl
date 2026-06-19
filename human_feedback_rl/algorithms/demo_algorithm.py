"""Demonstration-based reward learning via MaxEnt IRL.

A fixed set of expert trajectories is supplied at construction. Each iteration the
agent's fresh rollout serves as the model trajectories, the reward model is updated
against the expert set, and the agent is trained on the updated reward.

Three reward losses are available (see ``loss_type``), all of the form
``-mean(R(tau^E)) + partition_term``:

    maxent    L = -mean(R(tau^E)) + logsumexp(R(tau^M)) - log|M|
    maxent_2  L = -mean(R(tau^E)) + logsumexp(R(tau^E u tau^M)) - log|E u M|
    demo      L = -mean(R(tau^E)) + mean(R(tau^M))

Because tau^M are always fresh samples from the current policy, no importance
sampling correction is needed.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Optional

import numpy as np
import torch as th
from scipy.stats import spearmanr

from human_feedback_rl.common.base_algorithm import BaseAlgorithm
from human_feedback_rl.common.loggers import ExcludeFormatLogger, PrefixedLogger
from human_feedback_rl.common.reward_nets import make_reward_ensemble
from human_feedback_rl.common.trajectory_generators import TrajectoryGeneratorFromAgent
from human_feedback_rl.common.types import Trajectory


class DemoAlgorithm(BaseAlgorithm):
    """Alternating reward-learning (MaxEnt IRL) and agent-training loop."""

    STATUS_ARRIVED  = 0
    STATUS_COLLIDED = 1
    STATUS_OFFROAD  = 2
    STATUS_TIMEOUT  = 3
    STATUS_RUNNING  = 4

    VALID_LOSSES = ("maxent", "maxent_2", "demo")

    def __init__(
        self,
        env,
        agent,
        expert_trajectories: List[Trajectory],
        loss_type: str = "maxent",
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
        if loss_type not in self.VALID_LOSSES:
            raise ValueError(f"loss_type must be one of {self.VALID_LOSSES}, got {loss_type!r}.")

        super().__init__(env, agent, rng, log_folder=log_folder, output_formats=output_formats)

        self.expert_trajectories     = list(expert_trajectories)
        self.loss_type               = loss_type
        self.gradient_steps_rew      = gradient_steps_rew
        self.batch_size_expert       = batch_size_expert
        self.batch_size_model        = batch_size_model
        self.initial_agent_timesteps = initial_agent_timesteps
        self.exploration_frac        = exploration_frac
        self.temperature             = temperature
        self.trajectories            = []
        self.debug_dataset           = debug_dataset or {}

        # The debug dataset is a flat, sequentially-ordered list of transitions;
        # split it on `done` so we can compute return-level ranking metrics on it.
        self._debug_trajectories     = self._split_into_trajectories(self.debug_dataset)

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

        # Warm up the agent first so iteration-0 model trajectories come from a
        # non-random policy, improving the logsumexp partition estimate.
        if self.initial_agent_timesteps > 0:
            print(f"- Pre-warming agent for {self.initial_agent_timesteps} timesteps")
            self._train_agent(self.initial_agent_timesteps, log_interval)

        for iteration in range(n_iterations):
            t_iter = time.perf_counter()
            print(f"\nIteration {iteration}/{n_iterations - 1}")

            exploration_steps = int(self.exploration_frac * timesteps_per_iteration)
            print(f"- Collecting {timesteps_per_iteration} agent + {exploration_steps} exploration transitions")
            self.trajectories = self._sample_rollout(timesteps_per_iteration, exploration_steps)

            # Validate the reward model against ground truth before updating it.
            all_transitions = [t for traj in self.trajectories for t in traj]
            self._log_reward_validation(all_transitions, "reward_val/current_rollout")
            self._log_return_ranking(self.trajectories, "reward_val/current_rollout")
            if self.debug_dataset:
                self._log_reward_validation(self.debug_dataset, "reward_val/debug_dataset")
                self._log_return_ranking(self._debug_trajectories, "reward_val/debug_dataset")

            print("- Training reward model")
            self._train_reward_model()

            # Center rewards before agent training. For off-policy agents such as
            # SAC, older replay-buffer rewards still keep their old values; the
            # relabel debug below measures that stale-reward mismatch.
            self._normalize_reward_mean()
            self._log_replay_reward_staleness()

            print(f"- Training agent for {timesteps_per_iteration} timesteps")
            self._train_agent(timesteps_per_iteration, log_interval)

            self._log_iteration(t_iter, iteration)
            if checkpoint_dir is not None and (iteration + 1) % checkpoint_interval == 0:
                self._save_checkpoint(checkpoint_dir, iteration + 1)

        return self.trajectory_generator.agent

    # ------------------------------------------------------------------
    # Reward model training (MaxEnt IRL)
    # ------------------------------------------------------------------

    @staticmethod
    def _grad_norm(module) -> float:
        """Total L2 norm of the module's current parameter gradients."""
        return float(th.sqrt(sum(
            p.grad.pow(2).sum() for p in module.parameters() if p.grad is not None
        )))

    def _train_reward_model(self) -> None:
        if not self.trajectories:
            return

        t0 = time.perf_counter()

        def train_member(member, optimizer):
            member.train()
            norms = []
            for _ in range(self.gradient_steps_rew):
                loss = self._reward_loss(member)
                optimizer.zero_grad()
                loss.backward()
                norms.append(self._grad_norm(member))
                optimizer.step()
            return norms

        all_norms = []
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = [
                executor.submit(train_member, member, optimizer)
                for member, optimizer in zip(self.reward_model.members, self.optimizers)
            ]
            for future in as_completed(futures):
                all_norms.extend(future.result())

        t_train = time.perf_counter() - t0

        t0 = time.perf_counter()
        self._log_reward_loss_diagnostics()
        self.logger.record("reward/grad_norm",        float(np.mean(all_norms)), exclude="stdout")
        self.logger.record("reward/weight_norm",      self._param_norm(self.reward_model), exclude="stdout")
        self.logger.record("time/train_reward_model", t_train)
        self.logger.record_sum("time/loggings",       time.perf_counter() - t0)

    @staticmethod
    def _param_norm(module) -> float:
        """Total L2 norm of model parameters."""
        return float(th.sqrt(sum(p.detach().pow(2).sum() for p in module.parameters())))

    def _diagnostic_returns(self, member):
        """Deterministic expert/model trajectory returns for reward debugging."""
        expert_trajs = self._even_subset(self.expert_trajectories, self.batch_size_expert)
        model_trajs = self._even_subset(self.trajectories, self.batch_size_model)
        expert_returns = th.stack([self._traj_sum_reward(member, traj) for traj in expert_trajs])
        model_returns = th.stack([self._traj_sum_reward(member, traj) for traj in model_trajs])
        return expert_returns, model_returns

    @staticmethod
    def _even_subset(items, max_items: int):
        """Return a deterministic, evenly-spaced subset without advancing RNG state."""
        n_items = len(items)
        if max_items <= 0:
            max_items = 1
        if n_items <= max_items:
            return list(items)
        indices = np.linspace(0, n_items - 1, max_items, dtype=int)
        return [items[i] for i in indices]

    def _log_reward_loss_diagnostics(self) -> None:
        """Log loss-specific reward diagnostics that are meaningful for IRL."""
        if not self.trajectories:
            return

        self.reward_model.eval()
        with th.no_grad():
            expert_returns, model_returns = self._diagnostic_returns(self.reward_model)
            expert_term = -expert_returns.mean()
            model_mean = model_returns.mean()

            all_returns = th.cat([expert_returns, model_returns], dim=0)
            margin = expert_returns.mean() - model_mean
            abs_mean = all_returns.abs().mean()

            self.logger.record("reward/expert_return_mean", float(expert_returns.mean()), exclude="stdout")
            self.logger.record("reward/model_return_mean",  float(model_mean), exclude="stdout")
            self.logger.record("reward/expert_model_margin", float(margin), exclude="stdout")
            self.logger.record("reward/return_std",         float(all_returns.std(unbiased=False)), exclude="stdout")
            self.logger.record("reward/return_abs_mean",    float(abs_mean), exclude="stdout")
            self.logger.record("reward/return_min",         float(all_returns.min()), exclude="stdout")
            self.logger.record("reward/return_max",         float(all_returns.max()), exclude="stdout")

            if self.loss_type == "demo":
                loss = expert_term + model_mean
                self.logger.record("reward/loss",            float(loss), exclude="stdout")
                self.logger.record("reward/demo_margin",     float(margin), exclude="stdout")
                self.logger.record("reward/demo_scale_std",  float(all_returns.std(unbiased=False)), exclude="stdout")
                self.logger.record("reward/demo_scale_abs",  float(abs_mean), exclude="stdout")
                self.reward_model.train()
                return

            if self.loss_type == "maxent_2":
                partition = th.logsumexp(all_returns, dim=0) - np.log(len(all_returns))
                weights = th.softmax(all_returns, dim=0)
                expert_mass = weights[:len(expert_returns)].sum()
                model_mass = weights[len(expert_returns):].sum()
                top1_weight, top1_idx = weights.max(dim=0)
                ess = 1.0 / weights.pow(2).sum()

                self.logger.record("reward/loss",                         float(expert_term + partition), exclude="stdout")
                self.logger.record("reward/maxent2_partition_all",        float(partition), exclude="stdout")
                self.logger.record("reward/maxent2_expert_softmax_mass",  float(expert_mass), exclude="stdout")
                self.logger.record("reward/maxent2_model_softmax_mass",   float(model_mass), exclude="stdout")
                self.logger.record("reward/maxent2_top1_softmax_weight",  float(top1_weight), exclude="stdout")
                self.logger.record("reward/maxent2_top1_is_expert",       float((top1_idx < len(expert_returns)).item()), exclude="stdout")
                self.logger.record("reward/maxent2_effective_sample_size", float(ess), exclude="stdout")
                self.reward_model.train()
                return

            partition = th.logsumexp(model_returns, dim=0) - np.log(len(model_returns))
            weights = th.softmax(model_returns, dim=0)
            top_values = th.topk(weights, k=min(5, len(weights))).values
            ess = 1.0 / weights.pow(2).sum()

            self.logger.record("reward/loss",                          float(expert_term + partition), exclude="stdout")
            self.logger.record("reward/maxent_partition_model",        float(partition), exclude="stdout")
            self.logger.record("reward/maxent_top1_softmax_weight",    float(weights.max()), exclude="stdout")
            self.logger.record("reward/maxent_top5_softmax_mass",      float(top_values.sum()), exclude="stdout")
            self.logger.record("reward/maxent_effective_sample_size",  float(ess), exclude="stdout")
        self.reward_model.train()

    def _sample_returns(self, member):
        """Return (expert_returns, model_returns): per-trajectory return tensors
        for a random mini-batch of expert and current-model trajectories."""
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
        return expert_returns, model_returns

    def _reward_loss(self, member) -> th.Tensor:
        """IRL loss selected by ``self.loss_type``.

        All variants share the ``-mean(R(tau^E))`` term and differ only in the
        partition term. ``logsumexp(returns) - log(n)`` is a numerically stable
        estimate of the log partition function over the sampled distribution.
        """
        expert_returns, model_returns = self._sample_returns(member)
        expert_term = -expert_returns.mean()

        if self.loss_type == "demo":
            return expert_term + model_returns.mean()

        if self.loss_type == "maxent_2":
            all_returns = th.cat([model_returns, expert_returns], dim=0)
            return expert_term + th.logsumexp(all_returns, dim=0) - np.log(len(all_returns))

        # "maxent": partition over model samples only.
        return expert_term + th.logsumexp(model_returns, dim=0) - np.log(len(model_returns))

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
        """Shift each NormalizedRewardNet's EMA mean to zero on the current rollout.

        The reward model is double-wrapped: each ensemble member has its own
        NormalizedRewardNet and the ensemble is wrapped again. Both layers are
        recentered here; only the mean is changed, sigma stays at its initial value.
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

        # Centering each member leaves the outer wrapper near zero-mean; update it
        # explicitly to remove any residual bias.
        raw = self.reward_model.predict_unnormalized(obs, acts, status, done)
        self.reward_model.set_mean(raw.mean())

    def _log_replay_reward_staleness(self, batch_size: int = 2048) -> None:
        """Compare stored off-policy rewards with current reward-model predictions.

        SAC trains Q targets from replay-buffer rewards. Without relabelling, old
        transitions keep rewards produced by older reward-model parameters/stats.
        This debug metric estimates that mismatch without changing training.
        """
        if getattr(self.agent, "replay_buffer", None) is None:
            return

        reward_wrapper = self._find_relabel_debug_wrapper(self.trajectory_generator.venv)
        if reward_wrapper is None:
            return

        batch = reward_wrapper.sample_relabel_debug_batch(batch_size, self.rng)
        if batch is None:
            return

        obs, actions, status, done, stored_rewards = batch
        self.reward_model.eval()
        with th.no_grad():
            current_rewards = self.reward_model.predict(obs, actions, status, done)
        self.reward_model.train()

        delta = current_rewards - stored_rewards
        abs_delta = np.abs(delta)
        stored_std = float(np.std(stored_rewards))
        current_std = float(np.std(current_rewards))
        denom = current_std if current_std > 1e-8 else 1.0

        self.logger.record("replay_relabel_debug/sample_size",          len(stored_rewards), exclude="stdout")
        self.logger.record("replay_relabel_debug/stored_reward_mean",   float(np.mean(stored_rewards)), exclude="stdout")
        self.logger.record("replay_relabel_debug/current_reward_mean",  float(np.mean(current_rewards)), exclude="stdout")
        self.logger.record("replay_relabel_debug/stored_reward_std",    stored_std, exclude="stdout")
        self.logger.record("replay_relabel_debug/current_reward_std",   current_std, exclude="stdout")
        self.logger.record("replay_relabel_debug/delta_mean",           float(np.mean(delta)), exclude="stdout")
        self.logger.record("replay_relabel_debug/delta_std",            float(np.std(delta)), exclude="stdout")
        self.logger.record("replay_relabel_debug/delta_abs_mean",       float(np.mean(abs_delta)), exclude="stdout")
        self.logger.record("replay_relabel_debug/delta_abs_p95",        float(np.percentile(abs_delta, 95)), exclude="stdout")
        self.logger.record("replay_relabel_debug/staleness_ratio",      float(np.mean(abs_delta) / denom), exclude="stdout")
        self.logger.record("replay_relabel_debug/sign_flip_frac",       float(np.mean(np.sign(stored_rewards) != np.sign(current_rewards))), exclude="stdout")

        if stored_std > 1e-8 and current_std > 1e-8:
            corr = np.corrcoef(stored_rewards, current_rewards)[0, 1]
            self.logger.record("replay_relabel_debug/stored_current_corr", float(corr), exclude="stdout")

    @staticmethod
    def _find_relabel_debug_wrapper(venv):
        current = venv
        while current is not None:
            if hasattr(current, "sample_relabel_debug_batch"):
                return current
            current = getattr(current, "venv", None)
        return None

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
        """Log reward scale, status shortcuts, and ensemble disagreement."""
        _, pred_rewards, pred_std, status = self._run_reward_inference(transitions)

        arrived_mask  = status[:, self.STATUS_ARRIVED]  == 1
        collided_mask = status[:, self.STATUS_COLLIDED] == 1
        offroad_mask  = status[:, self.STATUS_OFFROAD]  == 1
        timeout_mask  = status[:, self.STATUS_TIMEOUT]  == 1
        running_mask  = status[:, self.STATUS_RUNNING]  == 1

        self.logger.record(f"{log_class}/reward_mean", float(np.mean(pred_rewards)))
        self.logger.record(f"{log_class}/reward_std",  float(np.std(pred_rewards)))
        self.logger.record(f"{log_class}/reward_min",  float(np.min(pred_rewards)))
        self.logger.record(f"{log_class}/reward_max",  float(np.max(pred_rewards)))

        self._record_masked_mean(f"{log_class}/reward_running",  pred_rewards, running_mask)
        self._record_masked_mean(f"{log_class}/reward_arrived",  pred_rewards, arrived_mask)
        self._record_masked_mean(f"{log_class}/reward_collided", pred_rewards, collided_mask)
        self._record_masked_mean(f"{log_class}/reward_offroad",  pred_rewards, offroad_mask)
        self._record_masked_mean(f"{log_class}/reward_timeout",  pred_rewards, timeout_mask)

        # Rising ensemble disagreement signals inputs that are becoming OOD for the reward model.
        self.logger.record(f"{log_class}/ensemble_std", float(np.mean(pred_std)))
        self._record_masked_mean(f"{log_class}/ensemble_std_running", pred_std, running_mask)

    def _record_masked_mean(self, key: str, values: np.ndarray, mask: np.ndarray) -> None:
        if np.any(mask):
            self.logger.record(key, float(np.mean(values[mask])))

    def _run_reward_inference(self, transitions):
        """Run the reward model in eval+no_grad mode; return (true, pred_mean, pred_std, status)."""
        self.reward_model.eval()
        with th.no_grad():
            true_rewards = np.array([t.true_reward for t in transitions], dtype=np.float32)
            obs          = np.array([t.observation  for t in transitions], dtype=np.float32)
            acts         = np.array([t.action       for t in transitions], dtype=np.float32)
            status       = np.array([t.next_status  for t in transitions], dtype=np.float32)
            done         = np.array([float(t.done)  for t in transitions], dtype=np.float32)
            pred_rewards = self.reward_model.predict(obs, acts, status, done)
            # Per-member predictions (N, n_members); std across members = disagreement.
            pred_std     = self.reward_model.predict_all(obs, acts, status, done).std(axis=1)
        self.reward_model.train()
        return true_rewards, pred_rewards, pred_std, status

    @staticmethod
    def _split_into_trajectories(transitions) -> List[Trajectory]:
        """Split a flat, sequentially-ordered transition list into trajectories on `done`."""
        trajectories, current = [], Trajectory()
        for t in transitions:
            current.add_transition(t)
            if t.done:
                trajectories.append(current)
                current = Trajectory()
        if len(current) > 0:
            trajectories.append(current)
        return trajectories

    def _log_return_ranking(self, trajectories, log_class: str) -> None:
        """Spearman rho between predicted and true trajectory returns (ranking quality)."""
        if len(trajectories) < 2:
            return
        true_returns, pred_returns = [], []
        for traj in trajectories:
            _, pred_rewards, _, _ = self._run_reward_inference(traj)
            true_returns.append(float(sum(t.true_reward for t in traj)))
            pred_returns.append(float(pred_rewards.sum()))
        rho, _ = spearmanr(true_returns, pred_returns)
        self.logger.record(f"{log_class}/spearman_returns", rho)

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
