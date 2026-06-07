"""
DemoAlgorithm: demonstration-based reward learning using MaxEnt IRL.

A fixed set of expert trajectories is passed at initialisation. At each
iteration the agent's current rollout serves as the model trajectories. The
reward model is updated by minimising the MaxEnt IRL loss:

    L = -mean(R_θ(τ^E)) + logsumexp(R_θ(τ^M)) - log(M)

Because τ^M are always fresh samples from the current policy, no importance
sampling correction is needed.

Model batch sampling uses two buffers to keep log Z calibrated even when the
agent converges to the expert distribution:
  - anchor_buffer: trajectories from the first n_anchor_iterations (permanent)
  - model_buffer:  rolling window of recent trajectories (FIFO, size model_buffer_size)
A fixed fraction (anchor_frac) of the model mini-batch is drawn from the anchor
buffer, guaranteeing that low-quality trajectories never disappear from the
partition function estimate.
"""

import collections
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, List, Optional, Union

import numpy as np
import torch as th

from human_feedback_rl.common.base_reward_learning_algorithm import BaseRewardLearningAlgorithm
from human_feedback_rl.common.reward_nets import make_reward_ensemble
from human_feedback_rl.common.types import Trajectory


class DemoAlgorithm(BaseRewardLearningAlgorithm):
    """
    Demonstration-based reward learning via MaxEnt IRL.

    Expert trajectories are passed once at initialisation and stay fixed for
    the entire training run.  At each iteration the algorithm:

      1. Collects agent trajectories via rollout (self.trajectories).
      2. Updates the reward model with:
            L = -mean(R_θ(τ^E)) + logsumexp(R_θ(τ^M)) - log(M)
         sampling mini-batches from the fixed expert set and the current rollout.
      3. Trains the agent with the updated reward model.
    """

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
        fragment_length: int = 1,
        temperature: float = 1,
        initial_queries: int = 0,
        initial_agent_timesteps: int = 0,
        exploration_frac: float = 0.0,
        exploration_eps: float = 0.5,
        query_schedule: Union[str, Callable[[float], float]] = "constant",
        reward_model_kwargs: Optional[dict] = None,
        model_buffer_size: int = 500,
        n_anchor_iterations: int = 3,
        anchor_frac: float = 0.4,
        grad_clip_rew: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
        log_folder: Optional[str] = None,
        output_formats: Optional[List] = None,
        debug_dataset: Optional[dict] = None,
    ):
        if not expert_trajectories:
            raise ValueError("expert_trajectories must be a non-empty list of Trajectory objects.")

        reward_model = make_reward_ensemble(env, **(reward_model_kwargs or {}))

        super().__init__(
            env=env,
            agent=agent,
            reward_model=reward_model,
            train_comparison_frac=1.0,
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
        )

        self.expert_trajectories      = list(expert_trajectories)
        self.gradient_steps_rew       = gradient_steps_rew
        self.batch_size_expert        = batch_size_expert
        self.batch_size_model         = batch_size_model
        self.initial_agent_timesteps  = initial_agent_timesteps

        self.model_buffer: collections.deque = collections.deque(maxlen=model_buffer_size)
        self.anchor_buffer: List[Trajectory] = []
        self.n_anchor_iterations              = n_anchor_iterations
        self.anchor_frac                      = anchor_frac

        self.grad_clip_rew = grad_clip_rew

        self.optimizers = [
            th.optim.Adam(m.parameters(), lr=lr_rew, weight_decay=l2_rew)
            for m in self.reward_model.members
        ]

    # ------------------------------------------------------------------
    # Optional hook implementations
    # ------------------------------------------------------------------

    def before_train(self, timesteps_per_iteration: int, log_interval: int) -> None:
        """Warm up the agent before the first reward model update.

        Runs only when initial_agent_timesteps > 0. Training the agent first
        ensures that the trajectories used in logsumexp(R_θ(τ^M)) at iteration 0
        come from a non-random policy, improving the partition function estimate.

        Pre-warmup anchor collection: one rollout is sampled from the initial
        (pre-warmup) policy and permanently added to the anchor buffer BEFORE
        warmup training begins. This guarantees genuinely low-quality trajectories
        are always present in log Z, regardless of how long the warmup runs.
        """
        if self.initial_agent_timesteps <= 0:
            return

        print(f"- Collecting pre-warmup anchor trajectories from initial policy ({timesteps_per_iteration} steps)")
        pre_warmup_trajs = self.sample_rollout(timesteps_per_iteration)
        self.anchor_buffer.extend(pre_warmup_trajs)
        print(f"  → {len(pre_warmup_trajs)} pre-warmup trajectories added to anchor buffer")

        print(f"- Pre-warming agent for {self.initial_agent_timesteps} timesteps before first reward model update")
        self.train_agent(self.initial_agent_timesteps, log_interval)

    # ------------------------------------------------------------------
    # Abstract hook implementations
    # ------------------------------------------------------------------

    def collect_feedback(self, _num_queries: int):
        """No feedback collection needed; expert data is fixed at init."""
        return [], []

    def push_data(self, _fragments, _feedback) -> None:
        """No dataset to update; model trajectories come from self.trajectories."""
        self.logger.record("dataset/expert_size",       len(self.expert_trajectories))
        self.logger.record("dataset/model_size",         len(self.trajectories))
        self.logger.record("dataset/model_buffer_size",  len(self.model_buffer))
        self.logger.record("dataset/anchor_buffer_size", len(self.anchor_buffer))

    def train_reward_model(self) -> None:
        """MaxEnt IRL reward model update using the current rollout as model trajectories."""
        if not self.trajectories:
            return

        self.model_buffer.extend(self.trajectories)
        if self.iteration < self.n_anchor_iterations:
            self.anchor_buffer.extend(self.trajectories)

        t0 = time.perf_counter()

        def train_member(member, optimizer):
            member.train()
            for _ in range(self.gradient_steps_rew):
                loss = self._compute_reward_loss(member)
                optimizer.zero_grad()
                loss.backward()
                if self.grad_clip_rew is not None:
                    th.nn.utils.clip_grad_norm_(member.parameters(), self.grad_clip_rew)
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

    def _compute_reward_loss(self, member) -> th.Tensor:
        """Loss used for reward model training. Override in subclasses."""
        return self._maxent_loss(member)

    def _evaluate_reward_model(self) -> float:
        """Reward model loss on a snapshot of the current expert and model batches."""
        if not self.trajectories:
            return float("nan")
        self.reward_model.eval()
        with th.no_grad():
            loss = self._compute_reward_loss(self.reward_model).item()
        self.reward_model.train()
        return loss

    def before_agent_training(self) -> None:
        """Normalize the reward model mean across current agent transitions."""
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

        raw = self.reward_model.predict_unnormalized(obs, acts, status, done)
        self.reward_model.set_mean(raw.mean())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _traj_sum_reward(self, member, traj: Trajectory) -> th.Tensor:
        """Sum of per-step rewards over a trajectory (supports gradients)."""
        obs         = th.tensor(np.array([t.observation  for t in traj]), dtype=th.float32)
        actions     = th.tensor(np.array([t.action       for t in traj]), dtype=th.float32)
        next_status = th.tensor(np.array([t.next_status  for t in traj]), dtype=th.float32)
        done        = th.tensor(np.array([float(t.done)  for t in traj]), dtype=th.float32)
        return member(obs, actions, next_status, done).sum()

    def _maxent_loss(self, member) -> th.Tensor:
        """MaxEnt IRL loss for one member network.

        L = -mean(R_θ(τ^E)) + logsumexp(R_θ(τ^M)) - log(M)

        τ^E: mini-batch from self.expert_trajectories
        τ^M: mini-batch from anchor_buffer (permanent early/low-quality trajectories)
             + model_buffer (rolling window of recent agent trajectories).
             Together they form a diverse off-policy sample that approximates log Z(θ).
             Expert trajectories are deliberately excluded: the gradient of log Z requires
             samples from p(τ|θ) ∝ exp(R_θ(τ)), not from the expert distribution.
        """
        # Expert mini-batch
        n_e     = min(self.batch_size_expert, len(self.expert_trajectories))
        exp_idx = self.rng.choice(len(self.expert_trajectories), size=n_e, replace=False)
        expert_returns = th.stack([
            self._traj_sum_reward(member, self.expert_trajectories[i])
            for i in exp_idx
        ])

        # Model mini-batch: anchor (permanent early trajectories) + rolling buffer.
        # The anchor guarantees low-quality trajectories stay in log Z even when
        # the current policy has fully converged to the expert distribution.
        model_pool = list(self.model_buffer)

        if self.anchor_buffer and model_pool:
            n_anchor = max(1, int(self.anchor_frac * self.batch_size_model))
            n_recent = self.batch_size_model - n_anchor

            anc_idx = self.rng.choice(len(self.anchor_buffer), size=min(n_anchor, len(self.anchor_buffer)), replace=False)
            rec_idx = self.rng.choice(len(model_pool),         size=min(n_recent,  len(model_pool)),         replace=False)

            model_returns = th.stack(
                [self._traj_sum_reward(member, self.anchor_buffer[i]) for i in anc_idx] +
                [self._traj_sum_reward(member, model_pool[i])          for i in rec_idx]
            )
        else:
            n_m       = min(self.batch_size_model, len(self.trajectories))
            model_idx = self.rng.choice(len(self.trajectories), size=n_m, replace=False)
            model_returns = th.stack([
                self._traj_sum_reward(member, self.trajectories[i])
                for i in model_idx
            ])

        # Numerically stable log partition function approximation
        log_z = th.logsumexp(model_returns, dim=0) - np.log(len(model_returns))

        return -expert_returns.mean() + log_z

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
