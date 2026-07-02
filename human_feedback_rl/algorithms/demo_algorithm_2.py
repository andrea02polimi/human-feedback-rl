"""Demonstration-based reward learning with a KL-regularized policy update.

Motivation
----------
With the batch-mixed log-sum-exp partition actually implemented by the reward
loss, the optimal reward is not ``log p_E`` but a *log-ratio* against the mixed
sampling distribution ``q_t``::

    R_t^*(tau) = log p_E(tau) - log q_t(tau) + c_t,
    q_t(tau)   = alpha * p_E(tau) + (1 - alpha) * p_{pi_t}(tau).

Standard SAC maximizes ``E_{pi}[R] + alpha_SAC * H(pi)``, whose ideal solution
is ``p_pi(tau) ∝ exp(R/alpha_SAC)``. Plugging the log-ratio reward in gives
``(p_E / q_t)^{1/alpha_SAC} != p_E`` in general, so SAC is *not* the policy
update that is coherent with this reward loss.

The coherent update maximizes reward minus KL to the sampling distribution::

    pi_new = argmax_pi  E_{tau~p_pi}[R_theta(tau)] - KL(p_pi || q_t),

whose closed-form solution is the target ``p_pi(tau) ∝ q_t(tau) e^{R_theta(tau)}``.
Because the mixed batch is drawn from ``q_t``, projecting that target onto the
policy class is a **weighted behavior cloning** update: reweight the batch by
``softmax(R_theta)`` and fit the policy by weighted maximum likelihood. If the
reward is at its ideal ``log(p_E / q_t)``, the target collapses back to ``p_E``.

Section 13 is implemented *standalone*: there is no SAC/PPO here. The agent is a
:class:`~human_feedback_rl.common.policies.SquashedGaussianPolicy` — a bounded
density on the action box, which is the measure-consistent choice for a reward
defined as a log-ratio of densities. It is trained purely by the weighted-BC
update of section 13.2/13.3; the environment is used only to sample rollouts.
"""

import time
from typing import Any, List, Optional

import numpy as np
import torch as th

from human_feedback_rl.algorithms.demo.checkpointing import CheckpointingMixin
from human_feedback_rl.algorithms.demo.imitation_metrics import (
    IMITATION_CLASSIFIER_L2,
    IMITATION_CLASSIFIER_LR,
    IMITATION_CLASSIFIER_STEPS,
    IMITATION_MAX_TRANSITIONS_PER_CLASS,
    ImitationMetricsMixin,
)
from human_feedback_rl.algorithms.demo.losses import RewardLossMixin
from human_feedback_rl.algorithms.demo.reward_diagnostics import RewardDiagnosticsMixin
from human_feedback_rl.algorithms.demo.reward_training import RewardTrainingMixin
from human_feedback_rl.algorithms.demo.rollout import RolloutMixin
from human_feedback_rl.common.base_algorithm import BaseAlgorithm
from human_feedback_rl.common.env_wrappers import EnvBufferingWrapper
from human_feedback_rl.common.policies import SquashedGaussianPolicy
from human_feedback_rl.common.reward_nets import make_reward_ensemble
from human_feedback_rl.common.trajectory_generators import _get_trajectories, rollout_agent
from human_feedback_rl.common.types import Trajectory


class DemoAlgorithm2(
    RewardLossMixin,
    RewardTrainingMixin,
    RolloutMixin,
    ImitationMetricsMixin,
    RewardDiagnosticsMixin,
    CheckpointingMixin,
    BaseAlgorithm,
):
    """Alternating batch-mixed IRL reward learning and weighted-BC policy update.

    The only losses that make the derivation of section 13 valid are the ones
    whose partition is estimated on the *mixed* batch ``B = B_E ∪ B_pi`` drawn
    from ``q_t`` (``maxent_2``) or the model-only surrogate (``maxent``). Ranking
    losses are rejected because they do not induce the ``q_t e^R`` target.
    """

    STATUS_ARRIVED = 0
    STATUS_COLLIDED = 1
    STATUS_OFFROAD = 2
    STATUS_TIMEOUT = 3
    STATUS_RUNNING = 4

    IMITATION_MAX_TRANSITIONS_PER_CLASS = IMITATION_MAX_TRANSITIONS_PER_CLASS
    IMITATION_CLASSIFIER_STEPS = IMITATION_CLASSIFIER_STEPS
    IMITATION_CLASSIFIER_LR = IMITATION_CLASSIFIER_LR
    IMITATION_CLASSIFIER_L2 = IMITATION_CLASSIFIER_L2

    # Section 13 requires a partition estimated on the q_t sample. Only the
    # batch-mixed partition losses satisfy this, so they are the valid choices.
    VALID_LOSSES = ("maxent_2", "maxent")

    def __init__(
        self,
        env,
        expert_trajectories: List[Trajectory],
        loss_type: str = "maxent_2",
        lr_rew: float = 0.001,
        gradient_steps_rew: int = 10,
        batch_size_expert: int = 32,
        batch_size_model: int = 64,
        l2_rew: float = 0.01,
        temperature: float = 1.0,
        # --- weighted behavior-cloning (section 13.2/13.3) ---
        lr_policy: float = 3e-4,
        gradient_steps_policy: int = 32,
        weight_temperature: float = 1.0,
        standardize_weights: bool = True,
        policy_kwargs: Optional[dict] = None,
        # -----------------------------------------------------
        initial_reward_timesteps: int = 0,
        reward_model_kwargs: Optional[dict] = None,
        rng: Optional[np.random.Generator] = None,
        log_folder: Optional[str] = None,
        output_formats: Optional[List] = None,
        debug_dataset: Optional[dict] = None,
        rollout_env=None,
        device: str = "cpu",
    ):
        if not expert_trajectories:
            raise ValueError("expert_trajectories must be a non-empty list of Trajectory objects.")
        if loss_type not in self.VALID_LOSSES:
            raise ValueError(
                f"loss_type must be one of {self.VALID_LOSSES} for the section-13 update "
                f"(the partition must be estimated on the q_t batch), got {loss_type!r}."
            )
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if weight_temperature <= 0:
            raise ValueError("weight_temperature (eta) must be positive.")
        if gradient_steps_rew <= 0 or gradient_steps_policy <= 0:
            raise ValueError("gradient_steps_rew and gradient_steps_policy must be positive.")
        if batch_size_expert <= 0 or batch_size_model <= 0:
            raise ValueError("Batch sizes must be positive.")

        # The standalone policy *is* the agent for the reused mixins (imitation
        # diagnostics call ``self.agent.predict`` / ``policy_action_log_probs``).
        self.policy = SquashedGaussianPolicy(
            env.observation_space, env.action_space, device=device, **(policy_kwargs or {})
        )
        super().__init__(env, self.policy, rng, log_folder=log_folder, output_formats=output_formats)

        self.expert_trajectories = list(expert_trajectories)
        self.loss_type = loss_type
        self.gradient_steps_rew = gradient_steps_rew
        self.batch_size_expert = batch_size_expert
        self.batch_size_model = batch_size_model
        self.initial_reward_timesteps = initial_reward_timesteps
        self.temperature = temperature

        self.lr_policy = lr_policy
        self.gradient_steps_policy = gradient_steps_policy
        self.weight_temperature = weight_temperature
        self.standardize_weights = standardize_weights

        # ``maxent_corrected``-only machinery kept as no-ops so the reused reward
        # mixins (which reference these) behave consistently.
        self.fragment_length = None
        self.trajectories = []
        self._maxent_corrected_steps = []
        self._num_env_steps = 0
        self.debug_dataset = debug_dataset or {}
        self._debug_rng = np.random.default_rng(0)
        self._debug_trajectories = self._split_into_trajectories(self.debug_dataset)

        self.reward_model = make_reward_ensemble(env, **(reward_model_kwargs or {}))

        # Rollout sampling: wrap the env with the tested buffering wrapper and
        # drive it with the standalone policy via ``rollout_agent``. No reward
        # wrapper is needed — weighted BC never feeds a reward into an RL learner,
        # and the buffer records the environment's true reward directly.
        self._buffering = EnvBufferingWrapper(rollout_env if rollout_env is not None else env)

        self.optimizers = [
            th.optim.Adam(member.parameters(), lr=lr_rew, weight_decay=l2_rew)
            for member in self.reward_model.members
        ]
        self.policy_optimizer = th.optim.Adam(self.policy.parameters(), lr=lr_policy)

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #

    def train(
        self,
        total_timesteps: int = 1_000_000,
        timesteps_per_iteration: int = 1024,
        log_interval: int = 1,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 10,
        imitation_diagnostics_interval: int = 10,
        scatter_interval: Optional[int] = None,
    ) -> Any:
        """Run the alternating reward-learning / weighted-BC loop."""
        if imitation_diagnostics_interval < 0:
            raise ValueError("imitation_diagnostics_interval must be non-negative.")
        if scatter_interval is None:
            scatter_interval = imitation_diagnostics_interval
        if scatter_interval < 0:
            raise ValueError("scatter_interval must be non-negative.")
        n_iterations = int(total_timesteps / timesteps_per_iteration)

        if self.initial_reward_timesteps > 0:
            print(f"- Collecting {self.initial_reward_timesteps} bootstrap transitions")
            self.trajectories = self._sample_rollout(self.initial_reward_timesteps)
            print("- Bootstrapping reward model")
            self._train_reward_model()

        for iteration in range(n_iterations):
            t_iter = time.perf_counter()
            print(f"\nIteration {iteration}/{n_iterations - 1}")

            # 1. Collect agent trajectories -> the p_{pi_t} part of q_t.
            print(f"- Collecting {timesteps_per_iteration} agent transitions")
            self.trajectories = self._sample_rollout(timesteps_per_iteration)

            should_log_imitation = imitation_diagnostics_interval > 0 and (
                iteration % imitation_diagnostics_interval == 0
                or iteration == n_iterations - 1
            )
            if should_log_imitation:
                self._log_imitation_diagnostics()
            self._log_expert_imitation_errors()

            all_transitions = [t for traj in self.trajectories for t in traj]
            self._log_validation_snapshot(all_transitions, "pre_update")

            # 2. Update the reward model with the batch-mixed partition loss.
            print("- Training reward model")
            self._train_reward_model()
            self._log_validation_snapshot(all_transitions, "post_update")
            self._log_outcome_returns()
            self._log_event_rates()

            should_log_scatter = self.debug_dataset and scatter_interval > 0 and (
                iteration % scatter_interval == 0 or iteration == n_iterations - 1
            )
            if should_log_scatter:
                self._log_return_scatter(
                    self._debug_trajectories,
                    "reward_val/debug_dataset/post_update",
                    iteration,
                )

            # 3. Policy update: weighted behavior cloning toward q_t * exp(R).
            print(f"- Weighted behavior cloning ({self.gradient_steps_policy} steps)")
            self._weighted_behavior_cloning()

            self._log_iteration(t_iter, iteration)
            if checkpoint_dir is not None and (iteration + 1) % checkpoint_interval == 0:
                self._save_checkpoint(checkpoint_dir, iteration + 1)

        return self.policy

    # ------------------------------------------------------------------ #
    # Rollout sampling (standalone, no RL learner)
    # ------------------------------------------------------------------ #

    def _sample_rollout(self, agent_steps: int, exploration_steps: int = 0) -> list:
        """Collect at least ``agent_steps`` transitions from the current policy.

        ``exploration_steps`` is accepted for signature compatibility but unused:
        the squashed-Gaussian policy is already stochastic, so section 13 samples
        directly from ``pi_phi`` without an epsilon-exploration mixture.
        """
        t0 = time.perf_counter()
        rollout_agent(self.policy, self._buffering, agent_steps, deterministic_policy=False)
        trajectories = _get_trajectories(self._buffering.pop_finished_trajectories(), agent_steps)
        t_sample = time.perf_counter() - t0
        self._num_env_steps += sum(len(traj) for traj in trajectories)

        t0 = time.perf_counter()
        true_rewards = [traj.total_reward() for traj in trajectories]
        model_rewards = [self._score_trajectory(traj) for traj in trajectories]
        lengths = [len(traj) for traj in trajectories]
        self.logger.record("rollout/mean_true_reward", float(np.mean(true_rewards)))
        self.logger.record("rollout/mean_model_reward", float(np.mean(model_rewards)))
        self.logger.record("rollout/mean_length", float(np.mean(lengths)))
        self._log_action_boundaries(trajectories)
        self.logger.record("time/sample_rollout", t_sample)
        self.logger.record_sum("time/loggings", time.perf_counter() - t0)
        return trajectories

    # ------------------------------------------------------------------ #
    # Weighted behavior cloning (section 13.2 / 13.3)
    # ------------------------------------------------------------------ #

    def _weighted_behavior_cloning(self) -> None:
        """Project ``q_t(tau) exp(R_theta(tau))`` onto the policy class.

        Draws the mixed batch ``B = B_E ∪ B_pi`` (samples from ``q_t``), computes
        the softmax importance weights ``w_j ∝ exp(R_theta(tau_j) / eta)``, then
        maximizes the weighted policy log-likelihood
        ``sum_j w_j sum_t log pi_phi(a_t^j | s_t^j)``.
        """
        if not self.trajectories:
            return
        t0 = time.perf_counter()

        expert_trajs, model_trajs = self._sample_trajectories()
        batch = list(expert_trajs) + list(model_trajs)

        # Fixed target weights: reward is frozen during the policy update.
        scores = np.array([self._trajectory_return(traj) for traj in batch], dtype=np.float64)
        weights = self._softmax_weights(scores)
        weights_th = th.as_tensor(weights, dtype=th.float32, device=self.policy.device)

        self.policy.train()
        losses = []
        for _ in range(self.gradient_steps_policy):
            log_probs = th.stack([self._policy_traj_log_prob(traj) for traj in batch])
            loss = -(weights_th * log_probs).sum()
            if not th.isfinite(loss):
                raise FloatingPointError("Non-finite weighted-BC loss.")
            self.policy_optimizer.zero_grad()
            loss.backward()
            self.policy_optimizer.step()
            losses.append(float(loss.detach()))
        self.policy.eval()

        self._log_bc_diagnostics(scores, weights, losses, len(expert_trajs))
        self.logger.record("time/weighted_bc", time.perf_counter() - t0)

    def _softmax_weights(self, scores: np.ndarray) -> np.ndarray:
        """Normalized target weights ``w_j ∝ exp(R_j / eta)``.

        When ``standardize_weights`` is set the scores are z-scored first so that
        ``eta`` acts as a temperature relative to the batch spread and the weights
        do not collapse onto a single trajectory when the raw reward scale drifts.
        """
        logits = np.asarray(scores, dtype=np.float64)
        if self.standardize_weights:
            std = logits.std()
            logits = (logits - logits.mean()) / (std + 1e-8)
        logits = logits / self.weight_temperature
        logits -= logits.max()
        w = np.exp(logits)
        return w / w.sum()

    def _trajectory_return(self, traj: Trajectory) -> float:
        """Raw (unnormalized) ``R_theta(tau)`` used to weight the batch."""
        obs = np.array([t.observation for t in traj])
        acts = np.array([t.action for t in traj])
        status = np.array([t.next_status for t in traj])
        done = np.array([float(t.done) for t in traj])
        return float(self.reward_model.predict_unnormalized(obs, acts, status, done).sum())

    def _policy_traj_log_prob(self, traj: Trajectory) -> th.Tensor:
        """Differentiable ``sum_t log pi_phi(a_t | s_t)`` for one trajectory."""
        obs = np.array([t.observation for t in traj], dtype=np.float32)
        actions = np.array([t.action for t in traj], dtype=np.float32).reshape(len(traj), -1)
        obs_tensor = self.policy._obs_tensor(obs)
        action_tensor = th.as_tensor(actions, dtype=th.float32, device=self.policy.device)
        return self.policy.log_prob(obs_tensor, action_tensor).sum()

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #

    def _log_bc_diagnostics(
        self,
        scores: np.ndarray,
        weights: np.ndarray,
        losses: List[float],
        n_expert: int,
    ) -> None:
        expert_scores = scores[:n_expert]
        agent_scores = scores[n_expert:]
        ess = float(1.0 / np.sum(weights ** 2))
        entropy = float(-np.sum(weights * np.log(weights + 1e-12)))
        self.logger.record("bc/loss", float(np.mean(losses)))
        self.logger.record("bc/mean_R_expert", float(np.mean(expert_scores)))
        if agent_scores.size:
            self.logger.record("bc/mean_R_agent", float(np.mean(agent_scores)))
        self.logger.record("bc/weight_ess", ess)
        self.logger.record("bc/weight_ess_fraction", ess / len(weights))
        self.logger.record("bc/weight_entropy", entropy)
        self.logger.record("bc/weight_max", float(weights.max()))
        # Fraction of target weight assigned to expert vs agent samples: a healthy
        # run should not put all mass on the expert (that would be plain BC).
        self.logger.record("bc/weight_mass_expert", float(weights[:n_expert].sum()))

    def _log_event_rates(self) -> None:
        """Log terminal-outcome frequencies of the current rollouts.

        ``DemoAlgorithm`` gets ``agent/event_rate/*`` for free from
        ``CustomLoggingCallback`` during ``agent.learn()``. This algorithm never
        runs an RL learner, so we compute the same rates directly from
        ``self.trajectories`` and publish them under identical keys for dashboard
        comparability.

        The denominator is the number of *genuine* terminal episodes (``done`` with
        a valid one-hot status), matching ``_log_outcome_returns``; truncated or
        malformed trajectories are excluded so an all-zero status is not misread
        as "arrived".
        """
        if not self.trajectories:
            return
        rate_names = {
            self.STATUS_ARRIVED: "successes",
            self.STATUS_COLLIDED: "collisions",
            self.STATUS_OFFROAD: "off_road",
            self.STATUS_TIMEOUT: "timeouts",
        }
        counts = {name: 0 for name in rate_names.values()}
        n_episodes = 0
        for traj in self.trajectories:
            if len(traj) == 0:
                continue
            last_status = np.asarray(traj[-1].next_status, dtype=np.float64)
            if not traj[-1].done or not np.isclose(last_status.sum(), 1.0):
                continue
            n_episodes += 1
            name = rate_names.get(int(np.argmax(last_status)))
            if name is not None:
                counts[name] += 1

        self.logger.record("agent/event_rate/n_episodes", n_episodes, exclude="stdout")
        if n_episodes == 0:
            return
        for name, count in counts.items():
            self.logger.record(f"agent/event_rate/{name}", count / n_episodes)

    def _save_checkpoint(self, checkpoint_dir: str, iteration: int) -> None:
        """Persist the reward model and the standalone policy (no SB3 agent)."""
        import os

        ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{iteration:04d}")
        os.makedirs(ckpt_path, exist_ok=True)
        th.save(self.reward_model.state_dict(), os.path.join(ckpt_path, "reward_model.pt"))
        th.save(self.policy.state_dict(), os.path.join(ckpt_path, "policy.pt"))
        th.save(
            {
                "iteration": iteration,
                "loss_type": self.loss_type,
                "temperature": self.temperature,
                "weight_temperature": self.weight_temperature,
                "standardize_weights": self.standardize_weights,
                "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
                "policy_optimizer": self.policy_optimizer.state_dict(),
            },
            os.path.join(ckpt_path, "training_state.pt"),
        )
        print(f"  checkpoint saved in {ckpt_path}")

    def _log_iteration(self, t_iter: float, iteration: int) -> None:
        t_log = time.perf_counter()
        self.logger.record("iterations", iteration)
        self.logger.record("agent/time/total_timesteps", self._num_env_steps)
        self.logger.record("time/total", time.perf_counter() - t_iter)
        self.logger.record_sum("time/loggings", time.perf_counter() - t_log)
        self.logger.dump()
