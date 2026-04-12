"""
Deep RL from Human Preferences  –  Christiano et al., 2017
https://arxiv.org/abs/1706.03741

On-policy variant using PPO as the base RL algorithm.

The original paper used on-policy methods (TRPO / A3C). This implementation
reproduces the same preference-based reward learning loop with PPO (SB3),
and serves as a comparison baseline against the off-policy SAC variant.

Key differences from the SAC version:
  - RolloutBuffer instead of ReplayBuffer: data is collected fresh each
    iteration and discarded after the policy update.
  - No replay buffer relabeling: the buffer is always current-iteration data,
    so re-labeling after RM training is not needed.
  - policy.forward() is called manually to obtain (actions, values, log_probs)
    required by the rollout buffer before compute_returns_and_advantage().
  - n_policy_train_steps is not a parameter: PPO runs a fixed number of epochs
    over the rollout buffer internally (controlled by PPO's n_epochs).

Training loop
─────────────
Pre-training phase:
  1. Collect initial segments using the current (randomly initialised) policy.
  2. Generate synthetic preferences from the true environment reward.
  3. Pre-train the reward model on those preferences.

Main loop (repeated until total_timesteps is reached):
  1. Collect n_rollout_steps transitions per environment using the current
     policy. Transitions are stored in the agent's rollout buffer with
     PREDICTED rewards. True rewards are kept for preference generation.
  2. compute_returns_and_advantage() from the last observation's value.
  3. Sample n_queries segment pairs; get synthetic preference labels.
  4. Train the reward model for reward_model_train_steps gradient steps.
  5. Train the policy (PPO update on the rollout buffer).
  6. Log metrics to wandb.
"""

from collections import deque
from typing import Callable, Deque, Dict, List, Tuple

import numpy as np
import torch

try:
    from sumo_gym_ego import EgoStatus
    _HAS_EGO_STATUS = True
except ImportError:
    _HAS_EGO_STATUS = False

from stable_baselines3.common.utils import obs_as_tensor

from human_feedback_rl.common import (
    ActiveFragmenter,
    EnsembleRewardModel,
    PreferenceDataset,
    PreferenceModelFromReward,
    RunningMeanStd,
    SB3MetricsLogger,
    SegmentPair,
    Trajectory,
    Transition,
    UnifiedLogger,
    setup_wandb_axes,
)
from human_feedback_rl.common.core import Preference


# ---------------------------------------------------------------------------
# Synthetic preference gatherer  (identical to SAC version)
# ---------------------------------------------------------------------------


class SyntheticGatherer:
    """Generates preference labels from the true environment reward."""

    def __init__(self):
        self._oracle = PreferenceModelFromReward()

    def gather(self, pairs: List[SegmentPair]) -> List[Preference]:
        return [self._oracle(pair) for pair in pairs]


# ---------------------------------------------------------------------------
# Query schedules  (identical to SAC version)
# ---------------------------------------------------------------------------

QUERY_SCHEDULES: Dict[str, Callable[[int, int], int]] = {
    "constant": lambda n_queries, t: n_queries,
    "inverse": lambda n_queries, t: max(1, n_queries // (t + 1)),
}


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------


class ChristianoPPOAlgorithm:
    """
    Implements Deep RL from Human Preferences (Christiano et al., 2017)
    using PPO as the base policy algorithm.

    Designed as a self-contained comparison against ChristianoAlgorithm
    (SAC-based). Shares the same reward model, preference dataset, and
    logging infrastructure; differs only in how the policy is trained.

    Parameters
    ----------
    env : VecEnv
        Vectorised environment.
    agent : stable_baselines3.PPO
        PPO policy with a RolloutBuffer.
    rng : np.random.Generator
        Random number generator for reproducibility.
    reward_model_n_networks : int
        Number of networks in the ensemble reward model.
    reward_model_hidden_size : int
        Hidden layer size for each reward network.
    reward_model_lr : float
        Learning rate for reward model optimizers.
    reward_model_l2 : float
        L2 regularisation weight.
    segment_length : int
        Length of each trajectory segment used for preference queries.
    preference_dataset_max_size : int
        Maximum preferences stored (circular buffer).
    query_schedule : str
        Key into QUERY_SCHEDULES.
    device : str
        PyTorch device for reward model tensors.
    """

    def __init__(
        self,
        env,
        agent,
        rng: np.random.Generator,
        reward_model_n_networks: int = 3,
        reward_model_hidden_size: int = 256,
        reward_model_lr: float = 3e-4,
        reward_model_l2: float = 1e-4,
        segment_length: int = 50,
        preference_dataset_max_size: int = 3000,
        query_schedule: str = "constant",
        device: str = "cpu",
    ):
        self.env = env
        self.agent = agent
        self.rng = rng
        self.segment_length = segment_length

        obs_dim: int = env.observation_space.shape[0]
        action_dim: int = env.action_space.shape[0]

        self.reward_model = EnsembleRewardModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_networks=reward_model_n_networks,
            hidden_size=reward_model_hidden_size,
            lr=reward_model_lr,
            l2_reg=reward_model_l2,
            device=device,
        )
        self.fragmenter = ActiveFragmenter(segment_length=segment_length, rng=rng)
        self.gatherer = SyntheticGatherer()
        self.preference_dataset = PreferenceDataset(max_size=preference_dataset_max_size)
        self.query_schedule_fn = QUERY_SCHEDULES[query_schedule]
        self.logger = UnifiedLogger(use_wandb=True)

        # Reward model gradient step counter (x-axis for rm/* metrics)
        self._rm_global_epochs: int = 0

        # Running z-score normaliser for predicted rewards
        self._reward_rms = RunningMeanStd()

        # Sliding windows for rollout metrics
        _w = 50
        self._window_true_rewards: Deque[float] = deque(maxlen=_w)
        self._window_ep_lengths: Deque[float] = deque(maxlen=_w)
        self._window_model_rewards: Deque[float] = deque(maxlen=_w)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        total_timesteps: int,
        n_initial_queries: int = 200,
        n_rollout_steps: int = 2048,
        n_queries_per_iter: int = 10,
        reward_model_train_steps: int = 200,
        reward_model_batch_size: int = 64,
    ) -> None:
        """
        Run the full Christiano training loop with PPO.

        Parameters
        ----------
        total_timesteps : int
            Total environment steps for the main loop.
        n_initial_queries : int
            Preference queries for reward model pre-training.
        n_rollout_steps : int
            Environment steps collected per iteration, per environment.
            Should match (or be a multiple of) PPO's n_steps.
        n_queries_per_iter : int
            Base number of preference queries per iteration.
        reward_model_train_steps : int
            Gradient steps on the reward model per iteration.
        reward_model_batch_size : int
            Mini-batch size for reward model training.
        """
        # Initialise SB3 internals (rollout buffer, logger, _last_obs, …)
        total_timesteps, _ = self.agent._setup_learn(total_timesteps)

        # ── Wandb setup ────────────────────────────────────────────────
        setup_wandb_axes()
        self._agent_logger = SB3MetricsLogger()
        try:
            self.agent.set_logger(self._agent_logger)
        except AttributeError:
            self.agent._logger = self._agent_logger

        # ── Phase 1: pre-training ──────────────────────────────────────
        self._pretraining_phase(
            n_initial_queries=n_initial_queries,
            reward_model_train_steps=reward_model_train_steps,
            reward_model_batch_size=reward_model_batch_size,
        )

        # ── Phase 2: main loop ─────────────────────────────────────────
        total_steps = 0
        iteration = 0

        print(f"Starting main loop (total_timesteps={total_timesteps})...")

        while total_steps < total_timesteps:
            # 1. Collect rollout into PPO's rollout buffer with predicted rewards
            trajectories, rollout_stats = self._collect_rollout(n_rollout_steps)
            total_steps += n_rollout_steps * self.env.num_envs

            # 2. Gather preferences
            n_queries = self.query_schedule_fn(n_queries_per_iter, iteration)
            pairs = self.fragmenter.sample_pairs(trajectories, n_queries)
            if pairs:
                for pref in self.gatherer.gather(pairs):
                    self.preference_dataset.add(pref)

            # 3. Train reward model
            if len(self.preference_dataset) >= reward_model_batch_size:
                rm_metrics = self.reward_model.train(
                    self.preference_dataset,
                    n_steps=reward_model_train_steps,
                    batch_size=reward_model_batch_size,
                    rng=self.rng,
                )
                self._rm_global_epochs += reward_model_train_steps
                if rm_metrics:
                    rm_metrics["reward_model/accuracy"] = self.reward_model.accuracy(
                        self.preference_dataset, reward_model_batch_size, self.rng
                    )
                    rm_metrics["reward_model/dataset_size"] = len(self.preference_dataset)
                    rm_metrics["reward_model/global_epochs"] = self._rm_global_epochs
            else:
                rm_metrics = {}

            # 4. Train policy (PPO update on the rollout buffer collected in step 1)
            self.agent.train()
            self._agent_logger.flush_to_wandb(fallback_step=total_steps)

            # 5. Log
            if rm_metrics:
                self.logger.log(rm_metrics)

            rollout_log = self._compute_rollout_metrics(
                trajectories, rollout_stats, iteration
            )
            if rollout_log:
                self.logger.log(rollout_log)

            self._print_progress(iteration, total_steps, rm_metrics)
            iteration += 1

        print("Training complete.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _pretraining_phase(
        self,
        n_initial_queries: int,
        reward_model_train_steps: int,
        reward_model_batch_size: int,
    ) -> None:
        """
        Collect initial segments and pre-train the reward model.
        Does NOT fill the rollout buffer (no RL update during pre-training).
        """
        steps_needed = n_initial_queries * 2 * self.segment_length
        n_steps_per_env = int(np.ceil(steps_needed / self.env.num_envs))

        print(
            f"[Pre-training] Collecting {steps_needed} steps "
            f"for {n_initial_queries} initial preference queries..."
        )
        trajectories = self._collect_raw_trajectories(n_steps_per_env)

        pairs = self.fragmenter.sample_pairs(trajectories, n_initial_queries)
        for pref in self.gatherer.gather(pairs):
            self.preference_dataset.add(pref)

        print(
            f"[Pre-training] Training reward model on "
            f"{len(self.preference_dataset)} preferences..."
        )
        pretrain_steps = reward_model_train_steps * 5
        metrics = self.reward_model.train(
            self.preference_dataset,
            n_steps=pretrain_steps,
            batch_size=reward_model_batch_size,
            rng=self.rng,
        )
        self._rm_global_epochs += pretrain_steps

        if metrics:
            acc = self.reward_model.accuracy(
                self.preference_dataset, reward_model_batch_size, self.rng
            )
            print(
                f"[Pre-training] loss={metrics.get('reward_model/loss', float('nan')):.4f}  "
                f"accuracy={acc:.3f}"
            )
            self.logger.log(
                {
                    **metrics,
                    "reward_model/accuracy": acc,
                    "reward_model/dataset_size": len(self.preference_dataset),
                    "reward_model/global_epochs": self._rm_global_epochs,
                }
            )

    def _collect_raw_trajectories(self, n_steps_per_env: int) -> List[Trajectory]:
        """
        Step through the env without filling the rollout buffer.
        Records Trajectory objects with true rewards for preference generation.
        """
        obs, _, completed, _ = self._rollout_loop(
            n_steps_per_env, add_to_buffer=False
        )
        self.agent._last_obs = obs
        return completed

    def _collect_rollout(
        self, n_steps_per_env: int
    ) -> Tuple[List[Trajectory], Dict[str, float]]:
        """
        Fill PPO's rollout buffer with n_steps_per_env steps per environment.
        Rewards stored are z-score-normalised PREDICTED rewards from the RM.
        Returns trajectories with TRUE rewards for preference generation.
        """
        # Reset the rollout buffer before filling it for this iteration.
        self.agent.rollout_buffer.reset()
        obs, _, completed, stats = self._rollout_loop(
            n_steps_per_env, add_to_buffer=True
        )
        self.agent._last_obs = obs
        self.agent.num_timesteps += n_steps_per_env * self.env.num_envs
        return completed, stats

    def _rollout_loop(
        self, n_steps_per_env: int, add_to_buffer: bool
    ) -> Tuple[np.ndarray, List[Trajectory], List[Trajectory], Dict[str, float]]:
        """
        Core collection loop shared by pre-training and main loop.

        When add_to_buffer=True, each step calls policy.forward() to obtain
        (actions, values, log_probs) needed by the PPO rollout buffer, then
        stores z-score-normalised predicted rewards.

        When add_to_buffer=False, actions are sampled with predict() and no
        buffer writes occur (lighter pre-training collection).

        Returns (final_obs, active_trajectories, completed_trajectories, stats).
        """
        n_envs = self.env.num_envs
        obs = self.agent._last_obs

        active_trajs: List[Trajectory] = [Trajectory() for _ in range(n_envs)]
        completed: List[Trajectory] = []
        model_reward_sum = 0.0
        model_reward_count = 0
        n_episodes = 0
        n_collisions = 0
        n_off_road = 0
        n_timeouts = 0
        n_successes = 0

        # episode_starts tracks whether the current obs is the first of an episode.
        # Needed by RolloutBuffer for GAE boundary detection.
        episode_starts = self.agent._last_episode_starts  # (n_envs,) bool

        for _ in range(n_steps_per_env):
            if add_to_buffer:
                # policy.forward() returns actions, values (V(s)), log_probs
                # required by PPO's rollout buffer for advantage computation.
                obs_tensor = obs_as_tensor(obs, self.agent.device)
                with torch.no_grad():
                    actions_t, values, log_probs = self.agent.policy.forward(obs_tensor)
                actions = actions_t.cpu().numpy()
                # Clip actions to the env's action space bounds.
                clipped_actions = np.clip(
                    actions,
                    self.env.action_space.low,
                    self.env.action_space.high,
                )
            else:
                clipped_actions, _ = self.agent.predict(obs, deterministic=False)
                actions = clipped_actions

            next_obs, true_rewards, dones, infos = self.env.step(clipped_actions)

            for i in range(n_envs):
                active_trajs[i].add(
                    Transition(
                        obs=obs[i].copy(),
                        action=actions[i].copy(),
                        true_reward=float(true_rewards[i]),
                        done=bool(dones[i]),
                    )
                )
                if dones[i]:
                    completed.append(active_trajs[i])
                    active_trajs[i] = Trajectory()
                    if _HAS_EGO_STATUS:
                        ego_status = infos[i].get("ego_status")
                        if ego_status is not None:
                            n_episodes += 1
                            n_collisions += int(ego_status == EgoStatus.COLLIDED.value)
                            n_off_road  += int(ego_status == EgoStatus.OFF_ROAD.value)
                            n_timeouts  += int(ego_status == EgoStatus.TIMEOUT.value)
                            n_successes += int(ego_status == EgoStatus.ARRIVED.value)

            if add_to_buffer:
                predicted_rewards = self.reward_model.predict_reward(obs, actions)
                # Update running stats with raw predictions, then normalize.
                self._reward_rms.update(predicted_rewards)
                normalized_rewards = self._reward_rms.normalize(predicted_rewards)

                # RolloutBuffer.add() signature:
                #   add(obs, action, reward, episode_start, value, log_prob)
                self.agent.rollout_buffer.add(
                    obs,
                    actions,
                    normalized_rewards,
                    episode_starts,
                    values,
                    log_probs,
                )
                # Track raw reward for monitoring.
                model_reward_sum += float(np.sum(predicted_rewards))
                model_reward_count += n_envs

            episode_starts = dones
            obs = next_obs

        # Compute returns and advantages using the value of the last observation.
        # This finalises the rollout buffer so PPO can call train().
        if add_to_buffer:
            obs_tensor = obs_as_tensor(obs, self.agent.device)
            with torch.no_grad():
                last_values = self.agent.policy.predict_values(obs_tensor)
            self.agent.rollout_buffer.compute_returns_and_advantage(
                last_values=last_values, dones=dones
            )
            # Persist episode_starts for the next iteration.
            self.agent._last_episode_starts = episode_starts

        # Include partial trajectories long enough for segment sampling.
        for traj in active_trajs:
            if len(traj) >= self.segment_length:
                completed.append(traj)

        stats: Dict[str, float] = {}
        if model_reward_count > 0:
            stats["mean_model_reward"] = model_reward_sum / model_reward_count
        if n_episodes > 0:
            stats["event_rate/collisions"] = n_collisions / n_episodes
            stats["event_rate/off_road"]   = n_off_road   / n_episodes
            stats["event_rate/timeouts"]   = n_timeouts   / n_episodes
            stats["event_rate/successes"]  = n_successes  / n_episodes

        return obs, active_trajs, completed, stats

    def _compute_rollout_metrics(
        self,
        trajectories: List[Trajectory],
        rollout_stats: Dict[str, float],
        iteration: int,
    ) -> Dict[str, float]:
        """Update sliding windows and return smoothed rollout metrics."""
        done_trajs = [
            t for t in trajectories
            if t.transitions and t.transitions[-1].done
        ]
        for traj in done_trajs:
            ep_return = sum(tr.true_reward for tr in traj.transitions)
            self._window_true_rewards.append(ep_return)
            self._window_ep_lengths.append(float(len(traj)))

        if "mean_model_reward" in rollout_stats:
            self._window_model_rewards.append(rollout_stats["mean_model_reward"])

        metrics: Dict[str, float] = {"rollout/iteration": float(iteration)}
        if self._window_true_rewards:
            metrics["rollout/smoothed_true_reward"] = float(
                np.mean(self._window_true_rewards)
            )
        if self._window_ep_lengths:
            metrics["rollout/smoothed_ep_length"] = float(
                np.mean(self._window_ep_lengths)
            )
        if self._window_model_rewards:
            metrics["rollout/smoothed_model_reward"] = float(
                np.mean(self._window_model_rewards)
            )
        for event_key in ("event_rate/collisions", "event_rate/off_road",
                          "event_rate/timeouts", "event_rate/successes"):
            if event_key in rollout_stats:
                metrics[f"rollout/{event_key}"] = rollout_stats[event_key]
        return metrics

    @staticmethod
    def _print_progress(
        iteration: int, total_steps: int, rm_metrics: Dict
    ) -> None:
        loss = rm_metrics.get("reward_model/loss", float("nan"))
        acc = rm_metrics.get("reward_model/accuracy", float("nan"))
        n_prefs = rm_metrics.get("reward_model/dataset_size", 0)
        print(
            f"[iter {iteration:4d}]  steps={total_steps:>10,}  "
            f"prefs={n_prefs:>5}  "
            f"rm_loss={loss:.4f}  "
            f"rm_acc={acc:.3f}"
        )