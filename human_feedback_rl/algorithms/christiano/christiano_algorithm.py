"""
Deep RL from Human Preferences  –  Christiano et al., 2017
https://arxiv.org/abs/1706.03741

Training loop
─────────────
Pre-training phase:
  1. Collect initial segments using the current (randomly initialised) policy.
  2. Generate synthetic preferences from the true environment reward.
  3. Pre-train the reward model on those preferences.

Main loop (repeated until total_timesteps is reached):
  1. Collect n_rollout_steps transitions per environment using the current policy.
     Transitions are stored in the agent's replay buffer with PREDICTED rewards.
     True rewards are kept in memory for preference generation.
  2. Sample n_queries segment pairs; get synthetic preference labels.
  3. Train the reward model for reward_model_train_steps gradient steps.
  4. Train the policy for n_policy_train_steps gradient steps.
  5. Log metrics to wandb.
"""

from collections import deque
from typing import Callable, Deque, Dict, List, Tuple

import numpy as np

try:
    from sumo_gym_ego import EgoStatus
    _HAS_EGO_STATUS = True
except ImportError:
    _HAS_EGO_STATUS = False

from human_feedback_rl.common import (
    ActiveFragmenter,
    EnsembleRewardModel,
    PreferenceDataset,
    PreferenceModelFromReward,
    SB3MetricsLogger,
    SegmentPair,
    Trajectory,
    Transition,
    UnifiedLogger,
    setup_wandb_axes,
)
from human_feedback_rl.common.core import Preference


# ---------------------------------------------------------------------------
# Synthetic preference gatherer
# ---------------------------------------------------------------------------


class SyntheticGatherer:
    """
    Generates preference labels from the true environment reward.
    Serves as a drop-in replacement for a human annotator.
    """

    def __init__(self):
        self._oracle = PreferenceModelFromReward()

    def gather(self, pairs: List[SegmentPair]) -> List[Preference]:
        return [self._oracle(pair) for pair in pairs]


# ---------------------------------------------------------------------------
# Query schedules
# ---------------------------------------------------------------------------

QUERY_SCHEDULES: Dict[str, Callable[[int, int], int]] = {
    # n_queries is constant throughout training
    "constant": lambda n_queries, t: n_queries,
    # n_queries decays as 1/(t+1); useful to concentrate queries early
    "inverse": lambda n_queries, t: max(1, n_queries // (t + 1)),
}


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------


class ChristianoAlgorithm:
    """
    Implements Deep RL from Human Preferences (Christiano et al., 2017).

    The algorithm learns a reward model from pairwise human preferences over
    short trajectory segments, then trains a policy with RL using the learned
    reward. Here human preferences are approximated synthetically using the
    true environment reward.

    Parameters
    ----------
    env : VecEnv
        Vectorised environment (e.g. from sre.make_vec_env).
    agent : stable_baselines3.SAC (or compatible off-policy algorithm)
        RL policy that exposes predict(), replay_buffer, and train().
    rng : np.random.Generator
        Random number generator for reproducibility.
    reward_model_n_networks : int
        Number of networks in the ensemble reward model.
    reward_model_hidden_size : int
        Hidden layer size for each reward network.
    reward_model_lr : float
        Learning rate for reward model optimizers.
    reward_model_l2 : float
        L2 regularisation weight for reward model parameters.
    segment_length : int
        Length k of each trajectory segment used for preference queries.
    preference_dataset_max_size : int
        Maximum number of preferences stored in the dataset (circular buffer).
    query_schedule : str
        Key into QUERY_SCHEDULES; controls how many queries are made per iter.
    device : str
        PyTorch device for reward model tensors ('cpu' or 'cuda').
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

        # Reward model gradient step counter (used as x-axis for rm/* metrics)
        self._rm_global_epochs: int = 0

        # Sliding windows for rollout metrics (last N episodes / iterations)
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
        n_rollout_steps: int = 2000,
        n_queries_per_iter: int = 10,
        reward_model_train_steps: int = 200,
        reward_model_batch_size: int = 64,
        n_policy_train_steps: int = 1000,
    ) -> None:
        """
        Run the full Christiano training loop.

        Parameters
        ----------
        total_timesteps : int
            Total number of environment steps for the main loop.
        n_initial_queries : int
            Number of preference queries for reward model pre-training.
        n_rollout_steps : int
            Environment steps collected per iteration, per environment.
        n_queries_per_iter : int
            Base number of preference queries per iteration (may decay with schedule).
        reward_model_train_steps : int
            Gradient steps on the reward model per iteration.
        reward_model_batch_size : int
            Mini-batch size for reward model training.
        n_policy_train_steps : int
            Gradient steps on the policy per iteration.
        """
        # Initialise SB3 internals (logger, _last_obs, num_timesteps, …)
        total_timesteps, _ = self.agent._setup_learn(total_timesteps)

        # ── Wandb setup ────────────────────────────────────────────────
        # Define per-group x-axes before any logging happens.
        setup_wandb_axes()

        # Replace SB3's logger so that agent.train() metrics can be forwarded
        # to wandb.  set_logger() is the public API; fall back to _logger for
        # versions where the property has no setter.
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
            # 1. Collect rollout; add to replay buffer with predicted rewards
            trajectories, rollout_stats = self._collect_rollout(n_rollout_steps)
            total_steps += n_rollout_steps * self.env.num_envs

            # 2. Gather preferences
            n_queries = self.query_schedule_fn(n_queries_per_iter, iteration)
            pairs = self.fragmenter.sample_pairs(trajectories, n_queries)
            if pairs:
                for pref in self.gatherer.gather(pairs):
                    self.preference_dataset.add(pref)

            # 3. Train reward model.
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

            # 4. Train policy; flush captured metrics to wandb after training.
            if self.agent.replay_buffer.size() >= self.agent.batch_size:
                self.agent.train(
                    gradient_steps=n_policy_train_steps,
                    batch_size=self.agent.batch_size,
                )
                self._agent_logger.flush_to_wandb(fallback_step=total_steps)

            # 5. Log
                # Reward model metrics (x-axis: global_epochs)
                if rm_metrics:
                    self.logger.log(rm_metrics)

                # Rollout metrics (x-axis: iteration)
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
        Does not add transitions to the replay buffer.
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
        # Use 5x more gradient steps for initial training to get a good starting point
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
        Step through the env for n_steps_per_env steps per environment.
        Records transitions with TRUE rewards but does NOT add to replay buffer.
        Updates self.agent._last_obs.
        """
        obs, trajectories, completed, _ = self._rollout_loop(
            n_steps_per_env, add_to_buffer=False
        )
        self.agent._last_obs = obs
        return completed

    def _collect_rollout(
        self, n_steps_per_env: int
    ) -> Tuple[List[Trajectory], Dict[str, float]]:
        """
        Step through the env for n_steps_per_env steps per environment.
        Stores transitions in the agent's replay buffer with PREDICTED rewards.
        Returns trajectories with TRUE rewards for preference generation.
        Updates self.agent._last_obs.
        """
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
        Core loop used by both _collect_raw_trajectories and _collect_rollout.

        Returns (final_obs, active_trajectories, completed_trajectories, stats).
        Completed trajectories are those that ended (done=True) or, for partial
        ones, that reached at least segment_length transitions.
        stats contains ``mean_model_reward`` (mean predicted reward per step)
        when add_to_buffer=True.
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

        for _ in range(n_steps_per_env):
            actions, _ = self.agent.predict(obs, deterministic=False)
            next_obs, true_rewards, dones, infos = self.env.step(actions)

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
                self.agent.replay_buffer.add(
                    obs, next_obs, actions, predicted_rewards, dones, infos
                )
                model_reward_sum += float(np.sum(predicted_rewards))
                model_reward_count += n_envs

            obs = next_obs

        # Include partial trajectories that are long enough for segment sampling
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

    def _relabel_replay_buffer(self, batch_size: int = 4096) -> None:
        """
        Re-predict rewards for every transition in the replay buffer using the
        current (updated) reward model, then overwrite ``replay_buffer.rewards``
        with z-score-normalised values.

        The running normaliser ``_reward_rms`` is NOT updated here: it was
        already updated with fresh rollout predictions this iteration, so the
        relabeling simply re-normalises using the current mean/std without
        double-counting.

        SB3's ReplayBuffer stores data with shape
            observations : (buffer_size, n_envs, obs_dim)
            actions      : (buffer_size, n_envs, action_dim)
            rewards      : (buffer_size, n_envs)
        where ``pos`` is the write pointer and ``full`` signals a wrapped buffer.
        """
        buf = self.agent.replay_buffer
        n_valid = buf.buffer_size if buf.full else buf.pos
        if n_valid == 0:
            return

        n_envs = buf.n_envs
        obs_dim = buf.observations.shape[-1]
        act_dim = buf.actions.shape[-1]

        # Flatten (n_valid, n_envs, dim) → (n_valid * n_envs, dim)
        obs_flat = buf.observations[:n_valid].reshape(-1, obs_dim)
        act_flat = buf.actions[:n_valid].reshape(-1, act_dim)
        n_total = obs_flat.shape[0]

        # Predict in batches to keep GPU/CPU memory bounded
        raw_rewards = np.empty(n_total, dtype=np.float32)
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            raw_rewards[start:end] = self.reward_model.predict_reward(
                obs_flat[start:end], act_flat[start:end]
            )

        # Normalise using current running stats (no rms update)
        std = float(np.sqrt(max(self._reward_rms.var, 0.0))) + self._reward_rms.epsilon
        norm_rewards = (raw_rewards - self._reward_rms.mean) / std

        # Overwrite rewards in-place; reshape back to (n_valid, n_envs)
        buf.rewards[:n_valid] = norm_rewards.reshape(n_valid, n_envs)

    def _compute_rollout_metrics(
        self,
        trajectories: List[Trajectory],
        rollout_stats: Dict[str, float],
        iteration: int,
    ) -> Dict[str, float]:
        """
        Update sliding-window buffers and return smoothed rollout metrics.

        true_reward and ep_length windows are updated per-episode (appending
        one value per completed episode).  model_reward window is updated
        per-iteration (one mean-per-step value per rollout).
        """
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