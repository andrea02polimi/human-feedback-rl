from collections import deque
from typing import Any, List, Optional

import numpy as np
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit

from human_feedback_rl.common import (
    ActiveFragmenter,
    EnsembleRewardModel,
    EnvRewardWrapper,
    InverseSchedule,
    Preference,
    PreferenceDataset,
    PreferenceModelFromReward,
    PrefixLogger,
    SegmentPair,
    Trajectory,
    Transition,
    UnifiedLogger,
    CustomLoggingCallback,
    encode_ego_status,
)
from .preference_trainer import RewardTrainerChristiano


# ---------------------------------------------------------------------------
# Wrapper that exposes true env rewards for preference labelling
# ---------------------------------------------------------------------------

class _TrueRewardEnvWrapper(EnvRewardWrapper):
    """
    Returns normalized RM rewards to the agent (stored in replay buffer) and
    records true env rewards in transitions for preference labelling.
    Completed episodes are drained with collect_trajectories().
    """

    def __init__(self, venv, reward_model, pessimism: float = 0.0):
        super().__init__(venv, reward_model)
        self.pessimism = pessimism
        self._ep_transitions: List[List[Transition]] = [[] for _ in range(venv.num_envs)]
        self._completed_trajectories: List[Trajectory] = []

    def reset(self):
        obs = super().reset()
        self._ep_transitions = [[] for _ in range(self.num_envs)]
        self._completed_trajectories = []
        return obs

    def step_wait(self):
        obs, env_rewards, dones, infos = self.venv.step_wait()

        if self._obs is not None and self._actions is not None:
            info_arr = np.stack([
                encode_ego_status(infos[i].get("ego_status", "running"))
                for i in range(self.num_envs)
            ])
            rm_rewards = self.reward_model.predict(self._obs, self._actions, pessimism=self.pessimism, info=info_arr)
            self._reward_stats.update(rm_rewards)
            normalized = (
                (rm_rewards - self._reward_stats.mean) / self._reward_stats.std
            ).astype(np.float32)

            for i in range(self.num_envs):
                self._ep_transitions[i].append(
                    Transition(
                        obs=self._obs[i].copy(),
                        action=self._actions[i].copy(),
                        reward=float(env_rewards[i]),
                        info={"ego_status": infos[i].get("ego_status", "running")},
                    )
                )
                if dones[i]:
                    self._completed_trajectories.append(
                        Trajectory(self._ep_transitions[i])
                    )
                    self._ep_transitions[i] = []
        else:
            normalized = np.zeros(self.num_envs, dtype=np.float32)

        self._obs = np.asarray(obs, dtype=np.float32)
        return obs, normalized, dones, infos

    def collect_trajectories(self) -> List[Trajectory]:
        """Drain and return all completed trajectories since the last call."""
        trajs = list(self._completed_trajectories)
        self._completed_trajectories = []
        return trajs


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------

class ChristianoSACAlgorithm:
    """

    Outer loop: collect rollouts → fragment → label → update datasets →
    train RM → train SAC. Small n_rollout_steps keeps RM training near on-policy
    while still benefiting from the replay buffer.
    """

    def __init__(
        self,
        env,
        agent,
        n_ensembles: int,
        segment_length: int,
        device: str = "cpu",
        lr_reward_model: float = 1e-4,
        max_dataset_size: int = 1e10,
        reward_model_batch_size: int = 32,
        num_pairs_initial: int = 100,
        num_pairs_final: int = 0,
        decay_pairs_schedule: float = 1.0,
        pessimism: float = 0.0,
    ):
        if hasattr(env.action_space, "n"):
            raise ValueError(
                "ChristianoSACAlgorithm requires a continuous action space. "
            )

        self.env = env
        self.agent = agent
        self.logger = UnifiedLogger()

        self.reward_model = EnsembleRewardModel(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            n_ensembles=n_ensembles,
            lr=lr_reward_model,
            device=device,
            discrete_actions=False,
        )

        self.preference_model = PreferenceModelFromReward(self.reward_model)

        self.reward_trainer = RewardTrainerChristiano(
            self.preference_model,
            logger=PrefixLogger(self.logger, "reward_model"),
            batch_size=reward_model_batch_size,
        )

        self.fragmenter = ActiveFragmenter(
            reward_model=self.reward_model,
            segment_length=segment_length,
        )

        self.preference_dataset = PreferenceDataset(max_dataset_size)
        self.preference_dataset_val = PreferenceDataset(max_dataset_size)

        self.pessimism = pessimism
        self.reward_wrapper = _TrueRewardEnvWrapper(self.env, self.reward_model, pessimism=pessimism)
        self.agent.set_env(self.reward_wrapper)
        self.agent.set_logger(PrefixLogger(self.logger, "agent"))

        self.schedule_num_pairs = InverseSchedule(
            initial_value=num_pairs_initial,
            final_value=num_pairs_final,
            decay_rate=decay_pairs_schedule,
        )

    def train(
        self,
        total_timesteps: int = 1_000_000,
        n_rollout_steps: int = 1_000,
        pretrain_iterations: int = 0,
        reward_model_epochs_per_it: int = 10,
        agent_log_interval: int = 1,
        segments_oversample_factor: int = 3,
        gradient_steps: int = -1,
        learning_starts: int = 1_000,
        replay_buffer_iterations: int = None,
        relabel_replay_buffer: bool = False,
        agent_update_every_n: int = 1,
        checkpoint_path: Optional[str] = None,
        checkpoint_window: int = 10,
    ) -> Any:
        """
        Args:
            total_timesteps: Total env steps.
            n_rollout_steps: Steps per outer iteration (smaller = fresher RM data).
            pretrain_iterations: Iterations where only the RM trains (no SAC updates).
            reward_model_epochs_per_it: RM training epochs per iteration.
            agent_log_interval: SAC logging interval for collect_rollouts.
            segments_oversample_factor: Oversample factor for the fragmenter.
            gradient_steps: SAC gradient updates per iteration (-1 = n_rollout_steps).
            learning_starts: Min buffer transitions before SAC training begins.
            replay_buffer_iterations: If set, buffer_size = K * n_rollout_steps.
                Smaller K = less staleness, less memory, faster relabeling.
            relabel_replay_buffer: Recompute all buffer rewards after each RM update.
            agent_update_every_n: SAC updates only every N iterations (RM updates every iter).
            checkpoint_path: Save best agent (by smoothed avg_true_reward) here (.zip appended by SB3).
            checkpoint_window: Window size for smoothed avg_true_reward in checkpointing.
        """
        if gradient_steps == -1:
            gradient_steps = n_rollout_steps

        # Override buffer_size from K * n_rollout_steps if requested.
        # Must be done before _setup_learn, which allocates the replay buffer.
        if replay_buffer_iterations is not None:
            self.agent.buffer_size = replay_buffer_iterations * n_rollout_steps
            print(
                f"[init] replay_buffer_iterations={replay_buffer_iterations}"
                f"  →  buffer_size={self.agent.buffer_size}"
                f"  (K={replay_buffer_iterations} × n_rollout_steps={n_rollout_steps})"
            )

        reward_model_global_epochs = 0

        # Initialise SAC internal state: _last_obs, replay_buffer, etc.
        total_timesteps, callback = self.agent._setup_learn(
            total_timesteps,
            callback=CustomLoggingCallback(),
            reset_num_timesteps=True,
        )
        callback.on_training_start(locals(), globals())

        # Fixed collection window per outer iteration
        train_freq = TrainFreq(n_rollout_steps, TrainFrequencyUnit.STEP)
        iteration = 0

        # Checkpoint state
        _reward_window: deque = deque(maxlen=checkpoint_window)
        _best_smoothed_reward: float = float("-inf")

        while self.agent.num_timesteps < total_timesteps:
            progress_remaining = 1.0 - self.agent.num_timesteps / total_timesteps
            num_pairs = int(self.schedule_num_pairs(progress_remaining))

            print(
                f"\n=== Iteration {iteration + 1}"
                f" (steps={self.agent.num_timesteps}/{total_timesteps}) ==="
            )

            # 1) Collect rollouts via SAC
            #    _TrueRewardEnvWrapper: normalized RM rewards → replay buffer,
            #    true rewards → accumulated for preference generation.
            print("[1/5] Collecting rollouts...")
            rollout = self.agent.collect_rollouts(
                self.reward_wrapper,
                callback=callback,
                train_freq=train_freq,
                replay_buffer=self.agent.replay_buffer,
                action_noise=self.agent.action_noise,
                learning_starts=learning_starts,
                log_interval=agent_log_interval,
            )
            if not rollout.continue_training:
                break

            trajectories = self.reward_wrapper.collect_trajectories()

            if not trajectories:
                iteration += 1
                continue

            ep_lengths   = [len(t.transitions) for t in trajectories]
            true_rewards = [t.total_reward()     for t in trajectories]
            avg_ep_length   = float(np.mean(ep_lengths))
            avg_true_reward = float(np.mean(true_rewards))
            print(
                f"      trajectories={len(trajectories)}"
                f"  ep_length=[{min(ep_lengths):.0f}/{avg_ep_length:.1f}/{max(ep_lengths):.0f}]"
                f"  true_reward=[{min(true_rewards):.2f}/{avg_true_reward:.2f}/{max(true_rewards):.2f}]"
            )

            # 2) Fragment trajectories into segment pairs
            print("[2/5] Fragmenting trajectories...")
            segment_pairs = self.fragmenter.fragment(trajectories=trajectories, num_pairs=num_pairs)
            print(f"      segment_pairs={len(segment_pairs)} (requested {num_pairs})")

            # 3) Generate preferences from true env rewards
            print("[3/5] Generating preferences...")
            segment_pairs, preferences = self._preferences_from_true_rewards(segment_pairs)
            debug_accuracy = self._compute_preference_accuracy(segment_pairs, preferences)
            print(f"      debug_accuracy={debug_accuracy:.3f}")

            # 4a) Split and accumulate into preference datasets
            train_pairs, train_prefs, val_pairs, val_prefs = self._train_val_split(
                segment_pairs, preferences
            )
            self.preference_dataset.push(train_pairs, train_prefs)
            self.preference_dataset_val.push(val_pairs, val_prefs)
            print(
                f"      dataset — train={len(self.preference_dataset)}"
                f"  val={len(self.preference_dataset_val)}"
            )

            # 4b) Train reward model
            print("[4/5] Training reward model...")
            self.reward_trainer.train(
                self.preference_dataset,
                num_epochs=reward_model_epochs_per_it,
            )
            reward_model_global_epochs += reward_model_epochs_per_it

            if relabel_replay_buffer:
                self._relabel_replay_buffer()
            else:
                self.reward_wrapper.reset_stats()

            loss_val_reward_model = self.reward_trainer.evaluate(self.preference_dataset_val)
            acc_train = self._compute_preference_accuracy(train_pairs, train_prefs)
            acc_val = self._compute_preference_accuracy(val_pairs, val_prefs)
            acc_train_global = self._compute_preference_accuracy(
                self.preference_dataset.pairs, self.preference_dataset.preferences
            )
            acc_val_global = self._compute_preference_accuracy(
                self.preference_dataset_val.pairs, self.preference_dataset_val.preferences
            )
            avg_ens_std, max_ens_std = self._compute_ensemble_uncertainty(trajectories)
            print(f"      loss_val={loss_val_reward_model:.4f}"
                  f"  ens_std=[avg={avg_ens_std:.3f}  max={max_ens_std:.3f}]")
            print(
                f"      accuracy — train_current={acc_train:.2f}"
                f"  val_current={acc_val:.2f}"
                f"  train_global={acc_train_global:.2f}"
            )

            is_pretraining = iteration < pretrain_iterations
            enough_data = self.agent.num_timesteps >= learning_starts
            agent_update_due = (iteration % agent_update_every_n == agent_update_every_n - 1)

            if is_pretraining:
                print(
                    f"[-/5] Pretraining phase: skipping SAC update"
                    f" for first {pretrain_iterations} iterations."
                )
            elif not enough_data:
                print(
                    f"[-/5] Collecting initial data"
                    f" (need {learning_starts} steps,"
                    f" have {self.agent.num_timesteps})."
                )
            elif not agent_update_due:
                rm_updates_until_agent = agent_update_every_n - 1 - (iteration % agent_update_every_n)
                print(
                    f"[-/5] RM updated, agent update in"
                    f" {rm_updates_until_agent} iteration(s)"
                    f" (every_n={agent_update_every_n})."
                )
            else:
                # 5) SAC gradient updates on the replay buffer
                print("[5/5] SAC policy update...")
                self.agent.train(
                    batch_size=self.agent.batch_size,
                    gradient_steps=gradient_steps,
                )

            # --- Logging ---
            avg_model_reward     = self._compute_avg_model_reward(trajectories)
            avg_model_reward_raw = self._compute_avg_model_reward(trajectories, pessimism=0.0)
            pessimistic_penalty  = avg_model_reward_raw - avg_model_reward
            buf = self.agent.replay_buffer
            buf_fill = (buf.pos if not buf.full else buf.buffer_size) / buf.buffer_size
            print(
                f"      agent_steps={self.agent.num_timesteps}"
                f"  buf={buf_fill:.0%}"
                f"  model_reward_raw={avg_model_reward_raw:.1f}"
                f"  pessimistic_penalty={pessimistic_penalty:.1f}"
                f"  avg_true_reward={avg_true_reward:.3f}"
            )

            self.logger.record("iterations", iteration + 1)
            self.logger.record("rollout/num_pairs", num_pairs)
            self.logger.record("rollout/avg_true_reward", avg_true_reward)
            self.logger.record("rollout/min_true_reward", float(min(true_rewards)))
            self.logger.record("rollout/max_true_reward", float(max(true_rewards)))
            self.logger.record("rollout/avg_model_reward", avg_model_reward)
            self.logger.record("rollout/avg_model_reward_raw", avg_model_reward_raw)
            self.logger.record("rollout/pessimistic_penalty", pessimistic_penalty)
            self.logger.record("rollout/avg_ep_length", avg_ep_length)
            self.logger.record("rollout/min_ep_length", float(min(ep_lengths)))
            self.logger.record("rollout/max_ep_length", float(max(ep_lengths)))
            self.logger.record("reward_model/ensemble_std_avg", avg_ens_std)
            self.logger.record("reward_model/ensemble_std_max", max_ens_std)
            self.logger.record("reward_model/global_epochs", reward_model_global_epochs)
            self.logger.record("reward_model/loss_validation", loss_val_reward_model)
            self.logger.record("reward_model/accuracy_train_current", acc_train)
            self.logger.record("reward_model/accuracy_val_current", acc_val)
            self.logger.record("reward_model/accuracy_train_global", acc_train_global)
            self.logger.record("reward_model/accuracy_val_global", acc_val_global)
            self.logger.record("reward_model/pretraining", is_pretraining)
            self.logger.record("reward_model/debug_accuracy", debug_accuracy)
            self.logger.record("agent/replay_buffer_fill", buf_fill)
            self.logger.record("agent/time/total_timesteps", self.agent.num_timesteps)
            self.logger.record("agent/update_due", agent_update_due)

            # --- Checkpoint ---
            if checkpoint_path is not None:
                _reward_window.append(avg_true_reward)
                smoothed_reward = float(np.mean(_reward_window))
                self.logger.record("rollout/smoothed_true_reward", smoothed_reward)
                if smoothed_reward > _best_smoothed_reward:
                    _best_smoothed_reward = smoothed_reward
                    self.agent.save(checkpoint_path)
                    print(
                        f"      [checkpoint] new best smoothed_reward={smoothed_reward:.3f}"
                        f"  saved → {checkpoint_path}.zip"
                    )

            self.logger.dump()

            iteration += 1

        callback.on_training_end()
        return self.agent

    # -----------------------------------------------------------------------
    # Replay buffer relabeling
    # -----------------------------------------------------------------------

    def _relabel_replay_buffer(self) -> None:
        """Rewrite all rewards in the replay buffer with the current RM and recompute norm stats."""
        buf = self.agent.replay_buffer
        n = buf.buffer_size if buf.full else buf.pos
        if n == 0:
            return

        obs = buf.observations[:n].reshape(n * buf.n_envs, -1)
        act = buf.actions[:n].reshape(n * buf.n_envs, -1)
        # Replay buffer does not store ego_status: pass zeros (treated as unknown)
        raw_rewards = self.reward_model.predict(obs, act, pessimism=self.pessimism, info=None)

        stats = self.reward_wrapper._reward_stats
        stats.__init__()
        stats.update(raw_rewards)

        normalized = ((raw_rewards - stats.mean) / stats.std).astype(np.float32)
        buf.rewards[:n] = normalized.reshape(n, buf.n_envs)

    # -----------------------------------------------------------------------
    # Preference helpers
    # -----------------------------------------------------------------------

    def _preferences_from_true_rewards(
        self, segment_pairs: List[SegmentPair]
    ) -> tuple[List[SegmentPair], List[Preference]]:
        filtered_pairs: List[SegmentPair] = []
        preferences: List[Preference] = []
        for pair in segment_pairs:
            r1, r2 = pair.seg1.total_reward(), pair.seg2.total_reward()
            if r1 == r2:
                continue
            filtered_pairs.append(pair)
            preferences.append(Preference((1, 0) if r1 > r2 else (0, 1)))
        return filtered_pairs, preferences

    def _compute_preference_accuracy(
        self, segment_pairs: List[SegmentPair], preferences: List[Preference]
    ) -> float:
        if not preferences:
            return 0.0
        correct = sum(
            1
            for pair, pref in zip(segment_pairs, preferences)
            if (self.preference_model.preference_probs(pair.seg1, pair.seg2).label[0] > 0.5)
            == (pref.label[0] > 0.5)
        )
        return correct / len(preferences)

    def _compute_avg_model_reward(self, trajectories: List[Trajectory], pessimism: Optional[float] = None) -> float:
        p = self.pessimism if pessimism is None else pessimism
        ep_model_rewards = [
            float(
                self.reward_model.predict(
                    np.array([t.obs for t in traj.transitions]),
                    np.array([t.action for t in traj.transitions]),
                    pessimism=p,
                    info=np.stack([
                        encode_ego_status(t.info.get("ego_status", "running"))
                        if t.info is not None else np.zeros(4, dtype=np.float32)
                        for t in traj.transitions
                    ]),
                ).sum()
            )
            for traj in trajectories
        ]
        return float(np.mean(ep_model_rewards))

    def _compute_ensemble_uncertainty(self, trajectories: List[Trajectory]) -> tuple[float, float]:
        """Returns (mean_std, max_std) of ensemble predictions across all trajectory steps."""
        all_stds = []
        for traj in trajectories:
            obs = np.array([t.obs    for t in traj.transitions])
            act = np.array([t.action for t in traj.transitions])
            info = np.stack([
                encode_ego_status(t.info.get("ego_status", "running"))
                if t.info is not None else np.zeros(4, dtype=np.float32)
                for t in traj.transitions
            ])
            var = self.reward_model.ensemble_variance(obs, act, info=info)  # (T,)
            all_stds.extend(np.sqrt(var).tolist())
        all_stds = np.array(all_stds)
        return float(all_stds.mean()), float(all_stds.max())

    def _train_val_split(
        self,
        pairs: List[SegmentPair],
        preferences: List[Preference],
        split_ratio: float = 0.7,
    ):
        n = len(pairs)
        indices = np.random.permutation(n)
        split_idx = int(n * split_ratio)
        train_idx, val_idx = indices[:split_idx], indices[split_idx:]
        return (
            [pairs[i] for i in train_idx],
            [preferences[i] for i in train_idx],
            [pairs[i] for i in val_idx],
            [preferences[i] for i in val_idx],
        )