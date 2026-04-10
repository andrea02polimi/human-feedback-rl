from typing import Any, List

import numpy as np
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
    Extends EnvRewardWrapper to also record the true environment reward in
    parallel with the RM reward supplied to the agent.

    The agent still receives normalized RM rewards (via the parent class).
    Completed true-reward trajectories are accumulated and can be drained with
    collect_trajectories(), which is called once per PPO rollout window.
    """

    def __init__(self, venv, reward_model):
        super().__init__(venv, reward_model)
        self._ep_transitions: List[List[Transition]] = [[] for _ in range(venv.num_envs)]
        self._completed_trajectories: List[Trajectory] = []

    def reset(self):
        obs = super().reset()
        self._ep_transitions = [[] for _ in range(self.num_envs)]
        self._completed_trajectories = []
        return obs

    def step_wait(self):
        obs, env_rewards, dones, infos = self.venv.step_wait()

        discrete = hasattr(self.venv.action_space, "n")

        if self._obs is not None and self._actions is not None:
            info_arr = np.stack([
                encode_ego_status(infos[i].get("ego_status", "running"))
                for i in range(self.num_envs)
            ])
            rm_rewards = self.reward_model.predict(self._obs, self._actions, info=info_arr)
            self._reward_stats.update(rm_rewards)
            normalized = (
                (rm_rewards - self._reward_stats.mean) / self._reward_stats.std
            ).astype(np.float32)

            for i in range(self.num_envs):
                action_i = int(self._actions[i]) if discrete else self._actions[i].copy()
                self._ep_transitions[i].append(
                    Transition(
                        obs=self._obs[i].copy(),
                        action=action_i,
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

class ChristianoPPOAlgorithm:

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
    ):
        self.env = env
        self.agent = agent

        self.logger = UnifiedLogger()

        discrete = hasattr(env.action_space, "n")
        self.discrete_actions = discrete
        self.reward_model = EnsembleRewardModel(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n if discrete else env.action_space.shape[0],
            n_ensembles=n_ensembles,
            lr=lr_reward_model,
            device=device,
            discrete_actions=discrete,
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

        self.reward_wrapper = _TrueRewardEnvWrapper(self.env, self.reward_model)
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
        pretrain_iterations: int = 0,
        reward_model_epochs_per_it: int = 10,
        agent_log_interval: int = 1,
        segments_oversample_factor: int = 3,
    ) -> Any:

        reward_model_global_epochs = 0

        # Initialise PPO internal state (resets num_timesteps, _last_obs, etc.)
        total_timesteps, callback = self.agent._setup_learn(
            total_timesteps,
            callback=CustomLoggingCallback(),
            reset_num_timesteps=True,
        )
        callback.on_training_start(locals(), globals())

        iteration = 0

        while self.agent.num_timesteps < total_timesteps:
            progress_remaining = 1.0 - self.agent.num_timesteps / total_timesteps
            num_pairs = int(self.schedule_num_pairs(progress_remaining))

            print(
                f"\n=== Iteration {iteration + 1}"
                f" (steps={self.agent.num_timesteps}/{total_timesteps}) ==="
            )

            # 1) Collect rollouts via PPO
            #    _TrueRewardEnvWrapper intercepts each step: passes normalized RM
            #    rewards to the rollout buffer and records true-reward transitions.
            print("[1/5] Collecting rollouts...")
            continue_training = self.agent.collect_rollouts(
                self.reward_wrapper,
                callback,
                self.agent.rollout_buffer,
                n_rollout_steps=self.agent.n_steps,
            )
            if not continue_training:
                break

            trajectories = self.reward_wrapper.collect_trajectories()

            if not trajectories:
                iteration += 1
                continue

            avg_ep_length = float(np.mean([len(t.transitions) for t in trajectories]))
            avg_true_reward = float(np.mean([t.total_reward() for t in trajectories]))
            print(
                f"      trajectories={len(trajectories)}"
                f"  avg_ep_length={avg_ep_length:.1f}"
                f"  avg_true_reward={avg_true_reward:.3f}"
            )

            # 2) Fragment trajectories into segment pairs
            print("[2/5] Fragmenting trajectories...")
            segment_pairs = self.fragmenter.fragment(
                trajectories=trajectories, num_pairs=num_pairs
            )
            print(f"      segment_pairs={len(segment_pairs)} (requested {num_pairs})")

            # 3) Generate preferences from true env rewards
            print("[3/5] Generating preferences...")
            segment_pairs, preferences = self._preferences_from_true_rewards(segment_pairs)
            debug_accuracy = self._compute_preference_accuracy(segment_pairs, preferences)
            print(f"      debug_accuracy={debug_accuracy:.3f}")

            train_pairs, train_prefs, val_pairs, val_prefs = self._train_val_split(
                segment_pairs, preferences
            )
            self.preference_dataset.push(train_pairs, train_prefs)
            self.preference_dataset_val.push(val_pairs, val_prefs)
            print(
                f"      dataset — train={len(self.preference_dataset)}"
                f"  val={len(self.preference_dataset_val)}"
            )

            # 4) Train reward model
            print("[4/5] Training reward model...")
            self.reward_trainer.train(
                self.preference_dataset,
                num_epochs=reward_model_epochs_per_it,
            )
            self.reward_wrapper.reset_stats()
            reward_model_global_epochs += reward_model_epochs_per_it

            loss_val_reward_model = self.reward_trainer.evaluate(self.preference_dataset_val)
            acc_train = self._compute_preference_accuracy(train_pairs, train_prefs)
            acc_val = self._compute_preference_accuracy(val_pairs, val_prefs)
            acc_train_global = self._compute_preference_accuracy(
                self.preference_dataset.pairs, self.preference_dataset.preferences
            )
            acc_val_global = self._compute_preference_accuracy(
                self.preference_dataset_val.pairs, self.preference_dataset_val.preferences
            )
            print(f"      loss_val={loss_val_reward_model:.4f}")
            print(
                f"      accuracy — train_current={acc_train:.2f}"
                f"  val_current={acc_val:.2f}"
                f"  train_global={acc_train_global:.2f}"
            )

            is_pretraining = iteration < pretrain_iterations
            if is_pretraining:
                print(
                    f"[-/5] Pretraining phase: skipping PPO update"
                    f" for first {pretrain_iterations} iterations."
                )
            else:
                # 5) PPO policy update on the collected rollout buffer
                print("[5/5] PPO policy update...")
                self.agent.train()

            # --- Logging ---
            avg_model_reward = self._compute_avg_model_reward(trajectories)
            print(
                f"      agent_steps={self.agent.num_timesteps}"
                f"  avg_model_reward={avg_model_reward:.3f}"
                f"  avg_true_reward={avg_true_reward:.3f}"
            )

            self.logger.record("iterations", iteration + 1)
            self.logger.record("rollout/num_pairs",          num_pairs)
            self.logger.record("rollout/avg_true_reward",    avg_true_reward)
            self.logger.record("rollout/avg_model_reward",   avg_model_reward)
            self.logger.record("rollout/avg_ep_length",      avg_ep_length)
            self.logger.record("reward_model/global_epochs", reward_model_global_epochs)
            self.logger.record("reward_model/loss_validation",         loss_val_reward_model)
            self.logger.record("reward_model/accuracy_train_current",  acc_train)
            self.logger.record("reward_model/accuracy_val_current",    acc_val)
            self.logger.record("reward_model/accuracy_train_global",   acc_train_global)
            self.logger.record("reward_model/accuracy_val_global",     acc_val_global)
            self.logger.record("reward_model/pretraining",             is_pretraining)
            self.logger.record("reward_model/debug_accuracy",          debug_accuracy)
            self.logger.record("agent/time/total_timesteps", self.agent.num_timesteps)
            self.logger.dump()

            iteration += 1

        callback.on_training_end()
        return self.agent

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

    def _compute_avg_model_reward(self, trajectories: List[Trajectory]) -> float:
        ep_model_rewards = [
            float(
                self.reward_model.predict(
                    np.array([t.obs for t in traj.transitions]),
                    np.array([t.action for t in traj.transitions]),
                ).sum()
            )
            for traj in trajectories
        ]
        return float(np.mean(ep_model_rewards))

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