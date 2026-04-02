import math
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
)
from .preference_trainer import RewardTrainerChristiano


class ChristianoAlgorithm:

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

        # The agent learns from the env wrapped with the reward model
        env_reward_wrapper = EnvRewardWrapper(self.env, self.reward_model)
        self.reward_wrapper = env_reward_wrapper
        self.agent.set_env(env_reward_wrapper)
        self.agent.set_logger(PrefixLogger(self.logger, "agent"))

        self.schedule_num_pairs = InverseSchedule(
            initial_value=num_pairs_initial,
            final_value=num_pairs_final,
            decay_rate=decay_pairs_schedule,
        )

    def train(self, 
            total_iterations: int = 10,
            pretrain_iterations: int = 0,
            agent_timesteps_per_it: int = 100_000, 
            reward_model_epochs_per_it: int = 10,
            agent_log_interval: int = 100,
            segments_oversample_factor: int = 3,
        ) -> Any:
        
        agent_global_timesteps = 0
        reward_model_global_epochs = 0

        for it in range(total_iterations):
            print(f"\n=== Iteration {it+1}/{total_iterations} ===")
            progress_remaining = 1 - (it - pretrain_iterations) / (total_iterations - pretrain_iterations) if it > pretrain_iterations else 1.0

            # 1) Sample trajectories with current agent
            print("[1/5] Collecting rollouts...")
            num_pairs = int(self.schedule_num_pairs(progress_remaining))
            total_segments_target = 2 * num_pairs * segments_oversample_factor 
            trajectories = self._collect_rollout(total_segments_target)

            avg_ep_length = float(np.mean([len(t.transitions) for t in trajectories]))
            avg_true_reward = float(np.mean([t.total_reward() for t in trajectories]))
            print(
                f"      trajectories={len(trajectories)}  avg_ep_length={avg_ep_length:.1f}  avg_true_reward={avg_true_reward:.3f}")

            # 2) Fragment trajectories
            print("[2/5] Fragmenting trajectories...")
            segment_pairs = self.fragmenter.fragment(trajectories=trajectories, num_pairs=num_pairs)

            print(f"      segment_pairs={len(segment_pairs)} (requested {num_pairs})")

            # 3) Generate preferences from true rewards
            print("[3/5] Generating preferences...")
            segment_pairs, preferences = self._preferences_from_true_rewards(segment_pairs)

            debug_accuracy = self._compute_preference_accuracy(segment_pairs, preferences)

            print(f"      debug_accuracy={debug_accuracy:.3f}")

            train_pairs, train_prefs, val_pairs, val_prefs = self._train_val_split(
                segment_pairs, preferences
            )
            self.preference_dataset.push(train_pairs, train_prefs)
            self.preference_dataset_val.push(val_pairs, val_prefs)

            print(f"      dataset — train={len(self.preference_dataset)}  val={len(self.preference_dataset_val)}")

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
                self.preference_dataset.pairs, self.preference_dataset.preferences)
            acc_val_global = self._compute_preference_accuracy(
                self.preference_dataset_val.pairs, self.preference_dataset_val.preferences)
            print(f"      loss_val={loss_val_reward_model:.4f}")
            print(
                f"      accuracy — train_current={acc_train:.2f}  val_current={acc_val:.2f}  train_global={acc_train_global:.2f}")


            is_pretraining = pretrain_iterations > 0 and it < pretrain_iterations
            if is_pretraining:
                print(f"[-/5] Pretraining phase: skipping agent training for first {pretrain_iterations} iterations.")
            else:
                # 5) Train agent
                print("[5/5] Training agent...")
                # Sync env state: _collect_rollout stepped self.env directly,
                # leaving env_reward_wrapper._obs and agent._last_obs stale.
                sync_obs = self.reward_wrapper.reset()
                self.agent._last_obs = sync_obs
                self.agent._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)
                self.agent.learn(
                    total_timesteps=agent_timesteps_per_it,
                    reset_num_timesteps=False,
                    log_interval=agent_log_interval,
                    callback=CustomLoggingCallback(),
                )
                agent_global_timesteps += agent_timesteps_per_it

            # --- Logging ---
            avg_model_reward = self._compute_avg_model_reward(trajectories)

            print(
                f"      agent_steps={agent_global_timesteps}  avg_model_reward={avg_model_reward:.3f}  avg_true_reward={avg_true_reward:.3f}")

            # rollout metrics
            self.logger.record("iterations", it + 1)
            self.logger.record("rollout/num_pairs",         num_pairs)
            self.logger.record("rollout/avg_true_reward",   avg_true_reward)
            self.logger.record("rollout/avg_model_reward",  avg_model_reward)
            self.logger.record("rollout/avg_ep_length",     avg_ep_length)

            # reward model metrics
            self.logger.record("reward_model/global_epochs", reward_model_global_epochs)
            self.logger.record("reward_model/loss_validation", loss_val_reward_model)
            self.logger.record("reward_model/accuracy_train_current", acc_train)
            self.logger.record("reward_model/accuracy_val_current",   acc_val)
            self.logger.record("reward_model/accuracy_train_global",  acc_train_global)
            self.logger.record("reward_model/accuracy_val_global",    acc_val_global)
            self.logger.record("reward_model/pretraining", is_pretraining)

            self.logger.record("reward_model/debug_accuracy", debug_accuracy)

            # agent metrics
            self.logger.record("agent/time/total_timesteps", agent_global_timesteps)

            self.logger.dump()

        return self.agent

    # -----------------------------------------------------------------------
    # Rollout collection
    # -----------------------------------------------------------------------

    def _collect_rollout(self, total_segments_target: int) -> List[Trajectory]:
        num_envs = self.env.num_envs
        current_transitions = [[] for _ in range(num_envs)]
        trajectories: List[Trajectory] = []
        total_segments = 0

        obs = self.env.reset()
        if isinstance(obs, tuple):  # gymnasium-style
            obs, _ = obs

        while total_segments < total_segments_target:
            action, _ = self.agent.predict(obs, deterministic=False)
            step_result = self.env.step(action)

            if len(step_result) == 5:  # gymnasium VecEnv
                next_obs, reward, terminated, truncated, _ = step_result
                done = terminated | truncated
            else:  # SB3 VecEnv
                next_obs, reward, done, _ = step_result

            for i in range(num_envs):
                current_transitions[i].append(
                    Transition(
                        obs=obs[i].copy(),
                        action=int(action[i]) if self.discrete_actions else action[i].copy(),
                        reward=float(reward[i]),
                    )
                )
                if done[i]:
                    traj_i = Trajectory(current_transitions[i])
                    trajectories.append(traj_i)
                    current_transitions[i] = []

                    length = traj_i.length()
                    num_segments = math.ceil(length / self.fragmenter.segment_length)
                    total_segments += num_segments
                    
            obs = next_obs

        # Include unfinished trajectories
        for i in range(num_envs):
            if current_transitions[i]:
                trajectories.append(Trajectory(current_transitions[i]))

        return trajectories

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
                continue  # skip ties: cross-entropy has no neutral label
            filtered_pairs.append(pair)
            preferences.append(Preference((1, 0) if r1 > r2 else (0, 1)))
        return filtered_pairs, preferences

    def _compute_preference_accuracy(
        self, segment_pairs: List[SegmentPair], preferences: List[Preference]
    ) -> float:
        if not preferences:
            return 0.0
        correct = sum(
            1 for pair, pref in zip(segment_pairs, preferences)
            if (self.preference_model.preference_probs(pair.seg1, pair.seg2).label[0] > 0.5) ==
               (pref.label[0] > 0.5)
        )
        return correct / len(preferences)

    def _compute_avg_model_reward(self, trajectories: List[Trajectory]) -> float:
        """Mean per-episode model reward across trajectories."""
        ep_model_rewards = [
            float(self.reward_model.predict(
                np.array([t.obs for t in traj.transitions]),
                np.array([t.action for t in traj.transitions]),
            ).sum())
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
    
