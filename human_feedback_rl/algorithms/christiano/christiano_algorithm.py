from typing import Any, List

import numpy as np
import wandb

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
        reward_training_epochs: int = 10,
        num_pairs_initial: int = 100,
        num_pairs_final: int = 0,
        decay_pairs_schedule: float = 1.0,
    ):
        self.env = env
        self.agent = agent
        self.reward_training_epochs = reward_training_epochs

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
            num_epochs=reward_training_epochs,
        )

        self.fragmenter = ActiveFragmenter(
            reward_model=self.reward_model,
            segment_length=segment_length,
        )

        self.preference_dataset = PreferenceDataset(max_dataset_size)
        self.preference_dataset_val = PreferenceDataset(max_dataset_size)

        # The agent learns from the env wrapped with the reward model
        env_reward_wrapper = EnvRewardWrapper(self.env, self.reward_model)
        self.agent.set_env(env_reward_wrapper)
        self.agent.set_logger(PrefixLogger(self.logger, "agent"))

        self.schedule_num_pairs = InverseSchedule(
            initial_value=num_pairs_initial,
            final_value=num_pairs_final,
            decay_rate=decay_pairs_schedule,
        )

    def train(self, 
              total_timesteps: int = 1_000_000, 
              num_iterations: int = 10,
              rolling_window: int = 100
              ) -> Any:
        
        per_iter_timesteps = int(total_timesteps / num_iterations)
        agent_global_steps = 0
        reward_model_global_epochs = 0

        for it in range(num_iterations):
            print(f"\n=== Iteration {it+1}/{num_iterations} ===")
            progress_remaining = 1 - it / num_iterations

            # 1) Sample trajectories with current agent
            print("[1/5] Collecting rollouts...")
            num_pairs = int(self.schedule_num_pairs(progress_remaining))
            tot_rollout_timesteps = num_pairs * self.fragmenter.segment_length * 2
            trajectories = self._collect_rollout(tot_rollout_timesteps)

            # 2) Fragment trajectories
            print("[2/5] Fragmenting trajectories...")
            segment_pairs = self.fragmenter.fragment(trajectories=trajectories, num_pairs=num_pairs)

            # 3) Generate preferences from true rewards
            print("[3/5] Generating preferences...")
            preferences = self._preferences_from_true_rewards(segment_pairs)

            train_pairs, train_prefs, val_pairs, val_prefs = self._train_val_split(
                segment_pairs, preferences
            )
            self.preference_dataset.push(train_pairs, train_prefs)
            self.preference_dataset_val.push(val_pairs, val_prefs)

            # 4) Train reward model
            print("[4/5] Training reward model...")
            self.reward_trainer.train(self.preference_dataset)

            reward_model_global_epochs += self.reward_training_epochs

            # 5) Train agent
            print("[5/5] Training agent...")
            self.agent.learn(
                total_timesteps=per_iter_timesteps, 
                reset_num_timesteps=False,
                callback=CustomLoggingCallback(window_size=rolling_window),
            )
            agent_global_steps += per_iter_timesteps

            # --- Logging ---
            avg_true_reward = float(np.mean([t.total_reward() for t in trajectories]))
            avg_ep_length = float(np.mean([len(t.transitions) for t in trajectories]))
            avg_model_reward = self._compute_avg_model_reward(trajectories)
            loss_val_reward_model = self.reward_trainer.evaluate(self.preference_dataset_val)

            # rollout metrics
            self.logger.record("iterations",        it + 1)
            self.logger.record("rollout/num_pairs",         num_pairs)
            self.logger.record("rollout/avg_true_reward",   avg_true_reward)
            self.logger.record("rollout/avg_model_reward",  avg_model_reward)
            self.logger.record("rollout/avg_ep_length",     avg_ep_length)

            # reward model metrics
            self.logger.record("reward_model/global_epochs", reward_model_global_epochs)
            self.logger.record("reward_model/loss_validation", loss_val_reward_model)
            self.logger.record("reward_model/accuracy_train_current",
                               self._compute_preference_accuracy(train_pairs, train_prefs))
            self.logger.record("reward_model/accuracy_val_current",
                               self._compute_preference_accuracy(val_pairs, val_prefs))
            self.logger.record("reward_model/accuracy_train_global",
                               self._compute_preference_accuracy(
                                   self.preference_dataset.pairs,
                                   self.preference_dataset.preferences))
            self.logger.record("reward_model/accuracy_val_global",
                               self._compute_preference_accuracy(
                                   self.preference_dataset_val.pairs,
                                   self.preference_dataset_val.preferences))

            # agent metrics
            self.logger.record("agent/time/total_timesteps", agent_global_steps)

            self.logger.dump()

        return self.agent

    # -----------------------------------------------------------------------
    # Rollout collection
    # -----------------------------------------------------------------------

    def _collect_rollout(self, total_timesteps_target: int) -> List[Trajectory]:
        num_envs = self.env.num_envs
        current_transitions = [[] for _ in range(num_envs)]
        trajectories: List[Trajectory] = []
        total_timesteps = 0

        obs = self.env.reset()
        if isinstance(obs, tuple):  # gymnasium-style
            obs, _ = obs

        while total_timesteps < total_timesteps_target:
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
                    trajectories.append(Trajectory(current_transitions[i]))
                    current_transitions[i] = []

            obs = next_obs
            total_timesteps += num_envs

        # Include unfinished trajectories
        for i in range(num_envs):
            if current_transitions[i]:
                trajectories.append(Trajectory(current_transitions[i]))

        return trajectories

    # -----------------------------------------------------------------------
    # Preference helpers
    # -----------------------------------------------------------------------

    def _preferences_from_true_rewards(self, segment_pairs: List[SegmentPair]) -> List[Preference]:
        preferences = []
        for pair in segment_pairs:
            r1, r2 = pair.seg1.total_reward(), pair.seg2.total_reward()
            preferences.append(Preference((1, 0) if r1 > r2 else (0, 1)))
        return preferences

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