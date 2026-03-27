from typing import Any, List
import numpy as np

from human_feedback_rl.common import *
from .reward_trainer_christiano import RewardTrainerChristiano

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

        # Inject logger into SB3
        self.agent.set_logger(PrefixLogger(self.logger, prefix="agent"))

        self.reward_model = EnsembleRewardModel(
            obs_dim = env.observation_space.shape[0], 
            action_dim = env.action_space.n, 
            n_ensembles = n_ensembles, 
            lr = lr_reward_model, 
            device = device
        )

        self.preference_model = PreferenceModelFromReward(self.reward_model)

        self.reward_trainer = RewardTrainerChristiano(
            self.preference_model,
            logger=PrefixLogger(self.logger),   
            batch_size=reward_model_batch_size,
            num_epochs=reward_training_epochs, 
        )

        self.fragmenter = ActiveFragmenter(
            reward_model=self.reward_model,
            segment_length=segment_length,
        )

        self.preference_dataset = PreferenceDataset(max_dataset_size)
        self.preference_dataset_val = PreferenceDataset(max_dataset_size)

        # the agent learns from the env wrapped with the reward model
        env_reward_wrapper = EnvRewardWrapper(self.env, self.reward_model)
        self.agent.set_env(env_reward_wrapper)

        # schedule for number of pairs to sample at each iteration
        self.schedule_num_pairs = InverseSchedule(
            initial_value=num_pairs_initial,
            final_value=num_pairs_final,
            decay_rate=decay_pairs_schedule,
        )

    def train(
        self,
        total_timesteps: int = 1e6,
        num_iterations: int = 10,
    ) -> Any:

        per_iter_timesteps = int(total_timesteps / num_iterations)

        global_agent_steps = 0
        global_reward_trainer_epochs = 0

        for it in range(num_iterations):
            print(f"\n=== Iteration {it+1}/{num_iterations} ===")
            progress_remaining = 1 - it / num_iterations

            # 1) Sample trajectories with current agent
            print("[1/5] Collecting rollouts...")
            num_pairs = int(self.schedule_num_pairs(progress_remaining))
            tot_rollout_timesteps = num_pairs * self.fragmenter.segment_length * 2
            trajectories = self._get_rollout(tot_rollout_timesteps)


            # 2) Fragment trajectories
            print("[2/5] Fragmenting trajectories...")      
            segment_pairs = self.fragmenter.fragment(
                trajectories=trajectories, 
                num_pairs=num_pairs
            )

            # 3) Generate preferences from true rewards
            print("[3/5] Generating preferences...")
            preferences = self._preferences_from_true_rewards(segment_pairs)

            # 3.1) Add to dataset --> the reward model is trained on all data collected so far
            train_pairs, train_prefs, val_pairs, val_prefs = self._split_segment_pairs_and_preference_data(
                segment_pairs, preferences
            )
            self.preference_dataset.push(train_pairs, train_prefs)
            self.preference_dataset_val.push(val_pairs, val_prefs)

            # 4) Train reward model
            print("[4/5] Training reward model...")
            loss = self.reward_trainer.train(
                self.preference_dataset,
            )
            validation_loss = self.reward_trainer.evaluate(self.preference_dataset_val)

            global_reward_trainer_epochs += self.reward_training_epochs

            # 5) Train agent
            print("[5/5] Training agent...")
            self.agent.learn(
                total_timesteps=per_iter_timesteps,
                reset_num_timesteps=False,
            )

            global_agent_steps += per_iter_timesteps

            # Logging
            avg_true_reward = np.mean([t.total_reward() for t in trajectories])
            avg_ep_length = np.mean([len(t.transitions) for t in trajectories])
            pref_model_current_accuracy_train = self._compute_preference_model_accuracy(train_pairs, train_prefs)
            pref_model_current_accuracy_val = self._compute_preference_model_accuracy(val_pairs, val_prefs)
            pref_model_global_accuracy_train = self._compute_preference_model_accuracy(self.preference_dataset.get_pairs(), self.preference_dataset.get_preferences())
            pref_model_global_accuracy_val = self._compute_preference_model_accuracy(self.preference_dataset_val.get_pairs(), self.preference_dataset_val.get_preferences())

            self.logger.record("rollout_trajectories/avg_true_reward", avg_true_reward)
            self.logger.record("rollout_trajectories/avg_episode_length", avg_ep_length)
            self.logger.record("rollout_trajectories/num_pairs", num_pairs)

            self.logger.record("reward_model/training_loss", loss)
            self.logger.record("reward_model/validation_loss", validation_loss)
            self.logger.record("reward_model/current_accuracy_train_wrt_true_reward", pref_model_current_accuracy_train)
            self.logger.record("reward_model/current_accuracy_val_wrt_true_reward", pref_model_current_accuracy_val)
            self.logger.record("reward_model/global_accuracy_train_wrt_true_reward", pref_model_global_accuracy_train)
            self.logger.record("reward_model/global_accuracy_val_wrt_true_reward", pref_model_global_accuracy_val)
            
            self.logger.record("timescales/iterations", it)
            self.logger.record("timescales/global_reward_trainer_epochs", global_reward_trainer_epochs)
            self.logger.record("timescales/global_agent_steps", global_agent_steps)

            # unified dump at each iteration
            self.logger.dump(it)

        return self.agent
    

    def _get_rollout(self, total_timesteps_target):
        
        num_envs = self.env.num_envs

        # Active trajectories (one per env)
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
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated | truncated
            else:  # classic SB3 VecEnv
                next_obs, reward, done, info = step_result

            for i in range(num_envs):
                current_transitions[i].append(
                    Transition(
                        obs=obs[i].copy(),
                        action=int(action[i]),
                        reward=float(reward[i]),
                    )
                )

                # If episode ends → close trajectory
                if done[i]:
                    trajectories.append(Trajectory(current_transitions[i]))
                    current_transitions[i] = []

            obs = next_obs
            total_timesteps += num_envs  # 🔥 critical fix

        # Optional: include unfinished trajectories
        for i in range(num_envs):
            if len(current_transitions[i]) > 0:
                trajectories.append(Trajectory(current_transitions[i]))

        return trajectories

    
    def _preferences_from_true_rewards(self, segment_pairs: List[SegmentPair]) -> List[Preference]:
        
        preferences: List[Preference] = []

        for pair in segment_pairs:
            r1 = pair.seg1.total_reward()
            r2 = pair.seg2.total_reward()

            if r1 > r2:
                preferences.append(Preference((1, 0)))
            else:
                preferences.append(Preference((0, 1)))

        return preferences
    

    def _compute_preference_model_accuracy(self, segment_pairs: List[SegmentPair], preferences: List[Preference]) -> float:
        
        correct = 0

        for pair, pref in zip(segment_pairs, preferences):
            pref_predicted = self.preference_model.preference_probs(pair.seg1, pair.seg2)
            if pref_predicted.label[0] > pref_predicted.label[1]:  # P(seg1 preferred) > P(seg2 preferred)
                pref_predicted_label = (1, 0)
            else:
                pref_predicted_label = (0, 1)
            
            if pref_predicted_label == pref.label:
                correct += 1

        return correct / len(preferences) if preferences else 0.0



    def _split_segment_pairs_and_preference_data(
            self,
            pairs: List[SegmentPair],
            preferences: List[Preference],
            split_ratio: float = 0.7,
        ):

        assert len(pairs) == len(preferences)

        # 1. create shuffled indices (synchronous shuffle)
        n = len(pairs)
        indices = np.random.permutation(n)

        pairs_shuffled = [pairs[i] for i in indices]
        prefs_shuffled = [preferences[i] for i in indices]

        # 2. split 70 / 30
        split_idx = int(n * split_ratio)

        train_pairs = pairs_shuffled[:split_idx]
        train_prefs = prefs_shuffled[:split_idx]

        val_pairs = pairs_shuffled[split_idx:]
        val_prefs = prefs_shuffled[split_idx:]

        return train_pairs, train_prefs, val_pairs, val_prefs