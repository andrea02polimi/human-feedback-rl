"""
ChristianoRLHF — Christiano et al. (2017) RLHF algorithm.

Like SB3 algorithms, all configuration parameters is passed to __init__ constructor.
Call train(output_dir) to start the full asynchronous pipeline.
"""

from human_feedback_rl.algorithms.base_trainer import BaseTrainer


class ChristianoAlgorithm(BaseTrainer):

    def __init__(
        self,
        env,
        agent,
        n_ensembles,
        n_max,
        length_segment,
        num_max_segment_pairs,

    ):
        self.env = env
        self.agent = agent
        
        env_reward_wrapper = EnvRewardWrapper(self.env)
        self.agent.set_env(env_reward_wrapper)
        
        self.reward_model = EnsembleRewardModel(n_ensembles)

        self.preference_model = PreferenceModelFromReward(reward_model)

        self.reward_trainer = RewardTrainerChristiano(preference_model)

        self.fragmenter = ActiveFragmenter(reward_model, length_segment, num_max_segment_pairs)

        self.preference_dataset = PreferenceDataset(n_max)


    def train(
        self,
        total_timesteps,
        num_iterations,
        num_traj_rollout,
    ) -> None:

        for it in range(num_iterations):

            # ---------------------------
            # 1) Collect trajectories
            # ---------------------------
            trajectories: List[Trajectory] = []

            for _ in range(num_traj_rollout):
                obs, _ = self.env.reset()
                self.agent.reset()

                terminated, truncated = False, False
                transitions = []

                while not (terminated or truncated):
                    action = self.agent.predict(obs)
                    next_obs, reward, terminated, truncated, info = self.env.step(action)

                    transitions.append(
                        Transition(obs=obs, action=action, reward=reward)
                    )

                    obs = next_obs

                trajectories.append(Trajectory(transitions))

            # ---------------------------
            # 2) Fragment trajectories
            # ---------------------------
            segment_pairs: List[SegmentPair] = self.fragmenter.fragment(trajectories)

            # ---------------------------
            # 3) Compute preferences
            # ---------------------------
            preferences: List[Preference] = []

            for pair in segment_pairs:
                r1 = pair.seg1.total_reward()
                r2 = pair.seg2.total_reward()

                if r1 > r2:
                    preferences.append(Preference((1, 0)))
                else:
                    preferences.append(Preference((0, 1)))

            self.preference_dataset.push(segment_pairs, preferences)

            self.reward_trainer.train(self.preference_dataset)
            
            self.agent.learn(
                total_timesteps=total_timesteps/num_iterations
            )

