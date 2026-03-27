"""
ChristianoAlgorithm — Christiano et al. (2017) RLHF algorithm.

Like SB3 algorithms, all configuration is passed to __init__.
Call train(...) to run the full synchronous pipeline.
"""

from typing import Any, List

import numpy as np
import wandb

from . import RewardTrainerChristiano
from human_feedback_rl.common import *



class ChristianoAlgorithm(BaseAlgorithm):

    def __init__(
        self,
        env,
        agent,
        n_ensembles: int,
        n_max: int,
        length_segment: int,
        num_max_segment_pairs: int,
        lr: float = 2e-4,
        device: str = "cpu",
    ):
        self.env = env
        self.agent = agent

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n  # Discrete(5)

        self.reward_model = EnsembleRewardModel(
            obs_dim, action_dim, n_ensembles, lr=lr, device=device
        )
        self.preference_model = PreferenceModelFromReward(self.reward_model)
        self.reward_trainer = RewardTrainerChristiano(self.preference_model)
        self.fragmenter = ActiveFragmenter(
            self.reward_model, length_segment, num_max_segment_pairs
        )
        self.preference_dataset = PreferenceDataset(n_max)

        env_reward_wrapper = EnvRewardWrapper(self.env, self.reward_model)
        self.agent.set_env(env_reward_wrapper)


    def train(
        self,
        total_timesteps: int,
        num_iterations: int,
        num_pairs_initial: int,
        num_pairs_final: int = 0,
        decay_pairs_schedule: float = 1.0,
    ) -> Any:
        
        timesteps_per_iter = int(total_timesteps / num_iterations)

        schedule = InverseSchedule(
            initial_value=num_pairs_initial,
            final_value=num_pairs_final,
            decay_rate=decay_pairs_schedule
        )

        for it in range(num_iterations):

            # ---------------------------
            # 1) Collect trajectories
            # ---------------------------
            trajectories = self._get_rollout(schedule(it/num_iterations))


            # ---------------------------
            # Log policy metrics (true env reward)
            # ---------------------------
            avg_true_reward = float(np.mean([t.total_reward() for t in trajectories]))
            avg_ep_length = float(np.mean([len(t.transitions) for t in trajectories]))
            print(
                f"[iter {it + 1}/{num_iterations}] "
                f"avg_true_reward={avg_true_reward:.3f}  avg_ep_length={avg_ep_length:.1f}"
            )
            if wandb.run is not None:
                wandb.log(
                    {
                        "policy/avg_true_reward": avg_true_reward,
                        "policy/avg_episode_length": avg_ep_length,
                    },
                    step=it + 1,
                )

            # ---------------------------
            # 2) Fragment trajectories
            # ---------------------------
            segment_pairs: List[SegmentPair] = self.fragmenter.fragment(trajectories)

            if not segment_pairs:
                continue

            # ---------------------------
            # 3) Compute preferences (oracle: true env reward)
            # ---------------------------
            preferences: List[Preference] = []

            for pair in segment_pairs:
                r1 = pair.seg1.total_reward()
                r2 = pair.seg2.total_reward()

                if r1 > r2:
                    preferences.append(Preference((1, 0)))
                elif r2 > r1:
                    preferences.append(Preference((0, 1)))
                else:
                    # Tie — label uniformly at random
                    if np.random.rand() < 0.5:
                        preferences.append(Preference((1, 0)))
                    else:
                        preferences.append(Preference((0, 1)))

            self.preference_dataset.push(segment_pairs, preferences)



            # ---------------------------
            # 4) Train reward model
            # ---------------------------
            loss = self.reward_trainer.train(self.preference_dataset)
            print(f"[iter {it + 1}/{num_iterations}] reward_model loss: {loss:.4f}")

            # ---------------------------
            # 5) Train agent on predicted rewards
            # ---------------------------
            self.agent.learn(
                total_timesteps=timesteps_per_iter,
                reset_num_timesteps=False,
            )

        return self.agent
    
    
    def _get_rollout(self, num_traj_rollout):
        
        trajectories: List[Trajectory] = []

        obs = self.env.reset()
        if isinstance(obs, tuple):  # gymnasium-style (obs, infos)
            obs, _ = obs

        for _ in range(num_traj_rollout):
            transitions = []
            done = np.zeros(self.env.num_envs, dtype=bool)

            while not done[0]:
                action, _ = self.agent.predict(obs, deterministic=False)
                step_result = self.env.step(action)

                if len(step_result) == 5:  # gymnasium VecEnv
                    next_obs, reward, terminated, truncated, info = step_result
                    done = terminated | truncated
                else:  # classic SB3 VecEnv
                    next_obs, reward, done, info = step_result

                transitions.append(
                    Transition(
                        obs=obs[0].copy(),
                        action=int(action[0]),
                        reward=float(reward[0]),
                    )
                )
                obs = next_obs

            trajectories.append(Trajectory(transitions))

            # Reset env[0] for the next rollout
            obs = self.env.reset()
            if isinstance(obs, tuple):
                obs, _ = obs

        return trajectories