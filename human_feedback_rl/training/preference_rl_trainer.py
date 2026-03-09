import os

import torch

from human_feedback_rl.concrete_experts.concrete_preference_expert import ConcreteTrajectoryPreferenceExpert
from human_feedback_rl.core import Step, Trajectory
from human_feedback_rl.feedback import PreferenceFeedback
from human_feedback_rl.replay_buffer import ReplayBuffer
from human_feedback_rl.training.base_trainer import BaseTrainer
from human_feedback_rl.utils.logging import Logger


class PreferenceRLTrainer(BaseTrainer):

    def __init__(self,
                 env,
                 policy,
                 expert_model,
                 reward_model,
                 policy_optimizer,
                 reward_optimizer):
        super().__init__(env, policy, expert_model, policy_optimizer)

        self.reward_model = reward_model

        self.reward_optimizer = reward_optimizer

        self.buffer = ReplayBuffer()

        self.pref_expert = ConcreteTrajectoryPreferenceExpert(env, expert_model)

        log_dir = os.path.join(self.base_log_dir, "preference_rl")

        self.logger = Logger(log_dir)

    def train(self,
              iterations=1000,
              query_interval=20):

        global_rewards = []
        global_lengths = []
        global_policy_losses = []

        for it in range(iterations):

            # rollout policy
            traj, stats = self.rollout()

            reward_sum = stats["reward_sum"]
            length = stats["length"]

            episode_match = stats["match"]
            episode_entropy = stats["entropy"]
            episode_kl = stats["kl"]

            self.buffer.add_trajectory(traj.steps)

            # query preference
            if it % query_interval == 0 and len(self.buffer.states) > 50:
                seg1, seg2 = self.buffer.sample_segments()

                feedback: PreferenceFeedback = self.pref_expert.query([seg1, seg2])

                pref = feedback.preferred_index

                loss = self.reward_model.update_reward_model(
                    seg1,
                    seg2,
                    pref,
                    self.reward_optimizer
                )

                print("reward model loss", loss)

            # relabel rewards
            if it > 20:
                self.buffer.relabel_rewards(self.reward_model)

            # update policy
            policy_loss = self.policy.update_policy(
                self.buffer,
                self.optimizer
            )

            avg_match = episode_match / length
            avg_entropy = episode_entropy / length
            avg_kl = episode_kl / length

            self.logger.log_episode(
                it,
                reward_sum,
                length,
                policy_loss,
                avg_kl,
                avg_match,
                avg_entropy,
            )

            global_rewards.append(reward_sum)
            global_lengths.append(length)
            global_policy_losses.append(policy_loss)

        print("\n=== TRAINING SUMMARY ===")

        print(
            f"episodes: {iterations}\n"
            f"avg reward: {sum(global_rewards) / len(global_rewards):.2f}\n"
            f"avg length: {sum(global_lengths) / len(global_lengths):.2f}\n"
            f"avg policy loss: {sum(global_policy_losses) / len(global_policy_losses):.4f}\n"
            f"max reward: {max(global_rewards):.2f}\n"
            f"min reward: {min(global_rewards):.2f}"
        )

    def rollout(self):

        obs, _ = self.env.reset()

        done = False

        steps = []

        reward_sum = 0
        length = 0

        episode_match = 0
        episode_entropy = 0
        episode_kl = 0

        while not done:
            state = obs

            logits, action_match, entropy, kl, state_tensor = self.forward_and_metrics(obs)

            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()

            obs, reward, terminated, truncated, _ = self.env.step(action)

            steps.append(Step(state, action))

            done = terminated or truncated

            reward_sum += reward
            length += 1

            episode_match += action_match
            episode_entropy += entropy
            episode_kl += kl

        trajectory = Trajectory(steps)

        stats = {
            "reward_sum": reward_sum,
            "length": length,
            "match": episode_match,
            "entropy": episode_entropy,
            "kl": episode_kl
        }

        return trajectory, stats