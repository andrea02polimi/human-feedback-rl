import os

import torch
import torch.nn.functional as f

from human_feedback_rl.concrete_experts.concrete_demonstration_expert import ConcreteStepDemonstrationExpert
from human_feedback_rl.training.base_trainer import BaseTrainer
from human_feedback_rl.core import Step
from human_feedback_rl.utils.logging import Logger


class ImitationTrainer(BaseTrainer):

    def __init__(self, env, policy, expert_model, optimizer):

        super().__init__(env, policy, expert_model, optimizer)

        self.demo_expert = ConcreteStepDemonstrationExpert(expert_model)

        log_dir = os.path.join(self.base_log_dir, "imitation")

        self.logger = Logger(log_dir)

        self.global_rewards = []
        self.global_lengths = []
        self.global_losses = []
        self.global_matches = []
        self.global_entropy = []
        self.global_kl = []

    # ------------------------------------------------

    def train(self, episodes):

        for episode in range(1, episodes + 1):

            obs, _ = self.env.reset()
            done = False

            reward_sum = 0
            loss_sum = 0
            length = 0

            episode_match = 0
            episode_entropy = 0
            episode_kl = 0

            while not done:
                logits, action_match, entropy, kl, state = self.forward_and_metrics(obs)

                step = Step(state, 0)

                feedback = self.demo_expert.query(step)

                expert_action = int(feedback.value)

                loss = f.cross_entropy(
                    logits,
                    torch.tensor([expert_action])
                )

                obs, reward, done, action = self.optimize_step(
                    logits,
                    loss
                )

                reward_sum += reward
                loss_sum += loss.item()
                length += 1

                episode_match += action_match
                episode_entropy += entropy
                episode_kl += kl

            if length > 0:
                avg_match = episode_match / length
                avg_entropy = episode_entropy / length
                avg_kl = episode_kl / length

                self.logger.log_episode(
                    episode,
                    reward_sum,
                    length,
                    loss_sum / length,
                    avg_kl,
                    avg_match,
                    avg_entropy,
                )

                self.global_rewards.append(reward_sum)
                self.global_lengths.append(length)
                self.global_losses.append(loss_sum / length)
                self.global_matches.append(avg_match)
                self.global_entropy.append(avg_entropy)
                self.global_kl.append(avg_kl)


    def save_model(self, path):

        os.makedirs("models", exist_ok=True)

        full_path = os.path.join("models", path)

        torch.save(self.policy.state_dict(), full_path)

        print(f"\nModel saved to {full_path}")

    def print_summary(self):

        print("\n====== TRAINING SUMMARY ======")

        print(
            f"Episodes: {len(self.global_rewards)}\n"
            f"Average reward: {sum(self.global_rewards) / len(self.global_rewards):.2f}\n"
            f"Max reward: {max(self.global_rewards):.2f}\n"
            f"Min reward: {min(self.global_rewards):.2f}\n"
            f"Average episode length: {sum(self.global_lengths) / len(self.global_lengths):.2f}\n"
            f"Average loss: {sum(self.global_losses) / len(self.global_losses):.4f}\n"
            f"Average action match: {sum(self.global_matches) / len(self.global_matches):.3f}\n"
            f"Average entropy: {sum(self.global_entropy) / len(self.global_entropy):.3f}\n"
            f"Average KL: {sum(self.global_kl) / len(self.global_kl):.3f}"
        )