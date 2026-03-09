import os

import torch
import torch.nn.functional as f

from human_feedback_rl.concrete_experts.Concrete_demonstration_expert import ConcreteStepDemonstrationExpert
from human_feedback_rl.training.base_trainer import BaseTrainer
from human_feedback_rl.Core import Step
from human_feedback_rl.utils.logging import Logger


class ImitationTrainer(BaseTrainer):

    def __init__(self, env, policy, expert_model, optimizer):

        super().__init__(env, policy, expert_model, optimizer)

        self.demo_expert = ConcreteStepDemonstrationExpert(expert_model)

        log_dir = os.path.join(self.base_log_dir, "imitation")

        self.logger = Logger(log_dir)

    # ------------------------------------------------

    def train(self, episodes):

        for episode in range(1, episodes + 1):

            obs, _ = self.env.reset()

            done = False

            reward_sum = 0
            loss_sum = 0
            # kl_sum = 0
            length = 0

            episode_match = 0
            episode_entropy = 0
            episode_kl = 0

            while not done:

                logits, action_match, entropy, kl, state = self.forward_and_metrics(obs)

                # --------------------------

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