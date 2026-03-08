import os

import torch
import torch.nn.functional as F

from human_feedback_rl.concrete_experts.Concrete_preference_expert import ConcreteStepPreferenceExpert
from human_feedback_rl.training.base_trainer import BaseTrainer
from human_feedback_rl.utils.logging import Logger
from human_feedback_rl.utils.losses import preference_loss
from human_feedback_rl.utils.sampling import sample_two_actions
from human_feedback_rl.Core import Step


class PreferenceTrainer(BaseTrainer):

    def __init__(self, env, policy, expert_model, optimizer):

        super().__init__(env, policy, expert_model, optimizer)

        self.pref_expert = ConcreteStepPreferenceExpert(env, expert_model)

        log_dir = os.path.join(self.base_log_dir, "preference")

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

                state = obs[0] if len(obs.shape) > 1 else obs

                state_tensor = torch.tensor(
                    state, dtype=torch.float32
                ).unsqueeze(0)

                logits = self.policy(state_tensor)

                # -------- METRICS --------

                with torch.no_grad():
                    expert_logits = self.expert_model.q_net(state_tensor)

                agent_probs = torch.softmax(logits, dim=1)
                expert_probs = torch.softmax(expert_logits, dim=1)

                agent_action = torch.argmax(agent_probs, dim=1)
                expert_action = torch.argmax(expert_probs, dim=1)

                action_match = (agent_action == expert_action).float().item()

                entropy = -(agent_probs * torch.log(agent_probs + 1e-8)).sum(dim=1).item()

                kl = F.kl_div(
                    torch.log_softmax(logits, dim=1),
                    expert_probs,
                    reduction="batchmean"
                ).item()

                # --------------------------

                a_i, a_j = sample_two_actions(logits)

                step_i = Step(state, a_i)
                step_j = Step(state, a_j)

                feedback = self.pref_expert.query([step_i, step_j])

                loss = preference_loss(
                    logits,
                    a_i,
                    a_j,
                    feedback.preferred_index
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                action = self.select_action(logits)

                obs, reward, terminated, truncated, _ = self.env.step(action)

                done = terminated or truncated

                # kl = self.compute_kl_on_expert_state(step)

                reward_sum += reward
                loss_sum += loss.item()
                # kl_sum += kl.item()
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
                # kl_sum / length,
                avg_kl,
                avg_match,
                avg_entropy,
            )