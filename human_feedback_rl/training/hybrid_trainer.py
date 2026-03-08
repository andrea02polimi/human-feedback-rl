import os

import torch
import torch.nn.functional as F

from human_feedback_rl.concrete_experts.Concrete_demonstration_expert import ConcreteStepDemonstrationExpert
from human_feedback_rl.concrete_experts.Concrete_preference_expert import ConcreteStepPreferenceExpert
from human_feedback_rl.training.base_trainer import BaseTrainer
from human_feedback_rl.utils.logging import Logger
from human_feedback_rl.utils.losses import imitation_loss, preference_loss
from human_feedback_rl.utils.history_sampling import sample_pair_from_history
from human_feedback_rl.Core import Step
from human_feedback_rl.utils.sampling import sample_topk_actions


class DemoPrefTrainer(BaseTrainer):

    def __init__(self,
                 env,
                 policy,
                 expert_model,
                 optimizer,
                 pref_weight=0.1):

        super().__init__(env, policy, expert_model, optimizer)

        self.pref_expert = ConcreteStepPreferenceExpert(env, expert_model)
        self.demo_expert = ConcreteStepDemonstrationExpert(expert_model)

        self.pref_weight = pref_weight

        log_dir = os.path.join(self.base_log_dir, "imitation_preference")

        self.logger = Logger(log_dir)

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
                )

                # --------------------------

                # ---------------------
                # DEMONSTRATION LOSS
                # ---------------------

                step = Step(state, 0)

                demo_feedback = self.demo_expert.query(step)

                expert_action = int(demo_feedback.value)

                demo_loss = imitation_loss(logits, expert_action)

                # ---------------------
                # PREFERENCE LOSS
                # ---------------------

                pref_loss = torch.tensor(0.0, device=logits.device)

                a_i, a_j = sample_topk_actions(logits)

                step_i = Step(state, a_i)
                step_j = Step(state, a_j)

                feedback = self.pref_expert.query([step_i, step_j])

                pref_loss = preference_loss(
                    logits,
                    a_i,
                    a_j,
                    feedback.preferred_index
                )

                # ---------------------

                # -----------------------------
                # WARMUP STRATEGY
                # -----------------------------

                if episode < 50:
                    loss = demo_loss

                else:

                    pref_ratio = min(0.4, (episode - 50) / 150)

                    loss = (1 - pref_ratio) * demo_loss + pref_ratio * pref_loss

                # -----------------------------
                # KL REGULARIZATION
                # -----------------------------

                loss = loss + 0.01 * kl

                # -----------------------------
                # OPTIMIZATION
                # -----------------------------

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                action = self.select_action(logits)

                obs, reward, terminated, truncated, _ = self.env.step(action)

                done = terminated or truncated

                reward_sum += reward
                loss_sum += loss.item()
                length += 1

                episode_match += action_match
                episode_entropy += entropy
                episode_kl += kl.item()

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