import os

from human_feedback_rl.feedback import PreferenceFeedback
from human_feedback_rl.concrete_experts.concrete_preference_expert import ConcreteStepPreferenceExpert
from human_feedback_rl.training.base_trainer import BaseTrainer
from human_feedback_rl.utils.logging import Logger
from human_feedback_rl.utils.losses import preference_loss
from human_feedback_rl.utils.sampling import sample_two_actions
from human_feedback_rl.core import Step


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

                logits, action_match, entropy, kl, state = self.forward_and_metrics(obs)

                # --------------------------

                a_i, a_j = sample_two_actions(logits)

                step_i = Step(state, a_i)
                step_j = Step(state, a_j)

                feedback: PreferenceFeedback = self.pref_expert.query([step_i, step_j])

                loss = preference_loss(
                    logits,
                    a_i,
                    a_j,
                    feedback.preferred_index
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
                # kl_sum / length,
                avg_kl,
                avg_match,
                avg_entropy,
            )