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

        # dataset aggregation
        self.dataset_states = []
        self.dataset_actions = []

        self.dataset_capacity = 50000

        self.train_step_counter = 0

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

            # β schedule: all'inizio usa più spesso l'esperto, poi l'agente
            beta = 0.97 ** episode
            beta = max(beta, 0.5)

            while not done:

                logits, action_match, entropy, kl, state = self.forward_and_metrics(obs)

                # azione dell'agente (sample dalla policy)
                if episode < 300:
                    agent_action = torch.multinomial(torch.softmax(logits, dim=1), 1).item()
                else:
                    agent_action = torch.argmax(logits, dim=1).item()

                # query all'esperto sullo stato corrente
                step = Step(state, None)
                feedback = self.demo_expert.query(step)
                expert_action = int(feedback.value)

                self.dataset_states.append(state.squeeze(0))
                self.dataset_actions.append(expert_action)

                self.train_step_counter += 1

                if len(self.dataset_states) > self.dataset_capacity:
                    self.dataset_states.pop(0)
                    self.dataset_actions.pop(0)

                batch_size = min(64, len(self.dataset_states))

                recent = min(10000, len(self.dataset_states))

                idx = torch.randint(
                    len(self.dataset_states) - recent,
                    len(self.dataset_states),
                    (batch_size,)
                )

                states = torch.stack([self.dataset_states[i] for i in idx])
                actions = torch.tensor([self.dataset_actions[i] for i in idx])

                if self.train_step_counter % 8 == 0:

                    logits = self.policy(states)

                    ce_loss = f.cross_entropy(
                        logits,
                        actions,
                        label_smoothing=0.05
                    )

                    with torch.no_grad():
                        expert_logits = self.expert_model.q_net(states)

                    kl = torch.nn.functional.kl_div(
                        torch.log_softmax(logits, dim=1),
                        torch.softmax(expert_logits, dim=1),
                        reduction="batchmean"
                    )

                    loss = ce_loss + 0.1 * kl

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                else:
                    loss = torch.tensor(0.0)

                # DAgger: mix tra azione esperto e agente
                if torch.rand(1).item() < beta:
                    action_to_env = expert_action
                else:
                    action_to_env = agent_action

                obs, reward, terminated, truncated, _ = self.env.step(action_to_env)
                done = terminated or truncated

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