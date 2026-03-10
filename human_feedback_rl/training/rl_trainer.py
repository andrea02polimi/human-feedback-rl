import os
import torch
from human_feedback_rl.utils.logging import Logger


class RLTrainer:

    def __init__(self, env, policy, reward_model, log_dir="runs/rl"):

        self.env = env
        self.policy = policy
        self.reward_model = reward_model

        self.optimizer = torch.optim.Adam(policy.parameters(), lr=5e-5)

        self.logger = Logger(log_dir)

        self.global_rewards = []
        self.global_lengths = []
        self.global_losses = []

    # ------------------------------------------

    def learned_reward(self, state, action):

        s = torch.tensor(state).float().unsqueeze(0)
        a = torch.tensor([action])

        with torch.no_grad():
            r = self.reward_model(s, a)

        return r.item()

    # ------------------------------------------

    def train(self, episodes=1000):

        for episode in range(1, episodes + 1):

            obs, _ = self.env.reset()
            done = False

            log_probs = []
            rewards = []

            reward_sum = 0
            length = 0

            while not done:

                state = torch.tensor(obs).float().unsqueeze(0)

                logits = self.policy(state)

                probs = torch.softmax(logits, dim=1)

                dist = torch.distributions.Categorical(probs)

                action = dist.sample()

                log_prob = dist.log_prob(action)

                obs, _, terminated, truncated, _ = self.env.step(action.item())

                done = terminated or truncated

                r = self.learned_reward(obs, action.item())

                log_probs.append(log_prob)
                rewards.append(r)

                reward_sum += r
                length += 1

            returns = []
            G = 0

            for r in reversed(rewards):
                G = r + 0.99 * G
                returns.insert(0, G)

            returns = torch.tensor(returns)

            loss = 0

            for log_prob, G in zip(log_probs, returns):
                loss += -log_prob * G

            loss /= len(log_probs)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # tensorboard logging
            self.logger.log_episode(
                episode,
                reward_sum,
                length,
                loss.item(),
                0,
                0,
                0
            )

            self.global_rewards.append(reward_sum)
            self.global_lengths.append(length)
            self.global_losses.append(loss.item())

    # ------------------------------------------

    def save_model(self, path):

        os.makedirs("models", exist_ok=True)

        full_path = os.path.join("models", path)

        torch.save(self.policy.state_dict(), full_path)

        print(f"\nModel saved to {full_path}")

    # ------------------------------------------

    def print_summary(self):

        print("\n====== RL TRAINING SUMMARY ======")

        print(
            f"Episodes: {len(self.global_rewards)}\n"
            f"Average reward: {sum(self.global_rewards) / len(self.global_rewards):.2f}\n"
            f"Max reward: {max(self.global_rewards):.2f}\n"
            f"Min reward: {min(self.global_rewards):.2f}\n"
            f"Average episode length: {sum(self.global_lengths) / len(self.global_lengths):.2f}\n"
            f"Average loss: {sum(self.global_losses) / len(self.global_losses):.4f}"
        )