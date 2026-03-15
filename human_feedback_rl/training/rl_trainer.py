import torch

from human_feedback_rl.training.base_trainer import BaseTrainer


class RLTrainer(BaseTrainer):

    def __init__(self, env, policy, reward_model, run_dir=None):

        optimizer = torch.optim.Adam(policy.parameters(), lr=5e-5)

        super().__init__(
            env=env,
            policy=policy,
            expert_model=None,
            optimizer=optimizer,
            run_dir=run_dir,
            name="rl"
        )

        self.reward_model = reward_model

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

            self.log_episode(
                episode,
                reward_sum,
                length,
                loss.item(),
                kl=0,
                match=0,
                entropy=0
            )