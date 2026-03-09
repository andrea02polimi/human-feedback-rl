import torch
import torch.nn as nn


class AgentPolicyNetwork(nn.Module):

    def __init__(self, obs_dim, n_actions):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        logits = self.net(x)
        return torch.clamp(logits, -10, 10)

    def update_policy(self, buffer, optimizer, batch_size=256):

        if len(buffer.states) < batch_size:
            return 0.0

        returns = self.compute_returns(buffer.rewards)

        idx = torch.randint(0, len(returns), (batch_size,))

        returns = torch.tensor(
            [returns[i] for i in idx],
            dtype=torch.float32
        )

        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        losses = []

        for k, i in enumerate(idx):
            s = buffer.states[i]
            a = buffer.actions[i]
            r = returns[k]

            state = torch.as_tensor(s, dtype=torch.float32).unsqueeze(0)

            logits = self.forward(state)

            log_probs = torch.log_softmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)

            log_prob = log_probs[0, a]

            entropy = -(probs * log_probs).sum()

            loss = -log_prob * r - 0.05 * entropy

            losses.append(loss)

        loss = torch.stack(losses).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()

        return loss.item()

    @staticmethod
    def compute_returns(rewards, gamma=0.99):
        returns = []
        r = 0

        for rew in reversed(rewards):
            r = rew + gamma * r
            returns.insert(0, r)

        return returns