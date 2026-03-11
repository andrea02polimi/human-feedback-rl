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
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)

    def update_policy(self, buffer, optimizer, expert_model, batch_size=256):
        states, actions, rewards = buffer.sample_batch(batch_size)

        advantages = torch.tensor(rewards, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = torch.clamp(advantages, -3, 3)

        losses = []

        for s, a, adv in zip(states, actions, advantages):
            state = torch.as_tensor(s, dtype=torch.float32).unsqueeze(0)

            logits = self.forward(state)

            log_probs = torch.log_softmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)

            log_prob = log_probs[0, a]

            entropy = -(probs * log_probs).sum()

            # policy gradient
            pg_loss = -log_prob * adv

            # expert distribution from DQN Q-values
            with torch.no_grad():
                obs_tensor = torch.as_tensor(s, dtype=torch.float32).unsqueeze(0)

                q_values = expert_model.q_net(obs_tensor)

                expert_probs = torch.softmax(q_values, dim=1)

            kl = torch.sum(
                expert_probs * (torch.log(expert_probs + 1e-8) - log_probs)
            )

            kl = torch.clamp(kl, 0, 5)

            loss = pg_loss - 0.05 * entropy + 0.1 * kl

            losses.append(loss)

        loss = torch.stack(losses).mean()

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)

        optimizer.step()

        return loss.item()