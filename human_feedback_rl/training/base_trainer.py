import torch
import torch.nn.functional as f

import os


class BaseTrainer:

    def __init__(self, env, policy, expert_model, optimizer):

        self.env = env
        self.policy = policy
        self.optimizer = optimizer
        self.expert_model = expert_model

        self.base_log_dir = os.path.join("tensorboard", "training")

        os.makedirs(self.base_log_dir, exist_ok=True)

    # ------------------------------------------------

    def compute_kl_on_expert_state(self, step):
        state = step.state

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        ).unsqueeze(0)

        agent_logits = self.policy(state_tensor)

        with torch.no_grad():
            expert_logits = self.expert_model.q_net(state_tensor)

        expert_probs = torch.softmax(expert_logits, dim=1)
        agent_log_probs = torch.log_softmax(agent_logits, dim=1)

        return f.kl_div(agent_log_probs, expert_probs, reduction="batchmean")

    # ------------------------------------------------

    @staticmethod
    def select_action(logits):
        probs = torch.softmax(logits, dim=1)
        return torch.multinomial(probs, 1).item()


    def optimize_step(self, logits, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        action = self.select_action(logits)

        obs, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        return obs, reward, done, action

    def forward_and_metrics(self, obs):
        state = obs[0] if len(obs.shape) > 1 else obs

        state_tensor = torch.tensor(
            state, dtype=torch.float32
        ).unsqueeze(0)

        logits = self.policy(state_tensor)

        with torch.no_grad():
            expert_logits = self.expert_model.q_net(state_tensor)

        agent_probs = torch.softmax(logits, dim=1)
        expert_probs = torch.softmax(expert_logits, dim=1)

        agent_action = torch.argmax(agent_probs, dim=1)
        expert_action = torch.argmax(expert_probs, dim=1)

        action_match = (agent_action == expert_action).float().item()

        entropy = -(agent_probs * torch.log(agent_probs + 1e-8)).sum(dim=1).item()

        kl = f.kl_div(
            torch.log_softmax(logits, dim=1),
            expert_probs,
            reduction="batchmean"
        ).item()

        return logits, action_match, entropy, kl, state_tensor