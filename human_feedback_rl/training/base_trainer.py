import torch
import torch.nn.functional as F

import os
import datetime


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

        return F.kl_div(agent_log_probs, expert_probs, reduction="batchmean")

    # ------------------------------------------------

    def select_action(self, logits):

        return torch.argmax(logits, dim=1).item()