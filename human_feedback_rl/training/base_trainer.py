from pathlib import Path
import torch
import torch.nn.functional as f

from human_feedback_rl.utils.logging import Logger


class BaseTrainer:

    def __init__(self, env, policy, expert_model, optimizer, run_dir=None, name="trainer"):

        self.env = env
        self.policy = policy
        self.expert_model = expert_model
        self.optimizer = optimizer

        self.run_dir = Path(run_dir) if run_dir else Path("runs")

        self.models_dir = self.run_dir / "models"
        self.tb_dir = self.run_dir / "tensorboard" / name

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.tb_dir.mkdir(parents=True, exist_ok=True)

        self.logger = Logger(self.tb_dir)

        # metriche comuni
        self.global_rewards = []
        self.global_lengths = []
        self.global_losses = []

    # ---------------------------------------------

    def log_episode(
        self,
        episode,
        reward,
        length,
        loss,
        kl=0,
        match=0,
        entropy=0
    ):

        self.logger.log_episode(
            episode,
            reward,
            length,
            loss,
            kl,
            match,
            entropy
        )

        self.global_rewards.append(reward)
        self.global_lengths.append(length)
        self.global_losses.append(loss)

    # ---------------------------------------------

    def save_model(self, path):

        path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(self.policy.state_dict(), path)

        print(f"\nModel saved to {path}")

    # ---------------------------------------------

    def print_summary(self):

        if len(self.global_rewards) == 0:
            return

        print("\n====== TRAINING SUMMARY ======")

        print(
            f"Episodes: {len(self.global_rewards)}\n"
            f"Average reward: {sum(self.global_rewards)/len(self.global_rewards):.2f}\n"
            f"Max reward: {max(self.global_rewards):.2f}\n"
            f"Min reward: {min(self.global_rewards):.2f}\n"
            f"Average episode length: {sum(self.global_lengths)/len(self.global_lengths):.2f}\n"
            f"Average loss: {sum(self.global_losses)/len(self.global_losses):.4f}"
        )

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