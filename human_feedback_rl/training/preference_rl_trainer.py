import os
import torch

from human_feedback_rl.concrete_experts.concrete_preference_expert import ConcreteTrajectoryPreferenceExpert
from human_feedback_rl.core import Step, Trajectory
from human_feedback_rl.feedback import PreferenceFeedback
from human_feedback_rl.replay_buffer import ReplayBuffer
from human_feedback_rl.training.base_trainer import BaseTrainer
from human_feedback_rl.utils.logging import Logger


class PreferenceRLTrainer(BaseTrainer):

    def __init__(
        self,
        env,
        policy,
        expert_model,
        reward_model,
        policy_optimizer,
        reward_optimizer
    ):
        super().__init__(env, policy, expert_model, policy_optimizer)

        self.reward_model = reward_model
        self.reward_optimizer = reward_optimizer

        self.buffer = ReplayBuffer()

        self.pref_expert = ConcreteTrajectoryPreferenceExpert(env, expert_model)

        log_dir = os.path.join(self.base_log_dir, "preference_rl")
        self.logger = Logger(log_dir)


    def pretrain_with_demonstrations(self, episodes=30):

        print("=== PRETRAINING WITH DEMONSTRATIONS ===")

        for ep in range(episodes):

            obs, _ = self.env.reset()
            obs = obs[0] if hasattr(obs, "shape") and len(obs.shape) > 1 else obs

            done = False

            while not done:

                state = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    expert_action, _ = self.expert_model.predict(obs, deterministic=True)

                if isinstance(expert_action, (list, tuple)):
                    expert_action = expert_action[0]

                if hasattr(expert_action, "shape"):
                    expert_action = expert_action.item()

                target = torch.tensor([expert_action], dtype=torch.long)

                logits = self.policy(state)

                loss = torch.nn.functional.cross_entropy(logits, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                obs, _, terminated, truncated, _ = self.env.step(expert_action)

                obs = obs[0] if hasattr(obs, "shape") and len(obs.shape) > 1 else obs

                done = terminated or truncated

        print("pretraining completed")

    def train(self, iterations=1000, query_interval=10):

        global_rewards = []
        global_lengths = []
        global_policy_losses = []

        # EARLY STOPPING
        best_reward = -float("inf")
        best_model_state = None

        reward_window = []
        window_size = 50

        collapse_patience = 500
        collapse_counter = 0

        for it in range(iterations):

            if len(self.buffer.states) < 2000:
                traj, _ = self.rollout()
                self.buffer.add_trajectory(traj)
                continue

            traj, stats = self.rollout()

            reward_sum = stats["reward_sum"]
            length = stats["length"]

            episode_match = stats["match"]
            episode_entropy = stats["entropy"]
            episode_kl = stats["kl"]

            self.buffer.add_trajectory(traj)

            if it % query_interval == 0:

                seg1, seg2 = self.buffer.sample_segments()

                feedback: PreferenceFeedback = self.pref_expert.query([seg1, seg2])

                pref = feedback.preferred_index

                for _ in range(10):
                    loss = self.reward_model.update_reward_model(
                        seg1,
                        seg2,
                        pref,
                        self.reward_optimizer
                    )

                    print("reward model loss", loss)

            if it % 200 == 0 and it > 0:
                self.buffer.relabel_rewards(self.reward_model, window=5000)

            policy_loss = 0

            for _ in range(3):
                policy_loss += self.policy.update_policy(
                    self.buffer,
                    self.optimizer,
                    self.expert_model
                )

            policy_loss /= 3

            self.logger.log_episode(
                it,
                reward_sum,
                length,
                policy_loss,
                episode_kl / max(length, 1),
                episode_match / max(length, 1),
                episode_entropy / max(length, 1)
            )

            global_rewards.append(reward_sum)

            reward_window.append(reward_sum)

            if len(reward_window) > window_size:
                reward_window.pop(0)

            avg_reward = sum(reward_window) / len(reward_window)

            if avg_reward > best_reward:
                best_reward = avg_reward
                best_model_state = {k: v.cpu() for k, v in self.policy.state_dict().items()}
                collapse_counter = 0
            else:
                collapse_counter += 1

            if collapse_counter > collapse_patience:
                print("\nTraining stopped: policy collapse detected")
                break

            global_lengths.append(length)
            global_policy_losses.append(policy_loss)

        print("\n=== TRAINING SUMMARY ===")

        print(
            f"episodes: {iterations}\n"
            f"avg reward: {sum(global_rewards) / len(global_rewards):.2f}\n"
            f"avg length: {sum(global_lengths) / len(global_lengths):.2f}\n"
            f"avg policy loss: {sum(global_policy_losses) / len(global_policy_losses):.4f}\n"
            f"max reward: {max(global_rewards):.2f}\n"
            f"min reward: {min(global_rewards):.2f}"
        )

        save_dir = os.path.join(self.base_log_dir, "models")
        os.makedirs(save_dir, exist_ok=True)

        model_path = os.path.join(save_dir, "agent_policy.pt")

        if best_model_state is not None:
            torch.save(best_model_state, model_path)
            print("Best model saved to:", model_path)
        else:
            torch.save(self.policy.state_dict(), model_path)
            print("Final model saved to:", model_path)


    def rollout(self):

        obs, _ = self.env.reset()

        obs = obs[0] if len(obs.shape) > 1 else obs

        done = False

        steps = []

        reward_sum = 0
        length = 0

        episode_match = 0
        episode_entropy = 0
        episode_kl = 0

        while not done:

            state = obs

            logits, action_match, entropy, kl, state_tensor = self.forward_and_metrics(obs)

            dist = torch.distributions.Categorical(logits=logits)

            action = dist.sample().item()

            obs, reward, terminated, truncated, _ = self.env.step(action)

            obs = obs[0] if hasattr(obs, "shape") and len(obs.shape) > 1 else obs

            steps.append(Step(state, action))

            done = terminated or truncated

            reward_sum += reward
            length += 1

            episode_match += action_match
            episode_entropy += entropy
            episode_kl += kl

        trajectory = Trajectory(steps)

        stats = {
            "reward_sum": reward_sum,
            "length": length,
            "match": episode_match,
            "entropy": episode_entropy,
            "kl": episode_kl
        }

        return trajectory, stats