import torch
import torch.nn.functional as f

from human_feedback_rl.training.base_trainer import BaseTrainer
from human_feedback_rl.experts.demonstration_expert import ConcreteStepDemonstrationExpert
from human_feedback_rl.core import Step


class ImitationTrainer(BaseTrainer):

    def __init__(self, env, policy, expert_model, optimizer, run_dir=None):

        super().__init__(
            env,
            policy,
            expert_model,
            optimizer,
            run_dir,
            name="imitation"
        )

        self.demo_expert = ConcreteStepDemonstrationExpert(expert_model)

        self.global_matches = []
        self.global_entropy = []
        self.global_kl = []

        self.dataset_states = []
        self.dataset_actions = []

        self.dataset_capacity = 50000
        self.train_step_counter = 0

    # ------------------------------------------------

    def train(self, episodes):

        for episode in range(1, episodes + 1):

            obs = self.env.reset()
            done = False

            reward_sum = 0
            loss_sum = 0
            length = 0

            episode_match = 0
            episode_entropy = 0
            episode_kl = 0

            beta = max(0.97 ** episode, 0.5)

            while not done:

                logits, action_match, entropy, kl, state = self.forward_and_metrics(obs)

                if episode < 300:
                    agent_action = torch.multinomial(
                        torch.softmax(logits, dim=1),
                        1
                    ).item()
                else:
                    agent_action = torch.argmax(logits, dim=1).item()

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
                        reduction="batchmean" # mean over the batch, kl is a scalar
                    )

                    loss = ce_loss + 0.1 * kl

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                else:
                    loss = torch.tensor(0.0)

                action_to_env = expert_action if torch.rand(1).item() < beta else agent_action

                obs, rewards, dones, infos = self.env.step([action_to_env])

                reward = rewards[0]
                done = dones[0]

                reward_sum += reward
                loss_sum += loss.item()
                length += 1

                episode_match += action_match
                episode_entropy += entropy
                episode_kl += kl

            avg_match = episode_match / length
            avg_entropy = episode_entropy / length
            avg_kl = episode_kl / length

            self.log_episode(
                episode,
                reward_sum,
                length,
                loss_sum / length,
                avg_kl,
                avg_match,
                avg_entropy
            )

            self.global_matches.append(avg_match)
            self.global_entropy.append(avg_entropy)
            self.global_kl.append(avg_kl)