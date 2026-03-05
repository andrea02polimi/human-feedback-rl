import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

from stable_baselines3 import DQN

from src.Core import Step
from src.concrete_experts.Concrete_demonstration_expert import (
    ConcreteStepDemonstrationExpert,
)


# -------------------------------------------------
# Policy network (agente)
# -------------------------------------------------

class Policy(nn.Module):

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
        return self.net(x)


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    env = gym.make("SumoEnv-v0")

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # agente da addestrare
    policy = Policy(obs_dim, n_actions)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    # -------------------------------------------------
    # expert dimostrazione (SB3)
    # -------------------------------------------------

    expert_model = DQN.load("expert_policies/model.zip")
    demonstration_expert = ConcreteStepDemonstrationExpert(
        env,
        expert_model,
    )

    # -------------------------------------------------
    # preference dataset
    # -------------------------------------------------

    # preference_dataset = PreferenceDataset("data/preferences.pkl")

    # -------------------------------------------------
    # training loop
    # -------------------------------------------------
    state, _ = env.reset()

    for step_id in range(100000):

        state_tensor = torch.tensor(
            state, dtype=torch.float32
        ).unsqueeze(0)

        logits = policy(state_tensor)

        probs = torch.softmax(logits, dim=1)

        action = torch.multinomial(probs, 1).item()

        # ---------------------------------------------
        # Step object
        # ---------------------------------------------

        step = Step(state, action)

        # ---------------------------------------------
        # demonstration feedback
        # ---------------------------------------------

        demo_feedback = demonstration_expert.query(step)

        expert_action = demo_feedback.action

        demo_target = torch.tensor(
            [expert_action], dtype=torch.long
        )

        demo_loss = torch.nn.functional.cross_entropy(
            logits,
            demo_target,
        )

        # ---------------------------------------------
        # preference feedback (dataset)
        # ---------------------------------------------



        # ---------------------------------------------
        # combined loss
        # ---------------------------------------------

        loss = demo_loss + 0.5 * pref_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ---------------------------------------------
        # environment step
        # ---------------------------------------------

        next_state, _, terminated, truncated, _ = env.step(
            action
        )

        state = next_state

        if terminated or truncated:
            state, _ = env.reset()


if __name__ == "__main__":
    main()