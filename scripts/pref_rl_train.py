from pathlib import Path

import torch

from human_feedback_rl.agents.policy_network import PolicyNetwork
from human_feedback_rl.data.preference_dataset import ReplayBuffer
from human_feedback_rl.models.reward_model import RewardModel
from human_feedback_rl.training.imitation_trainer import ImitationTrainer
from human_feedback_rl.training.preference_rl_trainer import PreferenceTrainer
from human_feedback_rl.utils.env_setup import build_env_and_expert
from human_feedback_rl.value_network import ValueNetwork

ROOT = Path(__file__).resolve().parents[1]

MODEL = ROOT / "2026-03-09_17-15-41_train_highway_fast_DQN/model.zip"

def main():

    env, expert_model = build_env_and_expert(MODEL)

    obs_dim = env.observation_space.shape[-1]
    n_actions = env.action_space.n

    policy = PolicyNetwork(obs_dim, n_actions)
    value = ValueNetwork(obs_dim)
    reward_model = RewardModel(obs_dim, n_actions)

    buffer = ReplayBuffer()

    policy_opt = torch.optim.Adam(policy.parameters(), lr=5e-5)
    value_opt = torch.optim.Adam(value.parameters(), lr=1e-4)
    reward_opt = torch.optim.Adam(reward_model.parameters(), lr=1e-4)

    imitation = ImitationTrainer(env, expert_model, policy, policy_opt)

    imitation.train(episodes=200)

    trainer = PreferenceTrainer(
        env,
        policy,
        value,
        reward_model,
        expert_model,
        buffer,
        policy_opt,
        value_opt,
        reward_opt
    )

    trainer.train()

    torch.save(policy.state_dict(), "policy.pt")

    env.close()

if __name__ == "__main__":
    main()