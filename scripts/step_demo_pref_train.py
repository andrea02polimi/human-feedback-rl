from pathlib import Path

import torch.optim as optim

from human_feedback_rl.utils.env_setup import build_env_and_expert, build_policy
from human_feedback_rl.training.hybrid_trainer import DemoPrefTrainer

ROOT = Path(__file__).resolve().parents[1]

MODEL = ROOT / "2026-03-09_17-15-41_train_highway_fast_DQN/model.zip"


def main():

    print("Loading environment and expert...")

    env, expert_model = build_env_and_expert(MODEL)

    policy = build_policy(env)

    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    trainer = DemoPrefTrainer(env, policy, expert_model, optimizer)

    trainer.train(episodes=200)

    env.close()


if __name__ == "__main__":
    main()