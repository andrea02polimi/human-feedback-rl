from pathlib import Path

import torch.optim as optim

from human_feedback_rl.training.imitation_trainer import ImitationTrainer
from human_feedback_rl.utils.env_setup import build_env_and_expert, build_policy


ROOT = Path(__file__).resolve().parents[1]

MODEL = ROOT / "2026-03-09_17-15-41_train_highway_fast_DQN/model.zip"


def main():

    print("Loading environment and expert...")

    env, expert_model = build_env_and_expert(MODEL)

    policy = build_policy(env)

    optimizer = optim.Adam(policy.parameters(), lr=5e-5)

    trainer = ImitationTrainer(
        env=env,
        policy=policy,
        expert_model=expert_model,
        optimizer=optimizer
    )

    trainer.train(episodes=2000)

    trainer.save_model("policy_final.pt")

    trainer.print_summary()

    env.close()

    env.close()


if __name__ == "__main__":
    main()