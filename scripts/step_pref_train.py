import torch
from human_feedback_rl.training.preference_trainer import PreferenceTrainer
from human_feedback_rl.utils.env_setup import build_env_and_expert, build_policy

MODEL = "2026-03-04_19-04-17_test_dqn_highway/model.zip"

def main():
    import os
    print("cwd:", os.getcwd())
    print("model exists:", os.path.exists(MODEL))

    env, expert_model = build_env_and_expert(MODEL)

    policy = build_policy(env)

    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    trainer = PreferenceTrainer(env, policy, expert_model, optimizer)

    trainer.train(episodes=200)

    env.close()


if __name__ == "__main__":
    main()