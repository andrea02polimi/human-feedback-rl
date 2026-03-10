from pathlib import Path

import torch
import torch.optim as optim

from human_feedback_rl.concrete_experts.concrete_preference_expert import ConcreteStepPreferenceExpert
from human_feedback_rl.preference_dataset import PreferenceDataset
from human_feedback_rl.reward_model import RewardModel
from human_feedback_rl.training.imitation_trainer import ImitationTrainer
from human_feedback_rl.training.preference_trainer import PreferenceTrainer
from human_feedback_rl.training.rl_trainer import RLTrainer
from human_feedback_rl.utils.env_setup import build_env_and_expert, build_policy
from human_feedback_rl.utils.preferences import collect_preferences

ROOT = Path(__file__).resolve().parents[1]

MODEL = ROOT / "2026-03-09_17-15-41_train_highway_fast_DQN/model.zip"

BC_MODEL = ROOT / "scripts/models/policy_bc.pt"


def main():

    print("Loading environment and expert...")

    env, expert_model = build_env_and_expert(MODEL)

    policy = build_policy(env)

    optimizer = optim.Adam(policy.parameters(), lr=1e-4)

    '''imitation = ImitationTrainer(
        env,
        policy,
        expert_model,
        optimizer
    )

    imitation.train(episodes=2000)

    imitation.save_model("policy_bc.pt")

    imitation.print_summary()'''

    if BC_MODEL.exists():

        print("Loading BC policy...")

        policy.load_state_dict(torch.load(BC_MODEL))


    pref_dataset = PreferenceDataset()

    pref_expert = ConcreteStepPreferenceExpert(env, expert_model)

    collect_preferences(env, policy, pref_expert, pref_dataset)

    obs_dim = env.observation_space.shape[-1]
    n_actions = env.action_space.n

    reward_model = RewardModel(obs_dim, n_actions)

    pref_trainer = PreferenceTrainer(
        reward_model,
        torch.optim.Adam(reward_model.parameters(), lr=1e-4),
        pref_dataset
    )

    pref_trainer.train(epochs=2000)

    rl_trainer = RLTrainer(
        env,
        policy,
        reward_model
    )

    rl_trainer.train(episodes=1000)

    rl_trainer.save_model("policy_rlhf.pt")

    rl_trainer.print_summary()

    env.close()

    env.close()


if __name__ == "__main__":
    main()