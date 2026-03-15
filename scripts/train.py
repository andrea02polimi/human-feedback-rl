import hydra
import torch
import torch.optim as optim

from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from human_feedback_rl.training.imitation_trainer import ImitationTrainer
from human_feedback_rl.training.preference_trainer import PreferenceTrainer
from human_feedback_rl.training.rl_trainer import RLTrainer

from human_feedback_rl.experts.preference_expert import ConcreteStepPreferenceExpert
from human_feedback_rl.data.preference_dataset import PreferenceDataset
from human_feedback_rl.models.reward_model import RewardModel

from human_feedback_rl.utils.env_setup import build_env_and_expert, build_policy
from human_feedback_rl.utils.preferences import collect_preferences


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):

    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    run_dir = Path(HydraConfig.get().runtime.output_dir)

    env, expert_model = build_env_and_expert(cfg)

    obs_dim = env.observation_space.shape[-1]
    n_actions = env.action_space.n

    policy = build_policy(env)

    # -----------------------------
    # Imitation learning
    # -----------------------------

    if cfg.pipeline.imitation:

        optimizer = optim.Adam(
            policy.parameters(),
            lr=cfg.imitation.policy_lr
        )

        imitation = ImitationTrainer(
            env,
            policy,
            expert_model,
            optimizer,
            run_dir
        )

        imitation.train(cfg.imitation.episodes)

        imitation.save_model(cfg.paths.bc_model)

        imitation.print_summary()

    # -----------------------------
    # Preferences
    # -----------------------------

    reward_model = None

    if cfg.pipeline.preferences:

        dataset = PreferenceDataset()

        pref_expert = ConcreteStepPreferenceExpert(env, expert_model)

        collect_preferences(
            env,
            policy,
            pref_expert,
            dataset,
            episodes=cfg.preferences.collect_episodes
        )

        reward_model = RewardModel(obs_dim, n_actions)

        pref_trainer = PreferenceTrainer(
            reward_model,
            optim.Adam(
                reward_model.parameters(),
                lr=cfg.preferences.lr
            ),
            dataset,
            run_dir
        )

        pref_trainer.train(cfg.preferences.epochs)

        torch.save(
            reward_model.state_dict(),
            cfg.paths.reward_model
        )

        print(f"\nReward model saved to {cfg.paths.reward_model}")

    # -----------------------------
    # RLHF
    # -----------------------------

    if cfg.pipeline.rlhf:

        if reward_model is None:

            reward_model = RewardModel(obs_dim, n_actions)

            reward_model.load_state_dict(
                torch.load(cfg.paths.reward_model)
            )

        rl_trainer = RLTrainer(
            env,
            policy,
            reward_model,
            run_dir
        )

        rl_trainer.train(cfg.rl.episodes)

        rl_trainer.save_model(cfg.paths.rl_model)

        rl_trainer.print_summary()

    env.close()

    print("\nTraining finished")
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()