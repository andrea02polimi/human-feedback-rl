"""
Imitation + RLHF training script.

Pipeline:
  imitation — behavioural cloning from DQN expert demonstrations
  rlhf      — REINFORCE using a separately trained reward model

For the Christiano et al. preference-learning pipeline see train_christiano.py.
"""

import hydra
import torch
import torch.optim as optim

from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from human_feedback_rl.training.imitation_trainer import ImitationTrainer
from human_feedback_rl.training.rl_trainer import RLTrainer
from human_feedback_rl.models.reward_model import RewardModel

from human_feedback_rl.utils.env_setup import build_env_and_expert, build_policy


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
    # RLHF (with pre-trained reward model)
    # -----------------------------

    if cfg.pipeline.rlhf:

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
