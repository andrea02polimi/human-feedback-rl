"""Checkpoint persistence for demonstration-based training."""

import os

import torch as th


class CheckpointingMixin:
    """Checkpoint methods used by ``DemoAlgorithm``."""

    def _save_checkpoint(self, checkpoint_dir: str, iteration: int) -> None:
        ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{iteration:04d}")
        os.makedirs(ckpt_path, exist_ok=True)
        th.save(self.reward_model.state_dict(), os.path.join(ckpt_path, "reward_model.pt"))
        th.save(
            {
                "iteration": iteration,
                "loss_type": self.loss_type,
                "temperature": self.temperature,
                "relabel_rewards": self.relabel_rewards,
                "normalize_agent_reward": self.normalize_agent_reward,
                "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
            },
            os.path.join(ckpt_path, "reward_training.pt"),
        )
        self.trajectory_generator.agent.save(os.path.join(ckpt_path, "agent"))
        if hasattr(self.trajectory_generator.agent, "save_replay_buffer"):
            self.trajectory_generator.agent.save_replay_buffer(
                os.path.join(ckpt_path, "replay_buffer.pkl")
            )
        print(f"  checkpoint saved in {ckpt_path}")
