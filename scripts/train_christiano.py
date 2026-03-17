"""
Entry point for Christiano et al. (2017) RLHF training.

Usage:
    python scripts/train_christiano.py
    python scripts/train_christiano.py algorithm=christiano_qnet
    python scripts/train_christiano.py preferences.oracle=qnet
"""
import multiprocessing as mp

import hydra
from omegaconf import DictConfig

from human_feedback_rl.algorithms.christiano.trainer import ChristianoTrainer


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    mp.set_start_method("spawn", force=True) # Questo impone che i processi figli vengano creati con spawn.
    ChristianoTrainer().train(cfg)


if __name__ == "__main__":
    main()
