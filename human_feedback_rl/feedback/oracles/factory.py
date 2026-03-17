"""
Factory function for building preference oracles from config.
"""

from omegaconf import DictConfig

from .base import BaseOracle
from .expert import ExpertOracle
from .human import HumanOracle
from human_feedback_rl.utils.env_setup import build_env_and_expert


def build_oracle(config: DictConfig) -> BaseOracle:
    """
    Build the appropriate oracle based on config.preferences.oracle.

    Supported values:
        "env_reward" — ExpertOracle using true environment rewards
        "qnet"       — ExpertOracle using expert DQN Q-values
        "human"      — HumanOracle (terminal prompt + pyglet window)
    """
    oracle = config.preferences.oracle

    label_mode = config.preferences.get("label_mode", "hard")

    if oracle == "env_reward":
        return ExpertOracle(mode="env_reward", label_mode=label_mode)
    elif oracle == "qnet":
        env, expert_model = build_env_and_expert(config)
        env.close()
        return ExpertOracle(mode="qnet", label_mode=label_mode, expert_model=expert_model)
    elif oracle == "human":
        return HumanOracle()
    else:
        raise ValueError(
            f"Unknown oracle {oracle!r}. Use 'env_reward', 'qnet', or 'human'."
        )
