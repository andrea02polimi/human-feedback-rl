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

    if oracle == "env_reward":
        return ExpertOracle(mode="env_reward")
    elif oracle == "qnet":
        env, expert_model = build_env_and_expert(config)
        env.close()
        return ExpertOracle(mode="qnet", expert_model=expert_model)
    elif oracle == "human":
        return HumanOracle()
    else:
        raise ValueError(
            f"Unknown oracle {oracle!r}. Use 'env_reward', 'qnet', or 'human'."
        )
