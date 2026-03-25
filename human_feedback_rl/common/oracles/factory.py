"""
Factory function for building preference oracles from config.
"""

from omegaconf import DictConfig

from .base import BaseOracle
from .expert import ExpertOracle
from .human import HumanOracle
from .logprob import LogProbOracle
from .ppo_logprob import PPOLogProbOracle
from human_feedback_rl.common.utils.env_setup import build_expert_only


def build_oracle(config: DictConfig) -> BaseOracle:
    """
    Build the appropriate oracle based on config.preferences.oracle.

    Supported values:
        "env_reward"    — ExpertOracle using true environment rewards
        "qnet"          — ExpertOracle using expert DQN Q-values
        "q_action"      — ExpertOracle using DQN Q-value at agent action
        "log_prob"      — LogProbOracle: Boltzmann over DQN Q-values
        "ppo_log_prob"  — PPOLogProbOracle: log π_exp(a_agent | s) via evaluate_actions
        "human"         — HumanOracle (terminal prompt + pyglet window)
    """
    oracle = config.preferences.oracle

    label_mode  = config.preferences.get("label_mode", "hard")
    temperature = config.preferences.get("oracle_temperature", 1.0)

    if oracle == "env_reward":
        return ExpertOracle(mode="env_reward", label_mode=label_mode)
    elif oracle == "qnet":
        env, expert_model = build_expert_only(config)
        env.close()
        return ExpertOracle(mode="qnet", label_mode=label_mode, expert_model=expert_model)
    elif oracle == "q_action":
        env, expert_model = build_expert_only(config)
        env.close()
        return ExpertOracle(mode="q_action", label_mode=label_mode, expert_model=expert_model)
    elif oracle == "log_prob":
        env, expert_model = build_expert_only(config)
        env.close()
        return LogProbOracle(label_mode=label_mode, expert_model=expert_model, temperature=temperature)
    elif oracle == "ppo_log_prob":
        env, expert_model = build_expert_only(config)
        env.close()
        return PPOLogProbOracle(label_mode=label_mode, expert_model=expert_model, temperature=temperature)
    elif oracle == "human":
        return HumanOracle()
    else:
        raise ValueError(
            f"Unknown oracle {oracle!r}. "
            f"Use 'env_reward', 'qnet', 'q_action', 'log_prob', 'ppo_log_prob', or 'human'."
        )
