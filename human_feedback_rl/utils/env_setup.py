from sumo_rl_ego.infra.builders.env_factory import build_env
from sumo_rl_ego.infra.builders.model_factory import load_model

from human_feedback_rl.agents.policy_network import AgentPolicyNetwork

from pathlib import Path
from omegaconf import OmegaConf, DictConfig

def load_config() -> DictConfig:
    root = Path(__file__).resolve().parents[3]
    cfg_env: DictConfig = OmegaConf.load(root / "sumo-rl-ego/experiments/configs/env/highway_fast.yaml")
    cfg_algo: DictConfig = OmegaConf.load(root / "sumo-rl-ego/experiments/configs/rl/dqn.yaml")
    return cfg_env, cfg_algo


def build_env_and_expert(model_dir, seed=0):

    cfg_env, cfg_algo = load_config()

    env = build_env(cfg_env, seed=seed)

    expert_model = load_model(
        env,
        cfg_algo,
        load_path=model_dir,
        seed=seed
    )

    return env, expert_model


def build_policy(env):

    obs_dim = env.observation_space.shape[-1]
    n_actions = env.action_space.n

    return AgentPolicyNetwork(obs_dim, n_actions)