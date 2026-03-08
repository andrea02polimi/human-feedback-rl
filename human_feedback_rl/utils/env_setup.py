from sumo_rl_ego.infra.builders.env_factory import build_env
from sumo_rl_ego.infra.builders.model_factory import load_model
from sumo_rl_ego.infra.loaders.config_loader import load_config_from_model

from human_feedback_rl.AgentPolicyNetwork import AgentPolicyNetwork


def build_env_and_expert(model_dir, seed=0):

    cfg = load_config_from_model(model_dir)

    env = build_env(cfg, seed=seed)

    expert_model = load_model(
        env,
        cfg,
        load_path=model_dir,
        seed=seed
    )

    return env, expert_model


def build_policy(env):

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    return AgentPolicyNetwork(obs_dim, n_actions)