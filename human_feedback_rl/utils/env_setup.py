
from pathlib import Path
from omegaconf import OmegaConf

from human_feedback_rl.agents.policy_network import AgentPolicyNetwork
import sumo_rl_ego as sre


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def build_env_and_expert(cfg):
    """
    Build SUMO environment and load expert model using the same
    logic used in the main training script.
    """

    model_path = PROJECT_ROOT / cfg.env.expert_model / "model.zip"

    run_cfg_path = PROJECT_ROOT / cfg.env.expert_model / ".hydra" / "config.yaml"

    run_cfg = OmegaConf.load(run_cfg_path)

    # build vectorized env exactly like in sre training
    env = sre.make_vec_env(
        run_cfg.env,
        n_envs=run_cfg.resources.n_envs,
        base_seed=cfg.seed,
    )

    # load expert model
    expert_model = sre.load_model(
        env,
        cfg = run_cfg,
        load_path=model_path,
        seed=run_cfg.seed,
        device=run_cfg.resources.device,
    )

    return env, expert_model


def build_policy(env):

    obs_dim = env.observation_space.shape[-1]
    n_actions = env.action_space.n

    return AgentPolicyNetwork(obs_dim, n_actions)