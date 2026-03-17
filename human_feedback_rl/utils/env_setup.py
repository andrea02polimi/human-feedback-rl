from pathlib import Path
from omegaconf import OmegaConf
import sumo_rl_ego as sre


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_expert_run_cfg(cfg):
    """Load the .hydra/config.yaml saved alongside the expert model checkpoint."""
    path = PROJECT_ROOT / cfg.env.expert_model / ".hydra" / "config.yaml"
    return OmegaConf.load(path)


def build_env_and_expert(cfg):
    """
    Create the SB3 VecEnv used for policy training and load the expert DQN.

    The number of parallel environments and device are taken from the expert
    model's own training config so that the observation/action spaces match
    exactly what the expert was trained on.

    Returns:
        env          — SB3 SubprocVecEnv
        expert_model — SB3 DQN with a .q_net attribute
    """
    run_cfg = _load_expert_run_cfg(cfg)
    model_path = PROJECT_ROOT / cfg.env.expert_model / "model.zip"

    env = sre.make_vec_env(
        run_cfg.env,
        n_envs=run_cfg.resources.n_envs,
        base_seed=cfg.seed,
    )

    expert_model = sre.load_model(
        env,
        cfg=run_cfg.algo,
        load_path=model_path,
        seed=run_cfg.seed,
        device=run_cfg.resources.device,
    )

    return env, expert_model


def build_single_env(cfg):
    """
    Create a single (non-vectorized) gymnasium environment.

    Used for:
      - obs/action space inspection (obs_dim, n_actions) without launching
        multiple SUMO instances.
      - Evaluation runs that require the standard gymnasium API
        (reset → (obs, info), step → (obs, rew, term, trunc, info)).
    """
    run_cfg = _load_expert_run_cfg(cfg)
    return sre.make_env(run_cfg.env, seed=cfg.seed)
