from pathlib import Path
from omegaconf import OmegaConf
from stable_baselines3 import DQN
import sumo_rl_ego as sre


PROJECT_ROOT = Path(__file__).resolve().parents[4]


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
        run_cfg.env.id,
        n_envs=run_cfg.env.n_envs,
        base_seed=cfg.seed,
    )

    expert_model = DQN.load(str(model_path), env=env, device=run_cfg.algo.algo_kwargs.device)

    return env, expert_model


def build_demo_env_and_expert(cfg):
    """
    Create a 1-env VecEnv + expert model for the demonstration worker.

    Uses a different seed from the policy worker to avoid duplicate episodes.
    A single env is sufficient because the demo worker processes one agent
    segment at a time (no need for parallelism).

    Returns:
        env          — SB3 VecEnv with n_envs=1
        expert_model — SB3 DQN with a .q_net attribute
    """
    run_cfg = _load_expert_run_cfg(cfg)
    model_path = PROJECT_ROOT / cfg.env.expert_model / "model.zip"

    env = sre.make_vec_env(
        run_cfg.env.id,
        n_envs=1,
        base_seed=cfg.seed + 1000,
    )

    expert_model = DQN.load(str(model_path), env=env, device=run_cfg.algo.algo_kwargs.device)

    return env, expert_model


def build_expert_only(cfg):
    """
    Load the expert DQN using a minimal 1-env VecEnv (opened and immediately
    usable for inference only — the env is returned so the caller can close it).

    Use this instead of build_env_and_expert when the env is not needed after
    loading (e.g. preference worker oracle), to avoid spawning n_envs SUMO
    instances that would be closed right away.

    Returns:
        env          — SB3 VecEnv with n_envs=1  (caller must close)
        expert_model — SB3 DQN with a .q_net attribute
    """
    run_cfg = _load_expert_run_cfg(cfg)
    model_path = PROJECT_ROOT / cfg.env.expert_model / "model.zip"

    env = sre.make_vec_env(
        run_cfg.env.id,
        n_envs=1,
        base_seed=cfg.seed,
    )

    expert_model = DQN.load(str(model_path), env=env, device=run_cfg.algo.algo_kwargs.device)

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
    return sre.make_env(run_cfg.env.id, seed=cfg.seed)
