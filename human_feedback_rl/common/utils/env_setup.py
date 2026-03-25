from pathlib import Path

from omegaconf import OmegaConf
import sumo_rl_ego as sre


def _load_policy_cfg(policy_id: str):
    """Load the config.yaml bundled with the registered expert model.

    The config lives at:
        <sumo_rl_ego package>/policies/models/<policy_id>/config.yaml

    Raises FileNotFoundError if policy_id is not registered.
    """
    models_dir = Path(sre.__file__).parent / "policies" / "models"
    cfg_path = models_dir / policy_id / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"No config found for policy_id={policy_id!r}. "
            f"Available models: {sre.list_models()}"
        )
    return OmegaConf.load(cfg_path)


def _env_kwargs(policy_cfg) -> dict:
    """
    Extract environment keyword arguments from a policy config.

    DQN config uses  env.env_args  (e.g. ego=discrete).
    PPO config uses  env.kwargs    (e.g. ego=continuous).
    Returns a plain dict suitable for **kwargs in make_env/make_vec_env.
    """
    if hasattr(policy_cfg.env, "kwargs"):
        return dict(OmegaConf.to_container(policy_cfg.env.kwargs, resolve=True))
    if hasattr(policy_cfg.env, "env_args"):
        return dict(OmegaConf.to_container(policy_cfg.env.env_args, resolve=True))
    return {}


def build_env_and_expert(cfg):
    """
    Create the SB3 VecEnv used for policy training and load the expert model.

    The number of parallel environments and env_id are taken from the expert
    model's own config so that the observation/action spaces match exactly
    what the expert was trained on.

    Returns:
        env          — SB3 SubprocVecEnv
        expert_model — SB3 model (DQN or PPO)
    """
    policy_cfg = _load_policy_cfg(cfg.env.expert_model)
    kwargs     = _env_kwargs(policy_cfg)

    env = sre.make_vec_env(
        policy_cfg.env.id,
        n_envs=policy_cfg.env.n_envs,
        base_seed=cfg.seed,
        **kwargs,
    )

    policy = sre.load_policy(cfg.env.expert_model, env=env)
    return env, policy.model


def build_expert_only(cfg):
    """
    Load the expert model with a minimal 1-env VecEnv (for inference only).

    Use this instead of build_env_and_expert when the env is not needed after
    loading (e.g. preference worker oracle), to avoid spawning n_envs SUMO
    instances that would be closed right away.

    Returns:
        env          — SB3 VecEnv with n_envs=1  (caller must close)
        expert_model — SB3 model (DQN or PPO)
    """
    policy_cfg = _load_policy_cfg(cfg.env.expert_model)
    kwargs     = _env_kwargs(policy_cfg)

    env = sre.make_vec_env(
        policy_cfg.env.id,
        n_envs=1,
        base_seed=cfg.seed,
        **kwargs,
    )

    policy = sre.load_policy(cfg.env.expert_model, env=env)
    return env, policy.model


def build_policy_env(cfg):
    """
    Create the VecEnv for the policy worker using the RLHF config n_envs.

    Unlike build_env_and_expert, this uses cfg.env.n_envs (the RLHF config)
    instead of the expert's training config n_envs, which would be wrong
    (the expert was trained with 16 envs; the RLHF policy should use 4).
    Does NOT load the expert model.

    Returns:
        env — SB3 SubprocVecEnv with n_envs=cfg.env.n_envs
    """
    policy_cfg = _load_policy_cfg(cfg.env.expert_model)
    kwargs     = _env_kwargs(policy_cfg)
    env = sre.make_vec_env(
        policy_cfg.env.id,
        n_envs=cfg.env.n_envs,
        base_seed=cfg.seed,
        **kwargs,
    )
    return env


def build_single_env(cfg):
    """
    Create a single (non-vectorized) gymnasium environment.

    Used for:
      - obs/action space inspection (obs_dim, action_feature_dim) without
        launching multiple SUMO instances.
      - Evaluation runs that require the standard gymnasium API
        (reset → (obs, info), step → (obs, rew, term, trunc, info)).
    """
    policy_cfg = _load_policy_cfg(cfg.env.expert_model)
    kwargs     = _env_kwargs(policy_cfg)
    return sre.make_env(policy_cfg.env.id, seed=cfg.seed, **kwargs)
