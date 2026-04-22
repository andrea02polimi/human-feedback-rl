from typing import Any, Callable, Optional
import numpy as np
from stable_baselines3.common.vec_env import VecEnv


def rollout_agent(
    policy: Any,
    venv: VecEnv,
    steps: int,
    deterministic_policy: bool = False,
) -> None:
    """
    Fa interagire l'agente con il VecEnv per un certo numero di step.

    Si assume che `venv` sia wrappato in modo tale da salvare automaticamente
    transizioni / traiettorie in un buffer interno.

    Args:
        policy: policy o modello SB3 con metodo `predict(obs, deterministic=...)`.
        venv: ambiente vettorializzato wrappato con un buffer.
        steps: numero totale di step da simulare.
        deterministic_policy: se True usa azioni deterministiche.
    """
    obs = venv.reset()
    state: Optional[np.ndarray] = None
    episode_starts = np.ones(venv.num_envs, dtype=bool)

    collected_steps = 0

    while collected_steps < steps:
        actions, state = policy.predict(
            obs,
            state=state,
            episode_start=episode_starts,
            deterministic=deterministic_policy,
        )

        obs, rewards, dones, infos = venv.step(actions)

        episode_starts = dones
        collected_steps += venv.num_envs