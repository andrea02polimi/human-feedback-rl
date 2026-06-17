from stable_baselines3.common.vec_env import VecEnv, VecMonitor
from stable_baselines3.common.base_class import BaseAlgorithm
from .reward_nets import RewardNet
from .loggers import NullLogger
from .env_wrappers import EnvRewardWrapper, EnvBufferingWrapper, PolicyExplorationWrapper
from . import types
import math
import numpy as np
import time
from typing import List, Any, Sequence, Optional
from .custom_logging_callback import CustomLoggingCallback
import torch as th


class TrajectoryGeneratorFromAgent:
    """Wrapper for training an SB3 agent on an arbitrary reward function."""

    def __init__(
        self,
        agent: BaseAlgorithm,
        reward_model: RewardNet,
        venv: VecEnv,
        exploration_frac: float = 0.0,
        exploration_eps: float = 0.5,
        logger=None,
        rng: np.random.Generator = None,
    ) -> None:

        self.logger = logger if logger is not None else NullLogger()
        self.agent        = agent

        self.rng = rng if rng is not None else np.random.default_rng()
        self.reward_model    = reward_model
        self.exploration_frac = exploration_frac

        # The BufferingWrapper records all trajectories, so we can return
        # them after training. This should come first (before the wrapper that
        # changes the reward function), so that we return the original environment
        # rewards.
        # When applying BufferingWrapper and RewardVecEnvWrapper, we should use `venv`
        # instead of `agent.get_env()` because SB3 may apply some wrappers to
        # `agent`'s env under the hood. In particular, in image-based environments,
        # SB3 may move the image-channel dimension in the observation space, making
        # `agent.get_env()` not match with `reward_fn`.

        self.buffering_wrapper = EnvBufferingWrapper(venv)

        self.venv = VecMonitor(EnvRewardWrapper(
            self.buffering_wrapper,
            reward_model=self.reward_model,
        ))

        self.agent.set_env(self.venv)

        algo_venv = self.agent.get_env()
        assert algo_venv is not None
        self.exploration_wrapper = PolicyExplorationWrapper(
            venv=algo_venv,
            policy=self.agent,
            exploration_eps=exploration_eps,
            rng=self.rng,
        )

    def train(self, steps: int, log_interval: int, **kwargs) -> None:
        """Train the agent for ``steps`` environment timesteps and log metrics."""
        if not self.buffering_wrapper.is_empty():
            raise RuntimeError(
                "There are transitions left in the buffer. "
                "Call AgentTrainer.sample() first to clear them.",
            )

        self.agent.learn(
            total_timesteps=steps,
            reset_num_timesteps=False,
            callback=CustomLoggingCallback(),
            log_interval=log_interval,
            **kwargs,
        )


    def sample(self, agent_steps, exploration_steps = 0) -> Sequence[types.Trajectory]:
        """Collect at least ``steps`` transitions and log rollout metrics."""
        t0 = time.perf_counter()
        agent_trajs  = self.buffering_wrapper.pop_finished_trajectories()
        avail_steps  = sum(len(traj) for traj in agent_trajs)

        if avail_steps < agent_steps:
            self.logger.log(
                f"-- Requested {agent_steps} transitions but only {avail_steps} in buffer."
                f" Sampling {agent_steps - avail_steps} additional transitions.",
            )
            algo_venv = self.agent.get_env()
            assert algo_venv is not None
            rollout_agent(
                policy=self.agent,
                venv=algo_venv,
                steps=agent_steps - avail_steps,
                deterministic_policy=False,
            )
            additional_trajs = self.buffering_wrapper.pop_finished_trajectories()
            agent_trajs = list(agent_trajs) + list(additional_trajs)

        agent_trajs  = _get_trajectories(agent_trajs, agent_steps)
        trajectories = list(agent_trajs)

        if exploration_steps > 0:
            algo_venv = self.agent.get_env()
            assert algo_venv is not None
            rollout_agent(
                policy=self.exploration_wrapper,
                venv=algo_venv,
                steps=exploration_steps,
                deterministic_policy=False,
            )
            exploration_trajs = self.buffering_wrapper.pop_finished_trajectories()
            exploration_trajs = _get_trajectories(exploration_trajs, exploration_steps)
            trajectories.extend(exploration_trajs)

        return trajectories


def _get_trajectories(
    trajectories: List[types.Trajectory],
    steps: int,
) -> List[types.Trajectory]:
    """Return enough trajectories to cover at least ``steps`` transitions."""
    if steps == 0:
        return []
    available_steps = sum(len(traj) for traj in trajectories)
    if available_steps < steps:
        raise RuntimeError(
            f"Asked for {steps} transitions but only {available_steps} available",
        )
    return trajectories


def rollout_agent(
    policy: Any,
    venv: VecEnv,
    steps: int,
    deterministic_policy: bool = False,
) -> None:
    obs = venv.reset()
    state: Optional[np.ndarray] = None
    episode_starts = np.ones(venv.num_envs, dtype=bool)

    collected_steps = 0
    finishing   = False
    active_envs = np.ones(venv.num_envs, dtype=bool)

    while True:
        actions, state = policy.predict(
            obs,
            state=state,
            episode_start=episode_starts,
            deterministic=deterministic_policy,
        )

        obs, rewards, dones, infos = venv.step(actions)

        episode_starts   = dones
        collected_steps += venv.num_envs

        if collected_steps >= steps:
            finishing = True

        if finishing:
            active_envs &= ~dones
            if not active_envs.any():
                break
