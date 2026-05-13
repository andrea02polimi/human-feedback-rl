from stable_baselines3.common.vec_env import VecEnv, VecMonitor
from stable_baselines3.common.base_class import BaseAlgorithm
from .reward_nets import RewardNet
from .loggers import PrefixedLogger, NullLogger
from .env_wrappers import EnvRewardWrapper, EnvBufferingWrapper, PolicyExplorationWrapper
from . import types
import numpy as np
import time
from typing import List, Tuple, Any, Dict, Sequence, Optional
from .custom_logging_callback import CustomLoggingCallback
import torch as th
from dataclasses import dataclass


@dataclass
class RolloutMetrics:
    true_rewards: List[float]
    model_rewards: List[float]
    lengths: List[int]
    mean_true_reward: float
    mean_model_reward: float
    mean_length: float
    time_sample: float


@dataclass
class TrainMetrics:
    total_timesteps: int
    current_timesteps: int
    time_train: float


class TrajectoryGeneratorFromAgent:
    """Wrapper for training an SB3 agent on an arbitrary reward function."""

    def __init__(
        self,
        agent: BaseAlgorithm,
        reward_model: RewardNet,
        venv: VecEnv,
        exploration_frac: float = 0.0,
        random_prob: float = 0.5,
        logger: PrefixedLogger = None,
        rng: np.random.Generator = None,
    ) -> None:

        self.logger = logger if logger is not None else NullLogger()
        self.agent = agent
        
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reward_model = reward_model
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
        
        # Unlike with BufferingWrapper, we should use `algorithm.get_env()` instead
        # of `venv` when interacting with `algorithm`.
        algo_venv = self.agent.get_env()
        assert algo_venv is not None
        # This wrapper will be used to ensure that rollouts are collected from a mixture
        # of `self.algorithm` and a policy that acts randomly. The samples from
        # `self.algorithm` are themselves stochastic if `self.algorithm` is stochastic.
        # Otherwise, they are deterministic, and action selection is only stochastic
        # when sampling from the random policy.
        self.exploration_wrapper = PolicyExplorationWrapper(
            policy=self.agent,
            venv=algo_venv,
            random_prob=random_prob,
            rng=self.rng,
        )

    def train(self, steps: int, log_interval: int, **kwargs) -> None:
        """Train the agent using the reward function specified during instantiation.

        Args:
            steps: number of environment timesteps to train for
            **kwargs: other keyword arguments to pass to BaseAlgorithm.train()

        Raises:
            RuntimeError: Transitions left in `self.buffering_wrapper`; call
                `self.sample` first to clear them.
        """
        
        if not self.buffering_wrapper.is_empty():
            raise RuntimeError(
                f"There are transitions left in the buffer. "
                "Call AgentTrainer.sample() first to clear them.",
            )

        num_timesteps_before = self.agent.num_timesteps

        t0 = time.perf_counter()
        self.agent.learn(
            total_timesteps=steps,
            reset_num_timesteps=False,
            callback=CustomLoggingCallback(),
            **kwargs,
        )

        return TrainMetrics(
            total_timesteps=self.agent.num_timesteps,
            current_timesteps=self.agent.num_timesteps-num_timesteps_before,
            time_train=time.perf_counter() - t0,
        )


    def sample(self, steps: int) -> Sequence[types.Trajectory]:
        t0 = time.perf_counter()
        agent_trajs = self.buffering_wrapper.pop_finished_trajectories()
        avail_steps = sum(len(traj) for traj in agent_trajs)

        exploration_steps = int(self.exploration_frac * steps)
        if self.exploration_frac > 0 and exploration_steps == 0:
            self.logger.warn(
                "No exploration steps included: exploration_frac = "
                f"{self.exploration_frac} > 0 but steps={steps} is too small.",
            )
        agent_steps = steps - exploration_steps

        if avail_steps < agent_steps:
            self.logger.log(
                f"Requested {agent_steps} transitions but only {avail_steps} in buffer."
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
        
        agent_trajs = _get_trajectories(agent_trajs, agent_steps)

        trajectories = list(agent_trajs)

        if exploration_steps > 0:
            self.logger.log(f"Sampling {exploration_steps} exploratory transitions.")
            
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
            # We call _get_trajectories separately on agent_trajs and exploration_trajs
            # and then just concatenate. This could mean we return slightly too many
            # transitions, but it gets the proportion of exploratory and agent
            # transitions roughly right.
            trajectories.extend(exploration_trajs)

        # logs
        true_rewards  = [traj.total_reward()  for traj in trajectories]   # se rews = env reward
        model_rewards = [self._score_trajectory(traj) for traj in trajectories]
        lengths       = [len(traj) for traj in trajectories]

        logs_metrics = RolloutMetrics(
            true_rewards=true_rewards,
            model_rewards=model_rewards,
            lengths=lengths,
            mean_true_reward=float(np.mean(true_rewards)),
            mean_model_reward=float(np.mean(model_rewards)),
            mean_length=float(np.mean(lengths)),
            time_sample=time.perf_counter() - t0,
        )

        return trajectories, logs_metrics


    def _score_trajectory(self, traj: types.Trajectory) -> float:
        obs         = np.array([t.observation for t in traj])
        acts        = np.array([t.action for t in traj])
        next_status = np.array([t.next_status for t in traj])
        done        = np.array([float(t.done) for t in traj])
        return self.reward_model.predict(obs, acts, next_status, done).sum()



def _get_trajectories(
    trajectories: List[types.Trajectory],
    steps: int,
    ) -> List[types.Trajectory]:
    
    """Get enough trajectories to have at least `steps` transitions in total."""
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
    finishing = False
    active_envs = np.ones(venv.num_envs, dtype=bool)  # quali env devono ancora finire

    while True:
        actions, state = policy.predict(
            obs,
            state=state,
            episode_start=episode_starts,
            deterministic=deterministic_policy,
        )

        obs, rewards, dones, infos = venv.step(actions)

        episode_starts = dones
        collected_steps += venv.num_envs

        if collected_steps >= steps:
            finishing = True

        if finishing:
            active_envs &= ~dones  # gli env che hanno fatto done non sono più attivi
            if not active_envs.any():
                break  # tutti gli env hanno finito il loro episodio