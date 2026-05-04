from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.base_class import BaseAlgorithm
from .reward_nets import RewardNet
from .loggers import MainLogger, make_sb3_logger
from .env_wrappers import EnvRewardWrapper, EnvBufferingWrapper, PolicyExplorationWrapper
from .custom_logging_callback import CustomLoggingCallback
from .metrics import log_hacking_signals
from . import types
import numpy as np
from typing import List, Any, Sequence, Optional


class TrajectoryGeneratorFromAgent:
    """Wrapper for training an SB3 agent on an arbitrary reward function."""

    def __init__(
        self,
        agent: BaseAlgorithm,
        reward_model: RewardNet,
        venv: VecEnv,
        rng: np.random.Generator,
        logger: MainLogger,
        exploration_frac: float = 0.0,
        random_prob: float = 0.5,
    ) -> None:

        self.agent        = agent
        self.main_logger  = logger
        self.reward_model = reward_model
        self.exploration_frac = exploration_frac
        self.rng = rng

        # Wire the SB3 agent's internal logger to forward ppo/* metrics to our MainLogger.
        # This intercepts PPO's own logger.dump() calls without subclassing PPO.
        agent.set_logger(make_sb3_logger(logger))

        # BufferingWrapper must come before RewardWrapper so that env (true) rewards
        # are captured, not the learned ones.
        self.buffering_wrapper = EnvBufferingWrapper(venv)

        self.venv = EnvRewardWrapper(
            self.buffering_wrapper,
            reward_model=self.reward_model,
        )

        self.agent.set_env(self.venv)

        algo_venv = self.agent.get_env()
        assert algo_venv is not None
        self.exploration_wrapper = PolicyExplorationWrapper(
            policy=self.agent,
            venv=algo_venv,
            random_prob=random_prob,
            rng=self.rng,
        )

    def train(self, steps: int, **kwargs) -> None:
        if not self.buffering_wrapper.is_empty():
            raise RuntimeError(
                "There are transitions left in the buffer. "
                "Call AgentTrainer.sample() first to clear them.",
            )
        self.agent.learn(
            total_timesteps=steps,
            reset_num_timesteps=False,
            callback=CustomLoggingCallback(main_logger=self.main_logger),
            **kwargs,
        )

    def sample(self, steps: int) -> Sequence[types.Trajectory]:
        agent_trajs = self.buffering_wrapper.pop_finished_trajectories()
        agent_trajs = agent_trajs[::-1]
        avail_steps = sum(len(traj) for traj in agent_trajs)

        exploration_steps = int(self.exploration_frac * steps)
        if self.exploration_frac > 0 and exploration_steps == 0:
            self.main_logger.log(
                f"WARNING: No exploration steps included — exploration_frac="
                f"{self.exploration_frac} but steps={steps} is too small."
            )
        agent_steps = steps - exploration_steps

        if avail_steps < agent_steps:
            self.main_logger.log(
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
            self.main_logger.log(f"Sampling {exploration_steps} exploratory transitions.")
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

        true_rewards  = [traj.total_reward()           for traj in trajectories]
        model_rewards = [self._score_trajectory(traj)  for traj in trajectories]
        lengths       = [len(traj)                     for traj in trajectories]

        self.main_logger.record("env/true_reward_mean", float(np.mean(true_rewards)))
        self.main_logger.record("env/ep_length_mean",   float(np.mean(lengths)))

        #log_ensemble_uncertainty(self.reward_model, trajectories, self.main_logger)
        log_hacking_signals(true_rewards, model_rewards, self.main_logger)

        return trajectories

    def _score_trajectory(self, traj: types.Trajectory) -> float:
        obs  = np.array([t.observation for t in traj])
        acts = np.array([t.action      for t in traj])
        return float(self.reward_model.predict(obs, acts).sum())


def _get_trajectories(
    trajectories: List[types.Trajectory],
    steps: int,
) -> List[types.Trajectory]:
    """Return enough trajectories to cover at least `steps` transitions."""
    if steps == 0:
        return []
    available_steps = sum(len(traj) for traj in trajectories)
    if available_steps < steps:
        raise RuntimeError(
            f"Asked for {steps} transitions but only {available_steps} available",
        )
    steps_cumsum = np.cumsum([len(traj) for traj in trajectories])
    idx = int((steps_cumsum >= steps).argmax())
    trajectories = trajectories[: idx + 1]
    assert sum(len(traj) for traj in trajectories) >= steps
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
    active_envs = np.ones(venv.num_envs, dtype=bool)

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
            active_envs &= ~dones
            if not active_envs.any():
                break