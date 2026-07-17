from stable_baselines3.common.vec_env import VecEnv, VecMonitor
from stable_baselines3.common.base_class import BaseAlgorithm
from gymnasium import spaces
from .reward_nets import RewardNet
from .loggers import NullLogger
from .env_wrappers import EnvRewardWrapper, EnvBufferingWrapper, PolicyExplorationWrapper
from . import types
import numpy as np
from typing import List, Any, Sequence, Optional
from .custom_logging_callback import CustomLoggingCallback, FixedIntervalDumpCallback
import torch as th


class TrajectoryGeneratorFromAgent:
    """Wrapper for training an SB3 agent on an arbitrary reward function.

    ``dump_timestep_interval``: when set, the agent's logs are dumped every N
    environment timesteps instead of SB3's native cadence (episodes for
    off-policy algorithms). This aligns the log x-values across seeds, which
    W&B grouped panels need to draw min/max bands on a custom x-axis.
    """

    def __init__(
        self,
        agent: BaseAlgorithm,
        reward_model: RewardNet,
        venv: VecEnv,
        exploration_eps: float = 0.5,
        logger=None,
        rng: np.random.Generator = None,
        sampling_venv: Optional[VecEnv] = None,
        dump_timestep_interval: Optional[int] = None,
    ) -> None:

        self.logger = logger if logger is not None else NullLogger()
        self.agent        = agent
        self.dump_timestep_interval = dump_timestep_interval

        self.rng = rng if rng is not None else np.random.default_rng()
        self.reward_model    = reward_model

        # The BufferingWrapper records all trajectories, so we can return
        # them after training. This should come first (before the wrapper that
        # changes the reward function), so that we return the original environment
        # rewards.
        # When applying BufferingWrapper and RewardVecEnvWrapper, we should use `venv`
        # instead of `agent.get_env()` because SB3 may apply some wrappers to
        # `agent`'s env under the hood. In particular, in image-based environments,
        # SB3 may move the image-channel dimension in the observation space, making
        # `agent.get_env()` not match with `reward_fn`.

        self._shared_sampling_env = sampling_venv is None
        if self._shared_sampling_env:
            self.buffering_wrapper = EnvBufferingWrapper(venv)
            training_env = self.buffering_wrapper
        else:
            self.buffering_wrapper = EnvBufferingWrapper(sampling_venv)
            training_env = venv

        self.reward_wrapper = EnvRewardWrapper(training_env, reward_model=self.reward_model)
        self.venv = VecMonitor(self.reward_wrapper)
        self.sampling_venv = (
            self.venv if self._shared_sampling_env else VecMonitor(self.buffering_wrapper)
        )

        self.agent.set_env(self.venv)

        algo_venv = self.agent.get_env()
        assert algo_venv is not None
        self.exploration_wrapper = PolicyExplorationWrapper(
            venv=self.sampling_venv,
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

        callbacks = [CustomLoggingCallback()]
        if self.dump_timestep_interval is not None:
            # Fixed-grid dumps replace SB3's native cadence entirely:
            # log_interval=None disables the episode/rollout-based dump.
            callbacks.append(FixedIntervalDumpCallback(self.dump_timestep_interval))
            log_interval = None

        self.agent.learn(
            total_timesteps=steps,
            reset_num_timesteps=False,
            callback=callbacks,
            log_interval=log_interval,
            **kwargs,
        )


    def sample(self, agent_steps, exploration_steps = 0) -> Sequence[types.Trajectory]:
        """Collect at least ``agent_steps`` transitions and log rollout metrics."""
        agent_trajs  = self.buffering_wrapper.pop_finished_trajectories()
        avail_steps  = sum(len(traj) for traj in agent_trajs)

        if avail_steps < agent_steps:
            self.logger.log(
                f"-- Requested {agent_steps} transitions but only {avail_steps} in buffer."
                f" Sampling {agent_steps - avail_steps} additional transitions.",
            )
            final_obs, final_dones = rollout_agent(
                policy=self.agent,
                venv=self.sampling_venv,
                steps=agent_steps - avail_steps,
                deterministic_policy=False,
            )
            if self._shared_sampling_env:
                self._sync_agent_observation(final_obs, final_dones)
            additional_trajs = self.buffering_wrapper.pop_finished_trajectories()
            agent_trajs = list(agent_trajs) + list(additional_trajs)

        agent_trajs  = _get_trajectories(agent_trajs, agent_steps)
        trajectories = list(agent_trajs)

        if exploration_steps > 0:
            final_obs, final_dones = rollout_agent(
                policy=self.exploration_wrapper,
                venv=self.sampling_venv,
                steps=exploration_steps,
                deterministic_policy=False,
            )
            if self._shared_sampling_env:
                self._sync_agent_observation(final_obs, final_dones)
            exploration_trajs = self.buffering_wrapper.pop_finished_trajectories()
            exploration_trajs = _get_trajectories(exploration_trajs, exploration_steps)
            trajectories.extend(exploration_trajs)

        return trajectories

    def _sync_agent_observation(self, obs: np.ndarray, dones: np.ndarray) -> None:
        """Keep SB3 state coherent when sampling uses its training environment."""
        self.agent._last_obs = obs
        self.agent._last_episode_starts = dones
        if self.agent._vec_normalize_env is not None:
            self.agent._last_original_obs = self.agent._vec_normalize_env.get_original_obs()


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
) -> tuple[np.ndarray, np.ndarray]:
    obs = venv.reset()
    state: Optional[np.ndarray] = None
    episode_starts = np.ones(venv.num_envs, dtype=bool)

    collected_steps = 0
    finishing   = False
    active_envs = np.ones(venv.num_envs, dtype=bool)
    buffering_wrapper = _find_buffering_wrapper(venv)

    while True:
        if buffering_wrapper is not None:
            recording_mask = active_envs if finishing else np.ones(venv.num_envs, dtype=bool)
            buffering_wrapper.set_recording_mask(recording_mask)

        actions, state = policy.predict(
            obs,
            state=state,
            episode_start=episode_starts,
            deterministic=deterministic_policy,
        )

        if buffering_wrapper is not None:
            buffering_wrapper.set_log_probs(policy_action_log_probs(policy, obs, actions))

        obs, rewards, dones, infos = venv.step(actions)

        episode_starts   = dones
        collected_steps += venv.num_envs

        if collected_steps >= steps:
            finishing = True

        if finishing:
            active_envs &= ~dones
            if not active_envs.any():
                break

    if buffering_wrapper is not None:
        buffering_wrapper.set_recording_mask(np.ones(venv.num_envs, dtype=bool))
    return obs, dones


def _find_buffering_wrapper(venv: VecEnv) -> Optional[EnvBufferingWrapper]:
    current = venv
    while current is not None:
        if isinstance(current, EnvBufferingWrapper):
            return current
        current = getattr(current, "venv", None)
    return None


def policy_action_log_probs(policy: Any, obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Evaluate actions under an SB3 PPO/SAC policy or exploration mixture."""
    if hasattr(policy, "action_log_prob"):
        return policy.action_log_prob(obs, actions)

    sb3_policy = policy.policy
    obs_tensor, _ = sb3_policy.obs_to_tensor(obs)

    with th.no_grad():
        if hasattr(sb3_policy, "evaluate_actions"): # PPO
            action_tensor = th.as_tensor(actions, device=sb3_policy.device)
            if isinstance(policy.action_space, spaces.Discrete):
                action_tensor = action_tensor.long().flatten()
                _, log_prob, _ = sb3_policy.evaluate_actions(obs_tensor, action_tensor)
            else:
                action_tensor = action_tensor.float()
                distribution = sb3_policy.get_distribution(obs_tensor)
                normal = getattr(distribution, "distribution", None)
                if isinstance(normal, th.distributions.Normal):
                    low = th.as_tensor(
                        policy.action_space.low, dtype=th.float32, device=sb3_policy.device
                    )
                    high = th.as_tensor(
                        policy.action_space.high, dtype=th.float32, device=sb3_policy.device
                    )
                    component_log_prob = normal.log_prob(action_tensor)
                    low_log_mass = th.special.log_ndtr((low - normal.loc) / normal.scale)
                    high_log_mass = th.special.log_ndtr((normal.loc - high) / normal.scale)
                    component_log_prob = th.where(
                        action_tensor <= low,
                        low_log_mass,
                        th.where(action_tensor >= high, high_log_mass, component_log_prob),
                    )
                    log_prob = component_log_prob.sum(dim=1)
                else:
                    _, log_prob, _ = sb3_policy.evaluate_actions(obs_tensor, action_tensor)
        elif hasattr(policy, "actor"): # SAC
            # SAC samples from a Gaussian and squashes through tanh, so its
            # action density lives in normalized [-1, 1] coordinates, not in the
            # environment's action space. Three steps recover the environment
            # log-density:
            #   1. Scale the environment action into [-1, 1] (the actor's
            #      output coordinates) before evaluating the log-probability.
            #   2. Evaluate the squashed distribution's log-prob there — SB3's
            #      SquashedDiagGaussian already includes the tanh Jacobian.
            #   3. Subtract log(action_scale) per dimension: the Jacobian of
            #      the affine map from normalized to environment coordinates.
            scaled_actions = sb3_policy.scale_action(np.asarray(actions))
            action_tensor = th.as_tensor(scaled_actions, dtype=th.float32, device=sb3_policy.device)
            mean, log_std, kwargs = policy.actor.get_action_dist_params(obs_tensor)
            distribution = policy.actor.action_dist.proba_distribution(mean, log_std, **kwargs)
            log_prob = distribution.log_prob(action_tensor)
            action_scale = (policy.action_space.high - policy.action_space.low) / 2.0
            log_prob = log_prob - float(np.log(action_scale).sum())
        else:
            raise TypeError(f"Unsupported policy type for log-prob evaluation: {type(policy).__name__}")

    return log_prob.detach().cpu().numpy().reshape(-1)
