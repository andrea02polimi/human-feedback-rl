from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.base_class import BaseAlgorithm
from .reward_nets import RewardNet
from .loggers import MainLogger
from .env_wrappers import EnvRewardWrapper, EnvBufferingWrapper, EnvExplorationWrapper
from . import types
import numpy as np
from typing import List, Tuple, Any, Dict, Sequence, Optional

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
        switch_prob: float = 0.5,
        random_prob: float = 0.5,
    ) -> None:
        
        self.agent = agent
        self.logger = logger
        
        self.reward_model = reward_model
        self.exploration_frac = exploration_frac
        self.rng = rng

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

        self.venv = EnvRewardWrapper(
            self.buffering_wrapper,
            reward_model=self.reward_model,
        )

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
        self.exploration_wrapper = EnvExplorationWrapper(
            policy=self.agent,
            venv=algo_venv,
            random_prob=random_prob,
            switch_prob=switch_prob,
            rng=self.rng,
        )

    def train(self, steps: int, **kwargs) -> None:
        """Train the agent using the reward function specified during instantiation.

        Args:
            steps: number of environment timesteps to train for
            **kwargs: other keyword arguments to pass to BaseAlgorithm.train()

        Raises:
            RuntimeError: Transitions left in `self.buffering_wrapper`; call
                `self.sample` first to clear them.
        """
        n_transitions = self.buffering_wrapper.n_transitions
        if n_transitions:
            raise RuntimeError(
                f"There are {n_transitions} transitions left in the buffer. "
                "Call AgentTrainer.sample() first to clear them.",
            )
        self.agent.learn(
            total_timesteps=steps,
            reset_num_timesteps=False,
            callback=self.log_callback,
            **kwargs,
        )

    def sample(self, steps: int) -> Sequence[types.Trajectory]:
        agent_trajs, _ = self.buffering_wrapper.pop_finished_trajectories()
        # We typically have more trajectories than are needed.
        # In that case, we use the final trajectories because
        # they are the ones with the most relevant version of
        # the agent.
        # The easiest way to do this will be to first invert the
        # list and then later just take the first trajectories:
        agent_trajs = agent_trajs[::-1]
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
            additional_trajs, _ = self.buffering_wrapper.pop_finished_trajectories()
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
            exploration_trajs, _ = self.buffering_wrapper.pop_finished_trajectories()
            exploration_trajs = _get_trajectories(exploration_trajs, exploration_steps)
            # We call _get_trajectories separately on agent_trajs and exploration_trajs
            # and then just concatenate. This could mean we return slightly too many
            # transitions, but it gets the proportion of exploratory and agent
            # transitions roughly right.
            trajectories.extend(exploration_trajs)

        return trajectories




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
    # We need the cumulative sum of trajectory lengths
    # to determine how many trajectories to return:
    steps_cumsum = np.cumsum([len(traj) for traj in trajectories])
    # Now we find the first index that gives us enough
    # total steps:
    idx = int((steps_cumsum >= steps).argmax())
    # we need to include the element at position idx
    trajectories = trajectories[: idx + 1]
    # sanity check
    assert sum(len(traj) for traj in trajectories) >= steps
    return trajectories


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