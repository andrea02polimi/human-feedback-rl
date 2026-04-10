"""Two-wrapper trajectory sampling, mirroring imitation's AgentTrainer pattern.

Architecture (innermost → outermost)::

    raw_env
      ↓
    BufferingWrapper   ← records (obs, action, TRUE reward, info) per step
      ↓
    EnvRewardWrapper   ← replaces rewards with normalised model rewards
      ↓
    agent (PPO / SAC …)

Trajectories returned by :meth:`AgentTrainer.sample` carry true environment
rewards and are used for preference comparison.  The agent trains on model
rewards via :meth:`AgentTrainer.train`.
"""
import math
from typing import List, Optional

import numpy as np
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper

from .core import Trajectory, Transition
from .custom_logging_callback import CustomLoggingCallback
from .env_reward_wrapper import EnvRewardWrapper
from .reward_model import EnsembleRewardModel


# ---------------------------------------------------------------------------
# BufferingWrapper
# ---------------------------------------------------------------------------

class BufferingWrapper(VecEnvWrapper):
    """VecEnv wrapper that records completed trajectories with true env rewards.

    Must be the innermost wrapper (placed directly around the raw environment,
    before :class:`EnvRewardWrapper`) so that the rewards it captures are the
    genuine environment rewards, not learned-model rewards.

    Equivalent to ``imitation.data.wrappers.BufferingWrapper``.
    """

    def __init__(self, venv: VecEnv) -> None:
        super().__init__(venv)
        self._current_transitions: List[List[Transition]] = [[] for _ in range(venv.num_envs)]
        self._finished_trajectories: List[Trajectory] = []
        self._current_obs: Optional[np.ndarray] = None
        self._current_actions: Optional[np.ndarray] = None

    @property
    def n_transitions(self) -> int:
        """Total transitions currently held in incomplete-episode buffers."""
        return sum(len(buf) for buf in self._current_transitions)

    def reset(self):
        obs = self.venv.reset()
        if isinstance(obs, tuple):      # gymnasium-style (obs, info)
            obs, _ = obs
        self._current_obs = np.asarray(obs, dtype=np.float32)
        self._current_transitions = [[] for _ in range(self.num_envs)]
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self._current_actions = np.asarray(actions)
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()

        if self._current_obs is not None and self._current_actions is not None:
            for i in range(self.num_envs):
                action_i = self._current_actions[i]
                # Discrete: actions[i] is a scalar → int; continuous: 1-D array → copy
                action = int(action_i) if np.ndim(action_i) == 0 else action_i.copy()
                info = (
                    {"ego_status": infos[i].get("ego_status", "running")}
                    if infos and isinstance(infos[i], dict)
                    else None
                )
                self._current_transitions[i].append(
                    Transition(
                        obs=self._current_obs[i].copy(),
                        action=action,
                        reward=float(rewards[i]),
                        info=info,
                    )
                )
                if dones[i] and self._current_transitions[i]:
                    self._finished_trajectories.append(
                        Trajectory(list(self._current_transitions[i]))
                    )
                    self._current_transitions[i] = []

        self._current_obs = np.asarray(obs, dtype=np.float32)
        return obs, rewards, dones, infos

    def pop_finished_trajectories(self) -> List[Trajectory]:
        """Return completed trajectories and clear all internal trajectory buffers.

        In addition to the finished trajectories, any in-progress (unfinished)
        episode buffers are also discarded.  This prevents transitions collected
        before a reward-model update from bleeding into the next training phase —
        the same invariant that imitation enforces via its two-return-value
        ``pop_finished_trajectories(finished, unfinished)``.

        After this call ``n_transitions == 0``, which is the precondition
        checked by :meth:`AgentTrainer.train`.
        """
        trajs = list(self._finished_trajectories)
        self._finished_trajectories = []
        # Discard in-progress episodes: they would otherwise mix pre- and
        # post-reward-model-update transitions inside a single trajectory.
        self._current_transitions = [[] for _ in range(self.num_envs)]
        return trajs


# ---------------------------------------------------------------------------
# AgentTrainer
# ---------------------------------------------------------------------------

class AgentTrainer:
    """Bundles an SB3 agent with trajectory buffering and reward wrapping.

    Provides two orthogonal operations:

    * :meth:`train` — run RL for *N* steps on model rewards (fills buffer).
    * :meth:`sample` — return trajectories with *true* rewards, collecting
      additional rollouts via ``predict()`` if the buffer is too small.

    This closely mirrors ``imitation.algorithms.preference_comparisons.AgentTrainer``.
    """

    def __init__(
        self,
        agent,
        venv: VecEnv,
        reward_model: EnsembleRewardModel,
        segment_length: int,
        agent_logger=None,
    ) -> None:
        """
        Args:
            agent: SB3 algorithm (e.g. PPO).
            venv: raw vectorised environment (before any reward wrapper).
            reward_model: ensemble used to predict model rewards.
            segment_length: used to count how many segments a trajectory
                contributes; controls when :meth:`sample` has collected enough.
            agent_logger: optional logger forwarded to the agent via
                ``agent.set_logger()``.
        """
        self.segment_length = segment_length

        # Build wrapper stack: raw_env → BufferingWrapper → EnvRewardWrapper
        self.buffering_wrapper = BufferingWrapper(venv)
        self.reward_wrapper = EnvRewardWrapper(self.buffering_wrapper, reward_model)

        self.agent = agent
        self.agent.set_env(self.reward_wrapper)
        if agent_logger is not None:
            self.agent.set_logger(agent_logger)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, steps: int, log_interval: int = 100, callback=None) -> None:
        """Run RL training for *steps* timesteps.

        The agent optimises against model rewards (provided by
        :class:`EnvRewardWrapper`).  Completed episodes accumulate in
        :class:`BufferingWrapper` with true rewards and can be retrieved
        via :meth:`sample` in the following iteration.

        Args:
            steps: number of environment timesteps to train for.
            log_interval: passed through to ``agent.learn()``.
            callback: optional SB3 callback.

        Raises:
            RuntimeError: if there are in-progress transitions still in the
                buffer, which means :meth:`sample` was not called beforehand.
                Call ``sample()`` first to clear the buffer.
        """
        n = self.buffering_wrapper.n_transitions
        if n > 0:
            raise RuntimeError(
                f"There are {n} in-progress transitions in the buffer. "
                "Call sample() before train() to clear them."
            )
        self.agent.learn(
            total_timesteps=steps,
            reset_num_timesteps=False,
            log_interval=log_interval,
            callback=callback or CustomLoggingCallback(),
        )

    def sample(self, n_steps: int) -> List[Trajectory]:
        """Return trajectories with true rewards covering ≥ *n_steps* transitions.

        Mirrors imitation's ``AgentTrainer.sample(steps)`` exactly: the budget
        is expressed in **transitions** (not segments), so the caller should pass:

            ``n_steps = ceil(transition_oversampling * 2 * num_pairs * segment_length)``

        This is the same formula imitation uses before calling
        ``trajectory_generator.sample(num_steps)``.

        Preferred source: trajectories already buffered from the last
        :meth:`train` call.  If the buffer is too small (including on the very
        first call before any training), the current policy is rolled out via
        ``predict()`` until the budget is met — equivalent to imitation's
        fallback ``generate_trajectories`` path.

        After any additional rollout ``agent._last_obs`` is synced so that
        the next :meth:`train` call starts from a consistent observation.

        Args:
            n_steps: minimum number of transitions (environment steps) required
                across all returned trajectories.

        Returns:
            List of :class:`Trajectory` objects carrying true env rewards.
        """
        trajs = self.buffering_wrapper.pop_finished_trajectories()

        # Prefer the most recent trajectories (latest policy version), mirroring
        # imitation's agent_trajs[::-1] inversion before _get_trajectories.
        trajs = trajs[::-1]

        if self._count_transitions(trajs) < n_steps:
            trajs = self._collect_until(trajs, n_steps)

        return trajs

    def reset_reward_stats(self) -> None:
        """Decay reward-normalisation statistics after a reward-model update."""
        self.reward_wrapper.reset_stats()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _count_transitions(self, trajectories: List[Trajectory]) -> int:
        """Total number of transitions across all trajectories.

        Mirrors imitation's ``sum(len(traj) for traj in agent_trajs)`` check
        inside ``AgentTrainer.sample()``.
        """
        return sum(traj.length() for traj in trajectories)

    def _collect_until(
        self,
        existing: List[Trajectory],
        n_steps: int,
    ) -> List[Trajectory]:
        """Roll out the current policy (predict-only) until transition budget is met.

        Uses ``agent.get_env()`` for stepping rather than ``self.reward_wrapper``
        directly, because SB3 may apply additional internal wrappers (e.g.
        VecTransposeImage) on top of the env passed to ``set_env()``.
        This matches imitation's pattern of always interacting with the agent
        through ``algorithm.get_env()``.
        """
        trajs = list(existing)

        algo_venv = self.agent.get_env()
        assert algo_venv is not None, "agent.get_env() returned None; was set_env() called?"

        # Start from agent's current obs; reset only if no obs is available.
        current_obs = self.agent._last_obs
        if current_obs is None:
            current_obs = algo_venv.reset()
            if isinstance(current_obs, tuple):
                current_obs, _ = current_obs

        while self._count_transitions(trajs) < n_steps:
            action, _ = self.agent.predict(current_obs, deterministic=False)
            # .step() calls step_async + step_wait through the full wrapper chain,
            # including BufferingWrapper which records true-reward transitions.
            step_result = algo_venv.step(action)
            current_obs = step_result[0]

            new_trajs = self.buffering_wrapper.pop_finished_trajectories()
            trajs.extend(new_trajs)

        # Sync agent internal state so the next learn() starts consistently.
        self.agent._last_obs = current_obs
        self.agent._last_episode_starts = np.zeros(
            (algo_venv.num_envs,), dtype=bool
        )

        return trajs
