import numpy as np
from collections import deque
from typing import List, Tuple, Any, Dict, Callable, Union, Optional

from stable_baselines3.common.vec_env import VecEnvWrapper, VecEnv

from .reward_nets import RewardEnsemble
from .types import Trajectory, Transition

# order: arrived, collided, off_road, timeout, running, teleported, removed_unknown
_STATUS_ONEHOT = {
    "arrived":         np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    "collided":        np.array([0, 1, 0, 0, 0, 0, 0], dtype=np.float32),
    "offroad":         np.array([0, 0, 1, 0, 0, 0, 0], dtype=np.float32),
    "timeout":         np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32),
    "running":         np.array([0, 0, 0, 0, 1, 0, 0], dtype=np.float32),
    "teleported":      np.array([0, 0, 0, 0, 0, 1, 0], dtype=np.float32),
    "removed_unknown": np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32),
}

def ego_status_to_onehot(status: str) -> np.ndarray:
    return _STATUS_ONEHOT.get(status, _STATUS_ONEHOT["running"])

class _RunningMeanStd:
    """Welford's online algorithm for running mean and variance."""

    def __init__(self):
        self.mean = 0.0
        self.var = 0.0
        self.count = 0

    def update(self, values: np.ndarray) -> None:
        for x in values.flat:
            self.count += 1
            delta = x - self.mean
            self.mean += delta / self.count
            self.var += (x - self.mean) * delta  # M2

    @property
    def std(self) -> float:
        if self.count < 2:
            return 1.0
        s = float(np.sqrt(max(0.0, self.var / (self.count - 1))))
        return s if s > 0 else 1.0


# ---------------------------------------------------------------------------
# Environment reward wrapper
# ---------------------------------------------------------------------------

class EnvRewardWrapper(VecEnvWrapper):
    """
    VecEnvWrapper that replaces environment rewards with EnsembleRewardModel
    predictions. Uses the pre-step observation for reward prediction to align
    with how the reward predictor was trained on (obs_t, a_t) pairs.

    Rewards are normalized to mean 0 / std 1 via a running estimate before
    being passed to the agent (Christiano et al. 2017, Section 2.2).
    """

    def __init__(self, venv: VecEnv, reward_model: RewardEnsemble, relabel_debug_size: int = 100_000):
        super().__init__(venv)
        self.reward_model = reward_model
        self._obs: np.ndarray | None = None
        self._actions: np.ndarray | None = None
        self._relabel_debug_buffer = deque(maxlen=relabel_debug_size)

    def reset(self):
        obs = self.venv.reset()
        self._obs = np.asarray(obs, dtype=np.float32)
        return obs

    def step_async(self, actions):
        self._actions = np.asarray(actions)
        self.venv.step_async(actions)

    def step_wait(self):
        obs, true_rew, dones, infos = self.venv.step_wait()

        if self._obs is not None and self._actions is not None:
            next_status = np.array([ego_status_to_onehot(i.get("ego_status", "running")) for i in infos])
            predicted_rew = self.reward_model.predict(self._obs, self._actions, next_status, dones.astype(np.float32))
            self._store_relabel_debug_batch(self._obs, self._actions, next_status, dones, predicted_rew)
        else:
            predicted_rew = np.zeros(len(obs), dtype=np.float32)

        self._obs = np.asarray(obs, dtype=np.float32)
        return obs, predicted_rew, dones, infos

    def _store_relabel_debug_batch(self, obs, actions, next_status, dones, rewards) -> None:
        for i in range(len(rewards)):
            self._relabel_debug_buffer.append((
                np.asarray(obs[i], dtype=np.float32).copy(),
                np.asarray(actions[i], dtype=np.float32).copy(),
                np.asarray(next_status[i], dtype=np.float32).copy(),
                float(dones[i]),
                float(rewards[i]),
            ))

    def sample_relabel_debug_batch(self, batch_size: int, rng: np.random.Generator):
        """Sample rewards as originally passed to the agent for stale-reward diagnostics."""
        n_items = len(self._relabel_debug_buffer)
        if n_items == 0:
            return None

        n_samples = min(batch_size, n_items)
        indices = rng.choice(n_items, size=n_samples, replace=False)
        samples = [self._relabel_debug_buffer[int(i)] for i in indices]
        obs, actions, next_status, dones, rewards = zip(*samples)
        return (
            np.asarray(obs, dtype=np.float32),
            np.asarray(actions, dtype=np.float32),
            np.asarray(next_status, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
            np.asarray(rewards, dtype=np.float32),
        )



class EnvBufferingWrapper(VecEnvWrapper):
    """VecEnvWrapper that records rollout transitions and groups them into Trajectory objects."""

    def __init__(self, venv: VecEnv, error_on_premature_reset: bool = True):
        super().__init__(venv)
        self.error_on_premature_reset = error_on_premature_reset

        self._initialized = False
        self._saved_actions = None

        # Completed (terminated) trajectories.
        self._finished_trajectories: List[Trajectory] = []

        # In-progress trajectory, one per parallel env.
        self._partial_trajectories: List[Trajectory] = []

        # Timestep counter per parallel env.
        self._timesteps: np.ndarray | None = None

        # Last observation seen per env.
        self._last_obs = None

    def is_empty(self):
        return len(self._finished_trajectories) == 0

    def step_async(self, actions):
        assert self._initialized, "Call reset() before stepping."
        assert self._saved_actions is None, "step_async called twice without step_wait."
        self._saved_actions = actions
        self.venv.step_async(actions)

    def step_wait(self):
        """Step the env and record one Transition per parallel env."""
        assert self._initialized, "Call reset() before stepping."
        assert self._saved_actions is not None, "step_wait called before step_async."

        actions = self._saved_actions
        self._saved_actions = None

        obs, true_rew, dones, infos = self.venv.step_wait()

        self._timesteps += 1

        for i in range(self.num_envs):
            transition = Transition(
                observation=self._last_obs[i],
                action=actions[i],
                true_reward=float(true_rew[i]),
                next_status=ego_status_to_onehot(infos[i].get("ego_status", "running")),
                done=bool(dones[i]),
            )

            self._partial_trajectories[i].add_transition(transition)

            if dones[i]:
                # Episode finished: store the completed trajectory and start a new one.
                self._finished_trajectories.append(self._partial_trajectories[i])
                self._partial_trajectories[i] = Trajectory()
                self._timesteps[i] = 0

        self._last_obs = obs
        return obs, true_rew, dones, infos

    def pop_finished_trajectories(self) -> List[Trajectory]:
        """Return and clear the trajectories completed since the last pop."""
        trajectories = self._finished_trajectories
        self._finished_trajectories = []
        return trajectories

    def reset(self, **kwargs):
        if (
            self._initialized
            and self.error_on_premature_reset
            and len(self._finished_trajectories) > 0
        ):
            raise RuntimeError("reset() called before the buffered trajectories were read.")

        self._initialized = True
        self._saved_actions = None

        obs = self.venv.reset(**kwargs)
        self._last_obs = obs

        self._timesteps = np.zeros(self.num_envs, dtype=int)
        self._partial_trajectories = [Trajectory() for _ in range(self.num_envs)]
        self._finished_trajectories = []

        return obs



class PolicyExplorationWrapper:
    """Epsilon-greedy exploration wrapper around a policy.

    On each `predict` call, with probability `exploration_eps` it samples random
    actions from the env's action space, otherwise it defers to the wrapped policy.
    Only stateless policies are supported.
    """

    def __init__(
        self,
        venv: VecEnv,
        policy: Callable,
        exploration_eps: float,
        rng: np.random.Generator,
    ):
        """
        Args:
            venv: vectorized env, used to sample random actions.
            policy: wrapped policy; must be callable and return (actions, state).
            exploration_eps: probability of sampling a random action per call.
            rng: random generator driving all random choices.
        """
        self.wrapped_policy = policy
        self.venv = venv
        self.exploration_eps = exploration_eps
        self.rng = rng

        # Seed the action space so random sampling is also driven by rng.
        seed = int(self.rng.integers(0, 2**31 - 1))
        self.venv.action_space.seed(seed)

    def predict(self, observation: np.ndarray, **kwargs) -> tuple:
        if self.rng.random() < self.exploration_eps:
            num_envs = len(observation)
            actions = np.stack([self.venv.action_space.sample() for _ in range(num_envs)])
            return actions, None
        return self.wrapped_policy.predict(observation, **kwargs)
