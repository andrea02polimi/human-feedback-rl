"""Shared fixtures: a SUMO-free fake VecEnv, trajectory factories, tiny reward nets."""

import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from human_feedback_rl.common.reward_nets import RewardNet, make_reward_ensemble
from human_feedback_rl.common.status import STATUS_NAMES, STATUS_DIM, ego_status_to_onehot
from human_feedback_rl.common.types import Trajectory, Transition

OBS_DIM = 4
ACT_DIM = 2

# Terminal statuses cycled by FakeVecEnv episode after episode.
TERMINAL_STATUSES = ("arrived", "collided", "offroad", "timeout")


class FakeVecEnv(VecEnv):
    """Deterministic SB3-compatible VecEnv emitting info["ego_status"].

    Episodes last exactly ``episode_len`` steps; each finished episode reports
    the next status from ``TERMINAL_STATUSES`` (per env), all other steps are
    "running". Rewards are a fixed linear function of observation and action.
    """

    def __init__(self, num_envs: int = 2, episode_len: int = 10, seed: int = 0):
        observation_space = spaces.Box(-1.0, 1.0, (OBS_DIM,), dtype=np.float32)
        action_space = spaces.Box(-1.0, 1.0, (ACT_DIM,), dtype=np.float32)
        super().__init__(num_envs, observation_space, action_space)
        self.episode_len = episode_len
        self.rng = np.random.default_rng(seed)
        self._steps = np.zeros(num_envs, dtype=int)
        self._episodes = np.zeros(num_envs, dtype=int)
        self._obs = None
        self._actions = None
        self.render_mode = None

    def _next_obs(self) -> np.ndarray:
        return self.rng.normal(size=(self.num_envs, OBS_DIM)).astype(np.float32).clip(-1, 1)

    def reset(self):
        self._steps[:] = 0
        self._obs = self._next_obs()
        return self._obs.copy()

    def step_async(self, actions):
        self._actions = np.asarray(actions, dtype=np.float32)

    def step_wait(self):
        self._steps += 1
        rewards = (0.1 * self._obs.sum(axis=1) + 0.01 * self._actions.sum(axis=1)).astype(
            np.float32
        )
        dones = self._steps >= self.episode_len
        infos = []
        for i in range(self.num_envs):
            if dones[i]:
                status = TERMINAL_STATUSES[self._episodes[i] % len(TERMINAL_STATUSES)]
                infos.append(
                    {"ego_status": status, "terminal_observation": self._obs[i].copy()}
                )
                self._steps[i] = 0
                self._episodes[i] += 1
            else:
                infos.append({"ego_status": "running"})
        self._obs = self._next_obs()
        return self._obs.copy(), rewards, dones, infos

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name)] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [getattr(self, method_name)(*args, **kwargs)] * self.num_envs

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs


class ConstantRewardNet(RewardNet):
    """Hand-computable reward: obs.sum() + 0.5 * action.sum() (+ optional constant)."""

    def __init__(self, observation_space=None, action_space=None, offset: float = 0.0):
        observation_space = observation_space or spaces.Box(-1, 1, (OBS_DIM,), dtype=np.float32)
        action_space = action_space or spaces.Box(-1, 1, (ACT_DIM,), dtype=np.float32)
        super().__init__(observation_space, action_space)
        self.offset = offset
        # One (unused) parameter so optimizers can be constructed against it.
        self._dummy = th.nn.Parameter(th.zeros(1))

    def forward(self, state, action, next_status=None, done=None):
        return state.sum(dim=1) + 0.5 * action.sum(dim=1) + self.offset


def make_trajectories(rng, lengths, obs_dim=OBS_DIM, act_dim=ACT_DIM, with_log_probs=False):
    """Deterministic trajectories with rotating terminal statuses."""
    trajs = []
    for ti, length in enumerate(lengths):
        traj = Trajectory()
        terminal = TERMINAL_STATUSES[ti % len(TERMINAL_STATUSES)]
        for j in range(length):
            done = j == length - 1
            traj.add_transition(
                Transition(
                    observation=rng.normal(size=obs_dim).astype(np.float32),
                    action=rng.normal(size=act_dim).astype(np.float32),
                    true_reward=float(ti + 0.1 * j),
                    next_status=ego_status_to_onehot(terminal if done else "running"),
                    done=done,
                    log_policy_prob=(-1.0 - 0.01 * j) if with_log_probs else None,
                )
            )
        trajs.append(traj)
    return trajs


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def fake_env():
    return FakeVecEnv(num_envs=2, episode_len=10)


@pytest.fixture
def tiny_reward_ensemble(fake_env):
    th.manual_seed(0)
    return make_reward_ensemble(fake_env, n_ensembles=2, net_arch=[8], activation_fn="tanh")


@pytest.fixture
def constant_reward_net():
    return ConstantRewardNet()
