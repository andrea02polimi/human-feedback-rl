import numpy as np
import pytest
from stable_baselines3 import SAC
from stable_baselines3.common.logger import KVWriter

from human_feedback_rl.common.custom_logging_callback import FixedIntervalDumpCallback
from human_feedback_rl.common.loggers import Logger
from human_feedback_rl.common.trajectory_generators import TrajectoryGeneratorFromAgent
from human_feedback_rl.common.reward_nets import make_reward_ensemble

from conftest import FakeVecEnv


class CapturingWriter(KVWriter):
    def __init__(self):
        self.writes = []

    def write(self, key_values, key_excluded, step=0):
        self.writes.append(dict(key_values))

    def close(self):
        pass


def _dump_timesteps(seed, dump_interval=64, total=512):
    """Total_timesteps values at which SAC dumps, with the fixed-grid callback."""
    env = FakeVecEnv(num_envs=2, episode_len=7 + seed, seed=seed)  # episode length varies per seed
    agent = SAC(
        "MlpPolicy", env, buffer_size=200, learning_starts=0, batch_size=16,
        train_freq=1, gradient_steps=1, policy_kwargs=dict(net_arch=[16]),
        seed=seed, verbose=0,
    )
    writer = CapturingWriter()
    agent.set_logger(Logger(folder=None, output_formats=[writer]))
    agent.learn(
        total_timesteps=total, log_interval=None,
        callback=FixedIntervalDumpCallback(dump_interval),
    )
    return [w["time/total_timesteps"] for w in writer.writes if "time/total_timesteps" in w]


def test_invalid_interval_raises():
    with pytest.raises(ValueError):
        FixedIntervalDumpCallback(0)


def test_dumps_land_on_fixed_grid():
    xs = _dump_timesteps(seed=0)
    assert xs, "no dumps recorded"
    # Every dump falls exactly when a 64-timestep boundary is crossed
    # (num_timesteps advances by n_envs=2, and 64 is a multiple of 2).
    assert xs == [64 * (i + 1) for i in range(len(xs))]


def test_dump_timesteps_identical_across_seeds():
    """The whole point: different seeds (and episode lengths) share x values."""
    xs_a = _dump_timesteps(seed=0)
    xs_b = _dump_timesteps(seed=1)
    assert xs_a == xs_b


def test_generator_grid_mode_disables_episode_dumps(rng):
    env = FakeVecEnv(num_envs=2, episode_len=10)
    agent = SAC(
        "MlpPolicy", env, buffer_size=200, learning_starts=0, batch_size=16,
        train_freq=1, gradient_steps=1, policy_kwargs=dict(net_arch=[16]),
        seed=0, verbose=0,
    )
    writer = CapturingWriter()
    agent.set_logger(Logger(folder=None, output_formats=[writer]))
    reward_model = make_reward_ensemble(env, n_ensembles=1, net_arch=[8])
    generator = TrajectoryGeneratorFromAgent(
        agent=agent, reward_model=reward_model, venv=env,
        rng=rng, dump_timestep_interval=64,
    )
    # log_interval=1 would dump every episode; grid mode must override it.
    generator.train(steps=256, log_interval=1)
    xs = [w["time/total_timesteps"] for w in writer.writes if "time/total_timesteps" in w]
    assert xs and all(x % 64 == 0 for x in xs)
