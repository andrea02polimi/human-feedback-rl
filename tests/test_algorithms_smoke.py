"""End-to-end smoke tests: each algorithm runs a tiny train() on the fake env."""

import numpy as np
import pytest
import torch as th
from stable_baselines3 import SAC

from human_feedback_rl.algorithms import HybridAlgorithm
from human_feedback_rl.common.replay_buffers import RewardRelabelReplayBuffer
from human_feedback_rl.common.reward_nets import make_reward_ensemble

from conftest import FakeVecEnv, make_trajectories

RM_KWARGS = dict(n_ensembles=2, net_arch=[8])


def _sac(env):
    return SAC(
        "MlpPolicy", env, buffer_size=500, learning_starts=0, batch_size=16,
        train_freq=1, gradient_steps=1, policy_kwargs=dict(net_arch=[16]),
        replay_buffer_class=RewardRelabelReplayBuffer, seed=0, verbose=0,
    )


def _hybrid(env, rng, **overrides):
    kwargs = dict(
        expert_trajectories=make_trajectories(rng, [10, 10, 10]),
        loss_type="demo_2",
        gradient_steps_rew=2,
        batch_size_expert=2,
        batch_size_model=2,
        batch_size_pref=4,
        total_queries=8,
        preference_fragment_length=3,
        relabel_rewards=True,
        reward_model_kwargs=RM_KWARGS,
        rng=np.random.default_rng(0),
        output_formats=[],
    )
    kwargs.update(overrides)
    return HybridAlgorithm(env, _sac(env), **kwargs)


def test_hybrid_algorithm_trains(rng):
    env = FakeVecEnv(num_envs=2, episode_len=10)
    algo = _hybrid(env, rng)
    agent = algo.train(
        total_timesteps=64, timesteps_per_iteration=32, log_interval=100, scatter_interval=0
    )
    assert agent is algo.agent
    assert len(algo.dataset_train) > 0
    assert algo.reward_model.normalization_std > 0


@pytest.mark.parametrize("loss_type", ["demo_1", "demo_2"])
def test_hybrid_demo_only_arm_trains_per_loss(loss_type, rng):
    env = FakeVecEnv(num_envs=2, episode_len=10)
    algo = _hybrid(env, rng, loss_type=loss_type, total_queries=0)
    agent = algo.train(
        total_timesteps=64, timesteps_per_iteration=32, log_interval=100, scatter_interval=0
    )
    assert agent is algo.agent
    assert algo.reward_model.normalization_std > 0


def test_hybrid_checkpoint_roundtrip(tmp_path, rng):
    env = FakeVecEnv(num_envs=2, episode_len=10)
    algo = _hybrid(env, rng)
    algo.save_checkpoint(str(tmp_path), 1)
    ckpt = tmp_path / "checkpoint_0001"
    assert (ckpt / "reward_model.pt").exists()
    assert (ckpt / "reward_training.pt").exists()
    assert (ckpt / "hybrid_training.pt").exists()
    assert (ckpt / "agent.zip").exists()
    assert (ckpt / "replay_buffer.pkl").exists()

    state_dict = th.load(ckpt / "reward_model.pt", weights_only=True)
    fresh = make_reward_ensemble(env, **RM_KWARGS)
    fresh.load_state_dict(state_dict)
    for key, value in fresh.state_dict().items():
        assert th.equal(value, state_dict[key]), key
