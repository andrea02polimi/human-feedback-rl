import pickle

import numpy as np
import pytest

from human_feedback_rl.common.replay_buffers import (
    RewardDiagnosticsReplayBuffer,
    RewardRelabelReplayBuffer,
)

from conftest import ConstantRewardNet, FakeVecEnv


def _filled_buffer(buffer_cls, relabel, n_steps=20, num_envs=2):
    env = FakeVecEnv(num_envs=num_envs, episode_len=5)
    buf = buffer_cls(
        buffer_size=64,
        observation_space=env.observation_space,
        action_space=env.action_space,
        n_envs=num_envs,
        relabel_rewards=relabel,
    )
    buf.set_reward_model(ConstantRewardNet())
    obs = env.reset()
    for _ in range(n_steps):
        actions = np.stack([env.action_space.sample() for _ in range(num_envs)])
        env.step_async(actions)
        next_obs, rewards, dones, infos = env.step_wait()
        buf.add(obs, next_obs, actions, rewards, dones, infos)
        obs = next_obs
    return buf


def _expected_reward(buf, obs, actions):
    # ConstantRewardNet on buffer-stored (unscaled) actions.
    return obs.sum(axis=1) + 0.5 * buf._unscale_actions(actions).sum(axis=1)


def test_relabel_enabled_returns_current_model_rewards():
    buf = _filled_buffer(RewardRelabelReplayBuffer, relabel=True)
    samples = buf.sample(16)
    obs = samples.observations.numpy()
    actions = buf._unscale_actions(samples.actions.numpy())
    expected = obs.sum(axis=1) + 0.5 * actions.sum(axis=1)
    assert np.allclose(samples.rewards.numpy().ravel(), expected, rtol=1e-4)


def test_relabel_disabled_returns_stored_rewards():
    buf = _filled_buffer(RewardRelabelReplayBuffer, relabel=False)
    samples = buf.sample(16)
    # Stored env rewards are 0.1*obs.sum + 0.01*act.sum — never equal to the
    # ConstantRewardNet output on the same inputs.
    obs = samples.observations.numpy()
    actions = buf._unscale_actions(samples.actions.numpy())
    model_rewards = obs.sum(axis=1) + 0.5 * actions.sum(axis=1)
    assert not np.allclose(samples.rewards.numpy().ravel(), model_rewards, rtol=1e-4)


def test_sample_reward_staleness_shapes(rng):
    buf = _filled_buffer(RewardDiagnosticsReplayBuffer, relabel=False)
    stored, current = buf.sample_reward_staleness(8, rng)
    assert stored.shape == current.shape == (8,)


def test_missing_reward_model_raises(rng):
    buf = _filled_buffer(RewardRelabelReplayBuffer, relabel=True)
    buf.set_reward_model(None)
    with pytest.raises(RuntimeError, match="reward model"):
        buf.sample(4)


def test_pickle_drops_reward_model():
    buf = _filled_buffer(RewardDiagnosticsReplayBuffer, relabel=False)
    restored = pickle.loads(pickle.dumps(buf))
    assert restored.reward_model is None
