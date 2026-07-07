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


class TestRelabelCache:
    def test_cached_rewards_match_live_relabelling_exactly(self):
        buf = _filled_buffer(RewardRelabelReplayBuffer, relabel=True)
        buf.refresh_relabel_cache()
        upper = buf.buffer_size if buf.full else buf.pos
        batch_inds = np.arange(upper)
        for env_idx in range(buf.n_envs):
            env_indices = np.full(upper, env_idx)
            cached = buf._relabelled_rewards(batch_inds, env_indices)
            live = buf._predict_rewards(batch_inds, env_indices)
            assert np.array_equal(cached, live)

    def test_entries_added_after_refresh_use_stored_rewards(self):
        buf = _filled_buffer(RewardRelabelReplayBuffer, relabel=True, n_steps=10)
        buf.refresh_relabel_cache()
        pos_at_refresh = buf.pos

        env = FakeVecEnv(num_envs=buf.n_envs, episode_len=5, seed=9)
        obs = env.reset()
        for _ in range(4):
            actions = np.stack([env.action_space.sample() for _ in range(buf.n_envs)])
            env.step_async(actions)
            next_obs, rewards, dones, infos = env.step_wait()
            buf.add(obs, next_obs, actions, rewards, dones, infos)
            obs = next_obs

        fresh_inds = np.array([pos_at_refresh, pos_at_refresh + 1])
        env_indices = np.zeros(2, dtype=int)
        got = buf._relabelled_rewards(fresh_inds, env_indices)
        assert np.array_equal(got, buf.rewards[fresh_inds, env_indices])

    def test_wraparound_freshness_mask(self):
        # Tiny buffer: refresh when full, then overwrite so fresh entries wrap.
        buf = _filled_buffer(RewardRelabelReplayBuffer, relabel=True, n_steps=6)
        small = RewardRelabelReplayBuffer(
            buffer_size=4,
            observation_space=buf.observation_space,
            action_space=buf.action_space,
            n_envs=1,
            relabel_rewards=True,
        )
        small.set_reward_model(ConstantRewardNet())
        env = FakeVecEnv(num_envs=1, episode_len=3, seed=2)
        obs = env.reset()

        def step_into(buffer):
            nonlocal obs
            actions = np.stack([env.action_space.sample()])
            env.step_async(actions)
            next_obs, rewards, dones, infos = env.step_wait()
            buffer.add(obs, next_obs, actions, rewards, dones, infos)
            obs = next_obs

        for _ in range(4):
            step_into(small)  # fill completely (pos wraps to 0)
        small.refresh_relabel_cache()
        for _ in range(3):
            step_into(small)  # overwrite positions 0, 1, 2 after the refresh

        env_indices = np.zeros(4, dtype=int)
        got = small._relabelled_rewards(np.arange(4), env_indices)
        stored = small.rewards[np.arange(4), env_indices]
        cached = small._relabel_cache[np.arange(4), 0]
        assert np.array_equal(got[:3], stored[:3])  # fresh (rewritten) entries
        assert got[3] == cached[3]                  # old entry still from cache

    def test_no_refresh_falls_back_to_live_relabelling(self):
        buf = _filled_buffer(RewardRelabelReplayBuffer, relabel=True)
        assert buf._relabel_cache is None
        batch_inds = np.arange(8)
        env_indices = np.zeros(8, dtype=int)
        got = buf._relabelled_rewards(batch_inds, env_indices)
        assert np.array_equal(got, buf._predict_rewards(batch_inds, env_indices))

    def test_pickle_drops_cache(self):
        buf = _filled_buffer(RewardRelabelReplayBuffer, relabel=True)
        buf.refresh_relabel_cache()
        restored = pickle.loads(pickle.dumps(buf))
        assert restored._relabel_cache is None
