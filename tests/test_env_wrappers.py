import numpy as np
import pytest

from human_feedback_rl.common.env_wrappers import (
    EnvBufferingWrapper,
    EnvRewardWrapper,
    PolicyExplorationWrapper,
)
from human_feedback_rl.common.status import STATUS_RUNNING

from conftest import ACT_DIM, ConstantRewardNet, FakeVecEnv


def _random_policy_step(env):
    return np.stack([env.action_space.sample() for _ in range(env.num_envs)])


class TestEnvBufferingWrapper:
    def test_records_full_trajectories(self):
        env = FakeVecEnv(num_envs=2, episode_len=5)
        wrapper = EnvBufferingWrapper(env)
        wrapper.reset()
        for _ in range(10):  # exactly two episodes per env
            wrapper.step_async(_random_policy_step(env))
            wrapper.step_wait()
        trajs = wrapper.pop_finished_trajectories()
        assert len(trajs) == 4
        assert all(len(t) == 5 for t in trajs)
        for traj in trajs:
            assert all(tr.next_status[STATUS_RUNNING] == 1 for tr in traj[:-1])
            assert traj[-1].done and traj[-1].next_status[STATUS_RUNNING] == 0

    def test_pre_step_observation_is_recorded(self):
        env = FakeVecEnv(num_envs=1, episode_len=3)
        wrapper = EnvBufferingWrapper(env)
        obs0 = wrapper.reset()
        wrapper.step_async(_random_policy_step(env))
        wrapper.step_wait()
        wrapper.step_async(_random_policy_step(env))
        wrapper.step_wait()
        wrapper.step_async(_random_policy_step(env))
        wrapper.step_wait()
        (traj,) = wrapper.pop_finished_trajectories()
        assert np.array_equal(traj[0].observation, obs0[0])

    def test_premature_reset_raises(self):
        env = FakeVecEnv(num_envs=1, episode_len=2)
        wrapper = EnvBufferingWrapper(env)
        wrapper.reset()
        for _ in range(2):
            wrapper.step_async(_random_policy_step(env))
            wrapper.step_wait()
        with pytest.raises(RuntimeError):
            wrapper.reset()

    def test_recording_mask_excludes_envs(self):
        env = FakeVecEnv(num_envs=2, episode_len=3)
        wrapper = EnvBufferingWrapper(env)
        wrapper.reset()
        wrapper.set_recording_mask(np.array([True, False]))
        for _ in range(3):
            wrapper.step_async(_random_policy_step(env))
            wrapper.step_wait()
        trajs = wrapper.pop_finished_trajectories()
        assert len(trajs) == 1  # only env 0 recorded


class TestEnvRewardWrapper:
    def test_rewards_replaced_with_model_prediction_on_pre_step_obs(self):
        env = FakeVecEnv(num_envs=2, episode_len=4)
        model = ConstantRewardNet()
        wrapper = EnvRewardWrapper(env, reward_model=model)
        obs = wrapper.reset()
        actions = _random_policy_step(env)
        wrapper.step_async(actions)
        _, rewards, _, _ = wrapper.step_wait()
        expected = obs.sum(axis=1) + 0.5 * actions.sum(axis=1)
        assert np.allclose(rewards, expected, rtol=1e-5)


class TestPolicyExplorationWrapper:
    class _FixedPolicy:
        def predict(self, observation, **kwargs):
            return np.zeros((len(observation), ACT_DIM), dtype=np.float32), None

        def action_log_prob(self, observation, actions):
            return np.full(len(observation), -0.5)

    def test_eps_zero_defers_to_policy(self, rng):
        env = FakeVecEnv(num_envs=2)
        wrapper = PolicyExplorationWrapper(env, self._FixedPolicy(), 0.0, rng)
        actions, _ = wrapper.predict(np.zeros((2, 4), dtype=np.float32))
        assert np.array_equal(actions, np.zeros((2, ACT_DIM)))

    def test_eps_one_samples_random_actions(self, rng):
        env = FakeVecEnv(num_envs=2)
        wrapper = PolicyExplorationWrapper(env, self._FixedPolicy(), 1.0, rng)
        actions, _ = wrapper.predict(np.zeros((2, 4), dtype=np.float32))
        assert actions.shape == (2, ACT_DIM)
        assert not np.array_equal(actions, np.zeros((2, ACT_DIM)))

    def test_action_log_prob_mixture(self, rng):
        env = FakeVecEnv(num_envs=1)
        obs = np.zeros((1, 4), dtype=np.float32)
        actions = np.zeros((1, ACT_DIM), dtype=np.float32)
        # Uniform density on [-1, 1]^ACT_DIM: log(1/2^d) = -d*log(2).
        uniform_log_prob = -ACT_DIM * np.log(2.0)
        policy_log_prob = -0.5

        eps1 = PolicyExplorationWrapper(env, self._FixedPolicy(), 1.0, rng)
        assert eps1.action_log_prob(obs, actions)[0] == pytest.approx(uniform_log_prob)

        eps0 = PolicyExplorationWrapper(env, self._FixedPolicy(), 0.0, rng)
        assert eps0.action_log_prob(obs, actions)[0] == pytest.approx(policy_log_prob)

        eps = 0.3
        mixed = PolicyExplorationWrapper(env, self._FixedPolicy(), eps, rng)
        expected = np.logaddexp(
            np.log1p(-eps) + policy_log_prob, np.log(eps) + uniform_log_prob
        )
        assert mixed.action_log_prob(obs, actions)[0] == pytest.approx(expected)
