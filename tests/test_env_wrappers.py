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


class _CountingVecEnv(FakeVecEnv):
    """FakeVecEnv that counts the steps actually taken in the environment."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.total_steps = 0

    def step_wait(self):
        self.total_steps += self.num_envs
        return super().step_wait()


class TestRolloutContinuation:
    """The episode SAC leaves half done must not disappear.

    With a shared environment ``agent.learn()`` almost always returns
    mid-episode, and the part already collected lives in
    ``_partial_trajectories``. A ``venv.reset()`` would erase it, and the old
    guard did not notice because it looked only at FINISHED trajectories --
    which ``sample()`` has just emptied with ``pop_finished_trajectories()``.
    """

    class _Fixed:
        """Deterministic policy; exposes action_log_prob to bypass SB3."""

        def predict(self, obs, state=None, episode_start=None, deterministic=False):
            return np.zeros((len(obs), ACT_DIM), dtype=np.float32), None

        def action_log_prob(self, obs, actions):
            return np.zeros(len(obs), dtype=np.float32)

    def _stepped_env(self, steps, episode_len=10):
        """An environment with an episode started and NOT finished."""
        env = EnvBufferingWrapper(FakeVecEnv(num_envs=1, episode_len=episode_len))
        env.reset()
        for _ in range(steps):
            env.step(np.zeros((1, ACT_DIM), dtype=np.float32))
        return env

    # --- the guard ----------------------------------------------------------

    def test_reset_raises_if_an_episode_is_in_progress(self):
        """What used to be lost silently is now an explicit error."""
        env = self._stepped_env(4)
        assert len(env._partial_trajectories[0]) == 4
        with pytest.raises(RuntimeError, match="still in progress"):
            env.reset()

    def test_the_message_says_how_many_transitions_would_be_lost(self):
        env = self._stepped_env(6)
        with pytest.raises(RuntimeError, match=r"\[6\] transitions"):
            env.reset()

    def test_reset_does_not_raise_with_episodes_closed(self):
        """The dedicated path, where rollout_agent always leaves all closed."""
        env = self._stepped_env(10)                  # episode exactly finished
        env.pop_finished_trajectories()
        assert all(len(t) == 0 for t in env._partial_trajectories)
        env.reset()                                   # must not raise

    # --- continuation -------------------------------------------------------

    def test_continuing_keeps_the_transitions_already_collected(self):
        from human_feedback_rl.common.trajectory_generators import rollout_agent

        env = self._stepped_env(4)                    # 4 steps out of 10
        rollout_agent(self._Fixed(), env, steps=1, start_obs=env.venv._obs)
        done = env.pop_finished_trajectories()
        assert done, "no trajectory finished"
        # SAC's 4 steps plus the 6 needed to close the episode
        assert len(done[0]) == 10, f"expected 10 steps, found {len(done[0])}"

    def test_from_clean_the_rollout_resets_as_before(self):
        """With no start_obs and no open episodes, behaviour is unchanged."""
        from human_feedback_rl.common.trajectory_generators import rollout_agent

        env = EnvBufferingWrapper(FakeVecEnv(num_envs=1, episode_len=10))
        rollout_agent(self._Fixed(), env, steps=1)
        done = env.pop_finished_trajectories()
        assert done and len(done[0]) == 10


class TestSampleEndToEnd:
    """The invariant that matters: nothing collected is thrown away.

    Testing ``rollout_agent`` in isolation is not enough: the defect came from
    the interplay between ``train()``, which leaves an episode open, and
    ``sample()``, which emptied the finished ones and then reset.
    """

    def _generator(self, rng, episode_len=10):
        from stable_baselines3 import SAC
        from human_feedback_rl.common.reward_nets import make_reward_ensemble
        from human_feedback_rl.common.trajectory_generators import (
            TrajectoryGeneratorFromAgent,
        )

        env = _CountingVecEnv(num_envs=1, episode_len=episode_len)
        agent = SAC(
            "MlpPolicy", env, buffer_size=200, learning_starts=0, batch_size=16,
            train_freq=1, gradient_steps=1, policy_kwargs=dict(net_arch=[16]),
            seed=0, verbose=0,
        )
        reward_model = make_reward_ensemble(env, n_ensembles=1, net_arch=[8])
        gen = TrajectoryGeneratorFromAgent(
            agent=agent, reward_model=reward_model, venv=env, rng=rng,
        )
        return gen, env

    def test_nessuna_transizione_persa_fra_train_e_sample(self, rng):
        gen, env = self._generator(rng)

        # 14 passi su episodi da 10: SAC si ferma a meta' del secondo
        gen.train(steps=14, log_interval=100)
        aperte_prima = sum(len(t) for t in gen.buffering_wrapper._partial_trajectories)
        assert aperte_prima > 0, "il test non esercita il caso che ci interessa"

        trajs = gen.sample(agent_steps=20)

        collected = sum(len(t) for t in trajs)
        still_open = sum(len(t) for t in gen.buffering_wrapper._partial_trajectories)
        # every step taken in the environment is either in a returned trajectory
        # or in an episode still open: none disappears
        assert collected + still_open == env.total_steps

    def test_the_returned_trajectories_are_whole_episodes(self, rng):
        """No truncation: demo_2 aggregates by sum, so length matters."""
        gen, _ = self._generator(rng)
        gen.train(steps=14, log_interval=100)
        trajs = gen.sample(agent_steps=20)
        assert trajs and all(len(t) == 10 for t in trajs)

