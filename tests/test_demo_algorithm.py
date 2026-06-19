import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from human_feedback_rl.algorithms import DemoAlgorithm
from human_feedback_rl.common.env_wrappers import EnvBufferingWrapper
from human_feedback_rl.common.loggers import configure_wandb_metrics
from human_feedback_rl.common.replay_buffers import RewardRelabelReplayBuffer
from human_feedback_rl.common.reward_nets import make_reward_ensemble
from human_feedback_rl.common.trajectory_generators import (
    policy_action_log_probs,
    rollout_agent,
)
from human_feedback_rl.common.types import Trajectory, Transition


class TinyEnv(gym.Env):
    observation_space = gym.spaces.Box(-10, 10, shape=(2,), dtype=np.float32)
    action_space = gym.spaces.Box(-2, 2, shape=(1,), dtype=np.float32)

    def __init__(self, horizon=5):
        self.t = 0
        self.horizon = horizon

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        self.t += 1
        terminated = self.t >= self.horizon
        observation = np.array([self.t, float(action[0])], dtype=np.float32)
        info = {"ego_status": "arrived" if terminated else "running"}
        return observation, float(1 - abs(action[0])), terminated, False, info


def make_vec_env():
    return DummyVecEnv([TinyEnv])


def make_expert_trajectory():
    transitions = []
    for i in range(5):
        arrived = i == 4
        status = [1, 0, 0, 0, 0, 0, 0] if arrived else [0, 0, 0, 0, 1, 0, 0]
        transitions.append(Transition(
            observation=np.array([i, 0], dtype=np.float32),
            action=np.array([0], dtype=np.float32),
            true_reward=1.0,
            next_status=np.array(status, dtype=np.float32),
            done=arrived,
        ))
    return Trajectory(transitions)


REWARD_KWARGS = {"n_ensembles": 1, "net_arch": [8], "activation_fn": "tanh"}


class DemoAlgorithmTest(unittest.TestCase):
    def test_wandb_metrics_use_semantic_axes_and_hide_secondary_panels(self):
        class FakeRun:
            def __init__(self):
                self.calls = []

            def define_metric(self, name, **kwargs):
                self.calls.append((name, kwargs))

        run = FakeRun()
        configure_wandb_metrics(run)
        definitions = dict(run.calls)

        self.assertEqual(
            definitions["agent/*"]["step_metric"],
            "agent/time/total_timesteps",
        )
        self.assertEqual(definitions["reward/*"]["step_metric"], "iterations")
        self.assertTrue(definitions["agent/action_rate/*"]["hidden"])
        self.assertTrue(
            definitions[
                "reward_val/debug_dataset/post_update/reward_arrived"
            ]["hidden"]
        )

    def test_ppo_log_prob_uses_tail_mass_for_clipped_actions(self):
        env = make_vec_env()
        agent = PPO(
            "MlpPolicy", env, n_steps=5, batch_size=5, n_epochs=1,
            device="cpu", verbose=0,
        )
        with th.no_grad():
            agent.policy.action_net.weight.zero_()
            agent.policy.action_net.bias.fill_(10.0)

        observation = env.reset()
        actions, _ = agent.predict(observation, deterministic=True)
        self.assertEqual(float(actions[0, 0]), float(env.action_space.high[0]))

        log_prob = policy_action_log_probs(agent, observation, actions)
        self.assertGreater(float(log_prob[0]), -1e-3)
        env.close()

    def test_multi_env_rollout_records_one_finishing_episode_per_env(self):
        class ZeroPolicy:
            @staticmethod
            def predict(observation, **kwargs):
                return np.zeros((len(observation), 1), dtype=np.float32), None

            @staticmethod
            def action_log_prob(observation, actions):
                return np.zeros(len(observation), dtype=np.float32)

        env = EnvBufferingWrapper(DummyVecEnv([
            lambda: TinyEnv(horizon=1),
            lambda: TinyEnv(horizon=4),
        ]))
        rollout_agent(ZeroPolicy(), env, steps=2)
        trajectories = env.pop_finished_trajectories()

        self.assertEqual(sorted(len(trajectory) for trajectory in trajectories), [1, 4])
        self.assertTrue(env._recording_mask.all())
        env.close()

    def test_maxent_corrected_uses_proposal_log_probability(self):
        algorithm = object.__new__(DemoAlgorithm)
        algorithm.loss_type = "maxent_corrected"
        algorithm.temperature = 1.0
        expert_traj = Trajectory([make_expert_trajectory()[0]])
        model_trajs = [
            Trajectory([Transition(None, None, 0.0, log_policy_prob=np.log(0.9))]),
            Trajectory([Transition(None, None, 0.0, log_policy_prob=np.log(0.1))]),
        ]
        expert_returns = th.tensor([0.0])
        model_returns = th.tensor([0.0, 2.0])
        algorithm._sample_returns = lambda member: (
            expert_returns, model_returns, [expert_traj], model_trajs
        )

        loss = algorithm._reward_loss(member=None)
        expected = th.logsumexp(
            model_returns - th.log(th.tensor([0.9, 0.1])), dim=0
        ) - np.log(2)
        self.assertAlmostEqual(float(loss), float(expected), places=6)

    def test_historical_losses_keep_original_formulas(self):
        algorithm = object.__new__(DemoAlgorithm)
        algorithm.temperature = 7.0
        expert_trajs = [Trajectory([make_expert_trajectory()[0]])]
        model_trajs = [Trajectory([make_expert_trajectory()[0]])]
        expert_returns = th.tensor([3.0])
        model_returns = th.tensor([1.0])
        algorithm._sample_returns = lambda member: (
            expert_returns, model_returns, expert_trajs, model_trajs
        )

        algorithm.loss_type = "maxent"
        self.assertAlmostEqual(float(algorithm._reward_loss(None)), -2.0)

        algorithm.loss_type = "maxent_2"
        expected_maxent_2 = -3.0 + float(th.logsumexp(th.tensor([1.0, 3.0]), 0)) - np.log(2)
        self.assertAlmostEqual(
            float(algorithm._reward_loss(None)), expected_maxent_2, places=6
        )

        for loss_type in ("demo", "demo_loss"):
            algorithm.loss_type = loss_type
            self.assertAlmostEqual(float(algorithm._reward_loss(None)), -2.0)

    def test_reward_transform_does_not_change_training_forward(self):
        env = make_vec_env()
        model = make_reward_ensemble(env, **REWARD_KWARGS)
        obs = np.zeros((2, 2), dtype=np.float32)
        actions = np.zeros((2, 1), dtype=np.float32)
        status = np.tile(np.array([0, 0, 0, 0, 1, 0, 0], dtype=np.float32), (2, 1))
        done = np.zeros(2, dtype=np.float32)
        tensors = model.preprocess(obs, actions, status, done)

        raw_before = model(*tensors).detach().clone()
        model.set_mean(10.0)
        raw_after = model(*tensors).detach().clone()
        self.assertTrue(th.equal(raw_before, raw_after))

        restored = make_reward_ensemble(env, **REWARD_KWARGS)
        restored.load_state_dict(model.state_dict())
        np.testing.assert_allclose(
            model.predict(obs, actions, status, done),
            restored.predict(obs, actions, status, done),
        )

        legacy_state = {
            key: value for key, value in model.state_dict().items()
            if not key.endswith(("_mean", "_std"))
        }
        legacy_restored = make_reward_ensemble(env, **REWARD_KWARGS)
        legacy_restored.load_state_dict(legacy_state)
        env.close()

    def test_ppo_rollout_records_log_prob_and_trains(self):
        train_env, rollout_env = make_vec_env(), make_vec_env()
        agent = PPO(
            "MlpPolicy", train_env, n_steps=5, batch_size=5, n_epochs=1,
            device="cpu", verbose=0,
        )
        algorithm = DemoAlgorithm(
            train_env,
            agent,
            [make_expert_trajectory()],
            rollout_env=rollout_env,
            gradient_steps_rew=1,
            batch_size_expert=1,
            batch_size_model=1,
            reward_model_kwargs=REWARD_KWARGS,
            output_formats=[],
        )

        trajectories = algorithm._sample_rollout(5)
        self.assertTrue(all(t.log_policy_prob is not None for t in trajectories[0]))
        self.assertIn("rollout/action_at_bound_fraction", algorithm.logger.name_to_value)
        algorithm.trajectories = trajectories
        transitions = [t for trajectory in trajectories for t in trajectory]
        algorithm._log_validation_snapshot(transitions, "pre_update")
        self.assertTrue(th.isfinite(algorithm._reward_loss(algorithm.reward_model.members[0])))
        algorithm._train_reward_model()
        algorithm._log_validation_snapshot(transitions, "post_update")
        self.assertIn(
            "reward_val/current_rollout/pre_update/reward_mean",
            algorithm.logger.name_to_value,
        )
        self.assertIn(
            "reward_val/current_rollout/post_update/reward_mean",
            algorithm.logger.name_to_value,
        )
        self.assertIn(
            "reward_val/current_rollout/post_update/gap_arrived_running",
            algorithm.logger.name_to_value,
        )
        self.assertIn(
            "reward/maxent_effective_sample_fraction",
            algorithm.logger.name_to_value,
        )
        algorithm._train_agent(5, 1)
        train_env.close()
        rollout_env.close()

    def test_sac_relabels_actual_replay_samples(self):
        train_env, rollout_env = make_vec_env(), make_vec_env()
        agent = SAC(
            "MlpPolicy",
            train_env,
            replay_buffer_class=RewardRelabelReplayBuffer,
            learning_starts=0,
            batch_size=4,
            buffer_size=50,
            train_freq=1,
            gradient_steps=1,
            device="cpu",
            verbose=0,
        )
        algorithm = DemoAlgorithm(
            train_env,
            agent,
            [make_expert_trajectory()],
            rollout_env=rollout_env,
            gradient_steps_rew=1,
            batch_size_expert=1,
            batch_size_model=1,
            reward_model_kwargs=REWARD_KWARGS,
            output_formats=[],
        )

        algorithm.trajectories = algorithm._sample_rollout(5)
        algorithm._train_reward_model()
        algorithm._train_agent(10, 1)

        replay = agent.replay_buffer
        self.assertGreater(replay.pos, 0)
        stored, current = replay.sample_reward_staleness(4, np.random.default_rng(0))
        self.assertEqual(stored.shape, current.shape)
        self.assertEqual(replay.sample(4).rewards.shape, (4, 1))

        replay.set_relabel_rewards(False)
        stored_sample = replay._get_samples(np.array([0])).rewards.cpu().numpy()
        np.testing.assert_allclose(stored_sample[:, 0], replay.rewards[0])
        algorithm.reward_model.set_mean(10.0)
        replay.set_relabel_rewards(True)
        relabelled_sample = replay._get_samples(np.array([0])).rewards.cpu().numpy()
        self.assertFalse(np.allclose(stored_sample, relabelled_sample))

        replay.set_relabel_rewards(False)
        algorithm._log_replay_reward_staleness(batch_size=4)
        self.assertEqual(
            algorithm.logger.name_to_value["replay_relabel_debug/relabel_enabled"], 0.0
        )
        with TemporaryDirectory() as tmp_dir:
            algorithm._save_checkpoint(tmp_dir, 1)
            checkpoint = Path(tmp_dir) / "checkpoint_0001"
            self.assertTrue((checkpoint / "reward_model.pt").exists())
            self.assertTrue((checkpoint / "reward_training.pt").exists())
            self.assertTrue((checkpoint / "replay_buffer.pkl").exists())
        replay.set_reward_model(None)
        replay.set_relabel_rewards(True)
        with self.assertRaisesRegex(RuntimeError, "attached reward model"):
            replay._get_samples(np.array([0]))
        train_env.close()
        rollout_env.close()


if __name__ == "__main__":
    unittest.main()
