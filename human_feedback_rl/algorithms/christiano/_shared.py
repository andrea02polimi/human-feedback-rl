"""
Utility functions shared between ChristianoAlgorithm and ChristianoAlgorithmDemo.

Kept separate to avoid duplicating identical logic in both algorithm files.
"""

from typing import List

import numpy as np
import torch as th

from human_feedback_rl.common.reward_nets import RewardEnsemble


def save_reward_model(reward_model: RewardEnsemble, path) -> None:
    """Save reward ensemble weights and architecture metadata to a checkpoint."""
    hidden_size = int(reward_model.state_dict()["members.0.net.0.weight"].shape[0])
    obs_dim = reward_model.observation_space.shape[0]
    act_dim = reward_model.action_space.shape[0]

    th.save(
        {
            "state_dict":  reward_model.state_dict(),
            "n_members":   len(reward_model.members),
            "obs_dim":     obs_dim,
            "act_dim":     act_dim,
            "hidden_size": hidden_size,
        },
        str(path),
    )
    print(f"Saved reward model → {path}")


def collect_debug_data(trajectory_generator, reward_model: RewardEnsemble, n_steps: int = 2000) -> dict:
    """
    Roll out the current policy and return per-step true vs predicted rewards.

    Returns a dict ready for notebook plotting:
        trajectories          — list of per-episode dicts with obs/actions/rewards
        episode_true_returns  — list[float]
        episode_pred_returns  — list[float]
    """
    from human_feedback_rl.common.trajectory_generators import rollout_agent

    algo_venv = trajectory_generator.agent.get_env()
    assert algo_venv is not None

    trajectory_generator.buffering_wrapper.pop_finished_trajectories()

    rollout_agent(
        policy=trajectory_generator.agent,
        venv=algo_venv,
        steps=n_steps,
        deterministic_policy=True,
    )
    trajs = trajectory_generator.buffering_wrapper.pop_finished_trajectories()

    reward_model.eval()
    result: dict = {
        "trajectories":          [],
        "episode_true_returns":  [],
        "episode_pred_returns":  [],
    }

    for traj in trajs:
        obs       = np.array([t.observation for t in traj], dtype=np.float32)
        acts      = np.array([t.action      for t in traj], dtype=np.float32)
        true_rews = np.array([t.true_reward  for t in traj], dtype=np.float32)

        pred_mean, pred_std = reward_model.predict_mean_std(obs, acts)

        result["trajectories"].append(
            {
                "obs":               obs,
                "actions":           acts,
                "true_rewards":      true_rews,
                "pred_rewards_mean": pred_mean.astype(np.float32),
                "pred_rewards_std":  pred_std.astype(np.float32),
            }
        )
        result["episode_true_returns"].append(float(true_rews.sum()))
        result["episode_pred_returns"].append(float(pred_mean.sum()))

    reward_model.train()
    return result


def compute_time_decay_weights(train_data: list, timestamp_idx: int, decay: float) -> np.ndarray:
    """Exponential time-decay weights — more recent items receive higher weight."""
    t_vals = np.array([item[timestamp_idx] for item in train_data], dtype=np.float32)
    t_norm = (t_vals - t_vals.min()) / (t_vals.max() - t_vals.min() + 1e-8)
    return np.exp(decay * t_norm)


def build_bootstrap_indices(
    rng: np.random.Generator, n_train: int, n_members: int
) -> List[np.ndarray]:
    """One bootstrap sample (with replacement) per ensemble member — standard bagging."""
    return [rng.choice(n_train, size=n_train, replace=True) for _ in range(n_members)]