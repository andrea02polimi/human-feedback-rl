"""
SB3-compatible components for the Christiano et al. RLHF pipeline.

  PredictedRewardVecWrapper  — VecEnvWrapper that replaces environment rewards
                               with reward-predictor predictions.  The policy
                               never sees the true environment reward (paper
                               requirement, Section 2.2).  True rewards are
                               forwarded through infos['true_reward'] so the
                               SegmentCollectorCallback can attach them to
                               Segment objects for the preference oracle.

  SegmentCollectorCallback   — SB3 callback that:
                                 1. collects trajectory segments from all envs
                                    and forwards them to segment_pipe
                                 2. reloads the reward predictor checkpoint
                                    every reload_interval gradient updates
                                 3. saves a policy checkpoint every
                                    save_interval gradient updates
                                 4. updates the shared env-steps counter used
                                    by the preference worker for query annealing
"""

import numpy as np
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnvWrapper

from learning_from_human_preferences.preferences.pref_db import Segment
from learning_from_human_preferences.reward_model.reward_predictor import (
    RewardPredictorEnsemble,
)


class PredictedRewardVecWrapper(VecEnvWrapper):
    """
    VecEnv wrapper that replaces environment rewards with reward-predictor
    predictions.

    Before reward_predictor_ready_event is set, returns zero rewards so the
    policy never observes the environment reward (Christiano et al. §2.2).

    True environment rewards are stored in infos['true_reward'] so the
    SegmentCollectorCallback can label each Segment correctly.
    """

    def __init__(self, venv, reward_predictor: RewardPredictorEnsemble,
                 reward_predictor_ready_event):
        super().__init__(venv)
        self.reward_predictor             = reward_predictor
        self.reward_predictor_ready_event = reward_predictor_ready_event

    def reset(self):
        return self.venv.reset()

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, true_rewards, dones, infos = self.venv.step_wait()

        # Pass true reward through info so segments can be labelled correctly.
        for i, info in enumerate(infos):
            info["true_reward"] = float(true_rewards[i])

        if self.reward_predictor_ready_event.is_set():
            predicted = self.reward_predictor.reward(obs)
            rewards   = predicted if np.all(np.isfinite(predicted)) else np.zeros(len(obs))
        else:
            rewards = np.zeros(len(obs), dtype=np.float32)

        return obs, rewards.astype(np.float32), dones, infos

    def reload(self, checkpoint_dir: str) -> None:
        """Load the latest reward predictor checkpoint from disk."""
        latest = RewardPredictorEnsemble.latest_checkpoint(checkpoint_dir)
        if latest:
            try:
                self.reward_predictor.load(latest)
            except Exception:
                pass


class SegmentCollectorCallback(BaseCallback):
    """
    SB3 training callback for the Christiano et al. pipeline.

    _on_step (called after every env step):
      - Appends the pre-step observation and the true env reward (from
        infos['true_reward']) to per-env segment buffers.
      - Emits a Segment (with .env_rewards attached) to segment_pipe when
        the buffer reaches segment_length or the episode ends.
      - Updates shared_env_steps for preference-worker query annealing.
      - Returns False to stop training when shutdown_event is set.

    _on_rollout_end (called once per A2C gradient update):
      - Reloads the reward predictor checkpoint every reload_interval updates.
      - Saves a policy checkpoint every save_interval updates.
    """

    def __init__(
        self,
        segment_pipe,
        segment_length: int,
        n_envs: int,
        shutdown_event,
        reward_predictor_ready_event,
        reward_predictor_wrapper: PredictedRewardVecWrapper,
        reward_predictor_checkpoint_dir: str,
        reload_interval: int,
        save_interval: int,
        policy_checkpoint_path: str,
        shared_env_steps,
        env_steps_offset: int = 0,
        agent_demo_pipe=None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.segment_pipe                    = segment_pipe
        self.segment_length                  = segment_length
        self.n_envs                          = n_envs
        self.shutdown_event                  = shutdown_event
        self.reward_predictor_ready_event    = reward_predictor_ready_event
        self.reward_predictor_wrapper        = reward_predictor_wrapper
        self.reward_predictor_checkpoint_dir = reward_predictor_checkpoint_dir
        self.reload_interval                 = reload_interval
        self.save_interval                   = save_interval
        self.policy_checkpoint_path          = policy_checkpoint_path
        self.shared_env_steps                = shared_env_steps
        self.env_steps_offset                = env_steps_offset

        self.agent_demo_pipe         = agent_demo_pipe

        self.current_segment_frames  = [[] for _ in range(n_envs)]
        self.current_segment_rewards = [[] for _ in range(n_envs)]
        self._gradient_step_count    = 0

    def _on_step(self) -> bool:
        if self.shutdown_event.is_set():
            return False

        # Keep shared counter current for preference-worker annealing.
        self.shared_env_steps.value = self.env_steps_offset + self.num_timesteps

        # obs_tensor holds the pre-step observation (set by SB3 before env.step).
        obs_tensor = self.locals.get("obs_tensor")
        if obs_tensor is None:
            return True
        obs_np = obs_tensor.cpu().numpy()                              # (n_envs, obs_dim)
        dones  = self.locals.get("dones", np.zeros(self.n_envs, dtype=bool))
        infos  = self.locals.get("infos", [{} for _ in range(self.n_envs)])

        for env_idx in range(self.n_envs):
            true_reward = float(infos[env_idx].get("true_reward", 0.0))
            self.current_segment_frames[env_idx].append(obs_np[env_idx].copy())
            self.current_segment_rewards[env_idx].append(true_reward)

            if (
                len(self.current_segment_frames[env_idx]) >= self.segment_length
                or dones[env_idx]
            ):
                frames  = self.current_segment_frames[env_idx]
                rewards = self.current_segment_rewards[env_idx]
                while len(frames) < self.segment_length:
                    frames.append(frames[-1].copy())
                    rewards.append(0.0)
                seg = Segment(frames[:self.segment_length])
                seg.env_rewards = rewards[:self.segment_length]
                try:
                    self.segment_pipe.put(seg, block=False)
                except Exception:
                    pass
                if self.agent_demo_pipe is not None:
                    try:
                        self.agent_demo_pipe.put(seg, block=False)
                    except Exception:
                        pass
                self.current_segment_frames[env_idx]  = []
                self.current_segment_rewards[env_idx] = []

        return True

    def _on_rollout_end(self) -> None:
        """Called once per A2C gradient update (after rollout collection)."""
        self._gradient_step_count += 1

        # Reload the latest reward predictor checkpoint periodically.
        if (
            self.reward_predictor_ready_event.is_set()
            and self._gradient_step_count % self.reload_interval == 0
        ):
            self.reward_predictor_wrapper.reload(self.reward_predictor_checkpoint_dir)

        # Save policy checkpoint periodically.
        if self._gradient_step_count % self.save_interval == 0:
            path = Path(self.policy_checkpoint_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(str(path))
            avg_reward = float(np.mean(self.model.rollout_buffer.rewards))
            print(
                f"[policy] gradient_step={self._gradient_step_count}"
                f"  env_steps={self.shared_env_steps.value}"
                f"  avg_predicted_reward={avg_reward:.3f}",
                flush=True,
            )
