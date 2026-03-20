"""
SB3 training callback for the Christiano et al. pipeline.

"""

import numpy as np
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from human_feedback_rl.common.segment import Segment
from human_feedback_rl.common.wrappers import PredictedRewardVecWrapper


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
        self.metrics_queue           = None   # set by _policy_worker after construction
        self.a2c_steps               = None   # set by _policy_worker after construction

        # buffer per costruire segmenti, uno per ogni ambiente
        self.current_segment_frames   = [[] for _ in range(n_envs)]
        self.current_segment_rewards  = [[] for _ in range(n_envs)]
        self.current_segment_actions  = [[] for _ in range(n_envs)]
        self._gradient_step_count     = 0

        # per-episode true reward tracking (env reward, not predicted)
        self._ep_true_reward_accum      = [0.0] * n_envs   # running sum per env
        self._completed_ep_true_rewards = []                # finished episodes this rollout

    def _on_step(self) -> bool:
        if self.shutdown_event.is_set():
            return False

        # Keep shared counter current for preference-worker annealing.
        self.shared_env_steps.value = self.env_steps_offset + self.num_timesteps

        # obs_tensor holds the pre-step observation (set by SB3 before env.step).
        # self.locals è un dizionare interno di SB3 che contiene tutte le variabili locali del loop di training
        obs_tensor = self.locals.get("obs_tensor")
        if obs_tensor is None:
            return True
        obs_np   = obs_tensor.cpu().numpy()                            # (n_envs, obs_dim)
        actions  = self.locals.get("actions")                          # (n_envs,) int
        dones    = self.locals.get("dones", np.zeros(self.n_envs, dtype=bool))
        infos    = self.locals.get("infos", [{} for _ in range(self.n_envs)])

        for env_idx in range(self.n_envs):
            true_reward = float(infos[env_idx].get("true_reward", 0.0))
            self.current_segment_frames[env_idx].append(obs_np[env_idx].copy())
            self.current_segment_rewards[env_idx].append(true_reward)
            if actions is not None:
                self.current_segment_actions[env_idx].append(int(actions[env_idx]))

            self._ep_true_reward_accum[env_idx] += true_reward
            if dones[env_idx]:
                self._completed_ep_true_rewards.append(self._ep_true_reward_accum[env_idx])
                self._ep_true_reward_accum[env_idx] = 0.0

            if (
                len(self.current_segment_frames[env_idx]) >= self.segment_length
                or dones[env_idx]
            ):
                frames  = self.current_segment_frames[env_idx]
                rewards = self.current_segment_rewards[env_idx]
                acts    = self.current_segment_actions[env_idx]
                while len(frames) < self.segment_length:
                    frames.append(frames[-1].copy())
                    rewards.append(0.0)
                    if acts:
                        acts.append(acts[-1])
                seg = Segment(frames[:self.segment_length])
                seg.env_rewards = rewards[:self.segment_length]
                if acts:
                    seg.actions = acts[:self.segment_length]
                try:
                    self.segment_pipe.put(seg, block=False)
                except Exception:
                    pass
                if self.agent_demo_pipe is not None:
                    try:
                        self.agent_demo_pipe.put(seg, block=False)
                    except Exception:
                        pass
                self.current_segment_frames[env_idx]   = []
                self.current_segment_rewards[env_idx]  = []
                self.current_segment_actions[env_idx]  = []

        return True

    def _on_rollout_end(self) -> None:
        """Called once per A2C gradient update (after rollout collection)."""
        self._gradient_step_count += 1

        # Keep the shared A2C step counter in sync (used as wandb X axis).
        if self.a2c_steps is not None:
            self.a2c_steps.value = self.num_timesteps

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
            avg_predicted_reward = float(np.mean(self.model.rollout_buffer.rewards))
            print(
                f"[policy] gradient_step={self._gradient_step_count}"
                f"  env_steps={self.shared_env_steps.value}"
                f"  avg_predicted_reward={avg_predicted_reward:.3f}",
                flush=True,
            )

        # Forward metrics to the main process for wandb logging.
        if self.metrics_queue is not None:
            rollout_rewards = self.model.rollout_buffer.rewards
            metrics = {
                "policy/mean_predicted_rew": float(np.mean(rollout_rewards)),
                "policy/std_predicted_rew":  float(np.std(rollout_rewards)),
            }

            # Episode metrics (available once at least one episode has completed).
            if self.model.ep_info_buffer:
                ep_rews = [info["r"] for info in self.model.ep_info_buffer]
                ep_lens = [info["l"] for info in self.model.ep_info_buffer]
                metrics["policy/mean_episode_avg_rew"]    = float(np.mean(ep_rews))
                metrics["policy/mean_episode_length"]     = float(np.mean(ep_lens))

            # True environment reward per episode (not visible to policy).
            if self._completed_ep_true_rewards:
                metrics["policy/mean_episode_avg_true_rew"] = float(
                    np.mean(self._completed_ep_true_rewards)
                )
                self._completed_ep_true_rewards = []

            # Training loss metrics logged by SB3 (value_loss, policy_gradient_loss, etc.)
            for key, value in self.model.logger.name_to_value.items():
                if key.startswith("train/"):
                    metrics[f"policy/{key}"] = float(value)

            try:
                self.metrics_queue.put_nowait(metrics)
            except Exception:
                pass
