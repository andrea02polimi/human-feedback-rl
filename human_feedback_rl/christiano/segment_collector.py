import numpy as np
import torch
from typing import Tuple

from learning_from_human_preferences.preferences.pref_db import Segment


class SegmentCollector:
    """
    Collects fixed-length trajectory segments from policy rollouts.

    Produces Segment objects whose `.frames` list contains one np.ndarray
    per environment step. Each frame is a flat observation vector (obs_dim,).

    The collector maintains its own state between calls so that segments are
    drawn from a continuous stream of experience rather than resetting the
    environment for every segment.

    Args:
        env:          Gymnasium (or SB3 VecEnv-like) environment
        segment_len:  number of steps per segment (paper default: 25)
    """

    def __init__(self, env, segment_len: int = 25):
        self.env = env
        self.segment_len = segment_len
        self._current_obs = None

    # ------------------------------------------------------------------

    def _reset(self):
        # SB3 VecEnv.reset() returns obs directly (no info tuple)
        obs = self.env.reset()
        self._current_obs = self._flatten_obs(obs)
        # Cache number of parallel envs for step calls
        raw = np.asarray(obs)
        self._n_envs = raw.shape[0] if raw.ndim > 1 else 1

    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_obs(obs) -> np.ndarray:
        """Return a 1-D observation array; handles vec-env batches (takes env 0)."""
        obs = np.asarray(obs)
        if obs.ndim > 1:
            obs = obs[0]
        return obs.copy()

    # ------------------------------------------------------------------

    def collect_segment(self, policy) -> Tuple[Segment, bool]:
        """
        Roll out `segment_len` steps with the given policy and return a Segment.

        Args:
            policy: callable that takes (1, obs_dim) torch tensor → logits

        Returns:
            (segment, episode_ended)
                segment:       Segment with `segment_len` frames
                episode_ended: True if the episode terminated during collection
        """
        if self._current_obs is None:
            self._reset()

        frames = []
        episode_ended = False

        for _ in range(self.segment_len):
            obs = self._current_obs
            frames.append(obs.copy())

            state = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = policy(state)
            action = torch.argmax(logits, dim=1).item()

            # SB3 VecEnv expects an array of actions, one per parallel env
            actions = np.array([action] * self._n_envs)
            # SB3 VecEnv.step() returns (obs, rewards, dones, infos) — 4 values
            next_obs, _, dones, _ = self.env.step(actions)
            done = bool(dones[0]) if hasattr(dones, '__len__') else bool(dones)

            if done:
                episode_ended = True
                self._reset()
            else:
                self._current_obs = self._flatten_obs(next_obs)

        return Segment(frames), episode_ended
