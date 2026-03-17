"""
DemonstrationCollector — encapsulates expert segment collection and
expert-vs-agent pair generation.

Extracted from _demonstration_worker in scripts/train_christiano.py.
"""

import queue
import random

import numpy as np

from human_feedback_rl.feedback.segment import Segment


class DemonstrationCollector:
    """
    Manages expert segment collection and pairing with agent segments
    for the demonstration worker subprocess.

    Args:
        segment_length:  number of frames per segment
        max_expert_segs: maximum expert segments to keep (circular buffer)
        num_envs:        number of parallel environments
    """

    def __init__(self, segment_length: int, max_expert_segs: int, num_envs: int):
        self._segment_length = segment_length
        self._max_expert_segs = max_expert_segs
        self._current_frames = [[] for _ in range(num_envs)]
        self._expert_buffer = []

    # ------------------------------------------------------------------

    def process_step(self, obs: np.ndarray, dones: np.ndarray) -> list:
        """
        Append observation frames per env; emit a Segment when the buffer
        reaches segment_length or the episode ends (with padding).

        Returns:
            list of completed Segment objects (may be empty)
        """
        completed = []
        num_envs = obs.shape[0]

        for env_idx in range(num_envs):
            self._current_frames[env_idx].append(obs[env_idx].copy())

            if (
                len(self._current_frames[env_idx]) >= self._segment_length
                or dones[env_idx]
            ):
                frames = self._current_frames[env_idx]
                while len(frames) < self._segment_length:
                    frames.append(frames[-1].copy())
                seg = Segment(frames[: self._segment_length])
                seg.env_rewards = [0.0] * self._segment_length  # unused for demo pairs
                completed.append(seg)
                self._current_frames[env_idx] = []

        return completed

    def add_to_buffer(self, segment: Segment) -> None:
        """Add a segment to the expert buffer (circular, max_expert_segs)."""
        if len(self._expert_buffer) < self._max_expert_segs:
            self._expert_buffer.append(segment)
        else:
            self._expert_buffer[random.randrange(self._max_expert_segs)] = segment

    def has_expert_segments(self) -> bool:
        return bool(self._expert_buffer)

    def try_pair(self, agent_demo_pipe, demo_pipe, max_pairs: int = 8) -> None:
        """
        Drain up to max_pairs agent segments from agent_demo_pipe; for each,
        pick a random expert segment and put (expert_frames, agent_frames)
        into demo_pipe.
        """
        for _ in range(max_pairs):
            try:
                agent_seg = agent_demo_pipe.get(block=False)
            except queue.Empty:
                break
            expert_seg = random.choice(self._expert_buffer)
            try:
                demo_pipe.put(
                    (expert_seg.frames, agent_seg.frames),
                    block=False,
                )
            except Exception:
                pass
