"""
DemoDatabase — circular buffer of (expert_frames, agent_frames) pairs.

Separate from PrefDB: demo pairs don't carry preference labels.
The margin ranking loss only needs the two segments, not a (p1, p2) label.
"""

import random
from typing import Iterator, List, Tuple

import numpy as np


class DemoDatabase:
    """
    Circular buffer storing expert-vs-agent segment pairs for the
    margin ranking loss.

    Each entry is (expert_frames, agent_frames) where frames is a
    list of np.ndarray observations of length segment_len.

    Args:
        maxlen: maximum number of pairs to keep; oldest are evicted first.
    """

    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self._pairs: List[Tuple] = []

    def append(self, expert_frames, agent_frames) -> None:
        self._pairs.append((expert_frames, agent_frames))
        if len(self._pairs) > self.maxlen:
            self._pairs.pop(0)

    def sample(self, batch_size: int) -> List[Tuple]:
        return random.sample(self._pairs, min(batch_size, len(self._pairs)))

    def __len__(self) -> int:
        return len(self._pairs)

    def __iter__(self) -> Iterator[Tuple]:
        return iter(self._pairs)
