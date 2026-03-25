"""
DemoDatabase — circular buffer of (frames, expert_actions, agent_actions) triples.

Separate from PrefDB: demo triples don't carry preference labels.
The action-conditioned margin ranking loss uses:
    L_demo = mean( relu( margin − (RP(obs, a_expert) − RP(obs, a_agent)) ) )
"""

import random
from typing import Iterator, List, Tuple

import numpy as np


class DemoDatabase:
    """
    Circular buffer storing agent-observed frames with paired expert and agent
    actions for the margin ranking loss.

    Each entry is (frames, expert_actions, agent_actions) where:
      - frames         — agent's observations, shape (T, obs_dim)
      - expert_actions — expert's chosen actions for those observations, shape (T,)
      - agent_actions  — agent's actual actions, shape (T,)

    Args:
        maxlen: maximum number of triples to keep; oldest are evicted first.
    """

    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self._pairs: List[Tuple] = []

    def append(self, frames, expert_actions, agent_actions) -> None:
        self._pairs.append((np.asarray(frames), np.asarray(expert_actions), np.asarray(agent_actions)))
        if len(self._pairs) > self.maxlen:
            self._pairs.pop(0)

    def sample(self, batch_size: int) -> List[Tuple]:
        return random.sample(self._pairs, min(batch_size, len(self._pairs)))

    def __len__(self) -> int:
        return len(self._pairs)

    def __iter__(self) -> Iterator[Tuple]:
        return iter(self._pairs)
