"""
Segment dataclass for trajectory segments used in the RLHF pipeline.

Note: env_rewards is NOT a dataclass field — it is monkey-patched onto
instances by workers when needed.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class Segment:
    frames: List[np.ndarray] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.frames)
