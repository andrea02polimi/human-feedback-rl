"""
Abstract base class for preference oracles.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple


class BaseOracle(ABC):
    @abstractmethod
    def label(self, seg1, seg2) -> Optional[Tuple[float, float]]:
        """Return (p1, p2) with p1+p2=1.0, or None to skip."""
        ...
