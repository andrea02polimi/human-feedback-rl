"""
Abstract base class for all RLHF training algorithms.
"""

from abc import ABC, abstractmethod



class BaseAlgorithm(ABC):
    """Common interface for all RLHF training algorithms."""

    @abstractmethod
    def train(self, *args, **kwargs):
        """Run the full training pipeline."""
        ...
