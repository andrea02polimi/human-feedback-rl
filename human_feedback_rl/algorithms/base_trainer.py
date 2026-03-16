"""
Abstract base class for all RLHF training algorithms.
"""

from abc import ABC, abstractmethod

from omegaconf import DictConfig


class BaseTrainer(ABC):
    """Common interface for all RLHF training algorithms."""

    @abstractmethod
    def train(self, cfg: DictConfig) -> None:
        """Run the full training pipeline."""
        ...
